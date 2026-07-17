"""
PyTorch Dataset classes for LitCVit experiments.

Three dataset variants:
  1. ETFlowDataset      – labelled images for fine-tuning & evaluation
  2. UnlabeledETFlow    – unlabelled images for self-supervised pre-training
                          (returns two augmented views of the same image)
  3. ETFlowSplit        – utility that wraps a numpy array with train/val/test splits
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    IMG_H, IMG_W, PROCESSED_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
)

# ─────────────────────────────────────────────────────────────────────────────
# Augmentations  (Section IV-B: data augmentations for contrastive learning)
# ─────────────────────────────────────────────────────────────────────────────

def random_byte_noise(img: torch.Tensor, noise_std: float = 5.0) -> torch.Tensor:
    """Add Gaussian noise to pixel values (simulates byte perturbations)."""
    noise = torch.randn_like(img) * noise_std
    return (img + noise).clamp(0, 255)

def random_crop_resize(img: torch.Tensor, min_ratio: float = 0.8) -> torch.Tensor:
    """Random crop + resize back to original size."""
    h, w = img.shape[-2:]
    ch   = int(h * (min_ratio + (1 - min_ratio) * torch.rand(1).item()))
    cw   = int(w * (min_ratio + (1 - min_ratio) * torch.rand(1).item()))
    r0   = torch.randint(0, h - ch + 1, (1,)).item()
    c0   = torch.randint(0, w - cw + 1, (1,)).item()
    cropped = img[:, r0:r0+ch, c0:c0+cw]
    return torch.nn.functional.interpolate(
        cropped.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
    ).squeeze(0)

def random_horizontal_flip(img: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    if torch.rand(1).item() < p:
        return img.flip(-1)
    return img

def augment(img: torch.Tensor) -> torch.Tensor:
    """
    Compose augmentations for contrastive pre-training.
    Input: (1, H, W) float tensor in [0, 255].

    Wrapped in no_grad: augmentation never needs gradients, and building an
    autograd graph for every tiny per-sample op was pure overhead multiplied
    across the whole dataset every epoch.
    """
    with torch.no_grad():
        img = random_byte_noise(img)
        img = random_crop_resize(img)
        img = random_horizontal_flip(img)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Labelled ET-flow Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ETFlowDataset(Dataset):
    """
    Returns (image_tensor, label) pairs for supervised training / evaluation.
    Images are normalised to [0, 1] as float32 tensors of shape (1, H, W).
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 augment_train: bool = False):
        assert len(images) == len(labels)
        self.images = images.astype(np.float32) / 255.0   # (N, H, W)
        self.labels = labels.astype(np.int64)
        self.augment = augment_train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.images[idx]).unsqueeze(0)  # (1, H, W)
        if self.augment:
            img = augment(img * 255.0) / 255.0
        return img, self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Unlabelled ET-flow Dataset (two views for contrastive pre-training)
# ─────────────────────────────────────────────────────────────────────────────

class UnlabeledETFlow(Dataset):
    """
    Returns (view_1, view_2) – two differently augmented views of the same image.
    Used during self-supervised pre-training with ntXent loss.
    """

    def __init__(self, images: np.ndarray):
        self.images = images.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.images[idx]).unsqueeze(0)  # (1, H, W)
        v1  = augment(img * 255.0) / 255.0
        v2  = augment(img * 255.0) / 255.0
        return v1, v2


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Train / Val / Test split utility
# ─────────────────────────────────────────────────────────────────────────────

class ETFlowSplit:
    """
    Flow-aware random split of pre-processed ET-flow image arrays.
    Ensures no cross-split leakage at the sample level.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 train_ratio: float = TRAIN_RATIO,
                 val_ratio:   float = VAL_RATIO,
                 seed: int          = RANDOM_SEED):
        N = len(images)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)

        n_train = int(N * train_ratio)
        n_val   = int(N * val_ratio)

        tr_idx  = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        te_idx  = idx[n_train + n_val:]

        self.train_images, self.train_labels = images[tr_idx], labels[tr_idx]
        self.val_images,   self.val_labels   = images[val_idx], labels[val_idx]
        self.test_images,  self.test_labels  = images[te_idx],  labels[te_idx]

        print(f"  Split sizes -> train: {len(tr_idx):,}  "
              f"val: {len(val_idx):,}  test: {len(te_idx):,}")

    def get_datasets(self, augment_train: bool = True):
        tr = ETFlowDataset(self.train_images, self.train_labels, augment_train)
        va = ETFlowDataset(self.val_images,   self.val_labels,   False)
        te = ETFlowDataset(self.test_images,  self.test_labels,  False)
        return tr, va, te

    def get_loaders(self, batch_size: int = 64, num_workers: int = 0,
                    class_balanced: bool = True):
        tr_ds, va_ds, te_ds = self.get_datasets(augment_train=True)

        if class_balanced:
            counts  = np.bincount(self.train_labels)
            weights = 1.0 / counts[self.train_labels]
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(weights).float(),
                num_samples=len(tr_ds), replacement=True
            )
            tr_loader = DataLoader(tr_ds, batch_size=batch_size,
                                   sampler=sampler, num_workers=num_workers,
                                   pin_memory=True)
        else:
            tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=True)

        va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
        te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
        return tr_loader, va_loader, te_loader


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: load pre-processed .npy files
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(name: str, processed_dir: str = PROCESSED_DIR):
    """
    Load a pre-processed dataset from .npy files.

    Args:
        name: "ustc", "5gad", or "tiissrc23"
    Returns:
        images (N, H, W), labels (N,), class_names list
    """
    prefixes = {
        "ustc":     "ustc_tfc2016",
        "5gad":     "5gad_2022",
        "tiissrc23":"tiissrc23",
    }
    prefix = prefixes.get(name)
    if prefix is None:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(prefixes)}")

    img_path = os.path.join(processed_dir, f"{prefix}_images.npy")
    lbl_path = os.path.join(processed_dir, f"{prefix}_labels.npy")
    cls_path = os.path.join(processed_dir, f"{prefix}_label_map.npy")

    if not os.path.exists(img_path):
        raise FileNotFoundError(
            f"Pre-processed data not found at '{img_path}'.\n"
            f"Run the appropriate preprocessing script first."
        )

    images = np.load(img_path)
    labels = np.load(lbl_path)
    classes = np.load(cls_path, allow_pickle=True).tolist()
    print(f"[Dataset] Loaded '{name}': {images.shape[0]:,} samples, "
          f"{len(classes)} classes.")
    return images, labels, classes


def get_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights for weighted cross-entropy loss."""
    n_classes = labels.max() + 1
    counts    = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts    = np.where(counts == 0, 1.0, counts)
    weights   = 1.0 / counts
    weights   = weights / weights.sum() * n_classes     # normalise
    return torch.from_numpy(weights).float()
