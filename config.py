"""
Configuration for LitCVit reproduction on 3 datasets:
  - USTC-TFC2016 (pretraining + fine-tuning)
  - 5GAD-2022    (fine-tuning only)
  - TII-SSRC-23  (fine-tuning only, CSV-based)
"""
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)   # …/New paper

# ── Dataset roots ────────────────────────────────────────────────────────────
USTC_DIR   = os.path.join(PARENT_DIR, "USTC-TFC2016-master", "USTC-TFC2016-master")
GAD5_DIR   = os.path.join(PARENT_DIR, "5GAD-main", "5GAD-main")
TIISSRC_CSV = os.path.join(PARENT_DIR, "TIISSRC23Dataset", "csv", "data.csv")

# ── Processed data output ─────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── ET-flow image parameters (Section IV-A of paper) ─────────────────────────
M_PACKETS    = 5     # packets per flow
P_ROWS       = 8     # rows per packet (bytes reshaped)
W_COLS       = 40    # bytes per row
IMG_H        = M_PACKETS * P_ROWS   # 40
IMG_W        = W_COLS               # 40
N_BYTES_PKT  = P_ROWS * W_COLS      # 320 bytes per packet

# ── Model hyperparameters (Section VI-B / Table II) ──────────────────────────
EMB_DIM      = 192     # d=192 best performance; use 128 for efficiency
RANK         = 16      # low-rank factorization rank r
BYTE_WIN     = 8       # byte-level attention window
PACKET_WIN   = 5       # packet-level attention window
DROPOUT      = 0.2     # classifier dropout
N_HEADS      = 4

# ── Pre-training hyperparameters ─────────────────────────────────────────────
PRETRAIN_EPOCHS  = 200
PRETRAIN_LR      = 3e-4
PRETRAIN_BS      = 128
PRETRAIN_WARMUP  = 10
TEMPERATURE      = 0.3      # τ = 0.3 (best from ablation)

# ── Fine-tuning hyperparameters ───────────────────────────────────────────────
FINETUNE_EPOCHS  = 500
FINETUNE_LR      = 1e-3
FINETUNE_BS      = 64
UNFREEZE_EPOCH   = 100      # unfreeze last 2 encoder blocks after epoch 100
FINETUNE_LR_HEAD = 1e-3
FINETUNE_LR_ENC  = 1e-4    # lower LR for unfrozen encoder blocks

# ── Evaluation ────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20
RANDOM_SEED = 42

# ── Device ───────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Dataset-specific settings ─────────────────────────────────────────────────
DATASET_CONFIGS = {
    "USTC-TFC2016": {
        "type": "pcap",
        "path": USTC_DIR,
        "n_classes": 20,      # 10 benign + 10 malware families
        "binary": False,      # multi-class
        "pretrain_eligible": True,
        "classes": [
            # benign
            "BitTorrent", "Facetime", "FTP", "Gmail", "MySQL",
            "Outlook", "Skype", "SMB", "Weibo", "WorldOfWarcraft",
            # malware
            "Cridex", "Geodo", "Htbot", "Miuref", "Neris",
            "Nsis-ay", "Shifu", "Tinba", "Virut", "Zeus",
        ],
        "benign_classes": [
            "BitTorrent", "Facetime", "FTP", "Gmail", "MySQL",
            "Outlook", "Skype", "SMB", "Weibo", "WorldOfWarcraft",
        ],
    },
    "5GAD-2022": {
        "type": "pcap",
        "path": GAD5_DIR,
        "n_classes": 11,      # normal + 10 attacks
        "binary": False,
        "pretrain_eligible": False,
        "classes": [
            "Normal",
            "AMFLookingForUDM", "automatedDropWithTimer",
            "automatedRedirectWithTimer", "CrashNRF",
            "FakeAMFDelete", "FakeAMFInsert", "GetAllNFs",
            "GetUserData", "randomAMFInsert", "randomDataDump",
        ],
    },
    "TII-SSRC-23": {
        "type": "csv",
        "path": TIISSRC_CSV,
        "n_classes": None,     # set dynamically
        "binary": False,
        "pretrain_eligible": False,
        "label_col": "Traffic Type",
        "numeric_cols_start": "Flow Duration",  # first numeric col
        "max_rows": 500_000,   # cap for memory
    },
}

PRETRAIN_DATASETS = ["USTC-TFC2016"]
FINETUNE_DATASETS = ["USTC-TFC2016", "5GAD-2022", "TII-SSRC-23"]
