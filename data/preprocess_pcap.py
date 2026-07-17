"""
Pre-processing Phase (Section IV-A of paper):
  Raw PCAP/pcapng -> ET-Flow Images (shape: IMG_H × IMG_W = 40 × 40)

For each flow:
  1. Extract up to M=5 packets (zero-pad if fewer).
  2. Remove Ethernet header; anonymise IP/port.
  3. Take first N_BYTES_PKT=320 bytes of each packet (zero-pad if shorter).
  4. Reshape packet bytes into (P_ROWS=8, W_COLS=40) matrix.
  5. Stack M packet matrices vertically -> (40, 40) ET-Flow image.
  6. Save as numpy uint8 array.

Usage:
    python preprocess_pcap.py  --dataset ustc   --out_dir processed_data/ustc
    python preprocess_pcap.py  --dataset 5gad   --out_dir processed_data/5gad
"""

import os
import sys
import glob
import struct
import binascii
import argparse
import hashlib
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    M_PACKETS, P_ROWS, W_COLS, N_BYTES_PKT, IMG_H, IMG_W,
    USTC_DIR, GAD5_DIR, PROCESSED_DIR, DATASET_CONFIGS
)

try:
    from scapy.all import PcapReader, Ether, IP, TCP, UDP, Raw
    SCAPY_OK = True
except ImportError:
    SCAPY_OK = False
    print("[WARNING] scapy not available – PCAP preprocessing will fail.")


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_ethernet(pkt_bytes: bytes) -> bytes:
    """Remove 14-byte Ethernet header if present; else return unchanged."""
    if len(pkt_bytes) >= 14:
        # Simple heuristic: check EtherType for IPv4 (0x0800)
        ethertype = struct.unpack(">H", pkt_bytes[12:14])[0]
        if ethertype == 0x0800:
            return pkt_bytes[14:]
    return pkt_bytes


def _anonymise(pkt_bytes: bytes) -> bytes:
    """
    Zero-out IP src/dst (bytes 12-19 of IP header) and TCP/UDP ports (0-3).
    Operates on the payload after Ethernet stripping.
    """
    ba = bytearray(pkt_bytes)
    # IP header: version/IHL at byte 0; src IP at [12:16], dst IP at [16:20]
    if len(ba) >= 20:
        ba[12:16] = b'\x00' * 4
        ba[16:20] = b'\x00' * 4
    # TCP/UDP ports: first 4 bytes of transport header
    ip_ihl = (ba[0] & 0x0F) * 4 if len(ba) >= 1 else 20
    if len(ba) >= ip_ihl + 4:
        ba[ip_ihl:ip_ihl + 4] = b'\x00' * 4
    return bytes(ba)


def packet_to_array(pkt_bytes: bytes) -> np.ndarray:
    """
    Convert raw packet bytes to a (P_ROWS, W_COLS) uint8 array.
    Truncate or zero-pad to N_BYTES_PKT bytes, then reshape.
    """
    raw = _anonymise(_strip_ethernet(pkt_bytes))
    # Truncate or pad to N_BYTES_PKT bytes
    if len(raw) >= N_BYTES_PKT:
        raw = raw[:N_BYTES_PKT]
    else:
        raw = raw + b'\x00' * (N_BYTES_PKT - len(raw))
    return np.frombuffer(raw, dtype=np.uint8).reshape(P_ROWS, W_COLS)


def flow_to_image(packets: list) -> np.ndarray:
    """
    Convert a list of raw packet byte-strings to a (IMG_H, IMG_W) uint8 image.
    Pads with zero matrices if fewer than M_PACKETS packets.
    """
    arrays = []
    for pkt in packets[:M_PACKETS]:
        arrays.append(packet_to_array(pkt))
    # Zero-pad missing packets
    while len(arrays) < M_PACKETS:
        arrays.append(np.zeros((P_ROWS, W_COLS), dtype=np.uint8))
    return np.vstack(arrays)   # (40, 40)


def _flow_key(pkt) -> tuple:
    """5-tuple flow key; direction-agnostic (sorted src/dst)."""
    try:
        src = (pkt[IP].src, pkt.sport if hasattr(pkt, 'sport') else 0)
        dst = (pkt[IP].dst, pkt.dport if hasattr(pkt, 'dport') else 0)
        proto = pkt[IP].proto
        return (min(src, dst), max(src, dst), proto)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PCAP -> ET-flow images
# ─────────────────────────────────────────────────────────────────────────────

def pcap_to_flows(pcap_path: str, max_flows: int = 5000) -> list:
    """
    Parse a pcap/pcapng file and return a list of (flow_id, packets) where
    packets is a list of raw bytes for packets in that flow.
    Only flows with ≥1 Raw payload packet are retained.
    """
    flows = defaultdict(list)
    try:
        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                key = _flow_key(pkt)
                if key is None:
                    continue
                if len(flows) >= max_flows and key not in flows:
                    continue
                # Get raw bytes (full packet from layer 2)
                try:
                    raw_bytes = bytes(pkt)
                    if len(raw_bytes) > 14:  # at least Ethernet + something
                        flows[key].append(raw_bytes)
                except Exception:
                    pass
    except Exception as e:
        print(f"[WARNING] Could not read {pcap_path}: {e}")
    # Filter flows that have payload
    result = [(k, v) for k, v in flows.items() if v]
    return result


def process_pcap_to_images(pcap_path: str, label: int, label_name: str,
                            max_flows: int = 3000) -> tuple:
    """
    Returns (images, labels) as numpy arrays.
    images: (N, IMG_H, IMG_W) uint8
    labels: (N,) int
    """
    flows = pcap_to_flows(pcap_path, max_flows)
    images, labels = [], []
    for _, pkts in flows:
        img = flow_to_image(pkts)
        images.append(img)
        labels.append(label)
    if not images:
        return np.empty((0, IMG_H, IMG_W), dtype=np.uint8), np.empty(0, dtype=np.int64)
    return np.stack(images, axis=0), np.array(labels, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# USTC-TFC2016 processor
# ─────────────────────────────────────────────────────────────────────────────

def process_ustc(out_dir: str, max_flows_per_class: int = 3000):
    """Process USTC-TFC2016 (Benign/ + Malware/ PCAP files)."""
    os.makedirs(out_dir, exist_ok=True)
    cfg    = DATASET_CONFIGS["USTC-TFC2016"]
    classes = cfg["classes"]
    label_map = {c: i for i, c in enumerate(classes)}

    all_images, all_labels = [], []
    print("\n[USTC-TFC2016] Processing PCAP files...")

    # Helper to find pcap for a class name
    def find_pcap(name):
        for sub in ["Benign", "Malware"]:
            p = os.path.join(USTC_DIR, sub, f"{name}.pcap")
            if os.path.exists(p):
                return p
            # Check .pcapng
            p2 = os.path.join(USTC_DIR, sub, f"{name}.pcapng")
            if os.path.exists(p2):
                return p2
        return None

    for cls_name in tqdm(classes, desc="USTC classes"):
        pcap_path = find_pcap(cls_name)
        if pcap_path is None:
            print(f"  [SKIP] {cls_name}: no PCAP found (may need to extract .7z)")
            continue
        lbl = label_map[cls_name]
        imgs, lbls = process_pcap_to_images(pcap_path, lbl, cls_name,
                                            max_flows_per_class)
        if imgs.shape[0] == 0:
            print(f"  [EMPTY] {cls_name}")
            continue
        all_images.append(imgs)
        all_labels.append(lbls)
        print(f"  {cls_name}: {imgs.shape[0]} flows")

    if not all_images:
        print("[ERROR] No flows extracted. Check PCAP file paths.")
        return

    X = np.concatenate(all_images, axis=0)
    y = np.concatenate(all_labels, axis=0)

    out = os.path.join(out_dir, "ustc_tfc2016")
    np.save(out + "_images.npy", X)
    np.save(out + "_labels.npy", y)
    np.save(out + "_label_map.npy", np.array(classes))
    print(f"\n[USTC-TFC2016] Saved {X.shape[0]} ET-flow images -> {out_dir}")
    return X, y, classes


# ─────────────────────────────────────────────────────────────────────────────
# 5GAD-2022 processor
# ─────────────────────────────────────────────────────────────────────────────

def process_5gad(out_dir: str, max_flows_per_class: int = 3000):
    """Process 5GAD-2022 (Normal-1UE/, Normal-2UE/, Attacks/)."""
    os.makedirs(out_dir, exist_ok=True)
    cfg    = DATASET_CONFIGS["5GAD-2022"]
    classes = cfg["classes"]
    label_map = {c: i for i, c in enumerate(classes)}

    all_images, all_labels = [], []
    print("\n[5GAD-2022] Processing PCAP/pcapng files...")

    # Normal traffic (class 0)
    normal_label = label_map["Normal"]
    for sub in ["Normal-1UE", "Normal-2UE"]:
        folder = os.path.join(GAD5_DIR, sub)
        pcaps  = glob.glob(os.path.join(folder, "*.pcap*"))
        for p in tqdm(pcaps, desc=f"5GAD {sub}", leave=False):
            imgs, lbls = process_pcap_to_images(
                p, normal_label, "Normal",
                max_flows=max_flows_per_class // max(len(pcaps), 1)
            )
            if imgs.shape[0]:
                all_images.append(imgs)
                all_labels.append(lbls)

    # Attack traffic
    attacks_dir = os.path.join(GAD5_DIR, "Attacks")
    for atk_name in tqdm(classes[1:], desc="5GAD attacks"):
        atk_dir = os.path.join(attacks_dir, atk_name)
        if not os.path.isdir(atk_dir):
            print(f"  [SKIP] {atk_name}: directory not found")
            continue
        pcaps = (glob.glob(os.path.join(atk_dir, "*.pcap")) +
                 glob.glob(os.path.join(atk_dir, "*.pcapng")))
        if not pcaps:
            print(f"  [SKIP] {atk_name}: no PCAP files")
            continue
        lbl = label_map[atk_name]
        for p in pcaps:
            imgs, lbls = process_pcap_to_images(
                p, lbl, atk_name,
                max_flows=max_flows_per_class // max(len(pcaps), 1)
            )
            if imgs.shape[0]:
                all_images.append(imgs)
                all_labels.append(lbls)

    if not all_images:
        print("[ERROR] No flows extracted for 5GAD.")
        return

    X = np.concatenate(all_images, axis=0)
    y = np.concatenate(all_labels, axis=0)

    out = os.path.join(out_dir, "5gad_2022")
    np.save(out + "_images.npy", X)
    np.save(out + "_labels.npy", y)
    np.save(out + "_label_map.npy", np.array(classes))
    print(f"\n[5GAD-2022] Saved {X.shape[0]} ET-flow images -> {out_dir}")
    return X, y, classes


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCAP -> ET-flow images")
    parser.add_argument("--dataset", choices=["ustc", "5gad", "all"],
                        default="all", help="Which dataset to process")
    parser.add_argument("--out_dir", default=PROCESSED_DIR,
                        help="Output directory for .npy files")
    parser.add_argument("--max_flows", type=int, default=3000,
                        help="Max flows per class")
    args = parser.parse_args()

    if not SCAPY_OK:
        print("[ERROR] scapy is not installed. Run: pip install scapy")
        sys.exit(1)

    if args.dataset in ("ustc", "all"):
        process_ustc(args.out_dir, args.max_flows)

    if args.dataset in ("5gad", "all"):
        process_5gad(args.out_dir, args.max_flows)
