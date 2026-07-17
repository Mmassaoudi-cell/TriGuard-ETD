"""Provenance-preserving PCAP extraction for USTC-TFC2016 and 5GAD-2022.

Unlike the legacy extractor, this module retains a source file and a canonical
five-tuple per generated flow, making a flow-disjoint split auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import py7zr
from scapy.all import IP, TCP, UDP, PcapReader

from data.preprocess_pcap import flow_to_image
from .data import create_split_bundle, fit_train_only_features

USTC_CLASSES = ["BitTorrent", "Facetime", "FTP", "Gmail", "MySQL", "Outlook", "Skype", "SMB", "Weibo", "WorldOfWarcraft",
                "Cridex", "Geodo", "Htbot", "Miuref", "Neris", "Nsis-ay", "Shifu", "Tinba", "Virut", "Zeus"]


def _capture_is_lfs_pointer(path: Path) -> bool:
    try:
        return path.stat().st_size < 1024 and path.read_text(encoding="utf-8", errors="ignore").startswith("version https://git-lfs.github.com/spec")
    except OSError:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_manifest(paths: list[Path]) -> list[dict]:
    """Size/hash/pointer audit for every selected raw capture."""
    records = []
    for path in sorted(paths):
        records.append({
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
            "git_lfs_pointer": bool(_capture_is_lfs_pointer(path)),
        })
    return records


def extract_ustc_archives(root: Path) -> list[Path]:
    """Extract USTC's locally available 7z files in place, once."""
    extracted = []
    for archive in root.rglob("*.7z"):
        marker = archive.with_suffix(".extracted.json")
        if not marker.exists():
            with py7zr.SevenZipFile(archive, "r") as z:
                z.extractall(path=archive.parent)
            marker.write_text(json.dumps({"archive": str(archive), "status": "extracted"}, indent=2), encoding="utf-8")
        extracted.append(archive)
    return extracted


def _key(packet) -> tuple | None:
    if IP not in packet:
        return None
    ip = packet[IP]
    src_port = int(getattr(packet, "sport", 0)); dst_port = int(getattr(packet, "dport", 0))
    a, b = (ip.src, src_port), (ip.dst, dst_port)
    return (min(a, b), max(a, b), int(ip.proto))


class _FlowAccumulator:
    """Bounded-memory per-flow accumulator.

    Full packet bytes are retained only for the flow's first `head_cap`
    packets (enough for the ET-flow image and the five-step timing sequence,
    per `flow_to_image`'s own `M_PACKETS` truncation). Every later packet only
    updates running scalar statistics, so a flow that spans an entire
    multi-gigabyte capture (e.g. a persistent 5G control-plane session) cannot
    make memory grow with total flow duration -- only with the number of
    distinct flows. Packets are assumed to arrive in capture (chronological)
    order, which PcapReader guarantees for a well-formed capture.
    """

    __slots__ = ("head", "head_cap", "count", "byte_sum", "size_min", "size_max", "size_sq_sum",
                "first_time", "last_time", "prev_time", "iat_sum", "iat_sq_sum", "iat_max",
                "pos_direction_count", "flag_sum", "first_port")

    def __init__(self, head_cap: int = 5):
        self.head_cap = head_cap
        self.head: list[tuple[float, bytes, int, int]] = []
        self.count = 0
        self.byte_sum = 0.0
        self.size_min = math.inf
        self.size_max = -math.inf
        self.size_sq_sum = 0.0
        self.first_time: float | None = None
        self.last_time: float | None = None
        self.prev_time: float | None = None
        self.iat_sum = 0.0
        self.iat_sq_sum = 0.0
        self.iat_max = 0.0
        self.pos_direction_count = 0
        self.flag_sum = 0
        self.first_port: int | None = None

    def add(self, t: float, raw: bytes, port: int, flags: int) -> None:
        size = len(raw)
        if self.count == 0:
            self.first_time = t
            self.first_port = port
        if len(self.head) < self.head_cap:
            self.head.append((t, raw, port, flags))
        self.count += 1
        self.byte_sum += size
        self.size_sq_sum += size * size
        self.size_min = min(self.size_min, size)
        self.size_max = max(self.size_max, size)
        if self.prev_time is not None:
            iat = max(t - self.prev_time, 0.0)
            self.iat_sum += iat
            self.iat_sq_sum += iat * iat
            self.iat_max = max(self.iat_max, iat)
        self.prev_time = t
        self.last_time = t
        if port == self.first_port:
            self.pos_direction_count += 1
        self.flag_sum += flags


def _flow_views(acc: "_FlowAccumulator", max_packets: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ET image, five packet sequence features, and direct flow stats
    computed over the *complete* flow from a bounded-memory accumulator."""
    raw_packets = [h[1] for h in acc.head]
    image = flow_to_image(raw_packets)
    seq = np.zeros((max_packets, 5), dtype=np.float32)
    for i, (t, raw, port, flags) in enumerate(acc.head[:max_packets]):
        iat = 0.0 if i == 0 else max(t - acc.head[i - 1][0], 0.0)
        direction = 1.0 if port == acc.first_port else -1.0
        seq[i] = [len(raw), iat, direction, flags, i / max_packets]
    count = acc.count
    size_mean = acc.byte_sum / count if count else 0.0
    size_var = max(acc.size_sq_sum / count - size_mean ** 2, 0.0) if count else 0.0
    duration = (acc.last_time - acc.first_time) if acc.first_time is not None else 0.0
    n_iat = max(count - 1, 0)
    iat_mean = acc.iat_sum / n_iat if n_iat else 0.0
    iat_var = max(acc.iat_sq_sum / n_iat - iat_mean ** 2, 0.0) if n_iat else 0.0
    direction_ratio = acc.pos_direction_count / count if count else 0.0
    stats = np.asarray([count, acc.byte_sum, duration, size_mean, math.sqrt(size_var),
                        acc.size_min if count else 0.0, acc.size_max if count else 0.0,
                        iat_mean, math.sqrt(iat_var), acc.iat_max, direction_ratio, acc.flag_sum], dtype=np.float32)
    return image, seq, stats


def _read_capture(path: Path, max_flows: int, packet_cap: int | None = None,
                  max_packets_read: int | None = None, head_cap: int = 5,
                  max_session_duration: float | None = None,
                  inactivity_timeout: float | None = None) -> dict[tuple, "_FlowAccumulator"]:
    """Read a capture into flow/session accumulators.

    By default, the unit is the source capture plus canonical bidirectional
    five-tuple.  Some 5GAD attacks are one long UDP five-tuple within one
    capture, making source-file/five-tuple train/validation/test support
    impossible for those classes.  When explicitly requested, this function
    creates temporal segments inside a five-tuple using a maximum duration or
    inactivity boundary.  That weaker protocol is recorded in metadata and must
    not be described as source-file-disjoint.
    """
    flows: dict[tuple, _FlowAccumulator] = {}
    active: dict[tuple, tuple] = {}
    segment_counter: dict[tuple, int] = {}
    sessionized = bool(max_session_duration or inactivity_timeout)

    def make_flow_key(base_key: tuple) -> tuple:
        if not sessionized:
            return base_key
        idx = segment_counter.get(base_key, 0)
        segment_counter[base_key] = idx + 1
        return (base_key, f"segment{idx:05d}")

    with PcapReader(str(path)) as reader:
        for packet_number, packet in enumerate(reader, start=1):
            if max_packets_read and packet_number > max_packets_read:
                break
            base_key = _key(packet)
            if base_key is None:
                continue
            key = active.get(base_key, base_key)
            acc = flows.get(key)
            if packet_cap and acc is not None and acc.count >= packet_cap:
                if sessionized:
                    acc = None
                else:
                    if len(flows) >= max_flows and all(v.count >= packet_cap for v in flows.values()):
                        break
                    continue
            if sessionized and acc is not None:
                t = float(packet.time)
                if max_session_duration and acc.first_time is not None and (t - acc.first_time) >= max_session_duration:
                    acc = None
                elif inactivity_timeout and acc.last_time is not None and (t - acc.last_time) > inactivity_timeout:
                    acc = None
            source_port = int(getattr(packet, "sport", 0))
            flags = int(packet[TCP].flags) if TCP in packet else 0
            if acc is None:
                if len(flows) >= max_flows:
                    continue
                key = make_flow_key(base_key)
                acc = _FlowAccumulator(head_cap=max(head_cap, 1))
                flows[key] = acc
                active[base_key] = key
            acc.add(float(packet.time), bytes(packet), source_port, flags)
    return flows


def prepare_ustc(root: str, output_dir: str, max_flows_per_class: int = 1500, seed: int = 42, extract_archives: bool = True,
                 packet_cap: int | None = None, max_packets_read: int | None = None) -> None:
    root_path = Path(root)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if extract_archives:
        extract_ustc_archives(root_path)
    class_files: dict[str, list[Path]] = {name: [] for name in USTC_CLASSES}
    for pcap in list(root_path.rglob("*.pcap")) + list(root_path.rglob("*.pcapng")):
        for label in USTC_CLASSES:
            if pcap.stem.lower().startswith(label.lower()) or label.lower() in {part.lower() for part in pcap.parts}:
                class_files[label].append(pcap); break
    images, timing, statistics, labels, groups = [], [], [], [], []
    available = [label for label in USTC_CLASSES if class_files[label]]
    mapping = {label: i for i, label in enumerate(available)}
    for label in available:
        remaining = max_flows_per_class
        for pcap in sorted(class_files[label]):
            if remaining <= 0: break
            flows = _read_capture(pcap, remaining, packet_cap, max_packets_read)
            for key, records in flows.items():
                image, seq, stats = _flow_views(records)
                images.append(image); timing.append(seq); statistics.append(stats); labels.append(mapping[label])
                groups.append(f"{pcap.resolve()}::{key}")
                remaining -= 1
                if remaining <= 0: break
    if not labels:
        raise RuntimeError("No USTC flows could be extracted")
    images = np.asarray(images, dtype=np.uint8); timing = np.asarray(timing, dtype=np.float32); statistics = np.asarray(statistics, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64); groups = np.asarray(groups)
    split = create_split_bundle(labels, groups, statistics, available, seed=seed)
    stats_scaled, stats_meta = fit_train_only_features(statistics, split)
    timing_scaled, timing_meta = fit_train_only_features(timing.reshape(len(timing), -1), split); timing_scaled = timing_scaled.reshape(timing.shape)
    mask = np.ones((len(labels), 3), dtype=np.bool_)
    np.savez_compressed(out / "ustc_multiview.npz", image=images, timing=timing_scaled, statistics=stats_scaled, labels=labels, groups=groups,
                        view_mask=mask, classes=np.asarray(available, dtype=object))
    split.save(out / "ustc_split", "USTC-TFC2016", seed)
    selected_files = [p for label in available for p in class_files[label]]
    metadata = {"dataset": "USTC-TFC2016", "classes_available": available, "class_files": {x: [str(p) for p in class_files[x]] for x in available},
                "capture_manifest": _capture_manifest(selected_files),
                "rows": int(len(labels)), "max_flows_per_class": max_flows_per_class, "packet_cap_per_flow": packet_cap, "max_packets_read_per_capture": max_packets_read,
                "flow_group": "absolute source PCAP path + bidirectional canonical 5-tuple",
                "preprocessing": {"statistics": stats_meta, "timing": timing_meta}, "raw_pcap_used": True}
    (out / "ustc_preprocessing.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(labels), "classes": available, "split": {x: len(getattr(split, x)) for x in ("train", "val", "test")}}, indent=2))


def validate_5gad(root: str) -> dict:
    files = list(Path(root).rglob("*.pcap")) + list(Path(root).rglob("*.pcapng"))
    pointers = [str(p) for p in files if _capture_is_lfs_pointer(p)]
    return {"captures": len(files), "git_lfs_pointers": len(pointers), "usable": bool(files) and not pointers,
            "message": "Fetch Git-LFS objects before preprocessing 5GAD." if pointers else "capture files appear usable", "examples": pointers[:5]}


GAD5_CLASSES = ["Normal", "AMFLookingForUDM", "automatedDropWithTimer",
                "automatedRedirectWithTimer", "CrashNRF", "FakeAMFDelete",
                "FakeAMFInsert", "GetAllNFs", "GetUserData", "randomAMFInsert",
                "randomDataDump"]


def prepare_5gad(root: str, output_dir: str, max_flows_per_class: int = 1500,
                 seed: int = 42, packet_cap: int | None = None,
                 max_packets_read: int | None = None,
                 max_session_duration: float | None = None,
                 inactivity_timeout: float | None = None) -> None:
    """Build a 5GAD cache from non-duplicated all-interface capture views."""
    root_path = Path(root)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    audit = validate_5gad(str(root_path))
    if not audit["usable"]:
        raise RuntimeError(audit["message"])
    class_files: dict[str, list[Path]] = {"Normal": sorted(
        [p for p in list((root_path / "Normal-1UE").glob("allcap*.pcapng")) +
         list((root_path / "Normal-2UE").glob("allcap*.pcapng")) if not _capture_is_lfs_pointer(p)])}
    for label in GAD5_CLASSES[1:]:
        candidates = sorted((root_path / "Attacks" / label).glob("Attacks_*.pcapng"))
        class_files[label] = [p for p in candidates if not _capture_is_lfs_pointer(p)]
    missing = [label for label in GAD5_CLASSES if not class_files.get(label)]
    if missing:
        raise RuntimeError(f"5GAD classes without usable canonical captures: {missing}")
    images, timing, statistics, labels, groups = [], [], [], [], []
    for class_id, label in enumerate(GAD5_CLASSES):
        remaining = max_flows_per_class
        for pcap in class_files[label]:
            if remaining <= 0: break
            flows = _read_capture(pcap, remaining, packet_cap, max_packets_read,
                                  max_session_duration=max_session_duration,
                                  inactivity_timeout=inactivity_timeout)
            for key, records in flows.items():
                image, seq, stats = _flow_views(records)
                images.append(image); timing.append(seq); statistics.append(stats); labels.append(class_id)
                groups.append(f"{pcap.resolve()}::{key}")
                remaining -= 1
                if remaining <= 0: break
    images = np.asarray(images, dtype=np.uint8); timing = np.asarray(timing, dtype=np.float32)
    statistics = np.asarray(statistics, dtype=np.float32); labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    split = create_split_bundle(labels, groups, statistics, GAD5_CLASSES, seed=seed)
    stats_scaled, stats_meta = fit_train_only_features(statistics, split)
    timing_scaled, timing_meta = fit_train_only_features(timing.reshape(len(timing), -1), split)
    mask = np.ones((len(labels), 3), dtype=np.bool_)
    np.savez_compressed(out / "5gad_multiview.npz", image=images,
                        timing=timing_scaled.reshape(timing.shape),
                        statistics=stats_scaled, labels=labels, groups=groups,
                        view_mask=mask, classes=np.asarray(GAD5_CLASSES, dtype=object))
    split.save(out / "5gad_split", "5GAD-2022", seed)
    selected_files = [p for label in GAD5_CLASSES for p in class_files[label]]
    metadata = {"dataset": "5GAD-2022", "rows": int(len(labels)),
                "classes": GAD5_CLASSES, "class_files": {x: [str(p) for p in class_files[x]] for x in GAD5_CLASSES},
                "capture_manifest": _capture_manifest(selected_files),
                "flow_group": ("absolute source PCAP path + bidirectional canonical five-tuple + temporal segment"
                               if (max_session_duration or inactivity_timeout)
                               else "absolute source PCAP path + bidirectional canonical five-tuple"),
                "packet_cap_per_flow": packet_cap, "max_packets_read_per_capture": max_packets_read,
                "max_session_duration_seconds": max_session_duration,
                "inactivity_timeout_seconds": inactivity_timeout,
                "source_file_disjoint": False if (max_session_duration or inactivity_timeout) else None,
                "preprocessing": {"statistics": stats_meta, "timing": timing_meta}, "raw_pcap_used": True,
                "selection": "Attack-specific captures and allcap normal captures only; redundant interface views excluded."}
    (out / "5gad_preprocessing.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(labels), "classes": GAD5_CLASSES,
                      "split": {x: len(getattr(split, x)) for x in ("train", "val", "test")}}, indent=2))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="dataset", required=True)
    u = sub.add_parser("ustc"); u.add_argument("--root", required=True); u.add_argument("--output-dir", default="artifacts/ustc_safe"); u.add_argument("--max-flows-per-class", type=int, default=1500); u.add_argument("--packet-cap", type=int, default=0, help="Optional per-flow packet cap; 0 reads complete flows"); u.add_argument("--max-packets-read", type=int, default=0, help="Optional capture-read cap for smoke tests; 0 reads complete captures"); u.add_argument("--seed", type=int, default=42); u.add_argument("--no-extract", action="store_true")
    gfull = sub.add_parser("5gad"); gfull.add_argument("--root", required=True); gfull.add_argument("--output-dir", default="artifacts/5gad_safe"); gfull.add_argument("--max-flows-per-class", type=int, default=1500); gfull.add_argument("--packet-cap", type=int, default=0); gfull.add_argument("--max-packets-read", type=int, default=0); gfull.add_argument("--seed", type=int, default=42); gfull.add_argument("--max-session-duration", type=float, default=0.0, help="Optional temporal segment duration in seconds for long 5GAD five-tuples; 0 disables"); gfull.add_argument("--inactivity-timeout", type=float, default=0.0, help="Optional temporal segmentation by inactivity gap in seconds; 0 disables")
    g = sub.add_parser("5gad-audit"); g.add_argument("--root", required=True)
    args = parser.parse_args()
    if args.dataset == "ustc": prepare_ustc(args.root, args.output_dir, args.max_flows_per_class, args.seed, not args.no_extract, args.packet_cap or None, args.max_packets_read or None)
    elif args.dataset == "5gad": prepare_5gad(args.root, args.output_dir, args.max_flows_per_class, args.seed, args.packet_cap or None, args.max_packets_read or None, args.max_session_duration or None, args.inactivity_timeout or None)
    else: print(json.dumps(validate_5gad(args.root), indent=2))
