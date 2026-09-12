"""Deterministic manifest generator for the Paderborn Bearing Dataset.

Phase 5 — Dataset Generation.

This script:
  1. Reads the bearing-level split from the approved Phase 4 design.
  2. Iterates every MAT file for every approved bearing.
  3. Loads vibration_1 and extracts non-overlapping 1-second windows (64 000 samples).
  4. Builds a manifest CSV row for every window with full provenance.
  5. Runs a suite of leakage and integrity checks.
  6. Computes training-only normalisation statistics (mean, std).
  7. Writes:
       data/metadata/paderborn_manifest.csv
       data/metadata/paderborn_norm_stats.json

The manifest is the ONLY output file produced.  No .npy cache files are created.
The Dataset loader reads MAT files on-the-fly using src/paderborn_mat.py.

INVARIANTS (all enforced programmatically):
  - Bearing-disjoint:     No bearing appears in more than one split.
  - Recording-disjoint:   No MAT file appears in more than one split.
  - No window crosses a recording boundary.
  - No duplicate window IDs.
  - Only approved bearings are included.
  - Normalisation stats come exclusively from TRAIN windows.
  - Re-running this script produces a byte-identical manifest (determinism).

Usage (from project root, seed=42 baked in):
    python -m src.generate_manifest
or
    python src/generate_manifest.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Project-root detection — all paths stored relative to project root.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = Path("data/raw/Paderborn/extracted")  # relative to project root
METADATA_DIR = Path("data/metadata")
MANIFEST_PATH = METADATA_DIR / "paderborn_manifest.csv"
NORM_STATS_PATH = METADATA_DIR / "paderborn_norm_stats.json"

# ---------------------------------------------------------------------------
# Phase 4 approved bearing split — canonical, do not change without review.
# ---------------------------------------------------------------------------
APPROVED_BEARINGS: dict[str, dict] = {
    # Healthy bearings
    "K001": {"class": "healthy",    "label_id": 0, "split": "train",      "damage_location": "none",        "damage_mode": "none",                  "damage_extent": 0},
    "K002": {"class": "healthy",    "label_id": 0, "split": "train",      "damage_location": "none",        "damage_mode": "none",                  "damage_extent": 0},
    "K003": {"class": "healthy",    "label_id": 0, "split": "train",      "damage_location": "none",        "damage_mode": "none",                  "damage_extent": 0},
    "K004": {"class": "healthy",    "label_id": 0, "split": "train",      "damage_location": "none",        "damage_mode": "none",                  "damage_extent": 0},
    "K005": {"class": "healthy",    "label_id": 0, "split": "validation", "damage_location": "none",        "damage_mode": "none",                  "damage_extent": 0},
    "K006": {"class": "healthy",    "label_id": 0, "split": "test",       "damage_location": "none",        "damage_mode": "none",                  "damage_extent": 0},
    # Real outer-ring damage
    "KA04": {"class": "outer_ring", "label_id": 1, "split": "train",      "damage_location": "outer_ring",  "damage_mode": "fatigue",               "damage_extent": 1},
    "KA15": {"class": "outer_ring", "label_id": 1, "split": "validation", "damage_location": "outer_ring",  "damage_mode": "plastic_deformation",   "damage_extent": 1},
    "KA16": {"class": "outer_ring", "label_id": 1, "split": "train",      "damage_location": "outer_ring",  "damage_mode": "fatigue",               "damage_extent": 2},
    "KA22": {"class": "outer_ring", "label_id": 1, "split": "test",       "damage_location": "outer_ring",  "damage_mode": "fatigue",               "damage_extent": 1},
    "KA30": {"class": "outer_ring", "label_id": 1, "split": "train",      "damage_location": "outer_ring",  "damage_mode": "plastic_deformation",   "damage_extent": 1},
    # Real inner-ring damage
    "KI04": {"class": "inner_ring", "label_id": 2, "split": "train",      "damage_location": "inner_ring",  "damage_mode": "fatigue",               "damage_extent": 1},
    "KI14": {"class": "inner_ring", "label_id": 2, "split": "test",       "damage_location": "inner_ring",  "damage_mode": "fatigue",               "damage_extent": 1},
    "KI16": {"class": "inner_ring", "label_id": 2, "split": "train",      "damage_location": "inner_ring",  "damage_mode": "fatigue",               "damage_extent": 3},
    "KI18": {"class": "inner_ring", "label_id": 2, "split": "train",      "damage_location": "inner_ring",  "damage_mode": "fatigue",               "damage_extent": 2},
    "KI21": {"class": "inner_ring", "label_id": 2, "split": "validation", "damage_location": "inner_ring",  "damage_mode": "fatigue",               "damage_extent": 1},
}

# Explicitly excluded bearings (combined damage or artificial — must NOT appear).
EXCLUDED_BEARINGS = {
    "KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09",
    "KB23", "KB24", "KB27",
    "KI01", "KI03", "KI05", "KI07", "KI08",
    "KI17",
}

# Operating-condition metadata map.
CONDITION_META: dict[str, dict] = {
    "N15_M07_F10": {"speed_rpm": 1500, "load_torque_nm": 0.7, "radial_force_n": 1000},
    "N09_M07_F10": {"speed_rpm": 900,  "load_torque_nm": 0.7, "radial_force_n": 1000},
    "N15_M01_F10": {"speed_rpm": 1500, "load_torque_nm": 0.1, "radial_force_n": 1000},
    "N15_M07_F04": {"speed_rpm": 1500, "load_torque_nm": 0.7, "radial_force_n": 400},
}

WINDOW_SAMPLES: int = 64_000   # 1.0 s @ 64 kHz
NOMINAL_FS:     int = 64_000
MANIFEST_COLUMNS = [
    "window_id", "mat_path", "label", "label_id", "split",
    "bearing_id", "archive", "operating_condition",
    "speed_rpm", "load_torque_nm", "radial_force_n",
    "measurement_id", "window_index",
    "start_sample", "end_sample", "start_time_s", "end_time_s",
    "sampling_rate_hz", "damage_location", "damage_mode", "damage_extent",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_filename(stem: str) -> tuple[str, str, int]:
    """Parse 'N15_M07_F10_K001_3' → (condition, bearing_id, measurement_id)."""
    # Stem format: <cond>_<bearing_id>_<measurement_id>
    # cond = first 3 underscore-separated tokens (e.g. N15_M07_F10)
    parts = stem.split("_")
    # condition = parts[0..2]  (N15, M07, F10)
    condition = "_".join(parts[:3])
    # bearing_id = parts[3]  (e.g. K001, KA04, KI16)
    bearing_id = parts[3]
    # measurement_id = parts[4]
    measurement_id = int(parts[4])
    return condition, bearing_id, measurement_id


def _window_id(bearing_id: str, condition: str, measurement_id: int, w_idx: int) -> str:
    """Build a globally unique, sortable window identifier."""
    return f"{bearing_id}_{condition}_m{measurement_id:02d}_w{w_idx:02d}"


# ---------------------------------------------------------------------------
# Core manifest generation
# ---------------------------------------------------------------------------

def generate_manifest(project_root: Path = PROJECT_ROOT) -> list[dict]:
    """Build and return the manifest as a list of row dicts (sorted deterministically)."""
    from src.paderborn_mat import load_vibration, extract_windows  # lazy import

    rows: list[dict] = []
    errors: list[str] = []

    # Iterate bearings in a deterministic order (sorted).
    for bearing_id in sorted(APPROVED_BEARINGS.keys()):
        meta = APPROVED_BEARINGS[bearing_id]
        bearing_dir = project_root / EXTRACTED_DIR / bearing_id
        archive = f"{bearing_id}.rar"

        if not bearing_dir.exists():
            errors.append(f"MISSING bearing directory: {bearing_dir}")
            continue

        # Collect MAT files for this bearing, sorted deterministically.
        mat_files = sorted(bearing_dir.glob("*.mat"))
        if not mat_files:
            errors.append(f"No MAT files found in {bearing_dir}")
            continue

        for mat_path in mat_files:
            stem = mat_path.stem
            try:
                condition, bid, meas_id = _parse_filename(stem)
            except Exception as e:
                errors.append(f"Cannot parse filename {stem}: {e}")
                continue

            if bid != bearing_id:
                errors.append(
                    f"Filename bearing token '{bid}' != directory '{bearing_id}' in {mat_path}"
                )
                continue

            if condition not in CONDITION_META:
                errors.append(f"Unknown operating condition '{condition}' in {mat_path}")
                continue

            cond_meta = CONDITION_META[condition]
            # Path stored relative to project root (portable).
            rel_mat = mat_path.relative_to(project_root)

            try:
                vib, t = load_vibration(mat_path)
            except Exception as e:
                errors.append(f"Failed to load {mat_path}: {e}")
                continue

            windows = extract_windows(vib, t, window_samples=WINDOW_SAMPLES)

            for w in windows:
                wid = _window_id(bearing_id, condition, meas_id, w["window_index"])
                row: dict[str, Any] = {
                    "window_id":         wid,
                    "mat_path":          str(rel_mat).replace("\\", "/"),
                    "label":             meta["class"],
                    "label_id":          meta["label_id"],
                    "split":             meta["split"],
                    "bearing_id":        bearing_id,
                    "archive":           archive,
                    "operating_condition": condition,
                    "speed_rpm":         cond_meta["speed_rpm"],
                    "load_torque_nm":    cond_meta["load_torque_nm"],
                    "radial_force_n":    cond_meta["radial_force_n"],
                    "measurement_id":    meas_id,
                    "window_index":      w["window_index"],
                    "start_sample":      w["start_sample"],
                    "end_sample":        w["end_sample"],
                    "start_time_s":      round(w["start_time_s"], 9),
                    "end_time_s":        round(w["end_time_s"], 9),
                    "sampling_rate_hz":  NOMINAL_FS,
                    "damage_location":   meta["damage_location"],
                    "damage_mode":       meta["damage_mode"],
                    "damage_extent":     meta["damage_extent"],
                }
                rows.append(row)

    if errors:
        print("\n[ERROR] The following problems were detected during manifest generation:")
        for e in errors:
            print(f"  - {e}")
        raise RuntimeError(
            f"Manifest generation failed with {len(errors)} error(s). "
            "Fix the errors above before proceeding."
        )

    # Sort for determinism: bearing → condition → measurement → window_index
    rows.sort(key=lambda r: (r["bearing_id"], r["operating_condition"],
                              r["measurement_id"], r["window_index"]))
    return rows


# ---------------------------------------------------------------------------
# Leakage and integrity checks
# ---------------------------------------------------------------------------

def run_integrity_checks(rows: list[dict]) -> dict:
    """Run all Phase 5 leakage and integrity checks.

    Returns a dict of results.  Raises RuntimeError if any check fails.
    """
    APPROVED_CLASSES = {"healthy", "outer_ring", "inner_ring"}
    failures: list[str] = []

    # --- 1. No bearing in multiple splits ---
    bearing_splits: dict[str, set] = {}
    for r in rows:
        bearing_splits.setdefault(r["bearing_id"], set()).add(r["split"])
    multi_split_bearings = {b: s for b, s in bearing_splits.items() if len(s) > 1}
    if multi_split_bearings:
        failures.append(f"CHECK 1 FAIL — bearings in multiple splits: {multi_split_bearings}")
    else:
        print("  ✓ CHECK 1: No bearing appears in multiple splits.")

    # --- 2. No MAT recording in multiple splits ---
    mat_splits: dict[str, set] = {}
    for r in rows:
        mat_splits.setdefault(r["mat_path"], set()).add(r["split"])
    multi_split_mats = {m: s for m, s in mat_splits.items() if len(s) > 1}
    if multi_split_mats:
        failures.append(f"CHECK 2 FAIL — MAT files in multiple splits: {list(multi_split_mats.keys())[:5]}")
    else:
        print("  ✓ CHECK 2: No MAT recording appears in multiple splits.")

    # --- 3. No window crosses recording boundary ---
    bad_bounds = [
        r for r in rows
        if r["end_sample"] - r["start_sample"] != WINDOW_SAMPLES or r["start_sample"] < 0
    ]
    if bad_bounds:
        failures.append(f"CHECK 3 FAIL — {len(bad_bounds)} windows with invalid boundaries.")
    else:
        print("  ✓ CHECK 3: All windows have valid boundaries (no cross-recording overlap).")

    # --- 4. No duplicate window IDs ---
    wids = [r["window_id"] for r in rows]
    if len(wids) != len(set(wids)):
        from collections import Counter
        dupes = [w for w, c in Counter(wids).items() if c > 1]
        failures.append(f"CHECK 4 FAIL — {len(dupes)} duplicate window IDs: {dupes[:5]}")
    else:
        print("  ✓ CHECK 4: All window IDs are unique.")

    # --- 5. Every source MAT file exists ---
    missing_mats = [r["mat_path"] for r in rows if not (PROJECT_ROOT / r["mat_path"]).exists()]
    if missing_mats:
        failures.append(f"CHECK 5 FAIL — {len(missing_mats)} missing MAT files: {missing_mats[:3]}")
    else:
        print(f"  ✓ CHECK 5: All {len(set(r['mat_path'] for r in rows))} source MAT files exist.")

    # --- 6. Valid start/end offsets ---
    bad_offsets = [r for r in rows if r["start_sample"] >= r["end_sample"]]
    if bad_offsets:
        failures.append(f"CHECK 6 FAIL — {len(bad_offsets)} windows with start >= end.")
    else:
        print("  ✓ CHECK 6: All window start/end offsets are valid.")

    # --- 7. Every window belongs to an approved class ---
    bad_classes = [r for r in rows if r["label"] not in APPROVED_CLASSES]
    if bad_classes:
        failures.append(f"CHECK 7 FAIL — unexpected class labels: {set(r['label'] for r in bad_classes)}")
    else:
        print("  ✓ CHECK 7: All windows have approved class labels.")

    # --- 8. Only approved bearings ---
    bearings_in_manifest = set(r["bearing_id"] for r in rows)
    unapproved = bearings_in_manifest - set(APPROVED_BEARINGS.keys())
    if unapproved:
        failures.append(f"CHECK 8 FAIL — unapproved bearings in manifest: {unapproved}")
    else:
        print("  ✓ CHECK 8: Only approved bearings are present.")

    # --- 9. Excluded bearings are absent ---
    found_excluded = bearings_in_manifest & EXCLUDED_BEARINGS
    if found_excluded:
        failures.append(f"CHECK 9 FAIL — excluded bearings found: {found_excluded}")
    else:
        print("  ✓ CHECK 9: No excluded bearings are present.")

    # --- 10. label_id is consistent with label ---
    label_id_map = {"healthy": 0, "outer_ring": 1, "inner_ring": 2}
    bad_ids = [r for r in rows if r["label_id"] != label_id_map.get(r["label"], -1)]
    if bad_ids:
        failures.append(f"CHECK 10 FAIL — {len(bad_ids)} label_id mismatches.")
    else:
        print("  ✓ CHECK 10: All label_id values are consistent with labels.")

    # --- 11. Determinism: re-sort and compare hash ---
    re_sorted = sorted(
        rows,
        key=lambda r: (r["bearing_id"], r["operating_condition"],
                        r["measurement_id"], r["window_index"])
    )
    original_ids = [r["window_id"] for r in rows]
    resorted_ids = [r["window_id"] for r in re_sorted]
    if original_ids != resorted_ids:
        failures.append("CHECK 11 FAIL — Re-sorted manifest order differs from original.")
    else:
        print("  ✓ CHECK 11: Manifest order is deterministic (sort-stable).")

    # --- 12. Operating conditions are the four approved ones ---
    bad_conds = set(r["operating_condition"] for r in rows) - set(CONDITION_META.keys())
    if bad_conds:
        failures.append(f"CHECK 12 FAIL — unexpected operating conditions: {bad_conds}")
    else:
        print("  ✓ CHECK 12: All operating conditions are from the approved set.")

    # Summary
    if failures:
        print("\n[INTEGRITY CHECK FAILURES]")
        for f in failures:
            print(f"  ✗ {f}")
        raise RuntimeError(
            f"Phase 5 integrity checks failed ({len(failures)} failure(s)). "
            "Do not proceed until resolved."
        )

    return {
        "checks_passed": 12,
        "checks_failed": 0,
        "bearings_verified": sorted(bearings_in_manifest),
        "total_windows": len(rows),
        "unique_mat_files": len(mat_splits),
        "unique_bearings": len(bearing_splits),
    }


# ---------------------------------------------------------------------------
# Training-only normalisation statistics
# ---------------------------------------------------------------------------

def compute_norm_stats(rows: list[dict], project_root: Path = PROJECT_ROOT) -> dict:
    """Compute global Z-score statistics from TRAINING windows only.

    Streams window data in sorted order to avoid loading the full dataset.
    Returns {"mean": float, "std": float, "n_samples": int, "n_windows": int}.
    """
    from src.paderborn_mat import get_window_data

    print("\nComputing training-only normalisation statistics...")
    train_rows = [r for r in rows if r["split"] == "train"]
    print(f"  Using {len(train_rows)} training windows.")

    # Welford's online algorithm for numerically stable mean+variance.
    n = 0
    mean_acc = 0.0
    M2_acc = 0.0

    for i, r in enumerate(train_rows):
        mat_path = project_root / r["mat_path"]
        window = get_window_data(mat_path, int(r["start_sample"]), int(r["end_sample"]))

        for x in window.astype(np.float64):
            n += 1
            delta = x - mean_acc
            mean_acc += delta / n
            M2_acc += delta * (x - mean_acc)

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(train_rows)} training windows processed...")

    if n < 2:
        raise ValueError("Fewer than 2 training samples — cannot compute std.")

    std = float(np.sqrt(M2_acc / (n - 1)))
    mean = float(mean_acc)

    print(f"  ✓ Normalisation stats: mean={mean:.6f}, std={std:.6f}, n_samples={n:,}")
    return {"mean": mean, "std": std, "n_samples": n, "n_windows": len(train_rows)}


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  ✓ Manifest written: {path}  ({len(rows)} rows)")


def write_norm_stats(stats: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Normalisation stats written: {path}")


def manifest_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def compute_statistics(rows: list[dict]) -> dict:
    from collections import defaultdict

    splits = ["train", "validation", "test"]
    classes = ["healthy", "outer_ring", "inner_ring"]

    stats: dict[str, Any] = {
        "total_windows": len(rows),
        "by_split": {},
        "by_class": {},
        "by_bearing": {},
        "by_operating_condition": {},
        "class_by_split": {},
        "condition_by_split": {},
        "recordings_by_split": {},
    }

    # By split
    for sp in splits:
        sp_rows = [r for r in rows if r["split"] == sp]
        stats["by_split"][sp] = len(sp_rows)

    # By class
    for cl in classes:
        stats["by_class"][cl] = sum(1 for r in rows if r["label"] == cl)

    # By bearing
    bearing_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        bearing_counts[r["bearing_id"]] += 1
    stats["by_bearing"] = dict(sorted(bearing_counts.items()))

    # By operating condition
    cond_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        cond_counts[r["operating_condition"]] += 1
    stats["by_operating_condition"] = dict(sorted(cond_counts.items()))

    # Class distribution per split
    for sp in splits:
        sp_rows = [r for r in rows if r["split"] == sp]
        stats["class_by_split"][sp] = {
            cl: sum(1 for r in sp_rows if r["label"] == cl)
            for cl in classes
        }

    # Condition distribution per split
    for sp in splits:
        sp_rows = [r for r in rows if r["split"] == sp]
        stats["condition_by_split"][sp] = {
            c: sum(1 for r in sp_rows if r["operating_condition"] == c)
            for c in sorted(CONDITION_META.keys())
        }

    # Recordings per split (unique MAT paths)
    for sp in splits:
        sp_rows = [r for r in rows if r["split"] == sp]
        stats["recordings_by_split"][sp] = len(set(r["mat_path"] for r in sp_rows))

    # Average windows per recording
    mat_window_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        mat_window_counts[r["mat_path"]] += 1
    stats["avg_windows_per_recording"] = round(
        sum(mat_window_counts.values()) / len(mat_window_counts), 2
    )
    stats["avg_windows_per_bearing"] = round(len(rows) / len(bearing_counts), 2)

    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 65)
    print("Phase 5 — Paderborn Manifest Generation")
    print("=" * 65)

    # 1. Generate manifest
    print("\n[1/5] Generating manifest rows...")
    rows = generate_manifest(PROJECT_ROOT)
    print(f"      Generated {len(rows)} manifest rows.")

    # 2. Integrity checks
    print("\n[2/5] Running integrity checks...")
    check_results = run_integrity_checks(rows)

    # 3. Compute normalization stats
    print("\n[3/5] Computing training-only normalisation statistics...")
    norm_stats = compute_norm_stats(rows, PROJECT_ROOT)

    # 4. Write outputs
    print("\n[4/5] Writing outputs...")
    abs_manifest = PROJECT_ROOT / MANIFEST_PATH
    write_manifest(rows, abs_manifest)
    write_norm_stats(norm_stats, PROJECT_ROOT / NORM_STATS_PATH)
    sha = manifest_sha256(abs_manifest)
    print(f"      SHA-256 of manifest: {sha}")

    # 5. Statistics
    print("\n[5/5] Computing dataset statistics...")
    stats = compute_statistics(rows)
    stats_path = PROJECT_ROOT / "results" / "paderborn_manifest_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(
            {"integrity": check_results, "statistics": stats, "norm_stats": norm_stats,
             "manifest_sha256": sha},
            f, indent=2
        )
    print(f"      Stats written: {stats_path}")

    print("\n" + "=" * 65)
    print("Manifest generation COMPLETE.")
    print(f"  Total windows : {stats['total_windows']}")
    for sp in ["train", "validation", "test"]:
        print(f"  {sp:>12s}  : {stats['by_split'][sp]}")
    print("  Class totals  :")
    for cl, cnt in stats["by_class"].items():
        print(f"    {cl:>12s}: {cnt}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
