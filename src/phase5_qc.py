"""Phase 5 QC — Quick quality-control inspection of extracted windows.

Verifies that the manifest and MAT-file extraction pipeline work correctly
by loading representative windows from each class and condition.

Outputs:
  results/qc_windows.json   — numerical summary of inspected windows.
  results/qc_waveforms.png  — small waveform grid (one row per class).

Usage (from project root):
    python -m src.phase5_qc
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.paderborn_mat import get_window_data, WINDOW_SAMPLES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "metadata" / "paderborn_manifest.csv"
NORM_STATS_PATH = PROJECT_ROOT / "data" / "metadata" / "paderborn_norm_stats.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_manifest() -> list[dict]:
    import csv
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_norm() -> tuple[float, float]:
    with open(NORM_STATS_PATH) as f:
        ns = json.load(f)
    return float(ns["mean"]), float(ns["std"])


def _get_window(row: dict) -> np.ndarray:
    mat_path = PROJECT_ROOT / row["mat_path"]
    return get_window_data(mat_path, int(row["start_sample"]), int(row["end_sample"]))


# ---------------------------------------------------------------------------
# Main QC logic
# ---------------------------------------------------------------------------

def pick_representative_rows(rows: list[dict]) -> list[dict]:
    """Select ~10 representative rows covering classes, conditions, and outliers."""
    selected = []

    # One from each class × each split combination
    classes = ["healthy", "outer_ring", "inner_ring"]
    splits = ["train", "validation", "test"]
    for cls in classes:
        for sp in splits:
            for r in rows:
                if r["label"] == cls and r["split"] == sp:
                    selected.append(r)
                    break  # first match per (class, split)

    # Add the known KI16 duration-outlier recordings
    for r in rows:
        if r["bearing_id"] == "KI16" and "KI16_4" in r["mat_path"]:
            selected.append(r)
            break
    for r in rows:
        if r["bearing_id"] == "KI16" and "KI16_9" in r["mat_path"]:
            selected.append(r)
            break

    # Deduplicate by window_id
    seen = set()
    deduped = []
    for r in selected:
        if r["window_id"] not in seen:
            seen.add(r["window_id"])
            deduped.append(r)

    return deduped


def inspect_window(row: dict, mean: float, std: float) -> dict:
    """Load one window, check shape, compute basic stats."""
    result: dict = {
        "window_id":     row["window_id"],
        "bearing_id":    row["bearing_id"],
        "label":         row["label"],
        "split":         row["split"],
        "mat_path":      row["mat_path"],
        "condition":     row["operating_condition"],
        "start_sample":  int(row["start_sample"]),
        "end_sample":    int(row["end_sample"]),
        "n_samples_expected": WINDOW_SAMPLES,
    }
    try:
        x = _get_window(row)
        x_norm = (x - mean) / max(std, 1e-8)

        result["status"] = "ok"
        result["n_samples_actual"] = len(x)
        result["shape_ok"] = len(x) == WINDOW_SAMPLES
        result["raw_mean"] = round(float(x.mean()), 6)
        result["raw_std"] = round(float(x.std()), 6)
        result["raw_rms"] = round(float(np.sqrt(np.mean(x ** 2))), 6)
        result["raw_min"] = round(float(x.min()), 6)
        result["raw_max"] = round(float(x.max()), 6)
        result["norm_mean"] = round(float(x_norm.mean()), 6)
        result["norm_std"] = round(float(x_norm.std()), 6)
        result["has_nan"] = bool(np.any(np.isnan(x)))
        result["has_inf"] = bool(np.any(np.isinf(x)))
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result


# ---------------------------------------------------------------------------
# Waveform plot
# ---------------------------------------------------------------------------

def plot_qc_waveforms(selected_rows: list[dict], inspection: list[dict], mean: float, std: float) -> None:
    """Create a compact waveform grid and save to results/qc_waveforms.png."""
    out_path = PROJECT_ROOT / "results" / "qc_waveforms.png"

    # Group by class
    classes = ["healthy", "outer_ring", "inner_ring"]
    class_rows: dict[str, list] = {c: [] for c in classes}
    for row, insp in zip(selected_rows, inspection):
        if insp["status"] == "ok" and row["label"] in classes:
            class_rows[row["label"]].append((row, insp))

    n_cols = max(len(v) for v in class_rows.values())
    n_rows = len(classes)

    fig, axes = plt.subplots(n_rows, max(n_cols, 1), figsize=(4 * max(n_cols, 1), 3 * n_rows),
                             squeeze=False)
    fig.suptitle("QC: Representative Windows per Class (Phase 5)", fontsize=12, y=1.01)

    cmap = {"healthy": "#2ecc71", "outer_ring": "#e74c3c", "inner_ring": "#3498db"}
    PLOT_SAMPLES = 4096  # show first 64 ms

    for row_idx, cls in enumerate(classes):
        pairs = class_rows[cls]
        for col_idx in range(max(n_cols, 1)):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor("#111")
            ax.tick_params(colors="#aaa", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")
            ax.set_ylabel("")
            ax.set_xlabel("")

            if col_idx < len(pairs):
                row_data, insp = pairs[col_idx]
                try:
                    x = _get_window(row_data)
                    x_norm = (x - mean) / max(std, 1e-8)
                    t_ms = np.arange(PLOT_SAMPLES) / 64.0  # ms
                    ax.plot(t_ms, x_norm[:PLOT_SAMPLES], lw=0.5, color=cmap[cls])
                    title = (f"{row_data['bearing_id']} {row_data['operating_condition']}\n"
                             f"split={row_data['split']}  rms={insp['raw_rms']:.3f}")
                    ax.set_title(title, fontsize=7, color="#ddd")
                    ax.set_xlabel("Time (ms)", fontsize=6, color="#aaa")
                    ax.set_ylabel("Normalised", fontsize=6, color="#aaa")
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center",
                            transform=ax.transAxes, color="red", fontsize=7)
                    ax.set_title(f"{cls} [col {col_idx}]", fontsize=7, color="#ddd")
            else:
                ax.set_visible(False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"  ✓ QC waveform plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("Phase 5 QC — Window Inspection")
    print("=" * 60)

    rows = _load_manifest()
    mean, std = _load_norm()
    print(f"  Manifest: {len(rows)} rows")
    print(f"  Norm stats: mean={mean:.6f}, std={std:.6f}")

    selected = pick_representative_rows(rows)
    print(f"\n  Inspecting {len(selected)} representative windows...")

    inspection = []
    all_ok = True
    for row in selected:
        result = inspect_window(row, mean, std)
        inspection.append(result)
        status = "✓" if result["status"] == "ok" else "✗"
        if result["status"] != "ok":
            all_ok = False
        shape_ok = result.get("shape_ok", False)
        has_nan = result.get("has_nan", False)
        has_inf = result.get("has_inf", False)
        rms = result.get("raw_rms", "N/A")
        print(f"  {status} {result['window_id']:40s}  "
              f"class={result['label']:12s}  split={result['split']:10s}  "
              f"shape_ok={shape_ok}  nan={has_nan}  inf={has_inf}  rms={rms}")

    # Save inspection results
    qc_path = PROJECT_ROOT / "results" / "qc_windows.json"
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(qc_path, "w") as f:
        json.dump({"n_inspected": len(inspection), "windows": inspection}, f, indent=2)
    print(f"\n  ✓ QC results saved: {qc_path}")

    # Waveform plot
    try:
        plot_qc_waveforms(selected, inspection, mean, std)
    except Exception as e:
        print(f"  ⚠ Waveform plot failed (non-fatal): {e}")

    print("\n" + "=" * 60)
    if all_ok:
        print("QC PASSED — All windows loaded and validated successfully.")
    else:
        print("QC WARNINGS — Some windows had errors. Check qc_windows.json.")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
