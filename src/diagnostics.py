"""Reproducible split and signal-distribution diagnostics (no model selection)."""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from src.paderborn_dataset import validate_bearing_disjoint_manifest
from src.paderborn_mat import get_window_data
from src.utils import load_config


def signal_features(x: np.ndarray) -> dict:
    mean, std = float(x.mean()), float(x.std())
    rms, peak = float(np.sqrt(np.mean(x * x))), float(np.max(np.abs(x)))
    centered = x - mean
    kurtosis = float(np.mean(centered ** 4) / max(float(np.mean(centered ** 2)) ** 2, 1e-12))
    return {"mean": mean, "std": std, "rms": rms, "peak_amplitude": peak, "crest_factor": peak / max(rms, 1e-12), "kurtosis": kurtosis}


def run(cfg: dict, output: Path) -> dict:
    manifest = Path(cfg["dataset"]["manifest"])
    split_report = validate_bearing_disjoint_manifest(manifest)
    with open(manifest, newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
    # A bounded, deterministic sample per bearing makes the report practical.
    grouped = defaultdict(list)
    for row in rows: grouped[row["bearing_id"]].append(row)
    per_bearing = {}
    for bearing, records in sorted(grouped.items()):
        sample = records[:min(20, len(records))]
        features = [signal_features(get_window_data(Path(r["mat_path"]), int(r["start_sample"]), int(r["end_sample"]))) for r in sample]
        per_bearing[bearing] = {"split": records[0]["split"], "label": records[0]["label"], "windows": len(records), "sampled_windows": len(sample), **{key: float(np.mean([f[key] for f in features])) for key in features[0]}}
    report = {"split_integrity": split_report, "per_bearing": per_bearing, "note": "Feature values are averages over at most the first 20 deterministic windows per bearing."}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2))
    _plot_feature_comparison(per_bearing, output.with_suffix(".png"))
    return report


def _plot_feature_comparison(per_bearing: dict, output: Path) -> None:
    """Save a compact per-bearing domain-shift comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names, values = list(per_bearing), per_bearing.values()
    colours = {"train": "#4C78A8", "validation": "#F58518", "test": "#54A24B"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), dpi=150)
    for ax, metric in zip(axes, ("mean", "rms", "crest_factor")):
        ax.bar(names, [v[metric] for v in values], color=[colours[v["split"]] for v in values])
        ax.set_title(metric.replace("_", " ").title()); ax.tick_params(axis="x", rotation=60); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(output); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="results/diagnostics/split_signal_report.json")
    args = parser.parse_args(); report = run(load_config(), Path(args.output)); print(json.dumps(report["split_integrity"], indent=2))
