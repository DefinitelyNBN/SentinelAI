"""Phase 6C-A — Controlled Frequency-Domain Ablation.

Scientific question:
    "Does restricting the STFT representation to a physically meaningful
    low-frequency region (0–1 kHz) improve bearing-disjoint validation
    classification?"

Reference model  : Frequency-Only v1 (full STFT, 0–32 kHz, Val Macro-F1 = 0.6152)
Experimental model: Frequency-Only v2 (STFT restricted to 0–1 kHz)

TEST SET IS LOCKED AND WILL NOT BE EVALUATED IN THIS SCRIPT.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
import yaml

# ──────────────────────────────────────────────────────────────────────────────
# SAFETY BANNER
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("TEST SET LOCKED — Phase 6C-A validation-only experiment.")
print("Only train and validation loaders will be instantiated.")
print("=" * 72)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_PATH  = PROJECT_ROOT / "configs" / "config.yaml"
MANIFEST     = PROJECT_ROOT / "data" / "metadata" / "paderborn_manifest.csv"
NORM_STATS   = PROJECT_ROOT / "data" / "metadata" / "paderborn_norm_stats.json"
OUT_DIR      = PROJECT_ROOT / "results" / "experiments" / "frequency_only_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR      = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

SEED = 42
torch.manual_seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# Manifest integrity check
# ──────────────────────────────────────────────────────────────────────────────
EXPECTED_SHA256 = "71916a2f142813455013024017bc05eb97406ff44d85c53da32efbb6f6d2ddaf"
with open(MANIFEST, "rb") as f:
    actual_sha256 = hashlib.sha256(f.read()).hexdigest()
assert actual_sha256 == EXPECTED_SHA256, (
    f"Manifest SHA-256 mismatch!\n  expected: {EXPECTED_SHA256}\n  actual:   {actual_sha256}"
)
print(f"Manifest SHA-256 verified: {actual_sha256}")

# ──────────────────────────────────────────────────────────────────────────────
# STFT frequency-band documentation
# ──────────────────────────────────────────────────────────────────────────────
FS        = cfg["dataset"]["sampling_rate"]     # 64 000 Hz
N_FFT     = cfg["model"]["n_fft"]               # 2 048
HOP       = cfg["model"]["hop_length"]          # 512
FREQ_RES  = FS / N_FFT                          # 31.25 Hz / bin
N_TOTAL   = N_FFT // 2 + 1                      # 1 025 bins (0 … Nyquist)
MAX_HZ    = 1_000.0
N_KEEP    = int(MAX_HZ / FREQ_RES) + 1          # 33 bins (inclusive)
N_KEEP    = min(N_KEEP, N_TOTAL)
HIGHEST_F = (N_KEEP - 1) * FREQ_RES            # 1 000.0 Hz

print(f"\nSTFT configuration:")
print(f"  sampling_rate : {FS} Hz")
print(f"  n_fft         : {N_FFT}")
print(f"  hop_length    : {HOP}")
print(f"  freq_resolution: {FREQ_RES:.4f} Hz/bin")
print(f"  total bins    : {N_TOTAL}  (0 – {(N_TOTAL - 1) * FREQ_RES:.0f} Hz)")
print(f"  retained bins : {N_KEEP}   (0 – {HIGHEST_F:.2f} Hz)")

# ──────────────────────────────────────────────────────────────────────────────
# Datasets (train + validation ONLY)
# ──────────────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.paderborn_dataset import PaderbornDataset
from torch.utils.data import DataLoader

with open(NORM_STATS) as f:
    ns = json.load(f)
mean_val, std_val = float(ns["mean"]), float(ns["std"])

train_ds = PaderbornDataset(
    MANIFEST, "train", project_root=PROJECT_ROOT,
    mean=mean_val, std=std_val, augment=False,
)
val_ds = PaderbornDataset(
    MANIFEST, "validation", project_root=PROJECT_ROOT,
    mean=mean_val, std=std_val, augment=False,
)
# Intentionally do NOT create or reference test_ds.

print(f"\nDatasets loaded:")
print(f"  train     : {len(train_ds)} windows ({train_ds.class_counts()})")
print(f"  validation: {len(val_ds)} windows ({val_ds.class_counts()})")

BS = cfg["training"]["batch_size"]
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True,  num_workers=0, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BS, shuffle=False, num_workers=0)

# ──────────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    dev = torch.device("mps")
elif torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")
print(f"\nDevice: {dev}")

# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
from src.models import FrequencyOnlyV2, count_parameters, model_size_kb

# Inject band_limit_hz into config so FrequencyOnlyV2 picks it up.
cfg["model"]["band_limit_hz"] = MAX_HZ

torch.manual_seed(SEED)
model = FrequencyOnlyV2(cfg).to(dev)
n_params = count_parameters(model)
sz_kb    = model_size_kb(model)
print(f"\nFrequency-Only v2: {n_params} parameters, {sz_kb:.2f} KB")

# ──────────────────────────────────────────────────────────────────────────────
# Training (identical hyper-parameters to Frequency-Only v1)
# ──────────────────────────────────────────────────────────────────────────────
CKPT_PATH = OUT_DIR / "checkpoint.pt"

from src.train import fit

best_val, history, train_time = fit(
    model, train_loader, val_loader, cfg, dev,
    checkpoint=CKPT_PATH,
    model_name="FrequencyOnly-v2",
)

best_epoch = max(h["epoch"] for h in history if abs(h["macro_f1"] - best_val["macro_f1"]) < 1e-9)

# ──────────────────────────────────────────────────────────────────────────────
# Confusion-matrix figure
# ──────────────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cm   = best_val["confusion_matrix"]
    cm_a = np.array(cm)
    classes = ["Healthy", "Outer Ring", "Inner Ring"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_a, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(3)); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Frequency-Only v2 — Validation Confusion Matrix")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm_a[i, j]), ha="center", va="center",
                    color="white" if cm_a[i, j] > cm_a.max() / 2 else "black")
    fig.tight_layout()
    fig_path = FIG_DIR / "cm_val_frequency_only_v2.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nConfusion-matrix figure saved: {fig_path}")
except Exception as e:
    print(f"Figure generation skipped: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Persist artefacts
# ──────────────────────────────────────────────────────────────────────────────
stft_params = {
    "sampling_rate_hz": FS,
    "n_fft": N_FFT,
    "hop_length": HOP,
    "window": "hann",
    "freq_resolution_hz_per_bin": FREQ_RES,
    "total_bins": N_TOTAL,
    "retained_bins": N_KEEP,
    "freq_range_hz": [0.0, float(HIGHEST_F)],
}

validation_metrics = {
    "accuracy":        best_val["accuracy"],
    "macro_f1":        best_val["macro_f1"],
    "precision_macro": best_val["precision_macro"],
    "recall_macro":    best_val["recall_macro"],
    "per_class":       best_val["per_class"],
    "confusion_matrix": best_val["confusion_matrix"],
    "num_samples":     best_val.get("num_samples"),
}
(OUT_DIR / "validation_metrics.json").write_text(json.dumps(validation_metrics, indent=2))

history_serializable = []
for h in history:
    rec = {}
    for k, v in h.items():
        if isinstance(v, dict):
            rec[k] = v
        else:
            rec[k] = float(v) if not isinstance(v, (int, str)) else v
    history_serializable.append(rec)
(OUT_DIR / "history.json").write_text(json.dumps(history_serializable, indent=2))

model_info = {
    "display_name": "Frequency-Only v2",
    "description": "FrequencyOnly with STFT restricted to 0–1 kHz band",
    "parameter_count": n_params,
    "model_size_kb": sz_kb,
    "training_time_s": train_time,
    "epochs_trained": len(history),
    "best_epoch": best_epoch,
    "checkpoint_path": str(CKPT_PATH),
    "stft_params": stft_params,
    "band_limit_hz": MAX_HZ,
}
(OUT_DIR / "model_info.json").write_text(json.dumps(model_info, indent=2))

# ──────────────────────────────────────────────────────────────────────────────
# Load Frequency-Only v1 reference
# ──────────────────────────────────────────────────────────────────────────────
with open(PROJECT_ROOT / "results" / "metrics" / "validation_experiments.json") as f:
    all_results = json.load(f)
v1 = all_results["models"]["frequency_only"]["validation"]
v1_f1    = v1["macro_f1"]
v1_acc   = v1["accuracy"]
v1_outer = v1["per_class"]["outer_ring"]["recall"]
v1_inner = v1["per_class"]["inner_ring"]["recall"]
v1_healthy = v1["per_class"]["healthy"]["recall"]

v2_f1    = best_val["macro_f1"]
v2_acc   = best_val["accuracy"]
v2_outer = best_val["per_class"]["outer_ring"]["recall"]
v2_inner = best_val["per_class"]["inner_ring"]["recall"]
v2_healthy = best_val["per_class"]["healthy"]["recall"]

delta_f1    = v2_f1    - v1_f1
delta_acc   = v2_acc   - v1_acc
delta_outer = v2_outer - v1_outer
delta_inner = v2_inner - v1_inner

improved = delta_f1 > 0.0
decision = "IMPROVED" if improved else "NO IMPROVEMENT"

# ──────────────────────────────────────────────────────────────────────────────
# Write summary files
# ──────────────────────────────────────────────────────────────────────────────
summary_json = {
    "phase": "6C-A",
    "experiment": "frequency_band_restriction_0_1khz",
    "manifest_sha256": actual_sha256,
    "stft_params": stft_params,
    "reference": {
        "name": "Frequency-Only v1",
        "macro_f1": v1_f1,
        "accuracy": v1_acc,
        "outer_ring_recall": v1_outer,
        "inner_ring_recall": v1_inner,
        "healthy_recall": v1_healthy,
    },
    "experiment_result": {
        "name": "Frequency-Only v2",
        "macro_f1": v2_f1,
        "accuracy": v2_acc,
        "outer_ring_recall": v2_outer,
        "inner_ring_recall": v2_inner,
        "healthy_recall": v2_healthy,
        "parameter_count": n_params,
        "model_size_kb": sz_kb,
        "training_time_s": train_time,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "confusion_matrix": best_val["confusion_matrix"],
    },
    "deltas": {
        "macro_f1": delta_f1,
        "accuracy": delta_acc,
        "outer_ring_recall": delta_outer,
        "inner_ring_recall": delta_inner,
    },
    "decision": decision,
    "test_set_evaluated": False,
}
(PROJECT_ROOT / "results" / "phase6c_frequency_ablation.json").write_text(
    json.dumps(summary_json, indent=2)
)

md_lines = [
    "# Phase 6C-A — Frequency-Domain Ablation",
    "",
    "**Project**: SentinelAI  ",
    "**Date**: September 2026  ",
    "**Phase**: 6C-A — Controlled Frequency-Domain Ablation  ",
    "**TEST SET**: Locked — not evaluated.",
    "",
    "---",
    "",
    "## Scientific Question",
    "",
    "> Does restricting the STFT representation to a physically meaningful",
    "> low-frequency region (0–1 kHz) improve bearing-disjoint validation",
    "> classification?",
    "",
    "---",
    "",
    "## Manifest Integrity",
    "",
    f"SHA-256: `{actual_sha256}`  ",
    "Status: **VERIFIED UNCHANGED**",
    "",
    "---",
    "",
    "## STFT Configuration",
    "",
    "| Parameter | Value |",
    "|---|---|",
    f"| Sampling rate | {FS} Hz |",
    f"| n_fft | {N_FFT} |",
    f"| hop_length | {HOP} |",
    f"| Window | Hann |",
    f"| Frequency resolution | {FREQ_RES:.4f} Hz/bin |",
    f"| Total bins (0–Nyquist) | {N_TOTAL} |",
    f"| **Retained bins (v2)** | **{N_KEEP}** (bins 0–{N_KEEP-1}) |",
    f"| **Frequency range retained** | **0 – {HIGHEST_F:.2f} Hz** |",
    "",
    "---",
    "",
    "## Experimental Control",
    "",
    "| Dimension | v1 | v2 |",
    "|---|---|---|",
    "| Architecture depth | identical | identical |",
    "| Channel counts | identical | identical |",
    "| Embedding dim | identical | identical |",
    "| Classifier | identical | identical |",
    "| Optimizer | AdamW | AdamW |",
    "| Learning rate | 0.001 | 0.001 |",
    "| Weight decay | 0.0001 | 0.0001 |",
    "| Batch size | 32 | 32 |",
    "| Max epochs | 30 | 30 |",
    "| Patience | 6 | 6 |",
    "| Loss | CrossEntropy | CrossEntropy |",
    "| Augmentation | none | none |",
    "| Normalization | training Z-score | training Z-score |",
    "| Seed | 42 | 42 |",
    "| **Experimental variable** | Full STFT (0–32 kHz) | **STFT 0–1 kHz only** |",
    "",
    "---",
    "",
    "## Results",
    "",
    "### Frequency-Only v2 — Model Info",
    "",
    f"- Parameter count : {n_params}",
    f"- Model size       : {sz_kb:.2f} KB",
    f"- Training time    : {train_time:.1f} s",
    f"- Epochs trained   : {len(history)}",
    f"- Best epoch       : {best_epoch}",
    "",
    "### Validation Metrics Comparison",
    "",
    "| Metric | v1 (reference) | v2 (experiment) | Delta |",
    "|---|---:|---:|---:|",
    f"| Macro-F1 | {v1_f1:.4f} | {v2_f1:.4f} | {delta_f1:+.4f} |",
    f"| Accuracy | {v1_acc:.4f} | {v2_acc:.4f} | {delta_acc:+.4f} |",
    f"| Outer-ring recall | {v1_outer:.4f} | {v2_outer:.4f} | {delta_outer:+.4f} |",
    f"| Inner-ring recall | {v1_inner:.4f} | {v2_inner:.4f} | {delta_inner:+.4f} |",
    f"| Healthy recall | {v1_healthy:.4f} | {v2_healthy:.4f} | {v2_healthy - v1_healthy:+.4f} |",
    "",
    "### Frequency-Only v2 — Validation Confusion Matrix",
    "",
    "```",
    str(best_val["confusion_matrix"]),
    "```",
    "",
    "---",
    "",
    "## Scientific Interpretation",
    "",
]

if improved:
    md_lines += [
        "The frequency-band restriction to 0–1 kHz **improved** validation Macro-F1",
        f"by {delta_f1:+.4f} relative to Frequency-Only v1.",
        "",
        "Quantitative improvement was observed in:",
        f"- Outer-ring recall: {v1_outer:.4f} → {v2_outer:.4f} ({delta_outer:+.4f})",
        f"- Inner-ring recall: {v1_inner:.4f} → {v2_inner:.4f} ({delta_inner:+.4f})",
        "",
        "Restricting the CNN's input to the low-frequency band appears to concentrate",
        "representational capacity in the region where bearing fault signatures",
        "(kinematic ball-pass frequencies, sidebands) are most discriminative,",
        "consistent with the hypothesis. Causal attribution remains unconfirmed;",
        "this is a single controlled experiment result.",
    ]
else:
    md_lines += [
        "The frequency-band restriction to 0–1 kHz did **NOT** improve validation",
        f"Macro-F1. Delta = {delta_f1:+.4f}.",
        "",
        "Retaining Frequency-Only v1 as the best validation model.",
        "",
        "Possible explanations:",
        "- With only 33 retained bins, the CNN may have insufficient spectral",
        "  resolution to discriminate fine harmonic structures.",
        "- The bearing fault signatures captured by v1 (with full spectrum) may",
        "  extend beyond 1 kHz for the training set bearing damage modes.",
        "- Reduction in input size may reduce spatial context in the 2D CNN.",
        "",
        "No further architectural modifications will be invented from this result.",
    ]

md_lines += [
    "",
    "---",
    "",
    "## Final Summary",
    "",
    "```",
    "REFERENCE:",
    f"Frequency-Only v1",
    f"Val Macro-F1 = {v1_f1:.4f}",
    "",
    "EXPERIMENT:",
    f"Frequency-Only v2",
    f"Val Macro-F1 = {v2_f1:.4f}",
    "",
    "DELTA:",
    f"{delta_f1:+.4f}",
    "",
    "Outer-ring recall:",
    f"v1 = {v1_outer:.4f}",
    f"v2 = {v2_outer:.4f}",
    "",
    f"FINAL DECISION:",
    decision,
    "```",
    "",
    "---",
    "",
    "**PHASE 6C-A COMPLETE — FREQUENCY-DOMAIN ABLATION FINISHED — TEST SET REMAINS LOCKED**",
]

(PROJECT_ROOT / "results" / "phase6c_frequency_ablation.md").write_text(
    "\n".join(md_lines)
)

# ──────────────────────────────────────────────────────────────────────────────
# Terminal summary
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("PHASE 6C-A RESULTS")
print("=" * 72)
print(f"\nSTFT band restriction: 0 – {HIGHEST_F:.0f} Hz ({N_KEEP} bins retained from {N_TOTAL})")
print(f"\nFrequency-Only v2:")
print(f"  Parameters : {n_params}")
print(f"  Size       : {sz_kb:.2f} KB")
print(f"  Train time : {train_time:.1f} s")
print(f"  Epochs     : {len(history)}")
print(f"  Best epoch : {best_epoch}")
print(f"\nValidation confusion matrix:")
for row in best_val["confusion_matrix"]:
    print(f"  {row}")
print()
print("REFERENCE:")
print("Frequency-Only v1")
print(f"Val Macro-F1 = {v1_f1:.4f}")
print()
print("EXPERIMENT:")
print("Frequency-Only v2")
print(f"Val Macro-F1 = {v2_f1:.4f}")
print()
print("DELTA:")
print(f"{delta_f1:+.4f}")
print()
print("Outer-ring recall:")
print(f"v1 = {v1_outer:.4f}")
print(f"v2 = {v2_outer:.4f}")
print()
print("FINAL DECISION:")
print(decision)
print()
print("PHASE 6C-A COMPLETE — FREQUENCY-DOMAIN ABLATION FINISHED — TEST SET REMAINS LOCKED")
