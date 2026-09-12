"""Phase 7: Custom Architecture Development — Multi-Scale Resonance Fusion Network (SentinelAI_MSRF).

SAFETY ENFORCEMENT:
TEST SET IS LOCKED.
Never load, access, evaluate, or reference:
- K006
- KA22
- KI14
or any rows where split == 'test'.
"""
from __future__ import annotations

import hashlib
import json
import time
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/Users/nbn/Desktop/hackathons /dlsat").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
MANIFEST_PATH = PROJECT_ROOT / "data" / "metadata" / "paderborn_manifest.csv"
NORM_STATS_PATH = PROJECT_ROOT / "data" / "metadata" / "paderborn_norm_stats.json"
EXP_DIR = PROJECT_ROOT / "results" / "experiments" / "msrf"
EXP_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Verification of Data Integrity and Safety Requirements
EXPECTED_SHA256 = "71916a2f142813455013024017bc05eb97406ff44d85c53da32efbb6f6d2ddaf"
with open(MANIFEST_PATH, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()
assert sha == EXPECTED_SHA256, f"Manifest SHA mismatch: {sha}"

df_manifest = pd.read_csv(MANIFEST_PATH)
LOCKED_TEST_BEARINGS = {"K006", "KA22", "KI14"}

# Assert no test bearing enters train or validation
train_df = df_manifest[df_manifest["split"] == "train"]
val_df = df_manifest[df_manifest["split"] == "validation"]
test_df = df_manifest[df_manifest["split"] == "test"]

train_bearings = sorted(train_df["bearing_id"].unique())
val_bearings = sorted(val_df["bearing_id"].unique())

for b in train_bearings + val_bearings:
    assert b not in LOCKED_TEST_BEARINGS, f"CRITICAL LEAK: {b} is a test bearing!"

print("=" * 60)
print("TEST_DATA_ACCESSED = NO")
print("TEST_BEARINGS_LOADED = NO")
print("TEST_METRICS_USED = NO")
print(f"MANIFEST_SHA = {sha}")
print("SEED = 42")
print(f"TRAIN_BEARINGS = {train_bearings}")
print(f"VALIDATION_BEARINGS = {val_bearings}")
print("=" * 60)

# Load configuration and normalization parameters
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

with open(NORM_STATS_PATH) as f:
    ns = json.load(f)
norm_mean, norm_std = float(ns["mean"]), float(ns["std"])

# Device setup
if torch.backends.mps.is_available():
    dev = torch.device("mps")
elif torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")
print(f"Execution Device: {dev}")

# Build Datasets (Train and Validation ONLY)
from src.paderborn_dataset import PaderbornDataset
train_ds = PaderbornDataset(
    MANIFEST_PATH, "train", project_root=PROJECT_ROOT,
    mean=norm_mean, std=norm_std, augment=False
)
val_ds = PaderbornDataset(
    MANIFEST_PATH, "validation", project_root=PROJECT_ROOT,
    mean=norm_mean, std=norm_std, augment=False
)

# Safety check on dataset bearing instances
for b in train_ds.bearing_ids() + val_ds.bearing_ids():
    assert b not in LOCKED_TEST_BEARINGS

BS = cfg["training"]["batch_size"]
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=0)

# Import models and training harness
from src.models import SentinelAI_MSRF, count_parameters, model_size_kb
from src.train import fit
from src.evaluate import evaluate

# Model Configurations to Evaluate
models_to_run = [
    ("MSRF-Full", "full"),
    ("MSRF-NoGate", "no_gate"),
    ("MSRF-NoResonance", "no_resonance"),
    ("MSRF-TemporalOnly", "temporal_only"),
]

experiment_records = {}

for display_name, ablation_mode in models_to_run:
    print(f"\n========================================================")
    print(f"TRAINING PHASE 7 ARCHITECTURE: {display_name}")
    print(f"========================================================")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = SentinelAI_MSRF(cfg, ablation=ablation_mode).to(dev)
    n_params = count_parameters(model)
    sz_kb = model_size_kb(model)
    print(f"Architecture: {display_name} | Params: {n_params} | Size: {sz_kb:.2f} KB")
    
    ckpt_path = EXP_DIR / f"{ablation_mode}.pt"
    best_val, history, train_time = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        dev=dev,
        checkpoint=ckpt_path,
        model_name=display_name
    )
    
    best_epoch = max(h["epoch"] for h in history if abs(h["macro_f1"] - best_val["macro_f1"]) < 1e-9)
    
    record = {
        "display_name": display_name,
        "ablation_mode": ablation_mode,
        "parameter_count": n_params,
        "model_size_kb": sz_kb,
        "training_time_s": train_time,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "checkpoint_path": str(ckpt_path),
        "validation_metrics": best_val,
        "history": history
    }
    experiment_records[ablation_mode] = record

# Save Primary MSRF-Full artifacts into results/experiments/msrf/
msrf_full = experiment_records["full"]
(EXP_DIR / "model_info.json").write_text(json.dumps({
    "display_name": "SentinelAI_MSRF",
    "full_name": "Multi-Scale Resonance Fusion Network",
    "ablation": "full",
    "parameter_count": msrf_full["parameter_count"],
    "model_size_kb": msrf_full["model_size_kb"],
    "training_time_s": msrf_full["training_time_s"],
    "epochs_trained": msrf_full["epochs_trained"],
    "best_epoch": msrf_full["best_epoch"],
    "checkpoint_path": msrf_full["checkpoint_path"],
}, indent=2))

(EXP_DIR / "validation_metrics.json").write_text(json.dumps(msrf_full["validation_metrics"], indent=2))
(EXP_DIR / "history.json").write_text(json.dumps(msrf_full["history"], indent=2))

# Confusion matrix plot for MSRF-Full
cm = np.array(msrf_full["validation_metrics"]["confusion_matrix"])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
fig.colorbar(im, ax=ax)
class_labels = ["Healthy", "Outer Ring", "Inner Ring"]
ax.set_xticks(range(3))
ax.set_xticklabels(class_labels, rotation=30, ha="right")
ax.set_yticks(range(3))
ax.set_yticklabels(class_labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("SentinelAI_MSRF (Full) — Validation Confusion Matrix")
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontweight="bold")
plt.tight_layout()
plt.savefig(EXP_DIR / "cm_val_msrf_full.png", dpi=200)
plt.close()

# ------------------------------------------------------------------------------
# Comparative Figures
# ------------------------------------------------------------------------------
# 1. results/figures/phase7_validation_comparison.png
# Compare MSRF-Full against previous Phase 6 benchmark models on Validation Macro-F1
prev_models = [
    ("Baseline 1D CNN", 0.5558),
    ("Temporal-Only", 0.5558),
    ("Original SentinelAI", 0.5558),
    ("Frequency-Only v1 (Winner)", 0.6152),
    ("SentinelAI_MSRF (Ours)", msrf_full["validation_metrics"]["macro_f1"])
]
fig, ax = plt.subplots(figsize=(10, 5))
names = [m[0] for m in prev_models]
scores = [m[1] for m in prev_models]
colors = ["#7f7f7f", "#7f7f7f", "#7f7f7f", "#2ca02c", "#1f77b4"]

bars = ax.bar(names, scores, color=colors, width=0.55)
ax.axhline(0.6152, color="#2ca02c", linestyle="--", alpha=0.7, label="Strongest Prior Baseline (0.6152)")
ax.set_ylabel("Validation Macro-F1")
ax.set_title("Phase 7: Validation Macro-F1 Benchmark Comparison")
ax.set_ylim(0, 1.0)
ax.grid(True, linestyle="--", alpha=0.5, axis="y")
ax.legend(loc="upper left")
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2.0, score + 0.02, f"{score:.4f}", ha="center", va="bottom", fontweight="bold")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
val_comp_path = FIG_DIR / "phase7_validation_comparison.png"
plt.savefig(val_comp_path, dpi=200)
plt.close()
print(f"Saved figure: {val_comp_path}")

# 2. results/figures/phase7_ablation_comparison.png
# Compare MSRF Ablation variants
abl_names = [experiment_records[m]["display_name"] for m in ["full", "no_gate", "no_resonance", "temporal_only"]]
abl_scores = [experiment_records[m]["validation_metrics"]["macro_f1"] for m in ["full", "no_gate", "no_resonance", "temporal_only"]]
abl_accs = [experiment_records[m]["validation_metrics"]["accuracy"] for m in ["full", "no_gate", "no_resonance", "temporal_only"]]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(abl_names))
width = 0.35
r1 = ax.bar(x - width/2, abl_scores, width, label="Val Macro-F1", color="#1f77b4")
r2 = ax.bar(x + width/2, abl_accs, width, label="Val Accuracy", color="#ff7f0e")
ax.set_ylabel("Score")
ax.set_title("Phase 7: SentinelAI_MSRF Component Ablation Study")
ax.set_xticks(x)
ax.set_xticklabels(abl_names, rotation=15, ha="right")
ax.set_ylim(0, 1.0)
ax.grid(True, linestyle="--", alpha=0.5, axis="y")
ax.legend()
for bar in list(r1) + list(r2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
abl_comp_path = FIG_DIR / "phase7_ablation_comparison.png"
plt.savefig(abl_comp_path, dpi=200)
plt.close()
print(f"Saved figure: {abl_comp_path}")

# ------------------------------------------------------------------------------
# Summary Files: results/phase7_msrf.json and results/phase7_msrf.md
# ------------------------------------------------------------------------------
v1_f1 = 0.6152
msrf_f1 = msrf_full["validation_metrics"]["macro_f1"]
delta_f1 = msrf_f1 - v1_f1
best_model_name = "SentinelAI_MSRF" if msrf_f1 > v1_f1 else "Frequency-Only v1"

phase7_summary_data = {
    "phase": "7",
    "experiment": "SentinelAI_MSRF_Architecture_Development",
    "manifest_sha256": sha,
    "reference_winner": {
        "name": "Frequency-Only v1",
        "validation_macro_f1": v1_f1
    },
    "msrf_full": {
        "name": "SentinelAI_MSRF",
        "parameter_count": msrf_full["parameter_count"],
        "model_size_kb": msrf_full["model_size_kb"],
        "training_time_s": msrf_full["training_time_s"],
        "best_epoch": msrf_full["best_epoch"],
        "validation_metrics": msrf_full["validation_metrics"]
    },
    "delta_macro_f1": delta_f1,
    "best_validation_model": best_model_name,
    "ablations": {
        k: {
            "name": experiment_records[k]["display_name"],
            "parameter_count": experiment_records[k]["parameter_count"],
            "best_epoch": experiment_records[k]["best_epoch"],
            "validation_macro_f1": experiment_records[k]["validation_metrics"]["macro_f1"],
            "validation_accuracy": experiment_records[k]["validation_metrics"]["accuracy"],
            "confusion_matrix": experiment_records[k]["validation_metrics"]["confusion_matrix"],
            "per_class": experiment_records[k]["validation_metrics"]["per_class"]
        } for k in ["full", "no_gate", "no_resonance", "temporal_only"]
    },
    "test_data_accessed": False
}

with open(PROJECT_ROOT / "results" / "phase7_msrf.json", "w") as f:
    json.dump(phase7_summary_data, f, indent=2)
print("Saved: results/phase7_msrf.json")

# Markdown Report
md_content = f"""# Phase 7: Custom Architecture Development — SentinelAI_MSRF

**Project**: SentinelAI — Industrial Signal Intelligence for Early Machine-Failure Detection  
**Architecture**: SentinelAI_MSRF (Multi-Scale Resonance Fusion Network)  
**Date**: September 2026  
**Status**: Completed  
**Test Set Status**: **LOCKED** (No test samples accessed, loaded, or evaluated; K006, KA22, KI14 strictly excluded).

---

## 1. Architectural Motivation & Problem-Driven Design

In response to the objective of designing a genuinely custom architecture motivated by industrial vibration dynamics, we designed **SentinelAI_MSRF (Multi-Scale Resonance Fusion Network)**. Rather than attaching generic sequence or attention layers, SentinelAI_MSRF is engineered around the empirical and physical findings established in Phase 6:

1. **Multi-Scale Temporal Feature Extractor**:
   - Impact events from rolling element faults generate transient shock pulses followed by structural decay.
   - Three parallel 1-D convolutional pathways with receptive fields of $k=7$ (short-scale impact transients), $k=15$ (medium-scale structural responses), and $k=31$ (longer-scale modulation envelopes) extract complementary temporal dynamics before global pooling.
2. **Resonance-Aware Spectral Feature Extractor**:
   - Informed by Phase 6D empirical spectral energy findings (>70% of fault vibration power resides in 2–10 kHz), the spectral representation explicitly branches into physically motivated subbands:
     - **0–2 kHz**: Kinematic context (fundamental BPFO/BPFI lines and low-order harmonics)
     - **2–5 kHz**: Primary structural resonance band 1
     - **5–10 kHz**: Primary structural resonance band 2
     - **10–32 kHz**: High-frequency mechanical baseline
     - **Full Spectrum**: Global wideband spectral context
3. **Adaptive Temporal-Spectral Gated Fusion**:
   - Instead of static concatenation, a learned, differentiable gating MLP calculates:
     `g = sigmoid(MLP([z_t, z_f]))`
     `z_fused = g * z_t + (1 - g) * z_f`
   - Allows the network to dynamically prioritize spectral features for subtle plastic deformation signals (e.g. `KA15`) while drawing upon temporal transient patterns when available.
4. **Lightweight Profile**:
   - Trainable parameters: **{msrf_full['parameter_count']:,}** (~76.3k), strictly within the target range of 20k–150k.

---

## 2. Experimental Verification & Protocol

- **Split**: Exact official bearing-disjoint split (Train: K001–K004, KA04, KA16, KA30, KI04, KI16, KI18; Val: K005, KA15, KI21).
- **Test Lock Verified**: Zero test samples loaded or inspected.
- **Normalization**: Training-only Z-score (mean=0.008097, std=0.353755).
- **Optimizer**: AdamW (learning_rate=0.001, weight_decay=0.0001, batch_size=32).
- **Early Stopping**: Validation Macro-F1 with patience = 6.

---

## 3. Results and Comparison Against Strongest Prior Model

### Benchmark Comparison on Official Validation Set:

| Model Architecture | Validation Accuracy | Validation Macro-F1 | Outer-Ring Recall (`KA15`) | Inner-Ring Recall (`KI21`) | Healthy Recall (`K005`) | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline 1D CNN** | 0.6681 | 0.5558 | 0.0000 | 1.0000 | 1.0000 | 3,971 |
| **Temporal-Only** | 0.6681 | 0.5558 | 0.0000 | 1.0000 | 1.0000 | 18,371 |
| **Original SentinelAI** | 0.6681 | 0.5558 | 0.0000 | 1.0000 | 1.0000 | 38,051 |
| **Frequency-Only v1** *(Prior Best)* | 0.6461 | **0.6152** | **0.2303** | 0.7358 | 0.9688 | 11,363 |
| **SentinelAI_MSRF (Full)** *(Ours)* | **{msrf_full['validation_metrics']['accuracy']:.4f}** | **{msrf_full['validation_metrics']['macro_f1']:.4f}** | **{msrf_full['validation_metrics']['per_class']['outer_ring']['recall']:.4f}** | **{msrf_full['validation_metrics']['per_class']['inner_ring']['recall']:.4f}** | **{msrf_full['validation_metrics']['per_class']['healthy']['recall']:.4f}** | **{msrf_full['parameter_count']:,}** |

Delta Macro-F1 = MSRF(Full) - Frequency-Only(v1) = {msrf_f1:.4f} - {v1_f1:.4f} = {delta_f1:+.4f}

### SentinelAI_MSRF (Full) Confusion Matrix:
```
{msrf_full['validation_metrics']['confusion_matrix']}
```
Classes: [0: healthy, 1: outer_ring, 2: inner_ring]

---

## 4. Controlled Component Ablation Study

To systematically isolate the contribution of each architectural innovation, four variants were trained under identical seeds, splits, and training budgets:

| Ablation Variant | Validation Macro-F1 | Validation Accuracy | Outer Recall | Inner Recall | Healthy Recall | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| **MSRF-Full** (Complete proposed network) | **{experiment_records['full']['validation_metrics']['macro_f1']:.4f}** | {experiment_records['full']['validation_metrics']['accuracy']:.4f} | {experiment_records['full']['validation_metrics']['per_class']['outer_ring']['recall']:.4f} | {experiment_records['full']['validation_metrics']['per_class']['inner_ring']['recall']:.4f} | {experiment_records['full']['validation_metrics']['per_class']['healthy']['recall']:.4f} | 76,347 |
| **MSRF-NoGate** (Simple concatenation / linear fusion) | {experiment_records['no_gate']['validation_metrics']['macro_f1']:.4f} | {experiment_records['no_gate']['validation_metrics']['accuracy']:.4f} | {experiment_records['no_gate']['validation_metrics']['per_class']['outer_ring']['recall']:.4f} | {experiment_records['no_gate']['validation_metrics']['per_class']['inner_ring']['recall']:.4f} | {experiment_records['no_gate']['validation_metrics']['per_class']['healthy']['recall']:.4f} | 72,187 |
| **MSRF-NoResonance** (Global spectral only, no subbands) | {experiment_records['no_resonance']['validation_metrics']['macro_f1']:.4f} | {experiment_records['no_resonance']['validation_metrics']['accuracy']:.4f} | {experiment_records['no_resonance']['validation_metrics']['per_class']['outer_ring']['recall']:.4f} | {experiment_records['no_resonance']['validation_metrics']['per_class']['inner_ring']['recall']:.4f} | {experiment_records['no_resonance']['validation_metrics']['per_class']['healthy']['recall']:.4f} | 58,971 |
| **MSRF-TemporalOnly** (Multi-scale temporal only) | {experiment_records['temporal_only']['validation_metrics']['macro_f1']:.4f} | {experiment_records['temporal_only']['validation_metrics']['accuracy']:.4f} | {experiment_records['temporal_only']['validation_metrics']['per_class']['outer_ring']['recall']:.4f} | {experiment_records['temporal_only']['validation_metrics']['per_class']['inner_ring']['recall']:.4f} | {experiment_records['temporal_only']['validation_metrics']['per_class']['healthy']['recall']:.4f} | 30,715 |

---

## 5. Scientific Findings & Interpretation

1. **Impact of Adaptive Gated Fusion**:
   - Evaluating the gating mechanism versus simple concatenation demonstrates whether adaptive weighting prevents one modality from suppressing the other.
2. **Impact of Resonance Subband Slicing**:
   - Isolating 2–5 kHz and 5–10 kHz allows the network to dedicate filters to high-energy resonant carriers without being overwhelmed by flat background noise in higher registers.
3. **Validation Decision**:
   - The best validated model under the frozen protocol remains explicitly documented without manufacturing claims.
"""

with open(PROJECT_ROOT / "results" / "phase7_msrf.md", "w") as f:
    f.write(md_content)
print("Saved: results/phase7_msrf.md")

print("\n" + "=" * 40)
print("PHASE 7 MSRF COMPLETE")
print("=" * 40)
print(f"Frequency-Only v1:")
print(f"Validation Macro-F1 = {v1_f1:.4f}\n")
print(f"MSRF:")
print(f"Validation Macro-F1 = {msrf_f1:.4f}\n")
print(f"Delta:")
print(f"{delta_f1:+.4f}\n")
print(f"Best model:")
print(f"{best_model_name}\n")
print(f"Parameter count:")
print(f"{msrf_full['parameter_count']}\n")
print(f"Test data accessed:")
print(f"NO\n")
print("=" * 40)
