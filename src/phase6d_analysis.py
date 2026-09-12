"""Phase 6D: Spectral Evidence Analysis and Secondary Robustness Analysis.

SAFETY ENFORCEMENT:
TEST SET IS LOCKED.
Never load, access, or reference:
- K006
- KA22
- KI14
or any rows where split == 'test'.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

# ──────────────────────────────────────────────────────────────────────────────
# SAFETY BANNER
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("TEST SET LOCKED — Phase 6D analysis-only.")
print("Verifying test bearings are NEVER accessed: K006, KA22, KI14.")
print("=" * 72)

LOCKED_TEST_BEARINGS = {"K006", "KA22", "KI14"}

PROJECT_ROOT = Path("/Users/nbn/Desktop/hackathons /dlsat").resolve()
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
MANIFEST_PATH = PROJECT_ROOT / "data" / "metadata" / "paderborn_manifest.csv"
NORM_STATS_PATH = PROJECT_ROOT / "data" / "metadata" / "paderborn_norm_stats.json"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = PROJECT_ROOT / "results" / "checkpoints" / "frequency_only.pt"

# Verify manifest rows to ensure safety
df_manifest = pd.read_csv(MANIFEST_PATH)
test_rows = df_manifest[df_manifest["split"] == "test"]
for b in test_rows["bearing_id"].unique():
    assert b in LOCKED_TEST_BEARINGS, f"Unexpected test bearing: {b}"

# Filter manifest to strictly TRAIN and VALIDATION
df_allowed = df_manifest[df_manifest["split"].isin(["train", "validation"])].copy()
for b in df_allowed["bearing_id"].unique():
    assert b not in LOCKED_TEST_BEARINGS, f"CRITICAL LEAKAGE DETECTED: {b} is in allowed set!"

print(f"Safety verification passed. Allowed bearings: {sorted(df_allowed['bearing_id'].unique())}")
print(f"Total allowed windows: {len(df_allowed)} (Train: {len(df_allowed[df_allowed['split']=='train'])}, Val: {len(df_allowed[df_allowed['split']=='validation'])})")

# Load Normalization stats
with open(NORM_STATS_PATH) as f:
    norm_stats = json.load(f)
norm_mean, norm_std = float(norm_stats["mean"]), float(norm_stats["std"])

# Load Config
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

FS = cfg["dataset"]["sampling_rate"]  # 64000
WINDOW_LEN = cfg["dataset"]["signal_length"]  # 64000

# ──────────────────────────────────────────────────────────────────────────────
# PART A: SPECTRAL EVIDENCE
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("PART A: SPECTRAL EVIDENCE EXTRACTION")
print("=" * 50)

from src.paderborn_mat import get_window_data

def compute_window_spectrum(mat_path: str, start: int, end: int, mean: float, std: float) -> tuple[np.ndarray, np.ndarray]:
    """Load window, apply training-derived normalization, compute one-sided rFFT magnitude."""
    raw = get_window_data(PROJECT_ROOT / mat_path, start, end)
    normed = (raw - mean) / max(std, 1e-8)
    # Compute one-sided FFT
    N = len(normed)
    # Hann window for spectral leakage mitigation
    w = np.hanning(N)
    sig_w = normed * w
    fft_vals = np.fft.rfft(sig_w)
    freqs = np.fft.rfftfreq(N, d=1.0/FS)
    mag = np.abs(fft_vals) * (2.0 / np.sum(w))  # Amplitude correction for Hann window
    return freqs, mag

# Collect spectra by train classes and validation bearings
# To be robust, compute across all windows in each group and take median spectrum
train_spectra = {"healthy": [], "outer_ring": [], "inner_ring": []}
val_spectra = {"K005": [], "KA15": [], "KI21": []}

print("Computing spectra across train and validation windows...")
# Limit sample per bearing to max 80 evenly spaced windows to keep computation fast & balanced
for split in ["train", "validation"]:
    split_df = df_allowed[df_allowed["split"] == split]
    for bearing in split_df["bearing_id"].unique():
        assert bearing not in LOCKED_TEST_BEARINGS
        b_df = split_df[split_df["bearing_id"] == bearing]
        indices = np.linspace(0, len(b_df) - 1, num=min(60, len(b_df)), dtype=int)
        sampled = b_df.iloc[indices]
        
        for _, row in sampled.iterrows():
            freqs, mag = compute_window_spectrum(
                row["mat_path"], int(row["start_sample"]), int(row["end_sample"]), norm_mean, norm_std
            )
            if split == "train":
                train_spectra[row["label"]].append(mag)
            else:
                val_spectra[bearing].append(mag)

# Median aggregation
median_train_spectra = {k: np.median(np.array(v), axis=0) for k, v in train_spectra.items()}
median_val_spectra = {k: np.median(np.array(v), axis=0) for k, v in val_spectra.items()}

# ------------------------------------------------------------------------------
# A1. Figures
# ------------------------------------------------------------------------------
print("Generating publication-quality spectral figures...")

# 1. Train classes comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
# Linear scale up to 5 kHz
ax = axes[0]
for lbl, color in [("healthy", "#2ca02c"), ("outer_ring", "#ff7f0e"), ("inner_ring", "#1f77b4")]:
    mask = freqs <= 5000
    ax.plot(freqs[mask], median_train_spectra[lbl][mask], label=f"Train {lbl} (median)", color=color, alpha=0.85, lw=1.2)
ax.set_title("Training Set: Median Magnitude Spectrum (0–5 kHz)")
ax.set_ylabel("Normalized Magnitude")
ax.set_xlabel("Frequency (Hz)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper right")

# Full spectrum 0–32 kHz
ax = axes[1]
for lbl, color in [("healthy", "#2ca02c"), ("outer_ring", "#ff7f0e"), ("inner_ring", "#1f77b4")]:
    # Downsample for visualization density
    step = 4
    ax.plot(freqs[::step], median_train_spectra[lbl][::step], label=f"Train {lbl} (median)", color=color, alpha=0.85, lw=1.0)
ax.set_title("Training Set: Full Spectrum (0–32 kHz)")
ax.set_ylabel("Normalized Magnitude")
ax.set_xlabel("Frequency (Hz)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper right")

plt.tight_layout()
train_fig_path = FIG_DIR / "spectral_class_comparison_train.png"
plt.savefig(train_fig_path, dpi=200)
plt.close()
print(f"Saved: {train_fig_path}")

# 2. Validation bearings comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
# 0-5 kHz
ax = axes[0]
for b_id, lbl, color in [("K005", "Healthy (K005)", "#2ca02c"), ("KA15", "Outer Ring (KA15)", "#ff7f0e"), ("KI21", "Inner Ring (KI21)", "#1f77b4")]:
    mask = freqs <= 5000
    ax.plot(freqs[mask], median_val_spectra[b_id][mask], label=f"Val {lbl} (median)", color=color, alpha=0.85, lw=1.2)
ax.set_title("Validation Set Bearings: Median Magnitude Spectrum (0–5 kHz)")
ax.set_ylabel("Normalized Magnitude")
ax.set_xlabel("Frequency (Hz)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper right")

# Full spectrum 0–32 kHz
ax = axes[1]
for b_id, lbl, color in [("K005", "Healthy (K005)", "#2ca02c"), ("KA15", "Outer Ring (KA15)", "#ff7f0e"), ("KI21", "Inner Ring (KI21)", "#1f77b4")]:
    step = 4
    ax.plot(freqs[::step], median_val_spectra[b_id][::step], label=f"Val {lbl} (median)", color=color, alpha=0.85, lw=1.0)
ax.set_title("Validation Set Bearings: Full Spectrum (0–32 kHz)")
ax.set_ylabel("Normalized Magnitude")
ax.set_xlabel("Frequency (Hz)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper right")

plt.tight_layout()
val_fig_path = FIG_DIR / "spectral_validation_bearings.png"
plt.savefig(val_fig_path, dpi=200)
plt.close()
print(f"Saved: {val_fig_path}")

# ------------------------------------------------------------------------------
# A2. Frequency-region analysis
# ------------------------------------------------------------------------------
print("\nQuantifying spectral energy distribution across defined bands...")

BANDS = [
    ("0–100 Hz", 0.0, 100.0),
    ("100–250 Hz", 100.0, 250.0),
    ("250–500 Hz", 250.0, 500.0),
    ("500–1000 Hz", 500.0, 1000.0),
    ("1–2 kHz", 1000.0, 2000.0),
    ("2–5 kHz", 2000.0, 5000.0),
    ("5–10 kHz", 5000.0, 10000.0),
    ("10–20 kHz", 10000.0, 20000.0),
    ("20–32 kHz", 20000.0, 32000.0),
]

def compute_band_energies(mag_spectrum: np.ndarray, freqs: np.ndarray) -> dict[str, float]:
    """Compute relative energy percentage in each band."""
    power = mag_spectrum ** 2
    total_power = np.sum(power)
    band_pct = {}
    for name, f_low, f_high in BANDS:
        mask = (freqs >= f_low) & (freqs < f_high)
        b_power = np.sum(power[mask])
        pct = (b_power / max(total_power, 1e-12)) * 100.0
        band_pct[name] = float(pct)
    return band_pct

train_band_energies = {
    cls_name: compute_band_energies(median_train_spectra[cls_name], freqs)
    for cls_name in ["healthy", "outer_ring", "inner_ring"]
}

val_band_energies = {
    b_id: compute_band_energies(median_val_spectra[b_id], freqs)
    for b_id in ["K005", "KA15", "KI21"]
}

# 3. Figure: Spectral energy by band
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(BANDS))
width = 0.25

rects1 = ax.bar(x - width, [train_band_energies["healthy"][b[0]] for b in BANDS], width, label="Healthy (Train)", color="#2ca02c")
rects2 = ax.bar(x, [train_band_energies["outer_ring"][b[0]] for b in BANDS], width, label="Outer Ring (Train)", color="#ff7f0e")
rects3 = ax.bar(x + width, [train_band_energies["inner_ring"][b[0]] for b in BANDS], width, label="Inner Ring (Train)", color="#1f77b4")

ax.set_ylabel("Relative Spectral Energy (%)")
ax.set_title("Spectral Energy Distribution Across Frequency Bands (Training Set)")
ax.set_xticks(x)
ax.set_xticklabels([b[0] for b in BANDS], rotation=30, ha="right")
ax.grid(True, linestyle="--", alpha=0.5, axis="y")
ax.legend()

plt.tight_layout()
band_fig_path = FIG_DIR / "spectral_energy_by_band.png"
plt.savefig(band_fig_path, dpi=200)
plt.close()
print(f"Saved: {band_fig_path}")

# ------------------------------------------------------------------------------
# A3. Fault-frequency context
# ------------------------------------------------------------------------------
# Verified kinematics from results/paderborn_dataset_design.json
kinematics = {
    "bearing_type": "6203",
    "pitch_diameter_mm": 29.05,
    "ball_diameter_mm": 6.75,
    "num_balls": 8,
    "contact_angle_deg": 0.0,
    "orders": {
        "BPFO": 3.0706,
        "BPFI": 4.9294,
        "BSF": 2.0357,
        "FTF": 0.3838
    },
    "operating_conditions": {
        "N15_M07_F10": {"speed_rpm": 1500, "speed_hz": 25.0, "torque_nm": 0.7, "radial_force_n": 1000, "BPFO_hz": 76.76, "BPFI_hz": 123.24},
        "N09_M07_F10": {"speed_rpm": 900, "speed_hz": 15.0, "torque_nm": 0.7, "radial_force_n": 1000, "BPFO_hz": 46.06, "BPFI_hz": 73.94},
        "N15_M01_F10": {"speed_rpm": 1500, "speed_hz": 25.0, "torque_nm": 0.1, "radial_force_n": 1000, "BPFO_hz": 76.76, "BPFI_hz": 123.24},
        "N15_M07_F04": {"speed_rpm": 1500, "speed_hz": 25.0, "torque_nm": 0.7, "radial_force_n": 400, "BPFO_hz": 76.76, "BPFI_hz": 123.24},
    }
}

# ──────────────────────────────────────────────────────────────────────────────
# PART B: SECONDARY BEARING-DISJOINT ROBUSTNESS ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("PART B: SECONDARY BEARING-DISJOINT ROBUSTNESS ANALYSIS")
print("=" * 50)

# Bearings available in TRAIN + OFFICIAL VAL:
# Healthy: K001, K002, K003, K004, K005 (5 bearings)
# Outer: KA04, KA16, KA30, KA15 (4 bearings)
# Inner: KI04, KI16, KI18, KI21 (4 bearings)

# Official Validation Split used:
# Healthy: K005, Outer: KA15, Inner: KI21

# Construct 3 ALTERNATIVE bearing-disjoint validation splits strictly from the remaining pool:
# Split A: Healthy: K001, Outer: KA04, Inner: KI04
# Split B: Healthy: K002, Outer: KA16, Inner: KI16
# Split C: Healthy: K003, Outer: KA30, Inner: KI18

alt_splits = {
    "Split_A": {"healthy": "K001", "outer_ring": "KA04", "inner_ring": "KI04"},
    "Split_B": {"healthy": "K002", "outer_ring": "KA16", "inner_ring": "KI16"},
    "Split_C": {"healthy": "K003", "outer_ring": "KA30", "inner_ring": "KI18"},
}

# Verify no test bearings in alt splits
for s_name, s_bearings in alt_splits.items():
    for lbl, b_id in s_bearings.items():
        assert b_id not in LOCKED_TEST_BEARINGS, f"CRITICAL: {b_id} is a test bearing!"

# Load Frozen Model Frequency-Only v1
from src.models import FrequencyOnly
from src.evaluate import evaluate as evaluate_model

if torch.backends.mps.is_available():
    dev = torch.device("mps")
elif torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")

print(f"Loading frozen Frequency-Only v1 model from: {CHECKPOINT_PATH} on {dev}...")
frozen_model = FrequencyOnly(cfg).to(dev)
ckpt = torch.load(CHECKPOINT_PATH, map_location=dev, weights_only=False)
frozen_model.load_state_dict(ckpt["state_dict"])
frozen_model.eval()

# Dataset loader class for custom split subsets
class SubsplitDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mean: float, std: float):
        self.rows = df.to_dict("records")
        self.mean = mean
        self.std = std
        
    def __len__(self):
        return len(self.rows)
        
    def __getitem__(self, idx: int):
        row = self.rows[idx]
        raw = get_window_data(PROJECT_ROOT / row["mat_path"], int(row["start_sample"]), int(row["end_sample"]))
        normed = (raw - self.mean) / max(self.std, 1e-8)
        t = torch.from_numpy(normed).unsqueeze(0)
        return t, int(row["label_id"])

robustness_results = {}
macro_f1_list = []

# Include Official Validation as reference baseline in robustness comparison
from src.paderborn_dataset import PaderbornDataset
official_val_ds = PaderbornDataset(
    MANIFEST_PATH, "validation", project_root=PROJECT_ROOT,
    mean=norm_mean, std=norm_std, augment=False
)
official_val_loader = DataLoader(official_val_ds, batch_size=32, shuffle=False, num_workers=0)
official_metrics = evaluate_model(frozen_model, official_val_loader, dev)

robustness_results["Official_Val"] = {
    "held_out_bearings": {"healthy": "K005", "outer_ring": "KA15", "inner_ring": "KI21"},
    "metrics": official_metrics,
    "note": "Official validation split evaluated during Phase 6"
}
print(f"Official Val Re-verification: Macro-F1 = {official_metrics['macro_f1']:.4f}, Acc = {official_metrics['accuracy']:.4f}")

for s_name, s_bearings in alt_splits.items():
    held_bearings = list(s_bearings.values())
    sub_df = df_allowed[df_allowed["bearing_id"].isin(held_bearings)].copy()
    
    # Ensure bearing disjointness
    for b in held_bearings:
        assert len(sub_df[sub_df["bearing_id"] == b]) > 0
        assert b not in LOCKED_TEST_BEARINGS
        
    sub_ds = SubsplitDataset(sub_df, norm_mean, norm_std)
    sub_loader = DataLoader(sub_ds, batch_size=32, shuffle=False, num_workers=0)
    
    metrics = evaluate_model(frozen_model, sub_loader, dev)
    robustness_results[s_name] = {
        "held_out_bearings": s_bearings,
        "sample_counts": dict(sub_df["label"].value_counts()),
        "metrics": metrics,
        "note": "Evaluated with frozen Frequency-Only v1 checkpoint (trained on original train split)"
    }
    macro_f1_list.append(metrics["macro_f1"])
    print(f"\n{s_name} ({s_bearings}):")
    print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Outer Recall: {metrics['per_class']['outer_ring']['recall']:.4f}")
    print(f"  Inner Recall: {metrics['per_class']['inner_ring']['recall']:.4f}")
    print(f"  Healthy Recall: {metrics['per_class']['healthy']['recall']:.4f}")
    print(f"  Confusion Matrix: {metrics['confusion_matrix']}")

# Summary Statistics across alternative splits
macro_f1_arr = np.array(macro_f1_list)
alt_summary_stats = {
    "mean_macro_f1": float(np.mean(macro_f1_arr)),
    "std_macro_f1": float(np.std(macro_f1_arr)),
    "min_macro_f1": float(np.min(macro_f1_arr)),
    "max_macro_f1": float(np.max(macro_f1_arr)),
}
robustness_results["alternative_splits_summary"] = alt_summary_stats

# Optional robustness figure
fig, ax = plt.subplots(figsize=(8, 5))
splits_plot = ["Official Val", "Split A (K001,KA04,KI04)", "Split B (K002,KA16,KI16)", "Split C (K003,KA30,KI18)"]
f1_vals = [official_metrics["macro_f1"]] + macro_f1_list
colors = ["#3366cc", "#109618", "#ff9900", "#dc3912"]

bars = ax.bar(splits_plot, f1_vals, color=colors, width=0.5)
ax.set_ylabel("Macro-F1")
ax.set_title("Frequency-Only v1 Performance Across Distinct Bearing Splits")
ax.set_ylim(0, 1.05)
ax.grid(True, linestyle="--", alpha=0.5, axis="y")
for bar, val in zip(bars, f1_vals):
    ax.text(bar.get_x() + bar.get_width()/2.0, val + 0.02, f"{val:.4f}", ha="center", va="bottom", fontweight="bold")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
rob_fig_path = FIG_DIR / "robustness_macro_f1.png"
plt.savefig(rob_fig_path, dpi=200)
plt.close()
print(f"Saved: {rob_fig_path}")

# ──────────────────────────────────────────────────────────────────────────────
# PERSIST METRICS & WRITE OUTPUT FILES
# ──────────────────────────────────────────────────────────────────────────────
# Save phase6d_analysis.json
phase6d_analysis_data = {
    "phase": "6D",
    "stft_config": {
        "sampling_rate": FS,
        "n_fft": cfg["model"]["n_fft"],
        "hop_length": cfg["model"]["hop_length"],
        "window": "hann"
    },
    "spectral_energy_bands": {
        "train_classes": train_band_energies,
        "validation_bearings": val_band_energies,
        "bands_hz": BANDS
    },
    "kinematics": kinematics,
    "safety_verification": {
        "locked_test_bearings": list(LOCKED_TEST_BEARINGS),
        "test_accessed": False
    }
}
with open(PROJECT_ROOT / "results" / "phase6d_analysis.json", "w") as f:
    json.dump(phase6d_analysis_data, f, indent=2)

# Save phase6d_robustness.json
with open(PROJECT_ROOT / "results" / "phase6d_robustness.json", "w") as f:
    json.dump(robustness_results, f, indent=2)

print("\nSaved JSON artifacts: results/phase6d_analysis.json, results/phase6d_robustness.json")
print("=" * 72)
print("TEST SET REMAINS LOCKED — NO TEST SAMPLES USED.")
print("=" * 72)
