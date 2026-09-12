"""Phase 7: SentinelAI-MSRF v2 — Full 30-Epoch Training.

STRICT PROTOCOL RULES:
1. TEST SET IS LOCKED — never access K006, KA22, KI14 or split=='test'.
2. Foreground execution only; no background, nohup, or detached processes.
3. Checkpoint saved only on validation Macro-F1 improvement.
4. Early stopping: patience=6 epochs with no val Macro-F1 improvement.
5. Report full validation metrics and confusion matrix after training.
6. DO NOT load or evaluate the test set.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path("/Users/nbn/Desktop/hackathons /dlsat").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST_PATH    = PROJECT_ROOT / "data" / "metadata" / "paderborn_manifest.csv"
NORM_STATS_PATH  = PROJECT_ROOT / "data" / "metadata" / "paderborn_norm_stats.json"
CONFIG_PATH      = PROJECT_ROOT / "configs" / "config.yaml"
CKPT_PATH        = PROJECT_ROOT / "results" / "checkpoints" / "msrf_v2_best.pt"
EXP_DIR          = PROJECT_ROOT / "results" / "experiments" / "msrf_v2"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ── Safety & audit ────────────────────────────────────────────────────────────
EXPECTED_SHA256 = "71916a2f142813455013024017bc05eb97406ff44d85c53da32efbb6f6d2ddaf"
with open(MANIFEST_PATH, "rb") as f:
    actual_sha256 = hashlib.sha256(f.read()).hexdigest()
assert actual_sha256 == EXPECTED_SHA256, f"Manifest SHA mismatch: {actual_sha256}"

df_manifest = pd.read_csv(MANIFEST_PATH)
LOCKED_TEST_BEARINGS = {"K006", "KA22", "KI14"}

train_df = df_manifest[df_manifest["split"] == "train"]
val_df   = df_manifest[df_manifest["split"] == "validation"]

train_bearings = sorted(train_df["bearing_id"].unique())
val_bearings   = sorted(val_df["bearing_id"].unique())

for b in train_bearings + val_bearings:
    assert b not in LOCKED_TEST_BEARINGS, f"CRITICAL LEAK: {b} is a test bearing!"

print("=" * 65)
print("PHASE 7 MSRF-V2 FULL TRAINING — PRE-FLIGHT AUDIT")
print("=" * 65)
print(f"MANIFEST_SHA        = {actual_sha256} (PASS)")
print(f"TEST_DATA_ACCESSED  = NO")
print(f"TEST_BEARINGS       = NEVER LOADED")
print(f"SEED                = 42")
print(f"TRAIN_BEARINGS      = {train_bearings}")
print(f"VALIDATION_BEARINGS = {val_bearings}")
print("=" * 65)

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Config & norm stats ───────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

with open(NORM_STATS_PATH) as f:
    ns = json.load(f)
norm_mean, norm_std = float(ns["mean"]), float(ns["std"])
assert abs(norm_mean - 0.008097) < 1e-4
assert abs(norm_std  - 0.353755) < 1e-4
print(f"Normalization: mean={norm_mean:.6f}, std={norm_std:.6f} (PASS)")

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    dev = torch.device("mps")
elif torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")
print(f"Compute device: {dev}")

# ── Model ─────────────────────────────────────────────────────────────────────
from src.models import SentinelAI_MSRF_v2, count_parameters, model_size_kb
from src.paderborn_dataset import PaderbornDataset
from src.evaluate import evaluate

torch.manual_seed(42)
model = SentinelAI_MSRF_v2(cfg, ablation="full").to(dev)
n_params = count_parameters(model)
sz_kb    = model_size_kb(model)
print(f"SentinelAI-MSRF v2 | {n_params:,} parameters | {sz_kb:.2f} KB")
assert 30_000 <= n_params <= 120_000, f"Parameter budget violated: {n_params}"

# ── Datasets & loaders ────────────────────────────────────────────────────────
train_ds = PaderbornDataset(
    MANIFEST_PATH, "train", project_root=PROJECT_ROOT,
    mean=norm_mean, std=norm_std, augment=False
)
val_ds = PaderbornDataset(
    MANIFEST_PATH, "validation", project_root=PROJECT_ROOT,
    mean=norm_mean, std=norm_std, augment=False
)

# Safety: verify no test bearing crept into datasets
for b in train_ds.bearing_ids() + val_ds.bearing_ids():
    assert b not in LOCKED_TEST_BEARINGS, f"CRITICAL: test bearing {b} in dataset!"

BS = cfg["training"]["batch_size"]
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True,  num_workers=0, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BS, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds)} windows | Val: {len(val_ds)} windows | Batch: {BS}")

# ── Optimiser & scheduler ─────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg["training"]["learning_rate"],
    weight_decay=cfg["training"]["weight_decay"],
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=cfg["training"].get("max_epochs", 30),
    eta_min=1e-5,
)
loss_fn = nn.CrossEntropyLoss()

MAX_EPOCHS = cfg["training"].get("max_epochs", 30)
PATIENCE   = cfg["training"]["patience"]

# ── Training loop ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"TRAINING: up to {MAX_EPOCHS} epochs | early-stop patience={PATIENCE}")
print("=" * 65)

best_val_f1   = -1.0
best_val_res  = None
no_improve    = 0
history       = []

total_t0 = time.time()

for epoch in range(1, MAX_EPOCHS + 1):
    epoch_t0 = time.time()
    model.train()
    losses = []
    for x_b, y_b in train_loader:
        x_b, y_b = x_b.to(dev), y_b.to(dev)
        optimizer.zero_grad()
        logits = model(x_b)
        loss   = loss_fn(logits, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

    scheduler.step()
    epoch_time = time.time() - epoch_t0
    train_loss = sum(losses) / len(losses)

    # Validation
    val_res = evaluate(model, val_loader, dev)
    val_f1  = val_res["macro_f1"]
    val_acc = val_res["accuracy"]

    improved = val_f1 > best_val_f1
    if improved:
        best_val_f1  = val_f1
        best_val_res = val_res
        no_improve   = 0
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "validation": val_res,
            "config": cfg,
            "model_name": "SentinelAI_MSRF_v2",
            "n_params": n_params,
            "size_kb": sz_kb,
        }, CKPT_PATH)
        marker = " *** BEST ***"
    else:
        no_improve += 1
        marker = f" (no improve {no_improve}/{PATIENCE})"

    print(
        f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val Macro-F1: {val_f1:.4f} | "
        f"{epoch_time:.1f}s{marker}"
    )

    history.append({
        "epoch": epoch,
        "train_loss": float(train_loss),
        "val_accuracy": float(val_acc),
        "val_macro_f1": float(val_f1),
        "improved": improved,
        "time_s": float(epoch_time),
    })

    if no_improve >= PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch}.")
        break

total_time = time.time() - total_t0

# ── Final report ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PHASE 7 MSRF-V2 FULL TRAINING COMPLETE")
print("=" * 65)
print(f"Total training time : {total_time:.1f}s")
print(f"Best Val Macro-F1   : {best_val_f1:.4f}")
print(f"Best Val Accuracy   : {best_val_res['accuracy']:.4f}")
print(f"Best checkpoint     : {CKPT_PATH}")
print()
print("Per-class (best checkpoint epoch):")
for c_name, c_m in best_val_res["per_class"].items():
    print(f"  {c_name:<12} | P: {c_m['precision']:.4f} | R: {c_m['recall']:.4f} | F1: {c_m['f1']:.4f}")
print()
print("Best validation confusion matrix:")
for row in best_val_res["confusion_matrix"]:
    print(" ", row)

# Comparison against Frequency-Only v1 baseline
print()
FREQ_ONLY_V1_VAL_F1 = 0.6152
delta = best_val_f1 - FREQ_ONLY_V1_VAL_F1
print(f"Reference (Frequency-Only v1) Val Macro-F1 = {FREQ_ONLY_V1_VAL_F1:.4f}")
print(f"SentinelAI-MSRF v2            Val Macro-F1 = {best_val_f1:.4f}")
print(f"Delta vs reference                         = {delta:+.4f}")
if delta > 0:
    print("VERDICT: IMPROVEMENT over Phase 6 best model.")
elif delta == 0:
    print("VERDICT: EQUAL to Phase 6 best model.")
else:
    print("VERDICT: BELOW Phase 6 best model.")

print()
print("TEST DATA ACCESSED  = NO (STRICTLY PRESERVED)")
print("=" * 65)

# ── Save full results JSON ────────────────────────────────────────────────────
results_dict = {
    "phase": "Phase 7 — SentinelAI-MSRF v2 Full Training",
    "manifest_sha256": actual_sha256,
    "test_set_accessed": False,
    "model": "SentinelAI_MSRF_v2",
    "n_params": n_params,
    "size_kb": sz_kb,
    "total_training_time_s": float(total_time),
    "config": {
        "lr": cfg["training"]["learning_rate"],
        "wd": cfg["training"]["weight_decay"],
        "batch_size": BS,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "grad_clip": 1.0,
        "seed": 42,
    },
    "best_epoch_validation": best_val_res,
    "best_val_macro_f1": best_val_f1,
    "reference_freq_only_v1_val_f1": FREQ_ONLY_V1_VAL_F1,
    "delta_vs_reference": float(delta),
    "training_history": history,
}

out_json = EXP_DIR / "phase7_msrf_v2_full_results.json"
with open(out_json, "w") as f:
    json.dump(results_dict, f, indent=2)
print(f"\nFull results saved: {out_json}")
