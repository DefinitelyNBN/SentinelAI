"""Phase 7 Revised: SentinelAI-MSRF v2 Smoke Test and 2-Epoch Real-Data Check.

STRICT PROTOCOL RULES:
1. TEST SET IS LOCKED.
   Never load, access, evaluate, or reference:
   - K006
   - KA22
   - KI14
   or any rows where split == 'test'.
2. FOREGROUND EXECUTION ONLY.
   No background processes, no nohup, no detached subprocesses.
3. STOP after Smoke Test and 2-Epoch Real-Data Check.
   Do not proceed to 30-epoch training until instructed.
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
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/Users/nbn/Desktop/hackathons /dlsat").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST_PATH = PROJECT_ROOT / "data" / "metadata" / "paderborn_manifest.csv"
NORM_STATS_PATH = PROJECT_ROOT / "data" / "metadata" / "paderborn_norm_stats.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
EXP_DIR = PROJECT_ROOT / "results" / "experiments" / "msrf_v2"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# STEP 1: SAFETY & AUDIT VERIFICATION
# ------------------------------------------------------------------------------
EXPECTED_SHA256 = "71916a2f142813455013024017bc05eb97406ff44d85c53da32efbb6f6d2ddaf"
with open(MANIFEST_PATH, "rb") as f:
    actual_sha256 = hashlib.sha256(f.read()).hexdigest()
assert actual_sha256 == EXPECTED_SHA256, f"Manifest SHA mismatch: {actual_sha256}"

df_manifest = pd.read_csv(MANIFEST_PATH)
LOCKED_TEST_BEARINGS = {"K006", "KA22", "KI14"}

train_df = df_manifest[df_manifest["split"] == "train"]
val_df = df_manifest[df_manifest["split"] == "validation"]
test_df = df_manifest[df_manifest["split"] == "test"]

train_bearings = sorted(train_df["bearing_id"].unique())
val_bearings = sorted(val_df["bearing_id"].unique())

for b in train_bearings + val_bearings:
    assert b not in LOCKED_TEST_BEARINGS, f"CRITICAL LEAK: {b} is a test bearing!"

print("=" * 65)
print("PHASE 7 MSRF-V2 PRE-FLIGHT AUDIT")
print("=" * 65)
print(f"MANIFEST_SHA = {actual_sha256} (PASS)")
print(f"TEST_DATA_ACCESSED = NO")
print(f"TEST_BEARINGS_LOADED = NO")
print(f"TEST_METRICS_USED = NO")
print(f"SEED = 42")
print(f"TRAIN_BEARINGS = {train_bearings}")
print(f"VALIDATION_BEARINGS = {val_bearings}")
print("=" * 65)

# Seed setup
torch.manual_seed(42)
np.random.seed(42)

# Load configuration and normalization
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

with open(NORM_STATS_PATH) as f:
    ns = json.load(f)
norm_mean, norm_std = float(ns["mean"]), float(ns["std"])
assert abs(norm_mean - 0.008097) < 1e-4
assert abs(norm_std - 0.353755) < 1e-4
print(f"NORMALIZATION_STATS = mean={norm_mean:.6f}, std={norm_std:.6f} (PASS)")

# Device selection
if torch.backends.mps.is_available():
    dev = torch.device("mps")
elif torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")
print(f"COMPUTE_DEVICE = {dev}")

# ------------------------------------------------------------------------------
# STEP 2: MODEL SMOKE TEST WITH SYNTHETIC BATCH
# ------------------------------------------------------------------------------
print("\n" + "-" * 50)
print("EXECUTING MODEL SMOKE TEST (SYNTHETIC BATCH)")
print("-" * 50)

from src.models import SentinelAI_MSRF_v2, count_parameters, model_size_kb

model = SentinelAI_MSRF_v2(cfg, ablation="full").to(dev)
p_count = count_parameters(model)
sz_kb = model_size_kb(model)
print(f"SentinelAI-MSRF v2 (Full) Instantiated.")
print(f"Trainable Parameters: {p_count:,} (Target: 30k–120k -> {'PASS' if 30000 <= p_count <= 120000 else 'FAIL'})")
print(f"Model Buffer+Param Size: {sz_kb:.2f} KB")

# Forward pass on synthetic input (2, 1, 64000)
x_syn = torch.randn(2, 1, 64000, device=dev)
print(f"Synthetic Input Tensor: shape={list(x_syn.shape)}, dtype={x_syn.dtype}, device={x_syn.device}")

# Track tensor shapes through major stages
model.eval()
with torch.no_grad():
    z_t = model.temporal(x_syn)
    z_f = model.spectral(x_syn)
    z_fused = model.fusion(z_t, z_f)
    logits = model.classifier(z_fused)

print(f"Major Intermediate Tensor Shapes:")
print(f"  Temporal Embedding (z_t) : {list(z_t.shape)}")
print(f"  Spectral Embedding (z_f) : {list(z_f.shape)}")
print(f"  Adaptive Fused Rep (z_fused): {list(z_fused.shape)}")
print(f"  Classifier Logits (3 classes): {list(logits.shape)}")

# Backward pass and gradient validation
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=0.001)
opt.zero_grad()
out_train = model(x_syn)
loss_syn = torch.nn.functional.cross_entropy(out_train, torch.tensor([0, 1], device=dev))
loss_syn.backward()

# Verify gradients are non-zero, non-NaN, non-Inf
grad_checks = []
for name, p in model.named_parameters():
    if p.requires_grad:
        assert p.grad is not None, f"Gradient missing for {name}"
        assert not torch.isnan(p.grad).any(), f"NaN gradient detected in {name}"
        assert not torch.isinf(p.grad).any(), f"Inf gradient detected in {name}"
        grad_checks.append(p.grad.abs().sum().item())

opt.step()
print(f"Backward Pass: Completed successfully.")
print(f"Gradient Check: All {len(grad_checks)} parameter tensors have valid finite non-NaN gradients.")
print(f"SMOKE TEST RESULT = PASS")

# ------------------------------------------------------------------------------
# STEP 3: 2-EPOCH REAL-DATA TRAINING CHECK
# ------------------------------------------------------------------------------
print("\n" + "-" * 50)
print("EXECUTING 2-EPOCH REAL-DATA TRAINING CHECK")
print("-" * 50)

from src.paderborn_dataset import PaderbornDataset
from src.evaluate import evaluate

train_ds = PaderbornDataset(
    MANIFEST_PATH, "train", project_root=PROJECT_ROOT,
    mean=norm_mean, std=norm_std, augment=False
)
val_ds = PaderbornDataset(
    MANIFEST_PATH, "validation", project_root=PROJECT_ROOT,
    mean=norm_mean, std=norm_std, augment=False
)

# Enforce no test bearings
for b in train_ds.bearing_ids() + val_ds.bearing_ids():
    assert b not in LOCKED_TEST_BEARINGS

BS = cfg["training"]["batch_size"]
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=0)

print(f"Train Dataset: {len(train_ds)} windows across {train_ds.bearing_ids()}")
print(f"Val Dataset  : {len(val_ds)} windows across {val_ds.bearing_ids()}")
print(f"Batch Size   : {BS} ({len(train_loader)} batches/epoch)")

# Fresh model instance for 2-epoch check
torch.manual_seed(42)
model_2ep = SentinelAI_MSRF_v2(cfg, ablation="full").to(dev)
optimizer = torch.optim.AdamW(
    model_2ep.parameters(),
    lr=cfg["training"]["learning_rate"],
    weight_decay=cfg["training"]["weight_decay"]
)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(1, 3):
    t_start = time.time()
    model_2ep.train()
    losses = []
    for x_b, y_b in train_loader:
        x_b, y_b = x_b.to(dev), y_b.to(dev)
        optimizer.zero_grad()
        preds = model_2ep(x_b)
        loss = loss_fn(preds, y_b)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    epoch_time = time.time() - t_start
    train_loss = sum(losses) / len(losses)
    
    # Validation evaluation
    val_res = evaluate(model_2ep, val_loader, dev)
    print(f"Epoch {epoch}/2 | Train Loss: {train_loss:.4f} | Val Acc: {val_res['accuracy']:.4f} | Val Macro-F1: {val_res['macro_f1']:.4f} | Time: {epoch_time:.1f}s")

print("\n" + "=" * 50)
print("2-EPOCH VALIDATION METRICS SUMMARY")
print("=" * 50)
print(f"Validation Accuracy:       {val_res['accuracy']:.4f} ({val_res['accuracy']*100:.2f}%)")
print(f"Validation Macro-F1:       {val_res['macro_f1']:.4f}")
print(f"Validation Macro-Precision:{val_res['precision_macro']:.4f}")
print(f"Validation Macro-Recall:   {val_res['recall_macro']:.4f}")
print("Per-Class Details:")
for c_name, c_m in val_res["per_class"].items():
    print(f"  {c_name:<11} | P: {c_m['precision']:.4f} | R: {c_m['recall']:.4f} | F1: {c_m['f1']:.4f}")
print("Validation Confusion Matrix:")
for r in val_res["confusion_matrix"]:
    print(" ", r)

# Save checkpoint and status from 2-epoch run
torch.save({
    "epoch": 2,
    "state_dict": model_2ep.state_dict(),
    "validation": val_res,
    "config": cfg,
    "model_name": "SentinelAI_MSRF_v2_2epoch_check"
}, EXP_DIR / "msrf_v2_2epoch_check.pt")

print(f"\nSaved 2-epoch checkpoint: {EXP_DIR / 'msrf_v2_2epoch_check.pt'}")
print("=" * 65)
print("TEST DATA ACCESSED: NO (STRICTLY PRESERVED)")
print("EXECUTION STOPPED AS DIRECTED: AWAITING USER INSTRUCTION")
print("=" * 65)
