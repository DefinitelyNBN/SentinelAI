"""Phase 6B Controlled Validation-Only Experiments.

Strict Test Set Rule:
THE TEST SET IS NEVER ACCESSED DURING THESE EXPERIMENTS.
All tuning and selection are conducted strictly on the validation set.

Experiments:
1. Experiment 6B-A: Full SentinelAI + Training-Derived Class-Weighted CrossEntropyLoss.
2. Experiment 6B-B: Controlled modification (if needed).
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models import SentinelAI, count_parameters, model_size_kb
from src.paderborn_dataset import build_datasets
from src.evaluate import evaluate, compute_metrics, CLASS_NAMES
from src.train import fit
from src.utils import device, load_config, save_json, set_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "results/experiments/phase6b"
EXP_DIR.mkdir(parents=True, exist_ok=True)


def plot_cm(cm: list[list[int]], title: str, save_path: Path):
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=150)
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm_arr.shape[1]),
        yticks=np.arange(cm_arr.shape[0]),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j, i, format(cm_arr[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black"
            )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def train_model_custom(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    dev: torch.device,
    checkpoint: Path,
    loss_fn: torch.nn.Module,
    model_name: str = "model",
    patience: int | None = None,
    lr: float | None = None
) -> tuple[dict, list[dict], float]:
    """Train with custom loss function and early stopping on val macro-F1."""
    import time
    t = cfg["training"]
    learning_rate = lr if lr is not None else t["learning_rate"]
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=t["weight_decay"])
    pat = patience if patience is not None else t.get("patience", 6)
    max_epochs = t.get("epochs", 30)

    best_score = -float("inf")
    stale = 0
    history = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print(f"\n--- Training {model_name} on {dev} (Max epochs={max_epochs}, Patience={pat}, LR={learning_rate}) ---")

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        train_loss = sum(losses) / max(len(losses), 1)
        val = evaluate(model, val_loader, dev)
        epoch_time = time.time() - t0
        val_f1 = val["macro_f1"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "epoch_time_s": epoch_time,
            **val
        }
        history.append(record)

        if val_f1 > best_score:
            best_score = val_f1
            stale = 0
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "validation": val,
                    "config": cfg,
                    "model_name": model_name
                },
                checkpoint
            )
            status = f" [BEST val macro-F1 = {best_score:.4f}]"
        else:
            stale += 1
            status = f" (stale={stale}/{pat})"

        print(
            f"Epoch {epoch:2d}/{max_epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Acc: {val['accuracy']:.4f} | "
            f"Val Macro-F1: {val['macro_f1']:.4f} | "
            f"Outer-Ring Recall: {val['per_class']['outer_ring']['recall']:.4f} | "
            f"Time: {epoch_time:.1f}s{status}"
        )

        if stale >= pat:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    total_time = time.time() - start_time
    state = torch.load(checkpoint, map_location=dev, weights_only=False)
    model.load_state_dict(state["state_dict"])
    return state["validation"], history, total_time


def run_experiment_6ba(cfg: dict, dev: torch.device, train_loader: DataLoader, val_loader: DataLoader) -> dict:
    """Experiment 6B-A: Training-derived class-weighted CrossEntropyLoss."""
    set_seed(42)
    # Calculate weights strictly from training counts:
    # healthy: 1278, outer_ring: 957, inner_ring: 956, total: 3191
    counts = [1278.0, 957.0, 956.0]
    total = sum(counts)
    weights = [total / (3.0 * c) for c in counts]
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=dev)
    print(f"\n[Experiment 6B-A] Training-derived class weights: {weight_tensor.tolist()}")

    model = SentinelAI(cfg, variant="sentinelai").to(dev)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    ckpt_path = EXP_DIR / "sentinelai_class_weighted.pt"

    val_metrics, history, train_time = train_model_custom(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        dev=dev,
        checkpoint=ckpt_path,
        loss_fn=loss_fn,
        model_name="Full SentinelAI (Class-Weighted)"
    )

    plot_cm(
        val_metrics["confusion_matrix"],
        "SentinelAI Class-Weighted - Val CM",
        EXP_DIR / "cm_val_class_weighted.png"
    )

    res = {
        "experiment": "6B-A",
        "description": "Full SentinelAI + training-derived class-weighted CrossEntropyLoss",
        "weights": weights,
        "validation_metrics": val_metrics,
        "history": history,
        "training_time_s": train_time,
        "checkpoint_path": str(ckpt_path)
    }
    save_json(res, EXP_DIR / "results_6ba_class_weighted.json")
    return res


def run_experiment_6bb(cfg: dict, dev: torch.device, train_loader: DataLoader, val_loader: DataLoader) -> dict:
    """Experiment 6B-B: SentinelAI with Projected Feature Fusion Layer."""
    from src.models import SentinelAI_ProjectedFusion
    set_seed(42)
    print("\n[Experiment 6B-B] SentinelAI with Projected Feature Fusion Layer")

    model = SentinelAI_ProjectedFusion(cfg).to(dev)
    loss_fn = torch.nn.CrossEntropyLoss()
    ckpt_path = EXP_DIR / "sentinelai_projected_fusion.pt"

    val_metrics, history, train_time = train_model_custom(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        dev=dev,
        checkpoint=ckpt_path,
        loss_fn=loss_fn,
        model_name="SentinelAI (Projected Fusion)"
    )

    plot_cm(
        val_metrics["confusion_matrix"],
        "SentinelAI Projected Fusion - Val CM",
        EXP_DIR / "cm_val_projected_fusion.png"
    )

    res = {
        "experiment": "6B-B",
        "description": "Full SentinelAI + Projected Feature Fusion Layer (independent LayerNorm+Linear projections)",
        "validation_metrics": val_metrics,
        "history": history,
        "training_time_s": train_time,
        "checkpoint_path": str(ckpt_path)
    }
    save_json(res, EXP_DIR / "results_6bb_projected_fusion.json")
    return res


def run_all_phase6b():
    cfg = load_config()
    dev = device()

    train_ds, val_ds, _, norm_mean, norm_std = build_datasets(cfg)
    train_ds.augment = False
    val_ds.augment = False

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(42)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )

    # 6B-A is already completed; load its results if available or run
    res_a_path = EXP_DIR / "results_6ba_class_weighted.json"
    if res_a_path.exists():
        with open(res_a_path) as f:
            res_a = json.load(f)
        print("Loaded existing Experiment 6B-A results.")
    else:
        print("=======================================================")
        print("PHASE 6B: RUNNING EXPERIMENT 6B-A (CLASS WEIGHTING)")
        print("=======================================================")
        res_a = run_experiment_6ba(cfg, dev, train_loader, val_loader)

    print("\nExperiment 6B-A Results:")
    print(f"  Val Macro-F1: {res_a['validation_metrics']['macro_f1']:.4f}")
    print(f"  Val Accuracy: {res_a['validation_metrics']['accuracy']:.4f}")
    print(f"  Outer Ring Recall: {res_a['validation_metrics']['per_class']['outer_ring']['recall']:.4f}")

    # Check if 6B-A improved validation Macro-F1 meaningfully over original 0.5558
    orig_val_f1 = 0.5558
    f1_a = res_a["validation_metrics"]["macro_f1"]
    
    print("\n=======================================================")
    print("PHASE 6B: RUNNING EXPERIMENT 6B-B (PROJECTED FUSION)")
    print("=======================================================")
    res_b = run_experiment_6bb(cfg, dev, train_loader, val_loader)
    print("\nExperiment 6B-B Results:")
    print(f"  Val Macro-F1: {res_b['validation_metrics']['macro_f1']:.4f}")
    print(f"  Val Accuracy: {res_b['validation_metrics']['accuracy']:.4f}")
    print(f"  Outer Ring Recall: {res_b['validation_metrics']['per_class']['outer_ring']['recall']:.4f}")

    return {"6B-A": res_a, "6B-B": res_b}


if __name__ == "__main__":
    run_all_phase6b()
