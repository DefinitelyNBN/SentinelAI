"""Training loop with early stopping for SentinelAI.

Improvements over Phase 6 version:
- Optional class-weighted CrossEntropyLoss (from training labels only)
- Optional CosineAnnealingLR scheduler
- Optional gradient clipping
- All new features are disabled by default for backward compatibility.
"""
from __future__ import annotations
import time
from pathlib import Path
import torch
from .evaluate import evaluate


def compute_class_weights(
    train_loader: torch.utils.data.DataLoader,
    num_classes: int = 3,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from the training set.

    Weights are computed ONLY from training labels. Never call this on
    validation or test loaders.

    Returns
    -------
    weights : torch.Tensor, shape (num_classes,), on ``device``
    """
    import numpy as np
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, y in train_loader:
        for label in y.tolist():
            if 0 <= label < num_classes:
                counts[label] += 1

    total = counts.sum().item()
    if total == 0:
        raise ValueError("Training loader is empty — cannot compute class weights.")

    # Inverse frequency: w_c = total / (num_classes * count_c)
    weights = total / (num_classes * counts.float().clamp(min=1))
    weights = weights / weights.sum() * num_classes  # normalise so mean weight = 1.0

    if device is not None:
        weights = weights.to(device)

    counts_str = ", ".join(f"{c}: {int(counts[c])}" for c in range(num_classes))
    print(f"  Class counts (train): {counts_str}")
    print(f"  Class weights:        {weights.cpu().tolist()}")
    return weights


def fit(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    cfg: dict,
    dev: torch.device,
    checkpoint: str | Path,
    model_name: str = "model",
) -> tuple[dict, list[dict], float]:
    """Train model with early stopping on validation macro-F1.

    Reads from cfg:
      cfg["training"]["learning_rate"]       (required)
      cfg["training"]["weight_decay"]        (required)
      cfg["training"]["epochs"]              (required)
      cfg["training"]["patience"]            (required)
      cfg["training"]["scheduler"]           "cosine" | "none"   (optional, default "none")
      cfg["training"]["grad_clip"]           float > 0           (optional, default 0 = disabled)
      cfg["model"]["use_class_weights"]      bool                (optional, default False)
      cfg["primary_metric"]                  str                 (optional, default "macro_f1")

    Returns
    -------
    best_val : dict
        Validation metrics from the best checkpoint epoch.
    history : list[dict]
        Epoch-by-epoch training and validation records.
    training_time_s : float
        Total training wall-clock time in seconds.
    """
    t = cfg["training"]
    m = cfg.get("model", {})

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=t["learning_rate"],
        weight_decay=t["weight_decay"],
    )

    # --- Loss function ---
    use_class_weights = m.get("use_class_weights", False)
    if use_class_weights:
        print(f"  Computing class weights from training data...")
        num_classes = cfg.get("dataset", {}).get("num_classes", 3)
        weights = compute_class_weights(train_loader, num_classes, dev)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        print(f"  Loss: Weighted CrossEntropyLoss")
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        print(f"  Loss: CrossEntropyLoss (unweighted)")

    # --- LR Scheduler ---
    primary_metric = cfg.get("primary_metric", "macro_f1")
    patience = t.get("patience", 6)
    max_epochs = t.get("epochs", t.get("max_epochs", 30))

    scheduler_type = t.get("scheduler", "none").lower()
    scheduler = None
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max_epochs, eta_min=1e-5
        )
        print(f"  Scheduler: CosineAnnealingLR (T_max={max_epochs}, eta_min=1e-5)")
    else:
        print(f"  Scheduler: none")

    # --- Gradient clipping ---
    grad_clip = float(t.get("grad_clip", 0.0))
    if grad_clip > 0:
        print(f"  Grad clip: max_norm={grad_clip}")

    best_score = -float("inf")
    stale = 0
    history = []
    checkpoint_path = Path(checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print(f"\n--- Training {model_name} on {dev} (Max epochs={max_epochs}, Patience={patience}) ---")

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
            losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        train_loss = sum(losses) / max(len(losses), 1)
        val = evaluate(model, val_loader, dev)
        epoch_time = time.time() - epoch_start
        val_score = val[primary_metric]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "epoch_time_s": epoch_time,
            **val,
        }
        history.append(record)

        is_best = val_score > best_score
        status_flag = ""
        if is_best:
            best_score = val_score
            stale = 0
            status_flag = f" [BEST - {primary_metric}={best_score:.4f}]"
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "validation": val,
                    "config": cfg,
                    "model_name": model_name,
                    "use_class_weights": use_class_weights,
                    "scheduler": scheduler_type,
                },
                checkpoint_path,
            )
        else:
            stale += 1
            status_flag = f" (stale={stale}/{patience})"

        print(
            f"Epoch {epoch:2d}/{max_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val['accuracy']:.4f} | "
            f"Val Macro-F1: {val['macro_f1']:.4f} | "
            f"Time: {epoch_time:.1f}s{status_flag}"
        )

        if stale >= patience:
            print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
            break

    total_time = time.time() - start_time
    print(f"Finished {model_name} in {total_time:.1f}s. Best val {primary_metric}: {best_score:.4f}")

    # Load best checkpoint weights back into model
    state = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model.load_state_dict(state["state_dict"])
    return state["validation"], history, total_time
