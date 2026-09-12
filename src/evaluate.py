"""Evaluation metrics calculation for SentinelAI Phase 6.

Pure NumPy/Python implementation of classification metrics:
- Accuracy
- Macro-F1, Macro-Precision, Macro-Recall
- Per-Class Precision, Recall, F1
- Confusion Matrix

Matches sklearn.metrics behavior with zero_division=0 without requiring external scikit-learn.
"""
from __future__ import annotations
import numpy as np
import torch

CLASS_NAMES = ["healthy", "outer_ring", "inner_ring"]

def compute_metrics(y_true: list[int], y_pred: list[int], num_classes: int = 3) -> dict:
    """Compute classification metrics and confusion matrix."""
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1

    total_samples = len(y_true)
    correct = sum(cm[i][i] for i in range(num_classes))
    acc = correct / max(total_samples, 1)

    precisions = []
    recalls = []
    f1s = []
    per_class = {}

    for i in range(num_classes):
        tp = cm[i][i]
        fp = sum(cm[k][i] for k in range(num_classes) if k != i)
        fn = sum(cm[i][k] for k in range(num_classes) if k != i)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
        per_class[name] = {
            "class_id": i,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1)
        }

    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    return {
        "accuracy": float(acc),
        "macro_f1": macro_f1,
        "precision_macro": macro_prec,
        "recall_macro": macro_rec,
        "per_class": per_class,
        "confusion_matrix": cm,
        "num_samples": total_samples
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, dev: torch.device) -> dict:
    model.eval()
    y_true = []
    y_pred = []
    for x, target in loader:
        logits = model(x.to(dev))
        preds = logits.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(target.tolist())

    return compute_metrics(y_true, y_pred, num_classes=len(CLASS_NAMES))
