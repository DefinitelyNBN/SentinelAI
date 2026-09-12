"""SentinelAI Phase 6 Experiment Execution Pipeline.

Executes:
1. Baseline 1D CNN
2. Temporal Only
3. Frequency Only
4. Full SentinelAI

Protocol:
- Seed: 42
- Manifest: data/metadata/paderborn_manifest.csv
- Normalization: training-only Z-score (data/metadata/paderborn_norm_stats.json)
- Augmentation: strictly disabled (augment=False)
- Early stopping: patience=6 on validation macro-F1
- Frozen test evaluation performed ONLY after validation model selection is complete.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluate import evaluate, CLASS_NAMES
from .models import (
    Baseline1DCNN,
    TemporalOnly,
    FrequencyOnly,
    SentinelAI,
    count_parameters,
    model_size_kb,
)
from .paderborn_dataset import build_datasets
from .train import fit
from .utils import device, save_json, set_seed

EXPERIMENT_ORDER = [
    ("baseline_1dcnn", "Baseline 1-D CNN"),
    ("temporal_only", "Temporal Only (1-D Conv)"),
    ("frequency_only", "Frequency Only (STFT 2-D Conv)"),
    ("sentinelai", "Full SentinelAI (Dual-Branch Fusion)")
]


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def instantiate_model(exp_key: str, cfg: dict) -> torch.nn.Module:
    if exp_key == "baseline_1dcnn":
        return Baseline1DCNN(num_classes=cfg["dataset"]["num_classes"])
    elif exp_key == "temporal_only":
        return TemporalOnly(cfg)
    elif exp_key == "frequency_only":
        return FrequencyOnly(cfg)
    elif exp_key == "sentinelai":
        return SentinelAI(cfg, variant="sentinelai")
    else:
        raise ValueError(f"Unknown experiment key: {exp_key}")


def plot_confusion_matrix(cm: list[list[int]], class_names: list[str], title: str, save_path: Path):
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=150)
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm_arr.shape[1]),
        yticks=np.arange(cm_arr.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
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


def plot_training_curves(histories: dict[str, list[dict]], save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Loss curves
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    for name, hist in histories.items():
        epochs = [h["epoch"] for h in hist]
        train_loss = [h["train_loss"] for h in hist]
        ax.plot(epochs, train_loss, marker="o", markersize=3, label=f"{name} (train)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross Entropy Loss")
    ax.set_title("Training Loss vs Epoch")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_dir / "loss_curves.png")
    plt.close(fig)

    # Validation Macro-F1 curves
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    for name, hist in histories.items():
        epochs = [h["epoch"] for h in hist]
        val_f1 = [h["macro_f1"] for h in hist]
        ax.plot(epochs, val_f1, marker="s", markersize=3, label=f"{name} (val F1)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Macro-F1")
    ax.set_title("Validation Macro-F1 vs Epoch")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_dir / "val_f1_curves.png")
    plt.close(fig)


def run_phase6_pipeline(cfg: dict) -> dict[str, Any]:
    """Execute complete Phase 6 training, ablation, and test evaluation."""
    seed = cfg.get("seed", 42)
    set_seed(seed)
    dev = device()

    manifest_path = Path(cfg["dataset"]["manifest"])
    norm_path = Path(cfg["dataset"]["norm_stats"])
    manifest_sha256 = file_sha256(manifest_path)

    # 1. Build Datasets
    print(f"Loading datasets from manifest: {manifest_path} (SHA256: {manifest_sha256[:12]}...)")
    train_ds, val_ds, test_ds, norm_mean, norm_std = build_datasets(cfg, norm_path)
    
    # Strictly enforce augment=False for all 4 primary experiments
    train_ds.augment = False
    val_ds.augment = False
    test_ds.augment = False

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]

    # Dataloaders (with deterministic seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=torch.Generator().manual_seed(seed)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    print(f"Dataset summary: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    print(f"Normalization (training only): mean={norm_mean:.6f}, std={norm_std:.6f}")

    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    checkpoints_dir = results_dir / "checkpoints"

    validation_results = {}
    histories = {}
    model_infos = {}

    # =========================================================================
    # PHASE 6A: TRAINING & VALIDATION SELECTION (TEST SET STRICTLY LOCKED)
    # =========================================================================
    print("\n=======================================================")
    print("PHASE 6A: TRAINING 4 MODELS (TEST SET STRICTLY LOCKED)")
    print("=======================================================")

    for exp_key, display_name in EXPERIMENT_ORDER:
        # Reset seed before each model initialization for absolute determinism
        set_seed(seed)
        model = instantiate_model(exp_key, cfg).to(dev)
        n_params = count_parameters(model)
        size_kb = model_size_kb(model)
        checkpoint_file = checkpoints_dir / f"{exp_key}.pt"

        print(f"\n[{display_name}] Parameters: {n_params:,} ({size_kb:.1f} KB)")
        best_val, history, train_time = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            dev=dev,
            checkpoint=checkpoint_file,
            model_name=display_name
        )

        validation_results[exp_key] = best_val
        histories[exp_key] = history
        model_infos[exp_key] = {
            "display_name": display_name,
            "parameter_count": n_params,
            "model_size_kb": size_kb,
            "training_time_s": train_time,
            "epochs_trained": len(history),
            "checkpoint_path": str(checkpoint_file)
        }

        # Save individual experiment outputs
        exp_dir = results_dir / "experiments" / exp_key
        exp_dir.mkdir(parents=True, exist_ok=True)
        save_json(best_val, exp_dir / "validation_metrics.json")
        save_json(history, exp_dir / "history.json")
        save_json(model_infos[exp_key], exp_dir / "model_info.json")
        plot_confusion_matrix(
            best_val["confusion_matrix"],
            CLASS_NAMES,
            f"{display_name} - Validation CM",
            figures_dir / f"cm_val_{exp_key}.png"
        )

    # Plot loss and F1 curves
    plot_training_curves(histories, figures_dir)

    # Save overall validation experiments summary
    val_summary = {
        "metadata": {
            "seed": seed,
            "device": str(dev),
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "manifest_sha256": manifest_sha256,
            "normalization": {"mean": norm_mean, "std": norm_std},
            "train_bearings": train_ds.bearing_ids(),
            "val_bearings": val_ds.bearing_ids(),
            "test_bearings": test_ds.bearing_ids(),
        },
        "models": {
            k: {
                "info": model_infos[k],
                "validation": validation_results[k]
            }
            for k, _ in EXPERIMENT_ORDER
        }
    }
    save_json(val_summary, results_dir / "metrics" / "validation_experiments.json")

    # =========================================================================
    # PHASE 6B: MAIN ABLATION ANALYSIS
    # =========================================================================
    primary_metric = cfg.get("primary_metric", "macro_f1")
    sentinelai_val_f1 = validation_results["sentinelai"][primary_metric]
    temporal_val_f1 = validation_results["temporal_only"][primary_metric]
    frequency_val_f1 = validation_results["frequency_only"][primary_metric]
    baseline_val_f1 = validation_results["baseline_1dcnn"][primary_metric]

    delta_freq_branch = sentinelai_val_f1 - temporal_val_f1
    delta_vs_baseline = sentinelai_val_f1 - baseline_val_f1

    ablation_summary = {
        "hypothesis": "Combining temporal and frequency representations will produce better bearing-damage classification than using either representation alone.",
        "primary_metric": primary_metric,
        "primary_ablation": {
            "description": "Full SentinelAI vs Temporal-Only (impact of adding STFT frequency branch)",
            "sentinelai_val_f1": sentinelai_val_f1,
            "temporal_only_val_f1": temporal_val_f1,
            "improvement_absolute": delta_freq_branch,
            "improvement_pct": (delta_freq_branch / max(temporal_val_f1, 1e-6)) * 100
        },
        "all_models_validation_comparison": {
            "baseline_1dcnn": baseline_val_f1,
            "temporal_only": temporal_val_f1,
            "frequency_only": frequency_val_f1,
            "sentinelai": sentinelai_val_f1
        },
        "sentinelai_vs_baseline": {
            "improvement_absolute": delta_vs_baseline,
            "improvement_pct": (delta_vs_baseline / max(baseline_val_f1, 1e-6)) * 100
        },
        "hypothesis_supported_on_validation": bool(
            sentinelai_val_f1 > temporal_val_f1 and sentinelai_val_f1 > frequency_val_f1
        )
    }
    save_json(ablation_summary, results_dir / "metrics" / "ablation.json")

    # =========================================================================
    # PHASE 6C: MODEL FREEZING AND FINAL TEST EVALUATION
    # =========================================================================
    print("\n=======================================================")
    print("PHASE 6C: MODEL FREEZING AND FINAL TEST SET EVALUATION")
    print("=======================================================")
    # Select best model based STRICTLY on validation macro-F1
    ranked_models = sorted(
        validation_results.keys(),
        key=lambda k: validation_results[k][primary_metric],
        reverse=True
    )
    selected_model_key = ranked_models[0]
    print(f"Validation Ranking (by {primary_metric}):")
    for rank, key in enumerate(ranked_models, 1):
        print(f"  {rank}. {model_infos[key]['display_name']}: {validation_results[key][primary_metric]:.4f}")
    print(f"\n--> Selected Model: {model_infos[selected_model_key]['display_name']} ({selected_model_key})")
    print("All architectural, hyperparameter, and checkpoint decisions are now FROZEN.")
    print("Evaluating all frozen checkpoints on the locked test set...\n")

    test_results = {}
    for exp_key, display_name in EXPERIMENT_ORDER:
        model = instantiate_model(exp_key, cfg).to(dev)
        checkpoint_file = checkpoints_dir / f"{exp_key}.pt"
        state = torch.load(checkpoint_file, map_location=dev, weights_only=False)
        model.load_state_dict(state["state_dict"])

        test_metrics = evaluate(model, test_loader, dev)
        test_results[exp_key] = test_metrics

        exp_dir = results_dir / "experiments" / exp_key
        save_json(test_metrics, exp_dir / "test_metrics.json")
        plot_confusion_matrix(
            test_metrics["confusion_matrix"],
            CLASS_NAMES,
            f"{display_name} - Test Set CM",
            figures_dir / f"cm_test_{exp_key}.png"
        )
        print(
            f"  {display_name:<38} | "
            f"Test Acc: {test_metrics['accuracy']:.4f} | "
            f"Test Macro-F1: {test_metrics['macro_f1']:.4f}"
        )

    test_summary = {
        "selected_model": selected_model_key,
        "selected_model_name": model_infos[selected_model_key]["display_name"],
        "primary_metric": primary_metric,
        "test_results": test_results
    }
    save_json(test_summary, results_dir / "metrics" / "test_evaluation.json")

    # =========================================================================
    # PHASE 6D: FINAL COMPARISON ARTIFACTS
    # =========================================================================
    final_dir = results_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    for exp_key, display_name in EXPERIMENT_ORDER:
        v = validation_results[exp_key]
        t = test_results[exp_key]
        info = model_infos[exp_key]
        row = {
            "model_key": exp_key,
            "model_name": display_name,
            "params": info["parameter_count"],
            "model_size_kb": round(info["model_size_kb"], 1),
            "train_time_s": round(info["training_time_s"], 1),
            "val_accuracy": round(v["accuracy"], 4),
            "val_macro_f1": round(v["macro_f1"], 4),
            "val_macro_precision": round(v["precision_macro"], 4),
            "val_macro_recall": round(v["recall_macro"], 4),
            "test_accuracy": round(t["accuracy"], 4),
            "test_macro_f1": round(t["macro_f1"], 4),
            "test_macro_precision": round(t["precision_macro"], 4),
            "test_macro_recall": round(t["recall_macro"], 4),
        }
        comparison_rows.append(row)

    save_json(comparison_rows, final_dir / "final_model_comparison.json")

    # Write CSV comparison
    with open(final_dir / "final_model_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    # Assemble comprehensive Phase 6 summary dict
    summary_data = {
        "phase": "Phase 6 — Training, Ablation, and Final Evaluation",
        "project": "SentinelAI",
        "dataset": {
            "name": cfg["dataset"]["name"],
            "task": cfg["dataset"]["subtrack"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "nominal_sampling_rate_hz": cfg["dataset"]["sampling_rate"],
            "window_samples": cfg["dataset"]["signal_length"],
            "window_duration_s": 1.0,
            "overlap_pct": 0.0,
            "normalization": {
                "method": "global_zscore_training_only",
                "mean": norm_mean,
                "std": norm_std
            },
            "split": {
                "train_bearings": train_ds.bearing_ids(),
                "val_bearings": val_ds.bearing_ids(),
                "test_bearings": test_ds.bearing_ids(),
                "train_windows": len(train_ds),
                "val_windows": len(val_ds),
                "test_windows": len(test_ds),
                "train_class_counts": train_ds.class_counts(),
                "val_class_counts": val_ds.class_counts(),
                "test_class_counts": test_ds.class_counts(),
            }
        },
        "reproducibility": {
            "seed": seed,
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "platform": platform.platform(),
            "device": str(dev),
            "optimizer": "AdamW",
            "learning_rate": cfg["training"]["learning_rate"],
            "weight_decay": cfg["training"]["weight_decay"],
            "batch_size": batch_size,
            "max_epochs": cfg["training"]["epochs"],
            "patience": cfg["training"]["patience"],
            "augmentation": "None (disabled)",
            "primary_metric": primary_metric
        },
        "model_comparison": comparison_rows,
        "ablation_findings": ablation_summary,
        "final_selection": {
            "selected_model_key": selected_model_key,
            "selected_model_name": model_infos[selected_model_key]["display_name"],
            "selection_criterion": f"Best validation {primary_metric}",
            "validation_macro_f1": validation_results[selected_model_key]["macro_f1"],
            "validation_accuracy": validation_results[selected_model_key]["accuracy"],
            "test_macro_f1": test_results[selected_model_key]["macro_f1"],
            "test_accuracy": test_results[selected_model_key]["accuracy"],
            "baseline_test_macro_f1": test_results["baseline_1dcnn"]["macro_f1"],
            "baseline_test_accuracy": test_results["baseline_1dcnn"]["accuracy"],
            "sentinelai_vs_baseline_test_improvement": round(
                test_results["sentinelai"]["macro_f1"] - test_results["baseline_1dcnn"]["macro_f1"], 4
            ),
            "sentinelai_vs_temporal_test_improvement": round(
                test_results["sentinelai"]["macro_f1"] - test_results["temporal_only"]["macro_f1"], 4
            )
        },
        "scientific_conclusions": {
            "main_hypothesis": "Combining temporal and frequency representations produces better bearing damage classification than using either alone.",
            "hypothesis_supported": bool(
                sentinelai_val_f1 > temporal_val_f1 and sentinelai_val_f1 > frequency_val_f1
            ),
            "did_sentinelai_beat_baseline": bool(sentinelai_val_f1 > baseline_val_f1),
            "caveats_and_limitations": [
                "Bearing-disjoint generalization is exceptionally challenging; machines possess distinct baseline vibration signatures and transfer characteristics.",
                "Results are strictly from the experimental Paderborn University bearing test rig under laboratory conditions.",
                "No claim is made of production deployment, real-time industrial readiness, or predictive maintenance integration."
            ]
        }
    }
    save_json(summary_data, results_dir / "phase6_experiment_summary.json")

    # Generate Markdown Summary
    md_summary = generate_markdown_summary(summary_data, validation_results, test_results, model_infos)
    with open(results_dir / "phase6_experiment_summary.md", "w", encoding="utf-8") as f:
        f.write(md_summary)

    print("\nPhase 6 artifacts successfully generated under results/")
    return summary_data


def generate_markdown_summary(data: dict, val: dict, test: dict, infos: dict) -> str:
    cmp = data["model_comparison"]
    sel = data["final_selection"]
    abl = data["ablation_findings"]
    sc = data["scientific_conclusions"]
    rep = data["reproducibility"]
    ds = data["dataset"]

    md = f"""# SentinelAI Phase 6: Training, Ablation, and Final Evaluation Summary

**Project**: SentinelAI — Industrial Signal Intelligence for Early Machine-Failure Detection  
**Task**: Three-class bearing-disjoint vibration classification (`healthy`, `outer_ring`, `inner_ring`)  
**Dataset**: Paderborn University Bearing Dataset  
**Primary Metric**: Validation Macro-F1 (for model selection)  
**Status**: COMPLETE  

---

## 1. Executive Summary & Core Results

| Model | Parameters | Size (KB) | Train Time (s) | Val Macro-F1 | Val Acc | Test Macro-F1 | Test Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for r in cmp:
        is_sel = " **(Selected)**" if r["model_key"] == sel["selected_model_key"] else ""
        md += f"| **{r['model_name']}**{is_sel} | {r['params']:,} | {r['model_size_kb']} | {r['train_time_s']} | **{r['val_macro_f1']:.4f}** | {r['val_accuracy']:.4f} | **{r['test_macro_f1']:.4f}** | {r['test_accuracy']:.4f} |\n"

    md += f"""
### Key Findings & Hypothesis Evaluation
- **Baseline Validation Macro-F1**: `{val['baseline_1dcnn']['macro_f1']:.4f}`
- **Temporal-Only Validation Macro-F1**: `{val['temporal_only']['macro_f1']:.4f}`
- **Frequency-Only Validation Macro-F1**: `{val['frequency_only']['macro_f1']:.4f}`
- **SentinelAI Validation Macro-F1**: `{val['sentinelai']['macro_f1']:.4f}`
- **SentinelAI vs Baseline Improvement (Val)**: `{abl['sentinelai_vs_baseline']['improvement_absolute']:+.4f}` ({abl['sentinelai_vs_baseline']['improvement_pct']:+.2f}%)
- **SentinelAI vs Temporal-Only Improvement (Val)**: `{abl['primary_ablation']['improvement_absolute']:+.4f}` ({abl['primary_ablation']['improvement_pct']:+.2f}%)
- **Selected Model**: **{sel['selected_model_name']}**
- **Final Test Macro-F1**: `{sel['test_macro_f1']:.4f}` (Accuracy: `{sel['test_accuracy']:.4f}`)
- **Central Hypothesis Supported**: **{"YES" if sc["hypothesis_supported"] else "NO"}**

---

## 2. Dataset & Split Specification

- **Input**: 1-D `vibration_1` signal sliced into 1.0 s windows (64,000 samples @ nominal 64 kHz, 0% overlap).
- **Manifest**: `{ds['manifest_path']}` (SHA-256: `{ds['manifest_sha256']}`)
- **Normalization**: Training-only global Z-score (Mean = `{ds['normalization']['mean']:.6f}`, Std = `{ds['normalization']['std']:.6f}`)
- **Splits** (Strictly Bearing-Disjoint, 0% leakage):
  - **Train**: {', '.join(ds['split']['train_bearings'])} ({ds['split']['train_windows']:,} windows)
  - **Validation**: {', '.join(ds['split']['val_bearings'])} ({ds['split']['val_windows']:,} windows)
  - **Test (Locked)**: {', '.join(ds['split']['test_bearings'])} ({ds['split']['test_windows']:,} windows)

---

## 3. Main Ablation Analysis

The central hypothesis posits that combining temporal and frequency-domain representations yields superior bearing damage classification than either representation alone.

1. **Full SentinelAI vs. Temporal-Only**:
   - Temporal-Only: `{val['temporal_only']['macro_f1']:.4f}` Macro-F1
   - Full SentinelAI: `{val['sentinelai']['macro_f1']:.4f}` Macro-F1
   - Delta from adding Frequency branch: `{abl['primary_ablation']['improvement_absolute']:+.4f}`

2. **Full SentinelAI vs. Frequency-Only**:
   - Frequency-Only: `{val['frequency_only']['macro_f1']:.4f}` Macro-F1
   - Full SentinelAI: `{val['sentinelai']['macro_f1']:.4f}` Macro-F1
   - Delta from adding Temporal branch: `{(val['sentinelai']['macro_f1'] - val['frequency_only']['macro_f1']):+.4f}`

3. **SentinelAI vs. 1D CNN Baseline**:
   - Baseline: `{val['baseline_1dcnn']['macro_f1']:.4f}` Macro-F1
   - SentinelAI: `{val['sentinelai']['macro_f1']:.4f}` Macro-F1
   - Delta vs Baseline: `{abl['sentinelai_vs_baseline']['improvement_absolute']:+.4f}`

---

## 4. Detailed Per-Class Breakdown

### Validation Set Per-Class Metrics
"""
    for exp_key, display_name in EXPERIMENT_ORDER:
        v = val[exp_key]
        md += f"\n#### {display_name} (Validation)\n"
        md += "| Class | Precision | Recall | F1-Score |\n|---|---:|---:|---:|\n"
        for cname in CLASS_NAMES:
            c = v["per_class"][cname]
            md += f"| {cname} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} |\n"
        md += f"\nConfusion Matrix:\n```\n{np.array(v['confusion_matrix'])}\n```\n"

    md += """
---

## 5. Final Locked Test Set Evaluation

> **LOCKED TEST SET RULE VERIFIED**: The test set was evaluated strictly once after all training, architecture decisions, and checkpoint selections were finalized.

"""
    for exp_key, display_name in EXPERIMENT_ORDER:
        t = test[exp_key]
        md += f"\n#### {display_name} (Test Set)\n"
        md += "| Class | Precision | Recall | F1-Score |\n|---|---:|---:|---:|\n"
        for cname in CLASS_NAMES:
            c = t["per_class"][cname]
            md += f"| {cname} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} |\n"
        md += f"\nConfusion Matrix:\n```\n{np.array(t['confusion_matrix'])}\n```\n"

    md += f"""
---

## 6. Reproducibility & Protocol Audit

- **Random Seed**: `{rep['seed']}`
- **Device**: `{rep['device']}`
- **PyTorch Version**: `{rep['pytorch_version']}`
- **Python Version**: `{rep['python_version']}`
- **Optimizer**: `{rep['optimizer']}` (LR: `{rep['learning_rate']}`, Weight Decay: `{rep['weight_decay']}`)
- **Batch Size**: `{rep['batch_size']}`
- **Max Epochs**: `{rep['max_epochs']}`, Patience: `{rep['patience']}`
- **Augmentation**: Disabled for all primary experiments
- **Checkpoints**: Saved under `results/checkpoints/`
- **Figures**: Saved under `results/figures/`

---

## 7. Scientific Honesty & Limitations

1. **Experimental Scope**: SentinelAI is evaluated as an experimental deep-learning signal-classification system on the benchmark Paderborn University bearing test rig.
2. **Generalization Challenge**: Real bearing damage classification across unseen bearings (bearing-disjoint) is challenging because individual mechanical units have distinct transfer functions and background noise baselines.
3. **No Unsubstantiated Claims**: No claim is made of production deployment, real-time edge processing, or predictive maintenance scheduling.
"""
    return md
