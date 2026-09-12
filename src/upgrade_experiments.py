"""Validation-first ablations for the compact ImprovedSentinelAI.

By default this command never instantiates the test dataset. ``--final-test``
is explicit and should only be used after choosing a fusion variant by
validation Macro-F1.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from src.improved_model import ImprovedSentinelAI
from src.paderborn_dataset import build_datasets, validate_bearing_disjoint_manifest
from src.train import fit
from src.evaluate import evaluate
from src.models import count_parameters, model_size_kb
from src.utils import load_config, device, set_seed


def run(cfg, epochs: int | None = None, final_test: bool = False, max_windows: int | None = None, output_dir: str = "results/experiments/improved_sentinelai"):
    cfg = json.loads(json.dumps(cfg))  # local immutable experiment snapshot
    if epochs is not None: cfg["training"]["epochs"] = epochs
    set_seed(cfg.get("seed", 42)); report = validate_bearing_disjoint_manifest(cfg["dataset"]["manifest"])
    train_ds, val_ds, test_ds, mean, std = build_datasets(cfg, include_test=final_test); train_ds.augment = False
    if max_windows is not None:
        if max_windows < 2: raise ValueError("max_windows must be at least 2")
        train_ds, val_ds = Subset(train_ds, range(min(max_windows, len(train_ds)))), Subset(val_ds, range(min(max_windows, len(val_ds))))
    dev = device(); bs = cfg["training"]["batch_size"]
    train = DataLoader(train_ds, batch_size=bs, shuffle=True, generator=torch.Generator().manual_seed(cfg["seed"]))
    val = DataLoader(val_ds, batch_size=bs, shuffle=False)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2)); (out / "split_manifest.json").write_text(json.dumps(report, indent=2)); (out / "normalization.json").write_text(json.dumps({"mean":mean,"std":std}, indent=2))
    results = {}
    for fusion in ("concat", "gated", "attention"):
        set_seed(cfg["seed"]); model = ImprovedSentinelAI(cfg, fusion).to(dev); checkpoint = out / f"{fusion}.pt"
        best, history, elapsed = fit(model, train, val, cfg, dev, checkpoint, f"ImprovedSentinelAI_{fusion}")
        results[fusion] = {"validation": best, "parameters": count_parameters(model), "model_size_kb": model_size_kb(model), "training_time_s": elapsed, "checkpoint": str(checkpoint)}
    selected = max(results, key=lambda name: results[name]["validation"]["macro_f1"])
    final = {"selection_metric":"validation_macro_f1", "selected_fusion":selected, "results":results, "test_evaluated":False}
    if final_test:
        assert test_ds is not None
        model = ImprovedSentinelAI(cfg, selected).to(dev); model.load_state_dict(torch.load(results[selected]["checkpoint"], map_location=dev, weights_only=False)["state_dict"])
        final["test_metrics"] = evaluate(model, DataLoader(test_ds, batch_size=bs, shuffle=False), dev); final["test_evaluated"] = True
    (out / "summary.json").write_text(json.dumps(final, indent=2)); return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--epochs", type=int); parser.add_argument("--final-test", action="store_true"); parser.add_argument("--max-windows", type=int); parser.add_argument("--output-dir", default="results/experiments/improved_sentinelai")
    args = parser.parse_args(); print(json.dumps(run(load_config(), args.epochs, args.final_test, args.max_windows, args.output_dir), indent=2))
