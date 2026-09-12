"""SentinelAI inference-only module.

Provides a clean, training-free API for loading a model checkpoint and
running predictions with optional explainability.

Usage
-----
    from src.inference import load_model, preprocess_signal, predict, explain

    cfg = load_config()
    model, meta = load_model(get_best_checkpoint(cfg), cfg)
    x = preprocess_signal(signal_1d, cfg, meta["mean"], meta["std"])
    result = predict(model, x, device)
    explanation = explain(model, x, device, result["class_id"])

All functions are side-effect-free and do not modify global state.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Class name order is alphabetically sorted (matches LABEL_ORDER in PaderbornDataset)
CLASS_NAMES: list[str] = ["healthy", "outer_ring", "inner_ring"]
CLASS_DISPLAY: list[str] = ["Healthy", "Outer-Ring Damage", "Inner-Ring Damage"]
CLASS_COLORS: list[str] = ["#2ecc71", "#e74c3c", "#f39c12"]


# ─────────────────────────────────────────────────────────────────────────────
# Configuration helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str | Path = "configs/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_best_checkpoint(cfg: dict, project_root: str | Path = ".") -> Path:
    """Return path to the best available checkpoint using the priority list in config.

    Falls back through:
        1. results/checkpoints/msrf_v2_best.pt
        2. results/checkpoints/frequency_only.pt
        3. results/checkpoints/sentinelai.pt
    Raises FileNotFoundError if none exist.
    """
    root = Path(project_root)
    priority = cfg.get("evaluation", {}).get("checkpoint_priority", [
        "results/checkpoints/msrf_v2_best.pt",
        "results/checkpoints/frequency_only.pt",
        "results/checkpoints/sentinelai.pt",
    ])
    for rel_path in priority:
        p = root / rel_path
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No checkpoint found. Tried: {priority}\n"
        "Train a model first: python -m src.experiments"
    )


def load_norm_stats(cfg: dict, project_root: str | Path = ".") -> tuple[float, float]:
    """Load training-derived normalization statistics from JSON file."""
    root = Path(project_root)
    stats_path = root / cfg["dataset"].get("norm_stats", "data/metadata/paderborn_norm_stats.json")
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Normalization stats not found: {stats_path}\n"
            "Run `python -m src.generate_manifest` to generate."
        )
    with open(stats_path) as f:
        ns = json.load(f)
    return float(ns["mean"]), float(ns["std"])


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str | Path,
    cfg: dict,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a model from a checkpoint and return it in eval mode.

    Parameters
    ----------
    checkpoint_path : path to .pt checkpoint file
    cfg : configuration dict (from load_config())
    device : torch.device (auto-detected if None)

    Returns
    -------
    model : nn.Module in eval mode
    meta : dict with keys: model_name, epoch, n_params, size_kb, validation
    """
    if device is None:
        device = _auto_device()

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "unknown")

    # Instantiate the right architecture based on model_name in the checkpoint
    model = _instantiate_model(model_name, cfg)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    from src.models import count_parameters, model_size_kb
    meta = {
        "model_name": model_name,
        "epoch": ckpt.get("epoch", "?"),
        "n_params": count_parameters(model),
        "size_kb": model_size_kb(model),
        "validation": ckpt.get("validation", {}),
        "checkpoint_path": str(ckpt_path),
    }
    return model, meta


def _instantiate_model(model_name: str, cfg: dict) -> torch.nn.Module:
    """Return a freshly-instantiated model for the given name."""
    from src.models import (
        SentinelAI_MSRF_v2,
        FrequencyOnly,
        SentinelAI,
        TemporalOnly,
        Baseline1DCNN,
    )
    from src.improved_model import ImprovedSentinelAI
    name_lower = model_name.lower()
    if "improved_sentinelai" in name_lower or "improvedsentinelai" in name_lower:
        fusion = next((kind for kind in ("concat", "gated", "attention") if name_lower.endswith(f"_{kind}")), None)
        return ImprovedSentinelAI(cfg, fusion)
    if "msrf_v2" in name_lower or "sentinelai_msrf_v2" in name_lower:
        return SentinelAI_MSRF_v2(cfg, ablation="full")
    elif "frequency_only" in name_lower or "frequencyonly" in name_lower:
        return FrequencyOnly(cfg)
    elif "temporal_only" in name_lower or "temporalonly" in name_lower:
        return TemporalOnly(cfg)
    elif "baseline" in name_lower:
        return Baseline1DCNN(cfg["dataset"]["num_classes"])
    elif "sentinelai" in name_lower:
        return SentinelAI(cfg, "sentinelai")
    else:
        # Best-effort: try MSRF_v2 first (most capable)
        return SentinelAI_MSRF_v2(cfg, ablation="full")


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Signal preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_signal(
    signal_1d: np.ndarray,
    cfg: dict,
    mean: float,
    std: float,
) -> torch.Tensor:
    """Validate, window, and normalize a raw vibration signal.

    Parameters
    ----------
    signal_1d : np.ndarray, 1-D float array (any length >= signal_length)
    cfg : config dict
    mean : training-derived mean (from paderborn_norm_stats.json)
    std : training-derived std (from paderborn_norm_stats.json)

    Returns
    -------
    x : torch.Tensor, shape (1, 1, signal_length), float32

    Raises
    ------
    ValueError : with a user-friendly message for any invalid input
    """
    # Validate shape
    if not isinstance(signal_1d, np.ndarray):
        raise ValueError(
            f"Expected a NumPy array, got {type(signal_1d).__name__}.\n"
            "Save your signal as numpy array: np.save('signal.npy', array)"
        )
    signal_1d = signal_1d.squeeze()
    if signal_1d.ndim != 1:
        raise ValueError(
            f"Expected a 1-D signal, got shape {signal_1d.shape}.\n"
            "The signal must be a single channel vibration waveform."
        )

    # Validate for NaN/Inf
    if np.any(np.isnan(signal_1d)):
        raise ValueError(
            "Input signal contains NaN values.\n"
            "Check your data file for missing or corrupted samples."
        )
    if np.any(np.isinf(signal_1d)):
        raise ValueError(
            "Input signal contains Inf values.\n"
            "Check your data file for sensor saturation or clipping."
        )

    signal_length = cfg["dataset"].get("signal_length", 64_000)

    # Validate length
    if len(signal_1d) < signal_length:
        raise ValueError(
            f"Signal is too short: {len(signal_1d)} samples "
            f"(minimum required: {signal_length} = 1 second at "
            f"{cfg['dataset'].get('sampling_rate', 64000)} Hz).\n"
            "Please provide at least 1 second of data."
        )

    # Take first window (truncate if longer)
    x = signal_1d[:signal_length].astype(np.float32)

    # Shared configurable preprocessing. Defaults exactly match legacy z-score.
    from src.preprocessing import preprocess_window
    x = preprocess_window(x, cfg, mean, std)

    # Add batch and channel dims → (1, 1, signal_length)
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)


def compute_stft_spectrogram(signal_1d: np.ndarray, cfg: dict) -> np.ndarray:
    """Compute log-magnitude STFT spectrogram for visualization.

    Returns
    -------
    spec : np.ndarray, shape (freq_bins, time_frames), float32
        Log-magnitude spectrogram (NOT normalized — for display only).
    """
    import scipy.signal as ss  # type: ignore

    n_fft = cfg["model"].get("n_fft", 2048)
    hop = cfg["model"].get("hop_length", 512)
    fs = cfg["dataset"].get("sampling_rate", 64_000)

    signal_length = cfg["dataset"].get("signal_length", 64_000)
    sig = signal_1d.squeeze()[:signal_length].astype(np.float32)

    freqs, times, Zxx = ss.stft(sig, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.log1p(np.abs(Zxx)).astype(np.float32)
    return mag, freqs, times


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run model inference and return structured prediction.

    Parameters
    ----------
    model : nn.Module in eval mode (from load_model)
    x : torch.Tensor, shape (1, 1, signal_length)
    device : torch.device (uses x.device if None)

    Returns
    -------
    result : dict with keys:
        class_id       : int (0, 1, or 2)
        class_name     : str ("healthy" | "outer_ring" | "inner_ring")
        class_display  : str ("Healthy" | "Outer-Ring Damage" | "Inner-Ring Damage")
        class_color    : str (hex color for UI)
        confidence     : float (model confidence = max softmax probability)
        probabilities  : list[float] [p_healthy, p_outer, p_inner]
        logits         : list[float]

    Note: "confidence" here refers to the model's softmax output for the
    predicted class, NOT a calibrated probability. The model has not been
    temperature-scaled or calibrated.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        logits = model(x.to(device))  # (1, num_classes)
        probs = torch.softmax(logits, dim=1).squeeze(0)  # (num_classes,)
        pred_id = int(probs.argmax().item())
        confidence = float(probs[pred_id].item())
        prob_list = probs.cpu().tolist()

    return {
        "class_id": pred_id,
        "class_name": CLASS_NAMES[pred_id],
        "class_display": CLASS_DISPLAY[pred_id],
        "class_color": CLASS_COLORS[pred_id],
        "confidence": confidence,
        "probabilities": prob_list,
        "logits": logits.squeeze(0).cpu().tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Explainability
# ─────────────────────────────────────────────────────────────────────────────

def explain(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device | None = None,
    target_class: int | None = None,
) -> dict[str, Any]:
    """Generate explainability maps for the given input.

    Attempts Grad-CAM on the spectral branch (requires SentinelAI_MSRF_v2).
    Falls back to input-gradient temporal saliency if Grad-CAM is unavailable.

    Parameters
    ----------
    model : nn.Module
    x : torch.Tensor, shape (1, 1, signal_length)
    device : torch.device
    target_class : int or None (if None, uses argmax of model prediction)

    Returns
    -------
    result : dict with keys:
        gradcam_available   : bool
        gradcam_heatmap     : np.ndarray (freq_bins, time_frames) or None
        temporal_saliency   : np.ndarray (signal_length,)
        target_class        : int
        note                : str describing the method used
    """
    if device is None:
        device = next(model.parameters()).device

    x_dev = x.to(device)

    # Determine target class
    if target_class is None:
        with torch.no_grad():
            logits = model(x_dev)
            target_class = int(logits.argmax(dim=1).item())

    from src.models import SpectrogramGradCAM, TemporalGradientSaliency

    # --- Grad-CAM (spectral branch) ---
    gradcam_heatmap = None
    gradcam_available = False
    gradcam_note = "Grad-CAM not available for this model architecture."
    try:
        cam = SpectrogramGradCAM(model)
        gradcam_heatmap = cam.generate(x_dev, target_class)
        cam.remove_hooks()
        gradcam_available = True
        gradcam_note = (
            "Grad-CAM from spectral branch (global encoder conv2). "
            "Bright regions indicate frequency bands most influential for this prediction."
        )
    except (AttributeError, RuntimeError) as e:
        gradcam_note = f"Grad-CAM unavailable ({e}). Using temporal saliency only."

    # --- Temporal input-gradient saliency ---
    try:
        temporal_saliency = TemporalGradientSaliency.generate(model, x_dev, target_class)
    except Exception:
        import numpy as np
        temporal_saliency = np.zeros(x.shape[-1], dtype=np.float32)

    return {
        "gradcam_available": gradcam_available,
        "gradcam_heatmap": gradcam_heatmap,
        "temporal_saliency": temporal_saliency,
        "target_class": target_class,
        "note": gradcam_note,
    }
