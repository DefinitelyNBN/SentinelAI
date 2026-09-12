"""Configurable, training-safe vibration preprocessing primitives."""
from __future__ import annotations

import numpy as np


class SignalValidationError(ValueError):
    """Raised when an input cannot be safely used as a vibration waveform."""


def validate_signal(signal: np.ndarray, minimum_samples: int | None = None) -> np.ndarray:
    """Return a finite float32 one-dimensional waveform or raise a clear error."""
    if not isinstance(signal, np.ndarray):
        raise SignalValidationError(f"Expected a NumPy array, got {type(signal).__name__}.")
    x = np.asarray(signal).squeeze()
    if x.ndim != 1:
        raise SignalValidationError(f"Expected one vibration channel, got shape {x.shape}.")
    if not np.issubdtype(x.dtype, np.number):
        raise SignalValidationError("Vibration samples must be numeric.")
    if not np.isfinite(x).all():
        raise SignalValidationError("Signal contains NaN or Inf values.")
    if minimum_samples is not None and len(x) < minimum_samples:
        raise SignalValidationError(
            f"Signal has {len(x):,} samples; at least {minimum_samples:,} are required."
        )
    return x.astype(np.float32, copy=False)


def apply_filter(signal: np.ndarray, cfg: dict, sampling_rate: int) -> np.ndarray:
    """Optionally apply a conservative zero-phase Butterworth filter.

    Filtering is deliberately opt-in. If enabled for training, normalization
    statistics must be regenerated from filtered training windows.
    """
    if not cfg.get("filter_enabled", False):
        return signal
    from scipy.signal import butter, sosfiltfilt

    kind = cfg.get("filter_type", "bandpass").lower()
    low, high = float(cfg.get("filter_low_hz", 100)), float(cfg.get("filter_high_hz", 20_000))
    nyquist = sampling_rate / 2
    if kind == "bandpass":
        if not 0 < low < high < nyquist:
            raise SignalValidationError("Bandpass cutoffs must satisfy 0 < low < high < Nyquist.")
        cutoff: float | list[float] = [low, high]
    elif kind == "highpass":
        if not 0 < low < nyquist:
            raise SignalValidationError("Highpass cutoff must be between 0 and Nyquist.")
        cutoff = low
    elif kind == "lowpass":
        if not 0 < high < nyquist:
            raise SignalValidationError("Lowpass cutoff must be between 0 and Nyquist.")
        cutoff = high
    else:
        raise SignalValidationError(f"Unsupported filter_type: {kind!r}.")
    sos = butter(int(cfg.get("filter_order", 4)), cutoff, btype=kind, fs=sampling_rate, output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def normalize(signal: np.ndarray, method: str, mean: float | None, std: float | None) -> np.ndarray:
    """Normalize with supplied training statistics; never estimates from one input."""
    method = method.lower()
    if method == "none":
        return signal.astype(np.float32, copy=False)
    if method != "zscore":
        raise SignalValidationError(f"Unsupported normalization_method: {method!r}.")
    if mean is None or std is None:
        raise SignalValidationError("Z-score normalization requires training-derived mean and std.")
    if not np.isfinite(std) or std <= 0:
        raise SignalValidationError("Normalization standard deviation must be positive and finite.")
    return ((signal - mean) / max(std, 1e-8)).astype(np.float32)


def preprocess_window(signal: np.ndarray, cfg: dict, mean: float | None, std: float | None) -> np.ndarray:
    """Validate, conservatively filter, crop to the configured window, normalize."""
    d, p = cfg["dataset"], cfg.get("preprocessing", {})
    length, fs = int(d["signal_length"]), int(d["sampling_rate"])
    x = validate_signal(signal, length)[:length]
    x = apply_filter(x, p, fs)
    return normalize(x, p.get("normalization_method", "zscore"), mean, std)
