"""Low-level MAT-file reader for the Paderborn University Bearing Dataset.

Responsibilities
----------------
- Load a single Paderborn MAT recording.
- Return the raw vibration_1 array.
- Return the HostService time-axis array.
- Extract a deterministic, zero-overlap 1-second window by sample index.

This module performs NO normalisation, NO augmentation, and NO splitting.
It does NOT write any file.  It is the single authoritative interface to the
MAT files used by both the manifest generator and the dataset loader.

Time-axis policy (Phase 4, approved)
--------------------------------------
Sample-index windowing only; no resampling.

Hardware acquisition is nominally 64 kHz with mean dt = 15.625 µs.
HostService timestamps show OS-level jitter, not true clock drift.
Resampling a 64 kHz signal to correct jitter would introduce interpolation
artefacts far larger than the jitter itself.  We therefore treat sample index
as the ground truth for window boundaries and use the stored HostService time
axis only for provenance metadata (start_time_s, end_time_s).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io as sio

# Nominal sampling rate (Hz) — used only for time-axis provenance.
NOMINAL_FS: int = 64_000
# Target window length in samples.
WINDOW_SAMPLES: int = 64_000  # 1.0 second @ 64 kHz


def _load_mat(mat_path: Path) -> object:
    """Load and return the top-level MATLAB struct from a Paderborn MAT file."""
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    stem = mat_path.stem
    if stem not in mat:
        raise KeyError(
            f"Expected top-level struct '{stem}' not found in {mat_path}. "
            f"Available keys: {[k for k in mat if not k.startswith('_')]}"
        )
    return mat[stem]


def load_vibration(mat_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vibration_array, time_axis_array) for a Paderborn MAT file.

    Parameters
    ----------
    mat_path : Path
        Absolute or project-root-relative path to the .mat file.

    Returns
    -------
    vib : np.ndarray, shape (N,), dtype float32
        Raw vibration_1 samples.
    t   : np.ndarray, shape (N,), dtype float64
        HostService time axis in seconds (same length as vib).

    Raises
    ------
    ValueError
        If vibration_1 or its time axis cannot be located.
    """
    data = _load_mat(Path(mat_path))

    # Locate vibration_1 in Y array
    vib = None
    x_index = None
    for y in data.Y:
        if getattr(y, "Name", None) == "vibration_1":
            vib = np.asarray(getattr(y, "Data"), dtype=np.float32).ravel()
            x_index = int(getattr(y, "XIndex", -1))
            break

    if vib is None:
        raise ValueError(f"vibration_1 not found in {mat_path}")

    # Locate the matching HostService time axis.
    # XIndex in the MAT struct is 1-based; X is an array of axis structs.
    t = None
    for x in data.X:
        if getattr(x, "Raster", None) == "HostService":
            t = np.asarray(getattr(x, "Data"), dtype=np.float64).ravel()
            break

    if t is None:
        raise ValueError(f"HostService time axis not found in {mat_path}")

    if len(vib) != len(t):
        # Truncate to the shorter of the two (safety guard)
        n = min(len(vib), len(t))
        vib, t = vib[:n], t[:n]

    return vib, t


def extract_windows(
    vib: np.ndarray,
    t: np.ndarray,
    window_samples: int = WINDOW_SAMPLES,
    stride_samples: int | None = None,
) -> list[dict]:
    """Extract deterministic, non-overlapping windows from a vibration array.

    Parameters
    ----------
    vib            : 1-D float32 array — raw vibration_1 samples.
    t              : 1-D float64 array — HostService time axis (seconds).
    window_samples : Number of samples per window (default 64 000 = 1 s).
    stride_samples : Stride between windows (default = window_samples → 0 % overlap).

    Returns
    -------
    List of dicts, each with keys:
        window_index   – 0-based window number within this recording.
        start_sample   – inclusive start index.
        end_sample     – exclusive end index  (= start_sample + window_samples).
        start_time_s   – time at start_sample from the HostService axis.
        end_time_s     – time at end_sample - 1 from the HostService axis.
        n_samples      – always equals window_samples.
    """
    if stride_samples is None:
        stride_samples = window_samples  # zero overlap

    n_total = len(vib)
    windows = []
    idx = 0
    w = 0
    while idx + window_samples <= n_total:
        end = idx + window_samples
        windows.append(
            {
                "window_index": w,
                "start_sample": idx,
                "end_sample": end,
                "start_time_s": float(t[idx]),
                "end_time_s": float(t[end - 1]),
                "n_samples": window_samples,
            }
        )
        idx += stride_samples
        w += 1

    return windows


def get_window_data(
    mat_path: Path,
    start_sample: int,
    end_sample: int,
) -> np.ndarray:
    """Load a single window from a MAT file by sample index.

    This is the function called by the Dataset __getitem__ at training time.
    It loads only the vibration_1 array (not the time axis) for efficiency.

    Parameters
    ----------
    mat_path     : Path to the source .mat file.
    start_sample : Inclusive start index.
    end_sample   : Exclusive end index.

    Returns
    -------
    np.ndarray, shape (end_sample - start_sample,), dtype float32.
    """
    vib, _ = load_vibration(Path(mat_path))
    window = vib[start_sample:end_sample]
    expected = end_sample - start_sample
    if len(window) != expected:
        raise ValueError(
            f"Window extraction returned {len(window)} samples, "
            f"expected {expected} (start={start_sample}, end={end_sample}, "
            f"signal_len={len(vib)}) in {mat_path}"
        )
    return window
