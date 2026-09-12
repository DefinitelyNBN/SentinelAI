"""SentinelAI — Industrial Signal Intelligence GUI.

A high-precision, dark industrial interface for vibration-based bearing fault classification
using the frozen Frequency Only (STFT 2-D Conv) deep learning model.
Strictly offline and frozen: loads no test datasets and performs no training.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import (
    explain,
    load_config,
    load_model,
    load_norm_stats,
    predict,
    preprocess_signal,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION & INJECTED INDUSTRIAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentinelAI — Industrial Signal Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* ── Reset & Typography ── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e2e8f0;
}

code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Background & Streamlit Overrides */
.stApp {
    background-color: #06090e;
    background-image: 
        radial-gradient(ellipse at top right, rgba(14, 165, 233, 0.04) 0%, transparent 60%),
        radial-gradient(ellipse at bottom left, rgba(16, 185, 129, 0.03) 0%, transparent 50%);
}

#MainMenu, header, footer { visibility: hidden; height: 0; }
[data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"] {
    background-color: #0a0f18;
    border-right: 1px solid #182234;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1280px;
}

/* ── Top Header Navigation ── */
.app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.85rem 1.4rem;
    background: #0b111a;
    border: 1px solid #1a2638;
    border-radius: 8px;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

.brand-title {
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #f8fafc;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.brand-title span.accent {
    color: #38bdf8;
}

.brand-subtitle {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
    margin-top: -2px;
}

.status-badge-online {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.25);
    padding: 0.35rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #34d399;
    letter-spacing: 0.06em;
}

.pulse-dot {
    width: 7px;
    height: 7px;
    background-color: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10b981;
}

/* ── Compact Status Strip ── */
.status-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    background: #0a0f19;
    border: 1px solid #162030;
    border-radius: 6px;
    padding: 0.65rem 1rem;
    margin-bottom: 1.5rem;
}

.status-cell {
    display: flex;
    flex-direction: column;
}

.status-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
}

.status-val {
    font-size: 0.85rem;
    font-weight: 600;
    color: #cbd5e1;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Hero Section ── */
.hero-box {
    padding: 1.25rem 0 1.5rem 0;
}

.hero-headline {
    font-size: 2.15rem;
    font-weight: 800;
    line-height: 1.15;
    color: #f8fafc;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
}

.hero-subheadline {
    font-size: 0.98rem;
    line-height: 1.5;
    color: #94a3b8;
    max-width: 820px;
    margin-bottom: 1rem;
}

.tech-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.65rem;
    background: #111a28;
    border: 1px solid #203046;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #38bdf8;
    text-transform: uppercase;
}

/* ── Engineering Instrument Card ── */
.instrument-card {
    background: #0a0f18;
    border: 1px solid #162234;
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1.5rem;
}

.instrument-card:hover {
    border-color: #1e3048;
}

.section-tag {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #38bdf8;
    margin-bottom: 0.25rem;
}

.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.75rem;
}

/* ── Custom Uploader Dropzone Styling ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] section {
    background-color: #080d16 !important;
    border: 1px dashed #1c2a3e !important;
    border-radius: 6px !important;
    padding: 1.25rem !important;
    transition: all 0.2s ease;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #38bdf8 !important;
    background-color: #0b1320 !important;
}
[data-testid="stFileUploader"] section button {
    background-color: #121d2d !important;
    color: #e2e8f0 !important;
    border: 1px solid #1e314b !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.35rem 0.85rem !important;
    transition: all 0.2s ease;
}
[data-testid="stFileUploader"] section button:hover {
    background-color: #1e3352 !important;
    border-color: #38bdf8 !important;
    color: #ffffff !important;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
}
[data-testid="stFileUploader"] section button span {
    color: #e2e8f0 !important;
}
[data-testid="stFileUploader"] span, 
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] p {
    color: #94a3b8 !important;
}
[data-testid="stFileUploader"] small {
    color: #64748b !important;
}
[data-testid="stFileUploaderFile"] {
    background-color: #0a111c !important;
    border: 1px solid #1a283e !important;
    border-radius: 4px !important;
}
[data-testid="stFileUploaderFile"] span {
    color: #cbd5e1 !important;
}

/* ── Confusion Matrix dark override ── */
.cm-container img {
    border-radius: 6px;
    filter: invert(0);
}
.cm-container [data-testid="stImage"],
.cm-container [data-testid="stImageContainer"] {
    background: #080d16 !important;
    border-radius: 6px;
}
.cm-container figure, .cm-container figure > div {
    background: #080d16 !important;
}

.upload-spec-row {
    display: flex;
    gap: 1.5rem;
    padding: 0.65rem 0.9rem;
    background: #06090e;
    border: 1px solid #162030;
    border-radius: 5px;
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.5rem;
}

.upload-spec-item {
    display: flex;
    align-items: center;
    gap: 0.35rem;
}

.upload-spec-item strong {
    color: #94a3b8;
}

.signal-loaded-box {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(14, 165, 233, 0.06);
    border: 1px solid rgba(14, 165, 233, 0.28);
    border-left: 4px solid #38bdf8;
    border-radius: 6px;
    padding: 0.85rem 1.15rem;
    margin-top: 0.75rem;
}

.signal-loaded-left {
    display: flex;
    flex-direction: column;
}

.signal-loaded-tag {
    font-size: 0.68rem;
    font-weight: 700;
    color: #38bdf8;
    letter-spacing: 0.1em;
}

.signal-loaded-name {
    font-size: 0.95rem;
    font-weight: 600;
    color: #f8fafc;
    font-family: 'JetBrains Mono', monospace;
}

.signal-loaded-meta {
    font-size: 0.75rem;
    color: #94a3b8;
}

.signal-loaded-badge {
    background: #102336;
    border: 1px solid #1e4060;
    color: #7dd3fc;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.3rem 0.7rem;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Pipeline Diagram (Empty State) ── */
.pipeline-diagram {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #080d15;
    border: 1px solid #152233;
    border-radius: 8px;
    padding: 1.5rem 1rem;
    margin: 1.5rem 0;
    overflow-x: auto;
}

.pipeline-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 0.75rem 1rem;
    background: #0e1624;
    border: 1px solid #1e2c42;
    border-radius: 6px;
    min-width: 140px;
}

.pipeline-node-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #38bdf8;
    text-transform: uppercase;
}

.pipeline-node-sub {
    font-size: 0.65rem;
    color: #64748b;
    margin-top: 2px;
}

.pipeline-arrow {
    color: #2b3e58;
    font-size: 1.25rem;
    font-weight: bold;
    padding: 0 0.5rem;
}

/* ── AI Diagnosis Centerpiece ── */
.diagnosis-container {
    background: #0a0f19;
    border: 1px solid #1a283e;
    border-radius: 8px;
    padding: 1.4rem;
    margin-bottom: 1.5rem;
}

.diagnosis-hero-card {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #0c1320;
    border: 1px solid #1e304b;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
}

.diag-pred-class {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    line-height: 1.1;
}

.diag-condition-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.65rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 0.35rem;
}

.diag-conf-box {
    text-align: right;
}

.diag-conf-num {
    font-size: 2.2rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
    color: #f8fafc;
}

.diag-conf-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
}

/* ── Custom Probability Bars ── */
.prob-card {
    background: #070c14;
    border: 1px solid #142032;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
}

.prob-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    font-weight: 600;
    color: #94a3b8;
    margin-bottom: 0.4rem;
}

.prob-header strong {
    color: #f1f5f9;
}

.prob-pct {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
}

.prob-bar-track {
    width: 100%;
    height: 8px;
    background: #111a28;
    border-radius: 4px;
    overflow: hidden;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
}

/* ── Official Benchmark Section ── */
.benchmark-container {
    background: #090e18;
    border: 1px solid #18263a;
    border-top: 3px solid #38bdf8;
    border-radius: 8px;
    padding: 1.4rem;
    margin-top: 2rem;
}

.benchmark-badge {
    background: #0e1e30;
    border: 1px solid #1e3a58;
    color: #38bdf8;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    display: inline-block;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 1rem 0;
}

.metric-cell {
    background: #060a11;
    border: 1px solid #152233;
    border-radius: 6px;
    padding: 0.9rem;
}

.metric-cell-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
}

.metric-cell-val {
    font-size: 1.5rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    color: #f8fafc;
    margin: 0.2rem 0;
}

.metric-cell-delta {
    font-size: 0.72rem;
    font-weight: 700;
    color: #10b981;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Comparison Bar ── */
.comp-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.65rem;
    font-size: 0.8rem;
}

.comp-name {
    width: 160px;
    font-weight: 600;
    color: #cbd5e1;
}

.comp-track {
    flex-grow: 1;
    background: #111a28;
    height: 18px;
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}

.comp-fill {
    height: 100%;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 0.5rem;
    font-size: 0.72rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 2.5rem 0 1rem 0;
    border-top: 1px solid #121d2c;
    margin-top: 3rem;
    font-size: 0.75rem;
    color: #475569;
}

.app-footer strong {
    color: #64748b;
}

/* ── Responsive rules ── */
@media (max-width: 768px) {
    .status-strip {
        grid-template-columns: repeat(2, 1fr);
    }
    .metric-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    .diagnosis-hero-card {
        flex-direction: column;
        align-items: flex-start;
        gap: 1rem;
    }
    .diag-conf-box {
        text-align: left;
    }
    .hero-headline {
        font-size: 1.6rem;
    }
    .pipeline-diagram {
        flex-direction: column;
        gap: 0.75rem;
    }
    .pipeline-arrow {
        transform: rotate(90deg);
        padding: 0.25rem 0;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE LOADING (CACHED, TRAINING-FREE, STRICTLY OFFLINE)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_frozen_resources():
    """Load the frozen FrequencyOnly checkpoint.

    This function never opens or evaluates the test dataset.
    """
    cfg = load_config(ROOT / "configs/config.yaml")
    checkpoint_path = ROOT / "results" / "checkpoints" / "frequency_only.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Frozen checkpoint not found at: {checkpoint_path}\n"
            "Official evaluation requires frequency_only.pt."
        )
    model, metadata = load_model(checkpoint_path, cfg)
    if type(model).__name__ != "FrequencyOnly":
        raise RuntimeError(
            f"Expected FrequencyOnly model, got: {type(model).__name__}"
        )
    mean, std = load_norm_stats(cfg, ROOT)
    return cfg, model, metadata, mean, std, checkpoint_path


@st.cache_data(show_spinner=False)
def load_locked_benchmark() -> dict:
    """Load the locked benchmark results from results/final_test_results.json."""
    results_path = ROOT / "results" / "final_test_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Benchmark results missing: {results_path}")
    with open(results_path, encoding="utf-8") as f:
        return json.load(f)


# Load resources safely
try:
    cfg, model, metadata, norm_mean, norm_std, ckpt_file = load_frozen_resources()
    benchmark_data = load_locked_benchmark()
except Exception as err:
    st.error(f"Initialization failure: {err}")
    st.stop()

sampling_rate = int(cfg["dataset"]["sampling_rate"])  # 64,000 Hz
signal_length = int(cfg["dataset"]["signal_length"])  # 64,000 samples


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS (ENGINEERING INSTRUMENT STYLE)
# ─────────────────────────────────────────────────────────────────────────────
def create_dark_plot(figsize=(7.5, 2.7)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#080d15")
    ax.set_facecolor("#080d15")
    ax.tick_params(colors="#64748b", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#182436")
        spine.set_linewidth(0.8)
    ax.grid(True, color="#121c2a", linestyle="--", linewidth=0.6, alpha=0.8)
    return fig, ax


def render_waveform_plot(raw_signal: np.ndarray, fs: int):
    fig, ax = create_dark_plot(figsize=(7.5, 2.7))
    t = np.arange(raw_signal.size) / fs
    ax.plot(t, raw_signal, color="#38bdf8", linewidth=0.65, alpha=0.95)
    ax.set_xlabel("Time (s)", color="#94a3b8", fontsize=8.5, labelpad=4)
    ax.set_ylabel("Amplitude (V)", color="#94a3b8", fontsize=8.5, labelpad=4)
    ax.set_xlim(0, t[-1])
    fig.tight_layout(pad=1.0)
    return fig


def render_stft_plot(model: torch.nn.Module, x_norm: torch.Tensor, fs: int):
    device = next(model.parameters()).device
    with torch.no_grad():
        spectrum = model.frequency.spectrogram(x_norm.to(device))[0, 0].cpu().numpy()
    freqs_khz = np.fft.rfftfreq(model.frequency.n_fft, d=1 / fs) / 1000.0
    times_s = np.arange(spectrum.shape[1]) * model.frequency.hop / fs

    fig, ax = create_dark_plot(figsize=(7.5, 2.7))
    mesh = ax.pcolormesh(
        times_s, freqs_khz, spectrum, shading="auto", cmap="inferno"
    )
    cb = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("log(1 + |STFT|)", color="#94a3b8", fontsize=8)
    cb.ax.tick_params(colors="#64748b", labelsize=7)
    cb.outline.set_edgecolor("#182436")
    ax.set_xlabel("Time (s)", color="#94a3b8", fontsize=8.5, labelpad=4)
    ax.set_ylabel("Frequency (kHz)", color="#94a3b8", fontsize=8.5, labelpad=4)
    ax.set_ylim(0, freqs_khz[-1])
    fig.tight_layout(pad=1.0)
    return fig


def render_attribution_plot(saliency: np.ndarray, raw_signal: np.ndarray, fs: int):
    fig, ax = create_dark_plot(figsize=(10, 2.5))
    t = np.arange(saliency.size) / fs
    ax.plot(t, raw_signal[:len(t)], color="#334155", linewidth=0.5, label="Waveform", alpha=0.6)
    ax.plot(t, saliency, color="#f59e0b", linewidth=0.8, label="Attribution |grad|")
    ax.fill_between(t, saliency, color="#f59e0b", alpha=0.15)
    ax.set_xlabel("Time (s)", color="#94a3b8", fontsize=8.5)
    ax.set_ylabel("Normalized Attribution", color="#94a3b8", fontsize=8.5)
    ax.set_xlim(0, t[-1])
    ax.legend(facecolor="#0c1320", edgecolor="#182436", fontsize=7.5, labelcolor="#cbd5e1", loc="upper right")
    fig.tight_layout(pad=1.0)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR (COMPACT SOFTWARE SPECIFICATION)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """<div style="padding: 0.5rem 0;">
<div style="font-size: 1.1rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">SENTINEL<span style="color:#38bdf8;">AI</span></div>
<div style="font-size: 0.7rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1.25rem;">Signal Intelligence</div>
<div style="background: #080d16; border: 1px solid #142032; border-radius: 6px; padding: 0.75rem; margin-bottom: 1rem;">
<div style="font-size: 0.68rem; color: #64748b; font-weight: 700; text-transform: uppercase;">Inference Engine</div>
<div style="font-size: 0.82rem; color: #10b981; font-weight: 600; margin-top: 2px;">● Ready</div>
<div style="font-size: 0.68rem; color: #64748b; font-weight: 700; text-transform: uppercase; margin-top: 0.6rem;">Model Architecture</div>
<div style="font-size: 0.82rem; color: #e2e8f0; font-weight: 500;">Frequency Only (STFT 2-D Conv)</div>
<div style="font-size: 0.68rem; color: #64748b; font-weight: 700; text-transform: uppercase; margin-top: 0.6rem;">Parameters</div>
<div style="font-size: 0.82rem; color: #e2e8f0; font-family: 'JetBrains Mono', monospace;">11,363</div>
<div style="font-size: 0.68rem; color: #64748b; font-weight: 700; text-transform: uppercase; margin-top: 0.6rem;">Checkpoint</div>
<div style="font-size: 0.78rem; color: #38bdf8; font-family: 'JetBrains Mono', monospace;">frequency_only.pt (Frozen)</div>
</div>
<div style="background: #080d16; border: 1px solid #142032; border-radius: 6px; padding: 0.75rem; margin-bottom: 1.25rem;">
<div style="font-size: 0.68rem; color: #64748b; font-weight: 700; text-transform: uppercase;">Processing Pipeline</div>
<div style="font-size: 0.78rem; color: #94a3b8; margin-top: 0.35rem; line-height: 1.5;">
Signal ➔ STFT ➔ 2-D CNN ➔ Classification
</div>
</div>
<div style="font-size: 0.75rem; color: #64748b; line-height: 1.45;">
<strong style="color: #94a3b8;">About:</strong> SentinelAI is an experimental deep learning system for bearing-condition classification from vibration signals.
</div>
</div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. APPLICATION HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="app-header">
    <div class="brand-title">
        <span>SENTINEL<span class="accent">AI</span></span>
        <div class="brand-subtitle">Industrial Signal Intelligence</div>
    </div>
    <div class="status-badge-online">
        <div class="pulse-dot"></div>
        MODEL ONLINE
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. STATUS STRIP
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="status-strip">
    <div class="status-cell">
        <div class="status-label">Model</div>
        <div class="status-val">Frequency Only</div>
    </div>
    <div class="status-cell">
        <div class="status-label">Checkpoint</div>
        <div class="status-val">FROZEN</div>
    </div>
    <div class="status-cell">
        <div class="status-label">Input</div>
        <div class="status-val">1-D VIBRATION</div>
    </div>
    <div class="status-cell">
        <div class="status-label">Sampling</div>
        <div class="status-val">64 kHz NOMINAL</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. HERO SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero-box">
    <div class="tech-badge">FROZEN MODEL • VALIDATED PIPELINE</div>
    <div class="hero-headline">Detect Bearing Faults From Vibration Signals</div>
    <div class="hero-subheadline">
        SentinelAI converts raw vibration measurements into time-frequency representations and classifies bearing condition using a compact deep learning model.
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PRIMARY INPUT AREA (MODE 1: LIVE SIGNAL ANALYSIS)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="instrument-card">
<div class="section-tag">Input Acquisition</div>
<div class="section-title">Signal Input</div>
<div style="font-size: 0.85rem; color: #94a3b8; margin-top: -0.4rem; margin-bottom: 0.85rem;">Upload a 1-D vibration waveform (.npy)</div>
<div class="upload-spec-row">
<div class="upload-spec-item"><strong>Window:</strong> 64,000 samples (1.00 s)</div>
<div class="upload-spec-item"><strong>Sampling Rate:</strong> 64 kHz nominal</div>
<div class="upload-spec-item"><strong>Resampling:</strong> None (exact rate required)</div>
</div>
</div>
""",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Select or drop a NumPy 1-D vibration waveform (.npy)",
    type=["npy"],
    label_visibility="collapsed",
    help="Upload a 1-D numpy array of vibration readings (.npy). Minimum 64,000 samples.",
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. EMPTY STATE (BEFORE UPLOAD)
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown(
        """<div style="text-align: center; padding: 1.5rem 0 2rem 0; margin-bottom: 1.5rem;">
<div style="font-size: 0.95rem; font-weight: 600; color: #cbd5e1; margin-bottom: 0.35rem;">Upload a vibration signal to begin analysis</div>
<div style="font-size: 0.8rem; color: #64748b; max-width: 580px; margin: 0 auto 1.5rem auto;">The model performs frozen single-window classification on a calibrated 64,000-sample (1.0 second) 1-D vibration segment.</div>
<div class="pipeline-diagram">
<div class="pipeline-node"><div class="pipeline-node-title">Raw Vibration</div><div class="pipeline-node-sub">1-D • 64 kHz • 1.0s</div></div>
<div class="pipeline-arrow">➔</div>
<div class="pipeline-node"><div class="pipeline-node-title">Normalization</div><div class="pipeline-node-sub">Training Global Z-Score</div></div>
<div class="pipeline-arrow">➔</div>
<div class="pipeline-node"><div class="pipeline-node-title">STFT</div><div class="pipeline-node-sub">n_fft 2048 • hop 512</div></div>
<div class="pipeline-arrow">➔</div>
<div class="pipeline-node"><div class="pipeline-node-title">Deep Features</div><div class="pipeline-node-sub">2-D Conv Spectrogram</div></div>
<div class="pipeline-arrow">➔</div>
<div class="pipeline-node"><div class="pipeline-node-title">Condition</div><div class="pipeline-node-sub">Healthy / Outer / Inner</div></div>
</div>
</div>""",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 6. SIGNAL PROCESSING & INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
else:
    # Validate and load raw signal
    try:
        raw_signal = np.load(uploaded_file, allow_pickle=False).astype(np.float32)
    except Exception:
        st.error("Please upload a NumPy .npy file containing a 1-D vibration signal.")
        st.stop()

    if raw_signal.ndim != 1 and (raw_signal.ndim != 2 or 1 not in raw_signal.shape):
        st.error(f"Expected a 1-D waveform, got shape {raw_signal.shape}.")
        st.stop()

    raw_signal = raw_signal.squeeze()

    if len(raw_signal) < signal_length:
        st.error(
            f"Signal must contain at least {signal_length:,} samples (got {len(raw_signal):,})."
        )
        st.stop()

    is_truncated = len(raw_signal) > signal_length
    analyzed_samples = raw_signal[:signal_length]

    # Signal Loaded Confirmation Banner
    st.markdown(
        f"""
    <div class="signal-loaded-box">
        <div class="signal-loaded-left">
            <div class="signal-loaded-tag">SIGNAL LOADED</div>
            <div class="signal-loaded-name">{uploaded_file.name}</div>
            <div class="signal-loaded-meta">
                {len(raw_signal):,} samples supplied &nbsp;•&nbsp; 64,000 samples analyzed (1.00 s window at 64 kHz)
            </div>
        </div>
        <div class="signal-loaded-badge">
            STATUS: VALIDATED
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if is_truncated:
        st.caption("ℹ️ Input truncated to the first 64,000 samples.")

    # Preprocess signal for the model
    try:
        x_tensor = preprocess_signal(analyzed_samples, cfg, norm_mean, norm_std)
    except Exception as err:
        st.error(f"Signal validation error: {err}")
        st.stop()

    # ─────────────────────────────────────────────────────────────────────────
    # 7. ANALYSIS WORKSPACE (RAW VIBRATION & STFT)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(
        """
    <div style="margin-top: 1.5rem; margin-bottom: 0.75rem;">
        <div class="section-tag">Instrument Displays</div>
        <div class="section-title">Signal Analysis</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_raw, col_stft = st.columns(2, gap="medium")

    with col_raw:
        st.markdown(
            """
        <div style="font-size: 0.85rem; font-weight: 700; color: #cbd5e1; margin-bottom: 0.35rem;">
            Raw Vibration
        </div>
        """,
            unsafe_allow_html=True,
        )
        fig_raw = render_waveform_plot(analyzed_samples, sampling_rate)
        st.pyplot(fig_raw, width="stretch")
        plt.close(fig_raw)

    with col_stft:
        st.markdown(
            """
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.35rem;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #cbd5e1;">
                Time-Frequency Representation
            </div>
            <div style="font-size: 0.72rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">
                STFT • n_fft 2048 • hop 512
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        fig_stft = render_stft_plot(model, x_tensor, sampling_rate)
        st.pyplot(fig_stft, width="stretch")
        plt.close(fig_stft)

    # ─────────────────────────────────────────────────────────────────────────
    # 8. DIAGNOSIS SECTION (VISUAL CENTERPIECE)
    # ─────────────────────────────────────────────────────────────────────────
    pred_res = predict(model, x_tensor)

    pred_class_id = pred_res["class_id"]
    pred_class_display = pred_res["class_display"].upper()
    confidence_pct = pred_res["confidence"] * 100.0
    probabilities = pred_res["probabilities"]

    # Condition determination (scientifically restrained)
    if pred_class_id == 0:
        cond_text = "Healthy"
        cond_color = "#10b981"
        cond_bg = "rgba(16, 185, 129, 0.12)"
        cond_border = "rgba(16, 185, 129, 0.35)"
        class_text_color = "#34d399"
    else:
        cond_text = "Damage class detected"
        cond_color = "#f43f5e"
        cond_bg = "rgba(244, 63, 94, 0.12)"
        cond_border = "rgba(244, 63, 94, 0.35)"
        class_text_color = "#fb7185" if pred_class_id == 1 else "#fbbf24"

    labels_info = [
        ("HEALTHY", probabilities[0], "#10b981", pred_class_id == 0),
        ("OUTER RING DAMAGE", probabilities[1], "#f43f5e", pred_class_id == 1),
        ("INNER RING DAMAGE", probabilities[2], "#f59e0b", pred_class_id == 2),
    ]

    prob_cards_html = []
    for label_name, p_val, p_color, is_pred in labels_info:
        pct_str = f"{p_val * 100.0:.1f}%"
        border_highlight = f"border: 1px solid {p_color}55;" if is_pred else "border: 1px solid #142032;"
        bg_highlight = f"background: {p_color}0a;" if is_pred else "background: #070c14;"
        indicator = f'<span style="font-size: 0.7rem; color: {p_color}; font-weight: 700;">★ PREDICTED</span>' if is_pred else ""
        bar_w = max(p_val * 100.0, 1.0)
        prob_cards_html.append(f"""<div class="prob-card" style="{bg_highlight} {border_highlight}">
<div class="prob-header">
<div><strong>{label_name}</strong> {indicator}</div>
<div class="prob-pct" style="color: {p_color};">{pct_str}</div>
</div>
<div class="prob-bar-track">
<div class="prob-bar-fill" style="width: {bar_w:.1f}%; background-color: {p_color};"></div>
</div>
</div>""")

    all_prob_cards = "".join(prob_cards_html)

    diagnosis_html = f"""<div class="diagnosis-container">
<div class="section-tag">Inference Result</div>
<div class="section-title">AI DIAGNOSIS</div>
<div class="diagnosis-hero-card" style="border-left: 5px solid {cond_color};">
<div>
<div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: #64748b;">Classified Bearing State</div>
<div class="diag-pred-class" style="color: {class_text_color};">{pred_class_display}</div>
<div class="diag-condition-pill" style="color: {cond_color}; background: {cond_bg}; border: 1px solid {cond_border};">CONDITION: {cond_text}</div>
</div>
<div class="diag-conf-box">
<div class="diag-conf-label">Confidence</div>
<div class="diag-conf-num">{confidence_pct:.1f}%</div>
<div style="font-size: 0.68rem; color: #64748b; margin-top: 2px;">max softmax output</div>
</div>
</div>
<div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b; margin-bottom: 0.6rem;">Class Probabilities</div>
{all_prob_cards}
<div style="font-size: 0.72rem; color: #64748b; margin-top: 0.5rem;">
Classification based on the frozen Frequency Only STFT model. Softmax outputs represent model distribution, not calibrated physical failure probabilities.
</div>
</div>"""

    st.markdown(diagnosis_html, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 9. MODEL INTERPRETATION (INPUT-GRADIENT ATTRIBUTION)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(
        """
    <div style="margin-top: 2rem; margin-bottom: 0.75rem;">
        <div class="section-tag">Explainability</div>
        <div class="section-title">MODEL INTERPRETATION</div>
        <div style="font-size: 0.85rem; color: #94a3b8; margin-top: -0.4rem; margin-bottom: 0.75rem;">
            Input-gradient attribution
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style="background: #080d16; border: 1px solid #162438; border-radius: 6px; padding: 0.75rem 1rem; font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.85rem;">
        Highlights input regions that most influence the selected class prediction. 
        <span style="color: #64748b;">(Note: This model uses direct input-gradient sensitivity; it does not use Grad-CAM.)</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    try:
        explanation = explain(model, x_tensor, target_class=pred_class_id)
        saliency_array = explanation["temporal_saliency"]
        if saliency_array is not None and np.any(saliency_array):
            fig_attr = render_attribution_plot(saliency_array, analyzed_samples, sampling_rate)
            st.pyplot(fig_attr, width="stretch")
            plt.close(fig_attr)
            st.caption(
                "Input-gradient saliency across the 64,000 sample analysis window. Peaks indicate temporal locations with highest sensitivity to the classification loss."
            )
        else:
            st.info("Attribution sensitivity map could not be derived for this waveform segment.")
    except Exception as err:
        st.info(f"Model attribution unavailable: {err}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. TECHNICAL DETAILS (COLLAPSIBLE SPECIFICATION)
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Technical Details", expanded=False):
    st.markdown(
        """<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.75rem; font-size: 0.82rem; padding: 0.5rem 0;">
<div style="background: #080d16; border: 1px solid #142032; border-radius: 5px; padding: 0.65rem 0.85rem;">
<div style="font-size: 0.68rem; color: #64748b; text-transform: uppercase; font-weight: 700;">Model Architecture</div>
<div style="font-weight: 600; color: #f1f5f9; margin-top: 2px;">Frequency Only (STFT 2-D Conv)</div>
</div>
<div style="background: #080d16; border: 1px solid #142032; border-radius: 5px; padding: 0.65rem 0.85rem;">
<div style="font-size: 0.68rem; color: #64748b; text-transform: uppercase; font-weight: 700;">Parameters</div>
<div style="font-weight: 600; color: #f1f5f9; margin-top: 2px; font-family: 'JetBrains Mono', monospace;">11,363</div>
</div>
<div style="background: #080d16; border: 1px solid #142032; border-radius: 5px; padding: 0.65rem 0.85rem;">
<div style="font-size: 0.68rem; color: #64748b; text-transform: uppercase; font-weight: 700;">Input Specification</div>
<div style="font-weight: 600; color: #f1f5f9; margin-top: 2px;">1-D vibration channel</div>
</div>
<div style="background: #080d16; border: 1px solid #142032; border-radius: 5px; padding: 0.65rem 0.85rem;">
<div style="font-size: 0.68rem; color: #64748b; text-transform: uppercase; font-weight: 700;">Sampling Rate</div>
<div style="font-weight: 600; color: #f1f5f9; margin-top: 2px;">64 kHz nominal</div>
</div>
<div style="background: #080d16; border: 1px solid #142032; border-radius: 5px; padding: 0.65rem 0.85rem;">
<div style="font-size: 0.68rem; color: #64748b; text-transform: uppercase; font-weight: 700;">Analysis Window</div>
<div style="font-weight: 600; color: #f1f5f9; margin-top: 2px;">64,000 samples / 1.000 s</div>
</div>
<div style="background: #080d16; border: 1px solid #142032; border-radius: 5px; padding: 0.65rem 0.85rem;">
<div style="font-size: 0.68rem; color: #64748b; text-transform: uppercase; font-weight: 700;">STFT Configuration</div>
<div style="font-weight: 600; color: #f1f5f9; margin-top: 2px; font-family: 'JetBrains Mono', monospace;">n_fft = 2048 • hop_length = 512</div>
</div>
<div style="background: #080d16; border: 1px solid #142032; border-radius: 5px; padding: 0.65rem 0.85rem;">
<div style="font-size: 0.68rem; color: #64748b; text-transform: uppercase; font-weight: 700;">Normalization</div>
<div style="font-weight: 600; color: #f1f5f9; margin-top: 2px;">Training-derived global z-score</div>
</div>
<div style="background: #080d16; border: 1px solid #142032; border-radius: 5px; padding: 0.65rem 0.85rem;">
<div style="font-size: 0.68rem; color: #64748b; text-transform: uppercase; font-weight: 700;">Norm Statistics</div>
<div style="font-weight: 600; color: #f1f5f9; margin-top: 2px; font-family: 'JetBrains Mono', monospace;">Mean: 0.0080971408 • Std: 0.3537545935</div>
</div>
<div style="background: #080d16; border: 1px solid #142032; border-radius: 5px; padding: 0.65rem 0.85rem; grid-column: span 2;">
<div style="font-size: 0.68rem; color: #64748b; text-transform: uppercase; font-weight: 700;">Checkpoint Artifact</div>
<div style="font-weight: 600; color: #38bdf8; margin-top: 2px; font-family: 'JetBrains Mono', monospace;">results/checkpoints/frequency_only.pt</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 11. BENCHMARK SECTION (MODE 2: OFFICIAL MODEL BENCHMARK)
# ─────────────────────────────────────────────────────────────────────────────
test_acc = benchmark_data["test_metrics"]["accuracy"] * 100.0
test_f1 = benchmark_data["test_metrics"]["macro_f1"] * 100.0
base_f1 = benchmark_data["baseline_comparison"]["baseline_test_macro_f1"] * 100.0
f1_diff = test_f1 - base_f1
rel_diff = benchmark_data["baseline_comparison"]["percentage_improvement_over_baseline"]

benchmark_html = f"""<div class="benchmark-container">
<div class="benchmark-badge">OFFICIAL BENCHMARK • LOCKED EVALUATION</div>
<div class="section-title" style="margin-bottom: 0.2rem;">BENCHMARK</div>
<div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 1.25rem;">
Locked evaluation on the official held-out test bearings (K006, KA22, KI14)
</div>
<div class="metric-grid">
<div class="metric-cell">
<div class="metric-cell-label">Test Macro-F1</div>
<div class="metric-cell-val" style="color: #38bdf8;">{test_f1:.2f}%</div>
<div class="metric-cell-delta">+{f1_diff:.2f} pp vs baseline</div>
</div>
<div class="metric-cell">
<div class="metric-cell-label">Mandatory Baseline</div>
<div class="metric-cell-val" style="color: #94a3b8;">{base_f1:.2f}%</div>
<div style="font-size: 0.72rem; color: #64748b;">Baseline 1D CNN</div>
</div>
<div class="metric-cell">
<div class="metric-cell-label">Relative Gain</div>
<div class="metric-cell-val" style="color: #10b981;">+{rel_diff:.2f}%</div>
<div style="font-size: 0.72rem; color: #64748b;">over baseline</div>
</div>
<div class="metric-cell">
<div class="metric-cell-label">Test Accuracy</div>
<div class="metric-cell-val" style="color: #f8fafc;">{test_acc:.2f}%</div>
<div style="font-size: 0.72rem; color: #64748b;">955 test windows</div>
</div>
</div>
<div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b; margin: 1.25rem 0 0.75rem 0;">
Macro-F1 Performance Comparison
</div>
<div class="comp-row">
<div class="comp-name">Mandatory Baseline</div>
<div class="comp-track">
<div class="comp-fill" style="width: {base_f1:.1f}%; background: #334155; color: #cbd5e1;">{base_f1:.2f}%</div>
</div>
</div>
<div class="comp-row">
<div class="comp-name" style="color: #38bdf8;">SentinelAI (Winner)</div>
<div class="comp-track">
<div class="comp-fill" style="width: {test_f1:.1f}%; background: #0284c7; color: #ffffff;">{test_f1:.2f}%</div>
</div>
</div>
<div style="font-size: 0.75rem; color: #64748b; margin-top: 0.75rem;">
Evaluation performed on 955 bearing-disjoint windows across unseen test bearings: K006 (Healthy: 320), KA22 (Outer Ring: 315), KI14 (Inner Ring: 320).
</div>
</div>"""

st.markdown(benchmark_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 12. OFFICIAL CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="margin-top: 1.5rem; margin-bottom: 0.75rem;">
    <div class="section-tag">Verification Artifact</div>
    <div class="section-title">Official Test Confusion Matrix</div>
    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: -0.4rem; margin-bottom: 1rem;">
        Frozen evaluation • K006 / KA22 / KI14 • Verified offline artifact
    </div>
</div>
""",
    unsafe_allow_html=True,
)

cm_path = ROOT / "results" / "figures" / "final_confusion_matrix.png"
if cm_path.exists():
    cm_col, _ = st.columns([1.6, 1])
    with cm_col:
        st.markdown(
            """<div class="cm-container" style="background: #080d16; border: 1px solid #1a273b; border-radius: 8px; padding: 0.5rem; margin-bottom: 0.4rem;">""",
            unsafe_allow_html=True,
        )
        st.image(
            str(cm_path),
            caption="Official locked test confusion matrix (Rows: Ground Truth, Columns: Predicted Class)",
            width="stretch",
        )
        st.markdown("""</div>""", unsafe_allow_html=True)
else:
    st.info("Official test confusion matrix artifact not found at results/figures/final_confusion_matrix.png.")


# ─────────────────────────────────────────────────────────────────────────────
# 13. PROFESSIONAL APPLICATION FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """<div class="app-footer">
<div style="font-size: 0.9rem; font-weight: 700; color: #94a3b8; letter-spacing: -0.01em;">SentinelAI &nbsp;|&nbsp; Industrial Signal Intelligence</div>
<div style="margin-top: 0.35rem; color: #475569;">Experimental research prototype &nbsp;•&nbsp; Paderborn Bearing Benchmark &nbsp;•&nbsp; Frozen Evaluation Pipeline</div>
</div>""",
    unsafe_allow_html=True,
)
