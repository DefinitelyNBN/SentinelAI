"""SentinelAI Streamlit demonstration with safe checkpoint-aware inference."""
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.inference import (load_config, get_best_checkpoint, load_model, load_norm_stats,
                           preprocess_signal, compute_stft_spectrogram, predict, explain)

st.set_page_config(page_title="SentinelAI", page_icon="⚙️", layout="wide")
st.title("SentinelAI")
st.caption("Bearing condition classification from one-channel vibration — experimental demo")

@st.cache_resource
def resources():
    cfg = load_config(ROOT / "configs/config.yaml")
    checkpoint = get_best_checkpoint(cfg, ROOT)
    model, meta = load_model(checkpoint, cfg)
    mean, std = load_norm_stats(cfg, ROOT)
    return cfg, model, meta, mean, std

try:
    cfg, model, meta, mean, std = resources()
except Exception as error:
    st.error(f"Model setup failed: {error}")
    st.stop()

with st.sidebar:
    st.header("Model information")
    st.write(f"Architecture: `{meta['model_name']}`")
    st.write(f"Checkpoint epoch: {meta['epoch']}")
    st.write(f"Parameters: {meta['n_params']:,}")
    st.write(f"Model size: {meta['size_kb']:.1f} KB")
    st.write(f"STFT: n_fft={cfg['model']['n_fft']}, hop={cfg['model']['hop_length']}")
    st.write(f"Preprocessing: {cfg['preprocessing']['normalization_method']}; filtering={'on' if cfg['preprocessing']['filter_enabled'] else 'off'}")

upload = st.file_uploader("Upload a one-dimensional `.npy` vibration waveform", type=["npy"])
sampling_rate = st.number_input("Sampling rate (Hz)", min_value=1, value=int(cfg['dataset']['sampling_rate']), step=1000)
if upload is None:
    st.info("Upload at least one second of vibration at the model's sampling rate.")
    st.stop()
if sampling_rate != cfg['dataset']['sampling_rate']:
    st.error(f"This checkpoint expects {cfg['dataset']['sampling_rate']} Hz; resampling is intentionally not performed in the demo.")
    st.stop()
try:
    raw = np.load(upload).astype(np.float32).squeeze()
    x = preprocess_signal(raw, cfg, mean, std)
except Exception as error:
    st.error(f"Could not process the uploaded signal: {error}")
    st.stop()

length = cfg['dataset']['signal_length']; duration = len(raw) / sampling_rate
st.caption(f"File: `{upload.name}` • Samples: {len(raw):,} • Duration: {duration:.3f} s • Sampling rate: {sampling_rate:,} Hz")
left, right = st.columns(2)
with left:
    st.subheader("Raw vibration")
    fig, ax = plt.subplots(); ax.plot(raw[:length], linewidth=.5); ax.set(xlabel="Sample", ylabel="Amplitude"); st.pyplot(fig, clear_figure=True)
with right:
    st.subheader("Processed vibration")
    fig, ax = plt.subplots(); ax.plot(x.squeeze().numpy(), linewidth=.5); ax.set(xlabel="Sample", ylabel="Training-normalized amplitude"); st.pyplot(fig, clear_figure=True)

try:
    spec, freqs, times = compute_stft_spectrogram(raw, cfg)
    st.subheader("Time-frequency representation")
    fig, ax = plt.subplots(figsize=(9, 3.5)); im = ax.pcolormesh(times, freqs / 1000, spec, shading="auto"); fig.colorbar(im, ax=ax, label="log magnitude"); ax.set(xlabel="Time (s)", ylabel="Frequency (kHz)"); st.pyplot(fig, clear_figure=True)
except Exception as error:
    st.warning(f"Spectrogram unavailable: {error}")

result = predict(model, x)
st.subheader("Prediction")
st.success(f"Predicted condition: {result['class_display']}")
st.caption(f"Model confidence: {result['confidence']:.1%} (maximum softmax score; not calibrated probability)")
for label, probability in zip(["Healthy", "Outer-ring damage", "Inner-ring damage"], result["probabilities"]):
    st.write(f"{label}: {probability:.1%}")
    st.progress(float(probability))

st.subheader("Explainability")
try:
    attribution = explain(model, x, target_class=result['class_id'])
    if attribution['gradcam_available']:
        fig, ax = plt.subplots(); ax.imshow(attribution['gradcam_heatmap'], origin="lower", aspect="auto", cmap="magma"); ax.set(xlabel="STFT time frame", ylabel="Frequency bin", title="Class-specific spectral Grad-CAM"); st.pyplot(fig, clear_figure=True)
    fig, ax = plt.subplots(); ax.plot(attribution['temporal_saliency'], linewidth=.5); ax.set(xlabel="Sample", ylabel="Normalized |gradient|", title="Temporal input-gradient saliency"); st.pyplot(fig, clear_figure=True)
    st.caption(attribution['note'] + " These maps indicate model sensitivity, not causal proof.")
except Exception as error:
    st.warning(f"Explainability could not be generated: {error}")
