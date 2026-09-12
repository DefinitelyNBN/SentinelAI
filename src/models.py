"""Model architectures for SentinelAI Phase 6 experiments.

Architectures:
1. Baseline1DCNN: Minimal 1-D CNN baseline (Conv1d -> ReLU -> MaxPool -> Conv1d -> ReLU -> MaxPool -> GAP -> Linear)
2. TemporalOnly: SentinelAI TemporalEncoder + Classifier
3. FrequencyOnly: STFT log-magnitude + FrequencyEncoder + Classifier
4. SentinelAI: Dual-branch feature fusion (Temporal + Frequency) + Classifier
"""
from __future__ import annotations
import torch
from torch import nn

class Baseline1DCNN(nn.Module):
    """Minimal, credible 1-D CNN baseline.
    
    raw 1-D signal -> Conv1D -> activation -> pooling -> Conv1D -> activation -> pooling -> GAP -> Linear -> 3 classes
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(4)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(32, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)


class TemporalEncoder(nn.Module):
    def __init__(self, channels: list[int], out_dim: int):
        super().__init__()
        layers = []
        cin = 1
        for cout in channels:
            layers += [
                nn.Conv1d(cin, cout, kernel_size=5, padding=2),
                nn.BatchNorm1d(cout),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ]
            cin = cout
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(cin, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x).mean(-1))


class FrequencyEncoder(nn.Module):
    def __init__(self, channels: list[int], out_dim: int, n_fft: int = 2048, hop: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        layers = []
        cin = 1
        for cout in channels:
            layers += [
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ]
            cin = cout
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(cin, out_dim)
        # Pre-register Hann window for physically defensible STFT without spectral leakage
        self.register_buffer("window", torch.hann_window(n_fft))

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, 1, L) -> squeeze channel dim for stft
        stft_out = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            return_complex=True
        )
        # Log magnitude spectrogram
        mag = torch.log1p(stft_out.abs()).unsqueeze(1)
        return mag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(self.spectrogram(x)).mean((-1, -2)))


class TemporalOnly(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        d = m["embedding_dim"]
        num_classes = cfg["dataset"]["num_classes"]
        self.temporal = TemporalEncoder(m["temporal_channels"], d)
        self.classifier = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(m["dropout"]),
            nn.Linear(d, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.temporal(x))


class FrequencyOnly(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        d = m["embedding_dim"]
        num_classes = cfg["dataset"]["num_classes"]
        self.frequency = FrequencyEncoder(m["frequency_channels"], d, m["n_fft"], m["hop_length"])
        self.classifier = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(m["dropout"]),
            nn.Linear(d, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.frequency(x))


class SentinelAI(nn.Module):
    def __init__(self, cfg: dict, variant: str = "sentinelai"):
        super().__init__()
        m = cfg["model"]
        d = m["embedding_dim"]
        num_classes = cfg["dataset"]["num_classes"]
        self.variant = variant
        if variant == "temporal":
            self.model = TemporalOnly(cfg)
        elif variant == "frequency":
            self.model = FrequencyOnly(cfg)
        else:
            self.temporal = TemporalEncoder(m["temporal_channels"], d)
            self.frequency = FrequencyEncoder(m["frequency_channels"], d, m["n_fft"], m["hop_length"])
            self.classifier = nn.Sequential(
                nn.Linear(2 * d, d),
                nn.ReLU(),
                nn.Dropout(m["dropout"]),
                nn.Linear(d, num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.variant in {"temporal", "frequency"}:
            return self.model(x)
        z = torch.cat([self.temporal(x), self.frequency(x)], dim=1)
        return self.classifier(z)


class SentinelAI_ProjectedFusion(nn.Module):
    """SentinelAI with projected feature fusion layer.
    
    Mitigates modality dominance by projecting temporal and frequency
    embeddings through independent LayerNorm + Linear projections before
    fused classification.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        d = m["embedding_dim"]
        num_classes = cfg["dataset"]["num_classes"]
        self.temporal = TemporalEncoder(m["temporal_channels"], d)
        self.frequency = FrequencyEncoder(m["frequency_channels"], d, m["n_fft"], m["hop_length"])
        
        self.proj_t = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU()
        )
        self.proj_f = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(m["dropout"]),
            nn.Linear(d, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zt = self.proj_t(self.temporal(x))
        zf = self.proj_f(self.frequency(x))
        z = torch.cat([zt, zf], dim=1)
        return self.fusion(z)


class FrequencyEncoderBandLimited(nn.Module):
    """FrequencyEncoder variant that restricts the STFT to a low-frequency band.

    The complete STFT is computed first (using the same n_fft / hop_length /
    Hann window as FrequencyEncoder), and then the frequency axis is sliced
    to retain only the bins whose centre frequency is <= max_freq_hz.

    No resampling or modification of the raw signal is performed.

    Parameters
    ----------
    channels : list[int]
        Conv2D channel progression (identical to FrequencyEncoder).
    out_dim : int
        Embedding dimension.
    n_fft : int
        FFT size (default 2048).
    hop : int
        Hop length (default 512).
    sampling_rate : int
        Signal sampling rate in Hz (needed to compute bin-to-Hz mapping).
    max_freq_hz : float
        Upper frequency limit in Hz.  Bins with centre freq > max_freq_hz
        are discarded before the CNN.
    """

    def __init__(
        self,
        channels: list[int],
        out_dim: int,
        n_fft: int = 2048,
        hop: int = 512,
        sampling_rate: int = 64_000,
        max_freq_hz: float = 1_000.0,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.sampling_rate = sampling_rate
        self.max_freq_hz = max_freq_hz

        # Compute retained bin indices.
        # Frequency resolution = sampling_rate / n_fft  [Hz/bin]
        # Bin k has centre frequency k * (sampling_rate / n_fft).
        # Retain bins k where k * freq_res <= max_freq_hz.
        freq_res = sampling_rate / n_fft                        # Hz per bin
        n_total = n_fft // 2 + 1                                # 0 … Nyquist
        n_keep = int(max_freq_hz / freq_res) + 1                # inclusive
        n_keep = min(n_keep, n_total)
        self.n_keep = n_keep                                     # saved for docs
        self.freq_res_hz = freq_res

        layers = []
        cin = 1
        for cout in channels:
            layers += [
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            cin = cout
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(cin, out_dim)
        self.register_buffer("window", torch.hann_window(n_fft))

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """Return band-limited log-magnitude spectrogram (B, 1, n_keep, T')."""
        stft_out = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            return_complex=True,
        )  # (B, n_fft//2+1, T')
        # Restrict to low-frequency band [0, max_freq_hz]
        stft_band = stft_out[:, : self.n_keep, :]               # (B, n_keep, T')
        mag = torch.log1p(stft_band.abs()).unsqueeze(1)          # (B, 1, n_keep, T')
        return mag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(self.spectrogram(x)).mean((-1, -2)))


class FrequencyOnlyV2(nn.Module):
    """Phase 6C-A experimental model: Frequency-Only with 0–1 kHz band restriction.

    Identical to FrequencyOnly in every respect (architecture depth, channel
    counts, embedding dimension, classifier, training hyper-parameters) except
    that the STFT representation is restricted to the 0–1 kHz region before
    the CNN.

    The sole experimental variable is the frequency-band restriction.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        d = m["embedding_dim"]
        num_classes = cfg["dataset"]["num_classes"]
        fs = cfg["dataset"].get("sampling_rate", 64_000)
        max_hz = cfg["model"].get("band_limit_hz", 1_000.0)
        self.frequency = FrequencyEncoderBandLimited(
            channels=m["frequency_channels"],
            out_dim=d,
            n_fft=m["n_fft"],
            hop=m["hop_length"],
            sampling_rate=fs,
            max_freq_hz=max_hz,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(m["dropout"]),
            nn.Linear(d, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.frequency(x))


class MultiScaleTemporalEncoder(nn.Module):
    """Parallel Conv1D feature extractor with varying receptive fields.
    
    Captures short-scale impact shocks (k=7), medium-scale structural responses (k=15),
    and long-scale modulation envelopes (k=31) from raw 1-D vibration signals.
    """
    def __init__(self, out_dim: int = 64):
        super().__init__()
        # Branch 1: Short scale (k=7)
        self.branch1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 24, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        # Branch 2: Medium scale (k=15)
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 24, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        # Branch 3: Long scale (k=31)
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=31, stride=2, padding=15),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 24, kernel_size=31, stride=2, padding=15),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(24 * 3, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.gap(self.branch1(x)).squeeze(-1)
        b2 = self.gap(self.branch2(x)).squeeze(-1)
        b3 = self.gap(self.branch3(x)).squeeze(-1)
        cat = torch.cat([b1, b2, b3], dim=-1)
        return self.proj(cat)


class SpectralSubbandBlock(nn.Module):
    """Subband CNN block with global spatial-temporal pooling."""
    def __init__(self, in_channels: int = 1, out_channels: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class ResonanceSpectralEncoder(nn.Module):
    """Resonance-aware spectral feature extractor with subband slicing.
    
    Extracts explicit representations across:
    - 0–2 kHz: Low-frequency / kinematic context (BPFO/BPFI fundamental range)
    - 2–5 kHz: Primary structural resonance band 1 (Phase 6D empirical finding)
    - 5–10 kHz: Primary structural resonance band 2 (>40% damage signal power)
    - 10–32 kHz: High-frequency mechanical baseline / background
    - Full spectrum: Global spectral context preserving wideband information
    """
    def __init__(
        self,
        out_dim: int = 64,
        n_fft: int = 2048,
        hop: int = 512,
        fs: int = 64000,
        use_resonance: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.fs = fs
        self.use_resonance = use_resonance
        self.register_buffer("window", torch.hann_window(n_fft))

        freq_res = fs / n_fft  # 31.25 Hz/bin
        self.idx_0_2k = (0, int(2000 / freq_res) + 1)
        self.idx_2_5k = (int(2000 / freq_res), int(5000 / freq_res) + 1)
        self.idx_5_10k = (int(5000 / freq_res), int(10000 / freq_res) + 1)
        self.idx_10_32k = (int(10000 / freq_res), n_fft // 2 + 1)

        # Global full-spectrum pathway
        self.enc_global = SpectralSubbandBlock(1, 16)  # 32 channels out

        if use_resonance:
            # Resonance and subband pathways
            self.enc_0_2k = SpectralSubbandBlock(1, 12)    # 24 channels
            self.enc_2_5k = SpectralSubbandBlock(1, 12)    # 24 channels
            self.enc_5_10k = SpectralSubbandBlock(1, 12)   # 24 channels
            self.enc_10_32k = SpectralSubbandBlock(1, 12)  # 24 channels
            in_proj = 32 + 24 * 4  # 128
        else:
            in_proj = 32

        self.proj = nn.Sequential(
            nn.Linear(in_proj, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stft_out = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            return_complex=True,
        )
        mag = torch.log1p(stft_out.abs()).unsqueeze(1)  # (B, 1, 1025, T)

        e_global = self.enc_global(mag)
        if self.use_resonance:
            f_0_2k = mag[:, :, self.idx_0_2k[0]:self.idx_0_2k[1], :].contiguous()
            f_2_5k = mag[:, :, self.idx_2_5k[0]:self.idx_2_5k[1], :].contiguous()
            f_5_10k = mag[:, :, self.idx_5_10k[0]:self.idx_5_10k[1], :].contiguous()
            f_10_32k = mag[:, :, self.idx_10_32k[0]:self.idx_10_32k[1], :].contiguous()

            e_0_2k = self.enc_0_2k(f_0_2k)
            e_2_5k = self.enc_2_5k(f_2_5k)
            e_5_10k = self.enc_5_10k(f_5_10k)
            e_10_32k = self.enc_10_32k(f_10_32k)
            cat = torch.cat([e_global, e_0_2k, e_2_5k, e_5_10k, e_10_32k], dim=-1)
        else:
            cat = e_global

        return self.proj(cat)


class AdaptiveGatedFusion(nn.Module):
    """Learned, differentiable gating mechanism between temporal and spectral representations."""
    def __init__(self, dim: int = 64, use_gate: bool = True):
        super().__init__()
        self.use_gate = use_gate
        self.proj_t = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU())
        self.proj_f = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU())
        if use_gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
                nn.Sigmoid(),
            )
        else:
            self.simple_fuse = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            )

    def forward(self, z_t: torch.Tensor, z_f: torch.Tensor) -> torch.Tensor:
        zt_p = self.proj_t(z_t)
        zf_p = self.proj_f(z_f)
        if self.use_gate:
            cat = torch.cat([zt_p, zf_p], dim=-1)
            g = self.gate_mlp(cat)
            z_fused = g * zt_p + (1.0 - g) * zf_p
        else:
            cat = torch.cat([zt_p, zf_p], dim=-1)
            z_fused = self.simple_fuse(cat)
        return z_fused


class SentinelAI_MSRF(nn.Module):
    """Multi-Scale Resonance Fusion Network (SentinelAI_MSRF).
    
    A custom problem-driven architecture for industrial bearing diagnostics combining:
    1. Multi-scale temporal feature extraction (k=7, 15, 31 parallel branches)
    2. Resonance-aware spectral feature extraction (0-2k, 2-5k, 5-10k, 10-32k + global)
    3. Adaptive gated temporal-spectral fusion
    4. Compact discriminative embedding projection (d=64)
    """
    def __init__(self, cfg: dict, ablation: str = "full"):
        super().__init__()
        m = cfg.get("model", {})
        d = 64  # Compact embedding dimension
        num_classes = cfg.get("dataset", {}).get("num_classes", 3)
        dropout = m.get("dropout", 0.2)
        n_fft = m.get("n_fft", 2048)
        hop = m.get("hop_length", 512)
        fs = cfg.get("dataset", {}).get("sampling_rate", 64000)

        self.ablation = ablation
        self.temporal = MultiScaleTemporalEncoder(out_dim=d)

        if ablation != "temporal_only":
            use_res = (ablation != "no_resonance")
            self.spectral = ResonanceSpectralEncoder(
                out_dim=d, n_fft=n_fft, hop=hop, fs=fs, use_resonance=use_res
            )
            use_gate = (ablation != "no_gate")
            self.fusion = AdaptiveGatedFusion(dim=d, use_gate=use_gate)

        self.classifier = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ablation == "temporal_only":
            z = self.temporal(x)
        else:
            z_t = self.temporal(x)
            z_f = self.spectral(x)
            z = self.fusion(z_t, z_f)
        return self.classifier(z)


class ChannelAttention1D(nn.Module):
    """Squeeze-and-Excitation channel attention for 1D feature maps."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1)
        return x * w


class ChannelAttention2D(nn.Module):
    """Squeeze-and-Excitation channel attention for 2D spectrogram representations."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class MultiScaleTemporalExtractorV2(nn.Module):
    """Multi-Scale Temporal Feature Extractor with channel attention and scale mixing.
    
    1. Parallel multi-scale receptive fields (k=7 short, k=15 medium, k=31 long).
    2. Channel-wise attention per branch to identify salient impact features.
    3. Scale-mixing 1D convolution across concatenated branches.
    4. Dual mean+max statistical pooling to preserve transient shock peaks.
    """
    def __init__(self, out_dim: int = 64, use_multi_scale: bool = True):
        super().__init__()
        self.use_multi_scale = use_multi_scale
        if use_multi_scale:
            self.b1 = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(16, 24, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(24),
                nn.ReLU(),
                ChannelAttention1D(24),
            )
            self.b2 = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(16, 24, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(24),
                nn.ReLU(),
                ChannelAttention1D(24),
            )
            self.b3 = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=31, stride=2, padding=15),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(16, 24, kernel_size=31, stride=2, padding=15),
                nn.BatchNorm1d(24),
                nn.ReLU(),
                ChannelAttention1D(24),
            )
            in_mixer = 24 * 3  # 72
        else:
            # Single-scale baseline path for ablation (k=15)
            self.b_single = nn.Sequential(
                nn.Conv1d(1, 24, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(24),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(24, 48, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(48),
                nn.ReLU(),
                ChannelAttention1D(48),
            )
            in_mixer = 48

        self.scale_mixer = nn.Sequential(
            nn.Conv1d(in_mixer, 40, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(40),
            nn.ReLU(),
            ChannelAttention1D(40),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(40 * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_multi_scale:
            h1 = self.b1(x)
            h2 = self.b2(x)
            h3 = self.b3(x)
            cat = torch.cat([h1, h2, h3], dim=1)
        else:
            cat = self.b_single(x)
        mixed = self.scale_mixer(cat)
        pool = torch.cat([self.gap(mixed).squeeze(-1), self.gmp(mixed).squeeze(-1)], dim=-1)
        return self.proj(pool)


class SpectralSubbandBlockV2(nn.Module):
    """Subband feature identification block with 2D channel attention and dual pooling."""
    def __init__(self, in_channels: int = 1, out_channels: int = 10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(),
            ChannelAttention2D(out_channels * 2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.conv2(h)
        # Dual mean + max pooling preserves both energy density and transient peak magnitudes
        feat = torch.cat([self.gap(h).flatten(1), self.gmp(h).flatten(1)], dim=-1)
        return feat


class ResonanceSpectralEncoderV2(nn.Module):
    """Resonance-aware spectral feature extractor v2.
    
    1. Slices STFT log-spectrogram into physically motivated subbands:
       - 0–2 kHz (kinematic ball-pass frequencies and low-order harmonics)
       - 2–5 kHz (structural resonance band 1)
       - 5–10 kHz (structural resonance band 2, containing >40% fault energy)
       - 10–32 kHz (transducer / mechanical baseline)
       - Global full-spectrum pathway (wideband context)
    2. Uses SpectralSubbandBlockV2 with 2D channel attention and dual mean+max pooling.
    3. Fuses all band representations into a unified spectral embedding.
    """
    def __init__(
        self,
        out_dim: int = 64,
        n_fft: int = 2048,
        hop: int = 512,
        fs: int = 64000,
        use_subbands: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.fs = fs
        self.use_subbands = use_subbands
        self.register_buffer("window", torch.hann_window(n_fft))

        freq_res = fs / n_fft  # 31.25 Hz/bin
        self.idx_0_2k = (0, int(2000 / freq_res) + 1)
        self.idx_2_5k = (int(2000 / freq_res), int(5000 / freq_res) + 1)
        self.idx_5_10k = (int(5000 / freq_res), int(10000 / freq_res) + 1)
        self.idx_10_32k = (int(10000 / freq_res), n_fft // 2 + 1)

        self.enc_global = SpectralSubbandBlockV2(1, 12)  # 24 * 2 = 48

        if use_subbands:
            self.enc_0_2k = SpectralSubbandBlockV2(1, 10)    # 20 * 2 = 40
            self.enc_2_5k = SpectralSubbandBlockV2(1, 10)    # 20 * 2 = 40
            self.enc_5_10k = SpectralSubbandBlockV2(1, 10)   # 20 * 2 = 40
            self.enc_10_32k = SpectralSubbandBlockV2(1, 10)  # 20 * 2 = 40
            in_proj = 48 + 40 * 4  # 208
        else:
            in_proj = 48

        self.proj = nn.Sequential(
            nn.Linear(in_proj, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stft_out = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            return_complex=True,
        )
        mag = torch.log1p(stft_out.abs()).unsqueeze(1)
        f_glob = self.enc_global(mag)

        if self.use_subbands:
            f_0_2k = self.enc_0_2k(mag[:, :, self.idx_0_2k[0]:self.idx_0_2k[1], :].contiguous())
            f_2_5k = self.enc_2_5k(mag[:, :, self.idx_2_5k[0]:self.idx_2_5k[1], :].contiguous())
            f_5_10k = self.enc_5_10k(mag[:, :, self.idx_5_10k[0]:self.idx_5_10k[1], :].contiguous())
            f_10_32k = self.enc_10_32k(mag[:, :, self.idx_10_32k[0]:self.idx_10_32k[1], :].contiguous())
            cat = torch.cat([f_glob, f_0_2k, f_2_5k, f_5_10k, f_10_32k], dim=-1)
        else:
            cat = f_glob

        return self.proj(cat)


class AdaptiveFeatureIdentificationFusionV2(nn.Module):
    """Learned Temporal-Spectral Feature Identification and Gated Residual Fusion."""
    def __init__(self, dim: int = 64, use_gate: bool = True):
        super().__init__()
        self.use_gate = use_gate
        self.proj_t = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU())
        self.proj_f = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU())

        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
                nn.Sigmoid(),
            )
            self.refine = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            )
        else:
            self.simple_fuse = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            )

    def forward(self, z_t: torch.Tensor, z_f: torch.Tensor) -> torch.Tensor:
        zt_p = self.proj_t(z_t)
        zf_p = self.proj_f(z_f)
        if self.use_gate:
            cat = torch.cat([zt_p, zf_p], dim=-1)
            g = self.gate(cat)
            fused = g * zt_p + (1.0 - g) * zf_p
            out = fused + self.refine(fused)
        else:
            cat = torch.cat([zt_p, zf_p], dim=-1)
            out = self.simple_fuse(cat)
        return out


class SentinelAI_MSRF_v2(nn.Module):
    """SentinelAI-MSRF v2: Multi-Scale Resonance Fusion Network (Revised).
    
    A problem-driven architecture for industrial bearing diagnostics:
    1. Multi-scale temporal feature extraction with channel attention & scale mixing
    2. Resonance-aware spectral feature extraction with dual mean+max pooling
    3. Learned temporal-spectral feature-gating & residual identification
    4. Compact parameter budget (~98k trainable parameters)
    """
    def __init__(self, cfg: dict, ablation: str = "full"):
        super().__init__()
        d = 64
        num_classes = cfg.get("dataset", {}).get("num_classes", 3)
        dropout = cfg.get("model", {}).get("dropout", 0.2)
        n_fft = cfg.get("model", {}).get("n_fft", 2048)
        hop = cfg.get("model", {}).get("hop_length", 512)
        fs = cfg.get("dataset", {}).get("sampling_rate", 64000)

        self.ablation = ablation
        use_multi_scale = (ablation != "no_temporal_multiscale")
        self.temporal = MultiScaleTemporalExtractorV2(out_dim=d, use_multi_scale=use_multi_scale)

        use_subbands = (ablation != "no_spectral_subbands")
        self.spectral = ResonanceSpectralEncoderV2(
            out_dim=d, n_fft=n_fft, hop=hop, fs=fs, use_subbands=use_subbands
        )

        use_gate = (ablation != "no_adaptive_gate")
        self.fusion = AdaptiveFeatureIdentificationFusionV2(dim=d, use_gate=use_gate)

        self.classifier = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_t = self.temporal(x)
        z_f = self.spectral(x)
        z_fused = self.fusion(z_t, z_f)
        return self.classifier(z_fused)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_kb(model: nn.Module) -> float:
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024.0


# ─────────────────────────────────────────────────────────────────────────────
# Explainability helpers
# ─────────────────────────────────────────────────────────────────────────────

class SpectrogramGradCAM:
    """Gradient-weighted Class Activation Map for the spectral branch.

    Hooks into the *second* Conv2D inside ``SpectralSubbandBlockV2.conv2``
    (the global full-spectrum pathway of ``ResonanceSpectralEncoderV2``).
    This layer sees the full log-magnitude spectrogram before dual pooling,
    making its activations well-localised in both frequency and time.

    Usage
    -----
    >>> cam = SpectrogramGradCAM(model)          # model: SentinelAI_MSRF_v2
    >>> heatmap = cam.generate(x, target_class=1)  # x: (1,1,64000)
    >>> cam.remove_hooks()

    Returns
    -------
    heatmap : np.ndarray, shape (freq_bins, time_frames), values in [0, 1]
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self._hooks: list = []
        self._attach()

    def _attach(self) -> None:
        """Register hooks on the global spectral encoder's second conv block."""
        try:
            target = self.model.spectral.enc_global.conv2
        except AttributeError:
            try:
                target = self.model.frequency.last_conv
            except AttributeError as exc:
                raise AttributeError(
                    "SpectrogramGradCAM requires an MSRF-v2 global spectral block "
                    "or ImprovedSentinelAI.frequency.last_conv."
                ) from exc

        def _save_acts(module, inp, out):
            self._acts = out.detach()

        def _save_grads(module, grad_in, grad_out):
            self._grads = grad_out[0].detach()

        self._hooks.append(target.register_forward_hook(_save_acts))
        self._hooks.append(target.register_full_backward_hook(_save_grads))

    def remove_hooks(self) -> None:
        """Remove all registered hooks (call after use to free memory)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(self, x: torch.Tensor, target_class: int) -> "np.ndarray":
        """Compute Grad-CAM heatmap for a single input tensor.

        Parameters
        ----------
        x : torch.Tensor, shape (1, 1, signal_length)
            Pre-processed, normalised input signal.
        target_class : int
            Class index (0=healthy, 1=outer_ring, 2=inner_ring).

        Returns
        -------
        heatmap : np.ndarray, shape (freq_bins, time_frames), float32, [0,1]
        """
        import numpy as np
        was_training = self.model.training
        self.model.eval()

        x = x.detach().requires_grad_(False)
        # Backpropagate the actual requested output logit. Earlier versions
        # differentiated an arbitrary feature-map channel, which was not a
        # faithful class explanation.
        self.model.zero_grad()

        # Forward pass — must be done with grad enabled for backward
        with torch.enable_grad():
            score = self.model(x)[0, target_class]
            score.backward()

        acts = self._acts  # (1, C, F', T')
        grads = self._grads  # (1, C, F', T')

        if acts is None or grads is None:
            raise RuntimeError("Grad-CAM hooks did not fire. Check model architecture.")

        # Global Average Pool over spatial dims → channel weights
        weights = grads.mean(dim=(-2, -1), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * acts).sum(dim=1).squeeze(0)       # (F', T')
        cam = torch.clamp(cam, min=0)                       # ReLU

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        heatmap = cam.cpu().numpy().astype(np.float32)

        if was_training:
            self.model.train()

        return heatmap


class TemporalGradientSaliency:
    """Input-gradient saliency for the temporal branch.

    For each input sample, computes |d(logit[target_class])/d(input)|,
    giving a pointwise measure of how sensitive the prediction is to
    each input sample position.

    This is NOT Grad-CAM; it does not require a specific layer. It is a
    simple gradient-based saliency map directly on the input waveform.

    Returns
    -------
    saliency : np.ndarray, shape (signal_length,), float32, [0, 1]
    """

    @staticmethod
    def generate(model: nn.Module, x: torch.Tensor, target_class: int) -> "np.ndarray":
        """
        Parameters
        ----------
        model : nn.Module with a .temporal branch (SentinelAI_MSRF_v2 or FrequencyOnly)
        x : torch.Tensor, shape (1, 1, L)
        target_class : int

        Returns
        -------
        saliency : np.ndarray, shape (L,), values in [0, 1]
        """
        import numpy as np
        was_training = model.training
        model.eval()
        model.zero_grad()

        x_in = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            logits = model(x_in)
            score = logits[0, target_class]
            score.backward()

        grad = x_in.grad  # (1, 1, L)
        if grad is None:
            return np.zeros(x.shape[-1], dtype=np.float32)

        sal = grad.abs().squeeze().cpu().numpy().astype(np.float32)
        g_min, g_max = sal.min(), sal.max()
        if g_max > g_min:
            sal = (sal - g_min) / (g_max - g_min)

        if was_training:
            model.train()

        return sal
