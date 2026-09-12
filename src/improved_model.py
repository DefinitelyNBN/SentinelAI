"""Compact, explainable dual-domain SentinelAI architecture.

Legacy models remain in ``src.models``. This module is intentionally isolated
so ablations can use the same data contract without silently changing them.
"""
from __future__ import annotations
import torch
from torch import nn


class ChannelAttention1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.net = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels), nn.Sigmoid())
    def forward(self, x): return x * self.net(x).unsqueeze(-1)


class ChannelAttention2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels), nn.Sigmoid())
    def forward(self, x): return x * self.net(x).unsqueeze(-1).unsqueeze(-1)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm1d(channels), nn.ReLU(), nn.Conv1d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm1d(channels))
        self.act = nn.ReLU()
    def forward(self, x): return self.act(x + self.body(x))


class TemporalEncoder(nn.Module):
    def __init__(self, channels: list[int], embedding_dim: int, reduction: int = 4):
        super().__init__()
        blocks, in_ch = [], 1
        for out_ch in channels:
            blocks += [nn.Conv1d(in_ch, out_ch, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(out_ch), nn.ReLU(), nn.MaxPool1d(2)]
            in_ch = out_ch
        self.features = nn.Sequential(*blocks, ResidualBlock1D(in_ch), ChannelAttention1D(in_ch, reduction))
        self.embed = nn.Linear(in_ch, embedding_dim)
    def forward(self, x): return self.embed(self.features(x).mean(-1))


class FrequencyEncoder(nn.Module):
    def __init__(self, channels: list[int], embedding_dim: int, n_fft: int, hop_length: int, reduction: int = 4):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        self.register_buffer("window", torch.hann_window(n_fft))
        blocks, in_ch = [], 1
        for out_ch in channels:
            blocks += [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.last_conv = nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False), nn.BatchNorm2d(in_ch), nn.ReLU(), ChannelAttention2D(in_ch, reduction))
        self.embed = nn.Linear(in_ch, embedding_dim)
    def spectrogram(self, x):
        z = torch.stft(x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)
        return torch.log1p(z.abs()).unsqueeze(1)
    def forward(self, x): return self.embed(self.last_conv(self.features(self.spectrogram(x))).mean((-1, -2)))


class FusionModule(nn.Module):
    """concat, per-feature gated, or two-token attention fusion."""
    def __init__(self, dim: int, fusion_type: str = "gated"):
        super().__init__(); self.fusion_type = fusion_type
        if fusion_type == "concat": self.project = nn.Sequential(nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.ReLU())
        elif fusion_type == "gated": self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        elif fusion_type == "attention": self.score = nn.Sequential(nn.Linear(dim, max(4, dim // 2)), nn.Tanh(), nn.Linear(max(4, dim // 2), 1))
        else: raise ValueError("fusion_type must be concat, gated, or attention")
    def forward(self, temporal, frequency):
        if self.fusion_type == "concat": return self.project(torch.cat([temporal, frequency], 1))
        if self.fusion_type == "gated":
            gate = self.gate(torch.cat([temporal, frequency], 1)); return gate * temporal + (1 - gate) * frequency
        tokens = torch.stack([temporal, frequency], 1); return (torch.softmax(self.score(tokens), 1) * tokens).sum(1)


class Classifier(nn.Module):
    def __init__(self, dim, num_classes, dropout):
        super().__init__(); self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, num_classes))
    def forward(self, x): return self.net(x)


class ImprovedSentinelAI(nn.Module):
    """Residual temporal + STFT spectral encoders with lightweight attention."""
    def __init__(self, cfg: dict, fusion_type: str | None = None):
        super().__init__(); m = cfg["model"]; d = int(m.get("embedding_dim", 64)); reduction = int(m.get("attention", {}).get("reduction", 4))
        temporal = m.get("improved_temporal_channels", [12, 24]); frequency = m.get("improved_frequency_channels", [8, 16])
        self.temporal = TemporalEncoder(temporal, d, reduction)
        self.frequency = FrequencyEncoder(frequency, d, int(m["n_fft"]), int(m["hop_length"]), reduction)
        self.fusion = FusionModule(d, fusion_type or m.get("fusion_type", "gated"))
        self.classifier = Classifier(d, int(cfg["dataset"]["num_classes"]), float(m.get("dropout", .2)))
    def forward(self, x): return self.classifier(self.fusion(self.temporal(x), self.frequency(x)))
