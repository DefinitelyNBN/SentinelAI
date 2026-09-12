"""Baseline model builder for SentinelAI Phase 6."""
from __future__ import annotations
from .models import Baseline1DCNN

def build_official_baseline(cfg: dict) -> Baseline1DCNN:
    """Build the required 1-D CNN baseline for Track 1 comparison."""
    num_classes = cfg.get("dataset", {}).get("num_classes", 3)
    return Baseline1DCNN(num_classes=num_classes)
