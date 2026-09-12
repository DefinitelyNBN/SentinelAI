"""Reproducibility and configuration utilities."""
from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import torch
import yaml

def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f: return yaml.safe_load(f)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    try: torch.use_deterministic_algorithms(True, warn_only=True)
    except RuntimeError: pass

def device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def save_json(value, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8") as f: json.dump(value,f,indent=2)
