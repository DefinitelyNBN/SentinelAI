"""Fast, synthetic tests for the upgraded components; no Paderborn test data."""
import copy
import csv
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
from src.improved_model import ImprovedSentinelAI, FusionModule
from src.paderborn_dataset import validate_bearing_disjoint_manifest
from src.preprocessing import SignalValidationError, preprocess_window
from src.utils import load_config


def small_config():
    cfg = copy.deepcopy(load_config())
    cfg["dataset"]["signal_length"] = 1024
    cfg["model"]["n_fft"] = 64
    cfg["model"]["hop_length"] = 16
    return cfg


def test_model_output_and_stft_shape():
    cfg = small_config(); model = ImprovedSentinelAI(cfg)
    x = torch.randn(2, 1, 1024)
    assert model(x).shape == (2, 3)
    assert model.frequency.spectrogram(x).shape[1] == 1


@pytest.mark.parametrize("kind", ["concat", "gated", "attention"])
def test_all_fusion_types(kind):
    result = FusionModule(16, kind)(torch.randn(3, 16), torch.randn(3, 16))
    assert result.shape == (3, 16)


def test_preprocessing_rejects_invalid_and_uses_train_stats():
    cfg = small_config()
    output = preprocess_window(np.ones(1024, dtype=np.float32), cfg, 1.0, 2.0)
    assert np.allclose(output, 0)
    with pytest.raises(SignalValidationError): preprocess_window(np.array([np.nan] * 1024), cfg, 0, 1)


def test_bearing_overlap_raises():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.csv"
        with open(path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["split", "bearing_id", "label", "label_id"]); writer.writeheader()
            writer.writerows([{"split":"train","bearing_id":"K001","label":"healthy","label_id":0}, {"split":"validation","bearing_id":"K001","label":"healthy","label_id":0}, {"split":"test","bearing_id":"K002","label":"outer_ring","label_id":1}])
        with pytest.raises(ValueError, match="Bearing-disjoint"):
            validate_bearing_disjoint_manifest(path)
