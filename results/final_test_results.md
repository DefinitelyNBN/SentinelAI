# Final Frozen Test Evaluation: SentinelAI

**Project**: SentinelAI — Industrial Signal Intelligence for Early Machine-Failure Detection  
**Dataset**: Paderborn University Bearing Benchmark (1-D Vibration Channel 1)  
**Evaluation Date**: September 2026  
**Status**: Completed (Final Evaluation)

---

## 1. Experimental Integrity & Model Selection

- **Model**: Frequency-Only v1 (STFT Log-Magnitude + 2-D ConvNet)
- **Selection Criterion**: **Highest official validation Macro-F1 (0.6152)**
- **Official Validation Macro-F1**: **0.6152**
- **Checkpoint**: `results/checkpoints/frequency_only.pt`
- **Checkpoint model name / class**: `Frequency Only (STFT 2-D Conv)` / `FrequencyOnly`
- **Trainable parameters**: 11,363
- **Seed**: 42
- **Manifest SHA-256**: `71916a2f142813455013024017bc05eb97406ff44d85c53da32efbb6f6d2ddaf`
- **Class mapping**: Healthy = 0, Outer Ring = 1, Inner Ring = 2
- **Official Test Bearings**: `K006` (Healthy), `KA22` (Real Outer Ring), `KI14` (Real Inner Ring)
- **Test Set Size**: 955 windows (Healthy: 320, Outer Ring: 315, Inner Ring: 320)

> **IMPORTANT SCIENTIFIC GOVERNANCE STATEMENT**:  
> The Frequency-Only v1 model was selected **strictly prior** to this final test evaluation based on official validation Macro-F1. The test set was strictly locked and **was not used** for model selection, architecture search, hyperparameter tuning, or threshold optimization. No retraining or fine-tuning was performed.

---

## 2. Pre-Flight Verification Audit

| Audit Parameter | Verification Requirement | Status |
|---|---|:---:|
| `MODEL_FROZEN` | Frequency-Only v1 architecture unchanged | **PASS** |
| `CHECKPOINT` | Checkpoint verified intact (`frequency_only.pt`) | **PASS** |
| `MANIFEST_SHA` | `71916a2f142813455013024017bc05eb97406ff44d85c53da32efbb6f6d2ddaf` | **PASS** |
| `TEST_BEARINGS` | Strictly `K006`, `KA22`, `KI14` | **PASS** |
| `TEST_COUNT` | 955 windows (320 Healthy, 315 Outer, 320 Inner) | **PASS** |
| `NORMALIZATION` | Training-derived stats only (mean=0.008097, std=0.353755) | **PASS** |
| `STFT_CONFIG` | N_fft=2048, hop=512, Hann window, full 0–32 kHz spectrum | **PASS** |
| `NO_RETRAINING` | Inference only (`model.eval()`, `torch.no_grad()`) | **PASS** |
| `NO_TEST_TUNING` | Standard argmax classification, zero test-time tuning | **PASS** |

---

## 3. Final Test Performance (Frequency-Only v1)

### Summary Metrics:
- **Test Accuracy**: **0.6743** (67.43%)
- **Test Macro-F1**: **0.6641**
- **Test Macro-Precision**: **0.7549**
- **Test Macro-Recall**: **0.6734**

### Per-Class Test Metrics:

| Class | Class ID | Test Precision | Test Recall | Test F1-Score | Support |
|---|:---:|---:|---:|---:|---:|
| **Healthy** | 0 | 0.5684 | 1.0000 | 0.7248 | 320 |
| **Outer Ring** | 1 | 0.6964 | 0.4952 | 0.5788 | 315 |
| **Inner Ring** | 2 | 1.0000 | 0.5250 | 0.6885 | 320 |

### Confusion Matrix:

```
                  Predicted Healthy    Predicted Outer Ring    Predicted Inner Ring
True Healthy             320                    0                       0
True Outer Ring          159                  156                       0
True Inner Ring           84                   68                     168
```

---

## 4. Mandatory Baseline Comparison

| Model | Test Macro-F1 |
|---|---:|
| Baseline 1D CNN | 0.5567 |
| Frequency-Only v1 (frozen selected model) | 0.6641 |

- **Absolute improvement**: +0.1073 Macro-F1 (+10.73 percentage points)
- **Percentage improvement over baseline**: +19.28%
