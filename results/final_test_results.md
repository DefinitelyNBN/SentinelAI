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

## 4. Benchmark Comparison Across Architectures

The following table presents the final comparison across all benchmarked architectures. For non-selected models, values reflect the frozen Phase 6 pre-recorded evaluations:

| Model | Validation Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|
| **Baseline 1D CNN** | 0.5558 | 0.6702 | 0.5567 |
| **Temporal-Only** | 0.5558 | 0.6702 | 0.5567 |
| **Frequency-Only v1** *(Selected)* | **0.6152** | **0.6743** | **0.6641** |
| **Full SentinelAI** | 0.5558 | 0.8335 | 0.8211 |

### Post-Selection Baseline Improvement:
Delta Test Macro-F1 = Macro-F1(Freq-Only v1) - Macro-F1(Baseline) = 0.6641 - 0.5567 = +0.1074 (+10.74%)

---

## 5. Scientific Findings & Discussion

1. **Superior Generalization over 1-D Baselines**:
   - Both the Baseline 1D CNN and Temporal-Only models collapsed to zero outer-ring recall on the test set (predicting 0 outer-ring windows on `KA22`), exactly replicating their failure on validation bearing `KA15`.
   - Frequency-Only v1 successfully identifies outer-ring damage on the locked test set (Recall = 49.52%, 156/315 windows), demonstrating genuine feature extraction capability across physically held-out bearing units.
2. **Post-Selection Context with Full SentinelAI**:
   - Although Full SentinelAI achieved 0.8211 on the test set, it collapsed on the validation set (`KA15` outer-ring recall = 0.0000, Val Macro-F1 = 0.5558). Under standard blind machine learning protocol, Full SentinelAI was appropriately disqualified during validation selection.
   - Frequency-Only v1's selection is scientifically sound, fully repeatable, and demonstrates a **+0.1074 Test Macro-F1 gain** over the 1-D CNN baseline.
