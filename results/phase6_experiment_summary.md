# SentinelAI Phase 6: Training, Ablation, and Final Evaluation Summary

**Project**: SentinelAI — Industrial Signal Intelligence for Early Machine-Failure Detection  
**Task**: Three-class bearing-disjoint vibration classification (`healthy`, `outer_ring`, `inner_ring`)  
**Dataset**: Paderborn University Bearing Dataset  
**Primary Metric**: Validation Macro-F1 (for model selection)  
**Status**: COMPLETE  

---

## 1. Executive Summary & Core Results

| Model | Parameters | Size (KB) | Train Time (s) | Val Macro-F1 | Val Acc | Test Macro-F1 | Test Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline 1-D CNN** | 3,971 | 15.5 | 111.7 | **0.5558** | 0.6681 | **0.5567** | 0.6702 |
| **Temporal Only (1-D Conv)** | 19,203 | 75.8 | 254.3 | **0.5558** | 0.6681 | **0.5567** | 0.6702 |
| **Frequency Only (STFT 2-D Conv)** **(Selected)** | 11,363 | 52.8 | 1008.0 | **0.6152** | 0.6461 | **0.6641** | 0.6743 |
| **Full SentinelAI (Dual-Branch Fusion)** | 30,307 | 127.5 | 639.8 | **0.5558** | 0.6681 | **0.8211** | 0.8335 |

### Key Findings & Hypothesis Evaluation
- **Baseline Validation Macro-F1**: `0.5558`
- **Temporal-Only Validation Macro-F1**: `0.5558`
- **Frequency-Only Validation Macro-F1**: `0.6152`
- **SentinelAI Validation Macro-F1**: `0.5558`
- **SentinelAI vs Baseline Improvement (Val)**: `+0.0000` (+0.00%)
- **SentinelAI vs Temporal-Only Improvement (Val)**: `+0.0000` (+0.00%)
- **Selected Model**: **Frequency Only (STFT 2-D Conv)**
- **Final Test Macro-F1**: `0.6641` (Accuracy: `0.6743`)
- **Central Hypothesis Supported**: **NO**

---

## 2. Dataset & Split Specification

- **Input**: 1-D `vibration_1` signal sliced into 1.0 s windows (64,000 samples @ nominal 64 kHz, 0% overlap).
- **Manifest**: `data/metadata/paderborn_manifest.csv` (SHA-256: `71916a2f142813455013024017bc05eb97406ff44d85c53da32efbb6f6d2ddaf`)
- **Normalization**: Training-only global Z-score (Mean = `0.008097`, Std = `0.353755`)
- **Splits** (Strictly Bearing-Disjoint, 0% leakage):
  - **Train**: K001, K002, K003, K004, KA04, KA16, KA30, KI04, KI16, KI18 (3,191 windows)
  - **Validation**: K005, KA15, KI21 (955 windows)
  - **Test (Locked)**: K006, KA22, KI14 (955 windows)

---

## 3. Main Ablation Analysis

The central hypothesis posits that combining temporal and frequency-domain representations yields superior bearing damage classification than either representation alone.

1. **Full SentinelAI vs. Temporal-Only**:
   - Temporal-Only: `0.5558` Macro-F1
   - Full SentinelAI: `0.5558` Macro-F1
   - Delta from adding Frequency branch: `+0.0000`

2. **Full SentinelAI vs. Frequency-Only**:
   - Frequency-Only: `0.6152` Macro-F1
   - Full SentinelAI: `0.5558` Macro-F1
   - Delta from adding Temporal branch: `-0.0594`

3. **SentinelAI vs. 1D CNN Baseline**:
   - Baseline: `0.5558` Macro-F1
   - SentinelAI: `0.5558` Macro-F1
   - Delta vs Baseline: `+0.0000`

---

## 4. Detailed Per-Class Breakdown

### Validation Set Per-Class Metrics

#### Baseline 1-D CNN (Validation)
| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| healthy | 1.0000 | 1.0000 | 1.0000 |
| outer_ring | 0.0000 | 0.0000 | 0.0000 |
| inner_ring | 0.5008 | 1.0000 | 0.6674 |

Confusion Matrix:
```
[[320   0   0]
 [  0   0 317]
 [  0   0 318]]
```

#### Temporal Only (1-D Conv) (Validation)
| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| healthy | 1.0000 | 1.0000 | 1.0000 |
| outer_ring | 0.0000 | 0.0000 | 0.0000 |
| inner_ring | 0.5008 | 1.0000 | 0.6674 |

Confusion Matrix:
```
[[320   0   0]
 [  0   0 317]
 [  0   0 318]]
```

#### Frequency Only (STFT 2-D Conv) (Validation)
| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| healthy | 0.8908 | 0.9688 | 0.9281 |
| outer_ring | 0.4650 | 0.2303 | 0.3080 |
| inner_ring | 0.5200 | 0.7358 | 0.6094 |

Confusion Matrix:
```
[[310   0  10]
 [ 38  73 206]
 [  0  84 234]]
```

#### Full SentinelAI (Dual-Branch Fusion) (Validation)
| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| healthy | 1.0000 | 1.0000 | 1.0000 |
| outer_ring | 0.0000 | 0.0000 | 0.0000 |
| inner_ring | 0.5008 | 1.0000 | 0.6674 |

Confusion Matrix:
```
[[320   0   0]
 [  0   0 317]
 [  0   0 318]]
```

---

## 5. Final Locked Test Set Evaluation

> **LOCKED TEST SET RULE VERIFIED**: The test set was evaluated strictly once after all training, architecture decisions, and checkpoint selections were finalized.


#### Baseline 1-D CNN (Test Set)
| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| healthy | 1.0000 | 1.0000 | 1.0000 |
| outer_ring | 0.0000 | 0.0000 | 0.0000 |
| inner_ring | 0.5039 | 1.0000 | 0.6702 |

Confusion Matrix:
```
[[320   0   0]
 [  0   0 315]
 [  0   0 320]]
```

#### Temporal Only (1-D Conv) (Test Set)
| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| healthy | 1.0000 | 1.0000 | 1.0000 |
| outer_ring | 0.0000 | 0.0000 | 0.0000 |
| inner_ring | 0.5039 | 1.0000 | 0.6702 |

Confusion Matrix:
```
[[320   0   0]
 [  0   0 315]
 [  0   0 320]]
```

#### Frequency Only (STFT 2-D Conv) (Test Set)
| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| healthy | 0.5684 | 1.0000 | 0.7248 |
| outer_ring | 0.6964 | 0.4952 | 0.5788 |
| inner_ring | 1.0000 | 0.5250 | 0.6885 |

Confusion Matrix:
```
[[320   0   0]
 [159 156   0]
 [ 84  68 168]]
```

#### Full SentinelAI (Dual-Branch Fusion) (Test Set)
| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| healthy | 1.0000 | 1.0000 | 1.0000 |
| outer_ring | 1.0000 | 0.4952 | 0.6624 |
| inner_ring | 0.6681 | 1.0000 | 0.8010 |

Confusion Matrix:
```
[[320   0   0]
 [  0 156 159]
 [  0   0 320]]
```

---

## 6. Reproducibility & Protocol Audit

- **Random Seed**: `42`
- **Device**: `mps`
- **PyTorch Version**: `2.14.0`
- **Python Version**: `3.13.13`
- **Optimizer**: `AdamW` (LR: `0.001`, Weight Decay: `0.0001`)
- **Batch Size**: `32`
- **Max Epochs**: `30`, Patience: `6`
- **Augmentation**: Disabled for all primary experiments
- **Checkpoints**: Saved under `results/checkpoints/`
- **Figures**: Saved under `results/figures/`

---

## 7. Scientific Honesty & Limitations

1. **Experimental Scope**: SentinelAI is evaluated as an experimental deep-learning signal-classification system on the benchmark Paderborn University bearing test rig.
2. **Generalization Challenge**: Real bearing damage classification across unseen bearings (bearing-disjoint) is challenging because individual mechanical units have distinct transfer functions and background noise baselines.
3. **No Unsubstantiated Claims**: No claim is made of production deployment, real-time edge processing, or predictive maintenance scheduling.
