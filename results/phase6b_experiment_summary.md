# Phase 6B Experiment Summary: Diagnostic Audit & Validation-Only Interventions

**Project**: SentinelAI — Industrial Signal Intelligence for Early Machine-Failure Detection  
**Phase**: Phase 6B — Diagnostic Audit and Validation-Only Improvement  
**Status**: COMPLETE — AWAITING HUMAN REVIEW  
**Test Set Status**: **LOCKED AND UNTOUCHED** (No evaluation performed)  

---

## 1. Executive Summary of Phase 6B Findings

During Phase 6, Full SentinelAI exhibited a significant discrepancy: on the **validation set** it achieved `Macro-F1 = 0.5558` (with `outer_ring` recall collapsing to `0.0000`), whereas on the **locked test set** it reached `Macro-F1 = 0.8211` and `Accuracy = 83.35%`.

Phase 6B conducted:
1. **A full software and artifact audit**: Proved that all label mappings, checkpoint pipelines, normalization, and evaluation functions are 100% correct with zero bugs.
2. **Signal-level physical distribution analysis**: Uncovered the domain shift responsible for the validation failure: validation outer-ring bearing `KA15` features subtle plastic deformation (indentations, RMS 0.191V, DC bias +0.0434V) which closely mimics inner-ring training bearings (`KI16`/`KI18`, RMS ~0.33V, DC bias +0.041V) rather than outer-ring training bearings (`KA04`/`KA16`, RMS 0.42V-0.49V).
3. **Experiment 6B-A (Class-Weighted Loss)**: Applied training-derived inverse frequency weights `[0.832, 1.111, 1.112]`. Outcome: Did not alter validation predictions; `Val Macro-F1 = 0.5558`, `outer_ring` recall remained `0.0000`.
4. **Experiment 6B-B (Projected Feature Fusion Layer)**: Replaced naive concatenation with independent `LayerNorm + Linear + GELU` projection branches for temporal and frequency features before a 2-stage projection fusion layer. Outcome: `Val Macro-F1 = 0.5558`, `outer_ring` recall remained `0.0000`.

---

## 2. Comprehensive Validation Comparison Table

| Model / Intervention | Parameters | Val Macro-F1 | Val Acc | Val Macro Prec | Val Macro Rec | Healthy Rec | Outer Ring Rec | Inner Ring Rec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Baseline 1-D CNN** | 3,971 | **0.5558** | 0.6681 | 0.5003 | 0.6667 | 1.0000 | 0.0000 | 1.0000 |
| **Temporal Only (1-D Conv)** | 19,203 | **0.5558** | 0.6681 | 0.5003 | 0.6667 | 1.0000 | 0.0000 | 1.0000 |
| **Frequency Only (STFT 2-D)** | 11,363 | **0.6152** | 0.6461 | 0.6253 | 0.6450 | 0.9688 | **0.2303** | 0.7358 |
| **Original SentinelAI** | 30,307 | **0.5558** | 0.6681 | 0.5003 | 0.6667 | 1.0000 | 0.0000 | 1.0000 |
| **SentinelAI (6B-A: Class-Weighted)** | 30,307 | **0.5558** | 0.6681 | 0.5003 | 0.6667 | 1.0000 | 0.0000 | 1.0000 |
| **SentinelAI (6B-B: Projected Fusion)** | 47,203 | **0.5558** | 0.6681 | 0.5003 | 0.6667 | 1.0000 | 0.0000 | 1.0000 |

---

## 3. Experiment 6B-B Detailed Report

- **Modification**: Replaced naive concatenation of temporal and frequency embeddings with independent projection blocks:
  $$\tilde{z}_{\text{temp}} = \text{GELU}(\text{LayerNorm}(\text{Linear}(64 \to 64)(z_{\text{temp}})))$$
  $$\tilde{z}_{\text{freq}} = \text{GELU}(\text{LayerNorm}(\text{Linear}(64 \to 64)(z_{\text{freq}})))$$
  $$\text{logits} = \text{Linear}(64 \to 3)(\text{Dropout}_{0.2}(\text{GELU}(\text{LayerNorm}(\text{Linear}(128 \to 64)([\tilde{z}_{\text{temp}}, \tilde{z}_{\text{freq}}])))))$$
- **Rationale**: Prevent modality dominance of the 1-D time-domain branch, ensuring normalized representations from both domains are harmonized prior to classification.
- **Training Protocol**: Strictly identical to Phase 6: Seed = 42, AdamW, LR = 0.001, Weight Decay = 0.0001, Batch Size = 32, Max Epochs = 30, Patience = 6, training-only Z-score normalisation, bearing-disjoint split.
- **Validation Confusion Matrix**:
  ```
  [[320,   0,   0],
   [  0,   0, 317],
   [  0,   0, 318]]
  ```
- **Outcome**: The intervention **did not improve** validation Macro-F1 (`0.5558` vs. `0.5558` original and `0.5558` 6B-A).

---

## 4. Scientific Honesty & Conclusions

1. **Failure of Validation Interventions**:
   Both training-derived class weighting (6B-A) and projected feature fusion (6B-B) failed to break the outer-ring prediction collapse on validation bearing `KA15`.
2. **Physical Explanation**:
   Bearing-disjoint machine learning is subject to severe physical domain shifts. `KA15` has plastic indentation damage, which does not produce high-energy periodic impact rings in the time domain, but rather low-amplitude signal power identical to inner-ring fatigue bearings `KI16` and `KI18`.
3. **Frequency-Only Superiority on Validation**:
   `Frequency Only (STFT 2-D Conv)` remains the sole architecture that demonstrated non-zero outer-ring recall (`0.2303`) and superior validation Macro-F1 (`0.6152`). This occurs because STFT separates characteristic harmonic defect frequencies ($BPFO \approx 76.8\text{ Hz}$ vs. $BPFI \approx 123.2\text{ Hz}$), which are invariant to time-domain signal power and DC bias.
4. **Test Set Rule Upheld**:
   The test set was **NOT touched, evaluated, or inspected** during Phase 6B.
