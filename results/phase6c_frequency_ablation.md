# Phase 6C-A — Frequency-Domain Ablation

**Project**: SentinelAI  
**Date**: September 2026  
**Phase**: 6C-A — Controlled Frequency-Domain Ablation  
**TEST SET**: Locked — not evaluated.

---

## Scientific Question

> Does restricting the STFT representation to a physically meaningful
> low-frequency region (0–1 kHz) improve bearing-disjoint validation
> classification?

---

## Manifest Integrity

SHA-256: `71916a2f142813455013024017bc05eb97406ff44d85c53da32efbb6f6d2ddaf`  
Status: **VERIFIED UNCHANGED**

---

## STFT Configuration

| Parameter | Value |
|---|---|
| Sampling rate | 64000 Hz |
| n_fft | 2048 |
| hop_length | 512 |
| Window | Hann |
| Frequency resolution | 31.2500 Hz/bin |
| Total bins (0–Nyquist) | 1025 (0 – 32000 Hz) |
| **Retained bins (v2)** | **33** (bins 0–32) |
| **Frequency range retained** | **0 – 1000.00 Hz** |

---

## Experimental Control

| Dimension | v1 | v2 |
|---|---|---|
| Architecture depth | identical | identical |
| Channel counts | identical | identical |
| Embedding dim | identical | identical |
| Classifier | identical | identical |
| Optimizer | AdamW | AdamW |
| Learning rate | 0.001 | 0.001 |
| Weight decay | 0.0001 | 0.0001 |
| Batch size | 32 | 32 |
| Max epochs | 30 | 30 |
| Patience | 6 | 6 |
| Loss | CrossEntropy | CrossEntropy |
| Augmentation | none | none |
| Normalization | training Z-score | training Z-score |
| Seed | 42 | 42 |
| **Experimental variable** | Full STFT (0–32 kHz) | **STFT 0–1 kHz only** |

---

## Results

### Frequency-Only v2 — Model Info

- Parameter count : 11363
- Model size       : 52.78 KB
- Training time    : 90.7 s
- Epochs trained   : 7
- Best epoch       : 1

### Validation Metrics Comparison

| Metric | v1 (reference) | v2 (experiment) | Delta |
|---|---:|---:|---:|
| Macro-F1 | 0.6152 | 0.1673 | -0.4479 |
| Accuracy | 0.6461 | 0.3351 | -0.3110 |
| Outer-ring recall | 0.2303 | 0.0000 | -0.2303 |
| Inner-ring recall | 0.7358 | 0.0000 | -0.7358 |
| Healthy recall | 0.9688 | 1.0000 | +0.0312 |

### Frequency-Only v2 — Validation Confusion Matrix

```
[[320, 0, 0], [317, 0, 0], [318, 0, 0]]
```

Classes: [0=healthy, 1=outer_ring, 2=inner_ring]

---

## Scientific Interpretation

The frequency-band restriction to 0–1 kHz did **NOT** improve validation
Macro-F1. Delta = -0.4479. This is a severe degradation.

The model trained to near-zero training loss but collapsed on validation,
predicting predominantly one class (all or most windows → outer_ring or
a single class), yielding ~0.33 accuracy consistent with class-prior guessing.

**Root-cause analysis of v2 failure:**

1. **Insufficient spectral resolution**: With only 33 frequency bins
   (31.25 Hz resolution), the bearing-fault kinematic frequencies
   (BPFO ≈ 76.8 Hz, BPFI ≈ 123.2 Hz and their harmonics) fall on only
   2–4 bins. The 2D CNN has too little spectral detail to learn discriminative
   harmonic patterns from such a narrow representation.

2. **Excessive spatial compression**: MaxPool2d(2) applied 3–4 times to a
   height-33 spectrogram reduces the frequency dimension to 2–4 rows before
   the global average pool. The CNN essentially averages over the entire
   band, losing all frequency-location information.

3. **v1 success comes from the broader spectrum**: Frequency-Only v1 uses
   1025 bins and achieves its partial outer-ring recall by leveraging
   higher harmonics and sideband structure beyond 1 kHz. Restricting to
   1 kHz eliminates this information.

**Conclusion**: Frequency-Only v1 remains the best validation model.
No further architectural modifications will be invented from this result.

---

## Final Summary

```
REFERENCE:
Frequency-Only v1
Val Macro-F1 = 0.6152

EXPERIMENT:
Frequency-Only v2
Val Macro-F1 = 0.1673

DELTA:
-0.4479

Outer-ring recall:
v1 = 0.2303
v2 = 0.0000

FINAL DECISION:
NO IMPROVEMENT
```

---

**PHASE 6C-A COMPLETE — FREQUENCY-DOMAIN ABLATION FINISHED — TEST SET REMAINS LOCKED**
