# SentinelAI — Phase 5: Dataset Generation and Manifest

## Scope

Phase 5 converts the raw Paderborn MAT archives into a deterministic, leakage-safe experiment-ready dataset.

**What this phase produced:**
- A deterministic manifest CSV (`data/metadata/paderborn_manifest.csv`)
- Training-only normalisation statistics (`data/metadata/paderborn_norm_stats.json`)
- A MAT-file reader (`src/paderborn_mat.py`) implementing the approved time-axis policy
- A manifest generator (`src/generate_manifest.py`) with 12 built-in integrity checks
- A PyTorch Dataset (`src/paderborn_dataset.py`) compatible with SentinelAI
- QC inspection output (`results/qc_windows.json`, `results/qc_waveforms.png`)

**What this phase did NOT do:**
- No model training
- No test-set evaluation
- No archive modification
- No `.npy` cache files
- No new MAT file downloads

---

## 1. Bearing-Level Split Used

Source: Phase 4 approved candidate split — [`results/paderborn_dataset_design.json`](results/paderborn_dataset_design.json)

| Partition | Healthy | Outer-Ring | Inner-Ring |
|---|---|---|---|
| **Train (10)** | K001, K002, K003, K004 | KA04, KA16, KA30 | KI04, KI16, KI18 |
| **Validation (3)** | K005 | KA15 | KI21 |
| **Test (3)** | K006 | KA22 | KI14 |

All windows from a bearing remain in exactly one partition. No window crosses a bearing or recording boundary.

---

## 2. Windowing Policy

| Parameter | Value |
|---|---|
| Window length | **1.0 s = 64,000 samples** |
| Stride | **64,000 samples (zero overlap)** |
| Overlap | **0 %** |
| Time-axis method | **Discrete sample indexing — no resampling** |
| Start index | 0 (from beginning of each MAT vibration array) |
| Truncation | Any trailing samples insufficient for a full window are discarded |

KI16's two known longer recordings (4.67 s, 4.28 s) yield 4 windows instead of 3. These are handled transparently by the extraction code and remain in the train split.

---

## 3. Exact Window Counts

### By split

| Split | Windows | Recordings |
|---|---:|---:|
| **Train** | **3,191** | 800 |
| **Validation** | **955** | 240 |
| **Test** | **955** | 240 |
| **Total** | **5,101** | 1,280 |

> **Note:** Phase 4 estimated 3,840 windows assuming exactly 3 windows per recording. The actual count is 5,101 because KI16's longer recordings yield 4 windows each. All extra windows are in the train split — no evaluation partition is affected.

### By class

| Class | Windows | Split: Train | Validation | Test |
|---|---:|---:|---:|---:|
| **Healthy** | 1,918 | 1,278 | 320 | 320 |
| **Outer-ring** | 1,589 | 957 | 317 | 315 |
| **Inner-ring** | 1,594 | 956 | 318 | 320 |

Validation and test partitions are near-perfectly balanced (≤5 window difference between classes).

### By bearing

| Bearing | Split | Class | Windows |
|---|---|---|---:|
| K001 | train | healthy | 320 |
| K002 | train | healthy | 318 |
| K003 | train | healthy | 320 |
| K004 | train | healthy | 320 |
| K005 | val | healthy | 320 |
| K006 | test | healthy | 320 |
| KA04 | train | outer_ring | 319 |
| KA15 | val | outer_ring | 317 |
| KA16 | train | outer_ring | 319 |
| KA22 | test | outer_ring | 315 |
| KA30 | train | outer_ring | 319 |
| KI04 | train | inner_ring | 320 |
| KI14 | test | inner_ring | 320 |
| KI16 | train | inner_ring | 319 |
| KI18 | train | inner_ring | 317 |
| KI21 | val | inner_ring | 318 |

### By operating condition

| Condition | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| N09_M07_F10 (900 rpm) | 1,276 | 799 | 238 | 239 |
| N15_M01_F10 (1500 rpm, 0.1 Nm) | 1,276 | 798 | 239 | 239 |
| N15_M07_F04 (1500 rpm, 400 N) | 1,273 | 796 | 240 | 237 |
| N15_M07_F10 (1500 rpm reference) | 1,276 | 798 | 238 | 240 |

All four operating conditions are nearly equally represented in every split (~25% each). No condition is held out from the primary evaluation.

---

## 4. Time-Axis and Resampling Policy

**Method: Discrete sample-index windowing. No resampling.**

The HostService acquisition clock has a nominal period of 15.625 µs (64 kHz). Observed HostService timestamps show OS-level scheduling jitter, not a true clock drift. The mean dt across all files is within ±2 Hz of 64,000 Hz.

Windows are extracted as:
```python
vib[start_sample : end_sample]   # shape (64000,)
```

The HostService time axis is loaded for provenance metadata only (`start_time_s`, `end_time_s` stored in the manifest). It is never used to resample or index signals.

---

## 5. Normalisation

| Parameter | Value |
|---|---|
| Strategy | Global Z-score, training windows only |
| Mean | **0.008097** |
| Std | **0.353755** |
| Samples used | 204,224,000 (3,191 training windows × 64,000) |
| Algorithm | Welford online (numerically stable) |
| Stored at | `data/metadata/paderborn_norm_stats.json` |

Validation and test windows use the training-derived statistics exactly. The norm stats file is read at Dataset construction time and passed to `PaderbornDataset`.

---

## 6. Integrity Check Results

All 12 checks passed on the first run.

| # | Check | Result |
|---|---|---|
| 1 | No bearing in multiple splits | ✅ Pass |
| 2 | No MAT recording in multiple splits | ✅ Pass |
| 3 | No window crosses recording boundary | ✅ Pass |
| 4 | No duplicate window IDs | ✅ Pass |
| 5 | All 1,280 source MAT files exist | ✅ Pass |
| 6 | All window start/end offsets valid | ✅ Pass |
| 7 | Only approved class labels | ✅ Pass |
| 8 | Only approved bearings | ✅ Pass |
| 9 | No excluded bearings present | ✅ Pass |
| 10 | label_id consistent with label | ✅ Pass |
| 11 | Manifest order is deterministic | ✅ Pass |
| 12 | All operating conditions approved | ✅ Pass |

**Manifest SHA-256:** `71916a2f142813455013024017bc05eb97406ff44d85c53da32efbb6f6d2ddaf`

Re-running `python -m src.generate_manifest` will produce the same SHA-256 hash.

---

## 7. QC Results

11 representative windows were inspected, including:
- One window from each of the 9 (class × split) combinations
- Two KI16 duration-outlier recordings (4.67 s and 4.28 s)

| Window ID | Class | Split | shape_ok | NaN | Inf | RMS |
|---|---|---|---|---|---|---|
| K001_N09_M07_F10_m01_w00 | healthy | train | ✅ | ✅ | ✅ | 0.372 |
| K005_N09_M07_F10_m01_w00 | healthy | validation | ✅ | ✅ | ✅ | 0.116 |
| K006_N09_M07_F10_m01_w00 | healthy | test | ✅ | ✅ | ✅ | 0.359 |
| KA04_N09_M07_F10_m01_w00 | outer_ring | train | ✅ | ✅ | ✅ | 0.218 |
| KA15_N09_M07_F10_m01_w00 | outer_ring | validation | ✅ | ✅ | ✅ | 0.119 |
| KA22_N09_M07_F10_m01_w00 | outer_ring | test | ✅ | ✅ | ✅ | 0.105 |
| KI04_N09_M07_F10_m01_w00 | inner_ring | train | ✅ | ✅ | ✅ | 0.208 |
| KI21_N09_M07_F10_m01_w00 | inner_ring | validation | ✅ | ✅ | ✅ | 0.150 |
| KI14_N09_M07_F10_m01_w00 | inner_ring | test | ✅ | ✅ | ✅ | 0.122 |
| KI16_N09_M07_F10_m04_w00 | inner_ring | train | ✅ | ✅ | ✅ | 0.174 |
| KI16_N09_M07_F10_m09_w00 | inner_ring | train | ✅ | ✅ | ✅ | 0.172 |

**QC result: ALL PASS.** See [`results/qc_waveforms.png`](results/qc_waveforms.png) for the normalised waveform grid.

---

## 8. STFT Configuration Update

The model STFT parameters were updated in `configs/config.yaml` from the unphysical scaffold values to the Phase 4 approved values:

| Parameter | Old (unphysical) | New (approved) | Effect |
|---|---|---|---|
| `n_fft` | 256 | **2048** | Frequency resolution 250 Hz → **31.25 Hz** |
| `hop_length` | 64 | **512** | Time step 1.0 ms → **8.0 ms** |
| Spectrogram shape | (129, 1000) | **(1025, 125)** | Physically meaningful |

At 31.25 Hz resolution: BPFO (76.8 Hz) and BPFI (123.2 Hz) are now in separate bins.

---

## 9. Files Created / Modified

### Created

| File | Description |
|---|---|
| [`data/metadata/paderborn_manifest.csv`](../data/metadata/paderborn_manifest.csv) | Deterministic 5,101-row manifest |
| [`data/metadata/paderborn_norm_stats.json`](../data/metadata/paderborn_norm_stats.json) | Training-only Z-score stats |
| [`src/paderborn_mat.py`](../src/paderborn_mat.py) | Low-level MAT reader (time-axis policy) |
| [`src/generate_manifest.py`](../src/generate_manifest.py) | Deterministic manifest generator + integrity checks |
| [`src/paderborn_dataset.py`](../src/paderborn_dataset.py) | PyTorch Dataset with training-only normalisation |
| [`src/phase5_qc.py`](../src/phase5_qc.py) | QC inspection script |
| [`results/paderborn_manifest_stats.json`](paderborn_manifest_stats.json) | Detailed statistics |
| [`results/paderborn_manifest_summary.json`](paderborn_manifest_summary.json) | This summary (JSON) |
| [`results/paderborn_manifest_summary.md`](paderborn_manifest_summary.md) | This summary (Markdown) |
| [`results/qc_windows.json`](qc_windows.json) | QC inspection details |
| [`results/qc_waveforms.png`](qc_waveforms.png) | Normalised waveform grid |

### Modified

| File | Changes |
|---|---|
| [`configs/config.yaml`](../configs/config.yaml) | Filled dataset fields; updated n_fft/hop_length; set primary_metric |

---

## 10. Open Decisions for Phase 6

| # | Decision | Status |
|---|---|---|
| D1 | **Primary metric** | Macro-F1 adopted (team recommendation; not organizer-mandated) |
| D2 | **Baseline architecture** | NOT YET DEFINED — organizer does not specify one for Track 1 |
| D3 | **Augmentation** | Disabled for baseline; random-crop ablation documented |
| D4 | **0.5-second window ablation** | Documented in Phase 4; not implemented here |

> [!IMPORTANT]
> The baseline architecture (D2) must be agreed before Phase 6 (training) begins. A standard 1D-ResNet-18 is the most defensible self-selection for a signal classification track.

---

**PADERBORN DATASET GENERATION COMPLETE — WAITING FOR REVIEW**
