# SentinelAI — Phase 4: Dataset Design and Preprocessing Specification

## Scope

This document specifies the complete experimental protocol for converting the Paderborn University Bearing Dataset into training-ready input for SentinelAI. It covers window design, normalization, split design, STFT analysis, window-count estimates, and organizer-constraint verification.

**What this document does NOT do:**
- It does not create the final manifest, splits, or processed data.
- It does not modify `configs/config.yaml`.
- It does not train any model.

---

## A. Verified Facts from the Dataset

### A1. Recording inventory

| Fact | Value |
|---|---|
| Archives extracted | 16 |
| MAT recordings total | 1,280 |
| Recordings per bearing | 80 (4 conditions × 20 measurements) |
| Operating conditions | 4 |
| Primary signal | `vibration_1` |
| Nominal sampling rate | 64 kHz |
| Observed effective rate (HostService dt = 1/64 000.02 s) | 63,998.9 – 64,000.9 Hz (ordinary files) |
| Documented recording duration | 4 seconds |
| Actual vibration sample count range | 255,996 – 299,038 |
| Duration outliers (KI16 only) | 4.672 s (299,038 samples) and 4.283 s (274,134 samples) |
| Load failures | 0 |
| Duplicate recordings | 0 |

### A2. Approved bearings

| Class | Bearings | Count |
|---|---|---|
| **Healthy** | K001, K002, K003, K004, K005, K006 | 6 |
| **Real outer-ring damage** | KA04, KA15, KA16, KA22, KA30 | 5 |
| **Real inner-ring damage** | KI04, KI14, KI16, KI18, KI21 | 5 |
| **Total** | | **16** |

Excluded: all artificially damaged bearings; KB23, KB24, KB27 (combined damage).

### A3. Operating conditions

| Code | Speed | Torque | Radial Force |
|---|---:|---:|---:|
| N15_M07_F10 | 1500 rpm | 0.7 Nm | 1000 N |
| N09_M07_F10 | 900 rpm | 0.7 Nm | 1000 N |
| N15_M01_F10 | 1500 rpm | 0.1 Nm | 1000 N |
| N15_M07_F04 | 1500 rpm | 0.7 Nm | 400 N |

All 16 bearings have all 4 conditions, all 20 measurements per condition.

### A4. Bearing kinematic fault frequencies (IBU 6203)

Parameters: Z = 8 balls, d = 6.75 mm, D = 29.05 mm, α = 0°.

| Frequency | Order (×shaft) | At 1500 rpm (25 Hz shaft) | At 900 rpm (15 Hz shaft) |
|---|---:|---:|---:|
| BPFO (Ball Pass, Outer) | 3.071 | **76.8 Hz** (period 13.0 ms) | **46.1 Hz** (period 21.7 ms) |
| BPFI (Ball Pass, Inner) | 4.929 | **123.2 Hz** (period 8.1 ms) | **73.9 Hz** (period 13.5 ms) |
| BSF (Ball Spin) | 2.036 | 50.9 Hz | 30.5 Hz |
| FTF (Cage/Fundamental Train) | 0.384 | 9.6 Hz (period 104 ms) | 5.8 Hz (period 173 ms) |

These frequencies are the direct physical targets for both temporal and STFT branch design.

---

## B. Organizer-Mandated Requirements

| Requirement | Status |
|---|---|
| Seed = 42 | ✅ **Mandatory — confirmed in document** |
| Fixed splits (train only, test untouched) | ✅ **Mandatory — confirmed in document** |
| Beat mandatory baseline, report comparison | ✅ **Mandatory — confirmed in document** |
| One-command reproducible run | ✅ **Mandatory — confirmed in document** |
| Track 1 dataset / sub-track | ❌ **NOT specified by organizer** |
| Track 1 fixed split (file) | ❌ **NOT specified by organizer** |
| Track 1 mandatory baseline | ❌ **NOT specified by organizer** |
| Track 1 primary metric | ❌ **NOT specified by organizer** |

> [!IMPORTANT]
> The hackathon document specifies mandatory rules (seed, fixed splits, baseline comparison) but leaves Track 1's dataset, split file, baseline architecture, and primary metric entirely unspecified. Unlike Tracks 2–5 which enumerate concrete baselines and metrics, Track 1 only states: *"Classify a 1-D or spectrogram signal into its category."*
> **These four items must be established, documented, and defended by the team, not invented or inferred from other tracks.**

---

## C. Experimental Design Recommendations

### C1. Signal Window Design

**Recommended window duration: 1.0 second (64,000 samples)**

#### Justification

| Criterion | 0.5 s window | **1.0 s window** (recommended) |
|---|---|---|
| Frequency resolution | 2.0 Hz | **1.0 Hz** |
| BPFO/BPFI separability | ✓ (separated by 30+ Hz) | ✓ (better margin) |
| Cage (FTF) cycles at 1500 rpm | 4.8 | **9.6** |
| Cage cycles at 900 rpm | 2.9 | **5.8** |
| Shaft revolutions at 1500 rpm | 12.5 | **25.0** |
| Windows per recording | 7 | **3** |
| Leakage risk from over-windowing | Moderate | **Low** |
| STFT frame count (n_fft=2048) | 62 | **125** |
| Defensibility | Adequate | **Strong** |

**Key physical argument:** Cage-related diagnostic information requires the FTF frequency to manifest in at least 5 complete cycles. At 900 rpm, the cage period is 173 ms, so a 1.0 s window captures 5.8 FTF cycles, while a 0.5 s window captures only 2.9 — marginally sufficient but weaker. At 1500 rpm, a 1.0 s window captures 9.6 cage cycles and 25 shaft revolutions, providing strong statistical confidence in the fault pattern.

The 1.0 s window is recommended as the primary choice. The 0.5 s window is retained as an ablation candidate.

### C2. Time-Axis Policy

**Method: Deterministic sample-index windowing (no resampling).**

The HostService clock in the Paderborn acquisition system shows mean Δt = 15.625 µs with hardware-level jitter. The effective sampling rate is uniformly 64,000 Hz (±2 Hz deviation) for normal files. The KI16 outlier files have the same clock rate but simply recorded longer.

**Policy:**
1. Load the raw `vibration_1` array directly from the MAT file.
2. Extract windows by sample index only: window `k` = samples `[k * L : (k+1) * L]` where `L = 64,000`.
3. Discard any trailing samples that do not fill a complete window.
4. Do not resample, stretch, or pad. The jitter in HostService timestamps is OS scheduling jitter, not a true sample-rate deviation.

This approach is simpler, avoids interpolation-induced spectral artefacts in a 64 kHz signal, and produces exactly deterministic arrays.

### C3. Windowing and Overlap

**Recommended: Zero overlap in all partitions.**

| Parameter | Value |
|---|---|
| Window length L | 64,000 samples (1.0 s) |
| Stride | 64,000 samples (= L, non-overlapping) |
| Overlap | 0% |

**Rationale:** Overlapping windows from the same recording are nearly identical signals with shared noise, bearing-ring resonance, and measurement artifact. Including them in evaluation would falsely inflate accuracy by testing the same physical signal multiple times. The experimental unit is the **bearing**, not the window — this constraint forbids any overlap policy that could make adjacent windows appear to be independent observations.

For training, non-overlapping windows from a 4-second recording give 3 windows, representing 3 disjoint temporal segments. This is conservative but rigorous.

> [!NOTE]
> A future augmentation-style "random crop" (randomly selecting start offset within each file at training time) is physically sound and can replace fixed windowing during training only — not during evaluation.

### C4. Normalization

**Strategy: Training-only global Z-score normalization**

```
x_normalized = (x_raw - μ_train) / σ_train
```

- `μ_train` and `σ_train` are computed once over all training-split windows.
- All validation and test windows use the training-derived statistics without modification.
- An optional per-window DC-offset removal step (subtract the window mean) prior to global scaling eliminates low-frequency accelerometer bias that may vary across operating conditions.

**Leakage rule:** No statistic from validation or test data may appear in any normalization calculation.

### C5. Augmentation

**Recommended: Disabled for baseline training.**

Augmentation should not be enabled until a clean baseline result is established. Three physically defensible methods are documented for later ablation:

| Method | Implementation | Physical Justification |
|---|---|---|
| Random phase shift | Random start offset within the MAT file (up to ±0.5 s) | Rotating machinery is stationary-process; phase of fault impacts is arbitrary |
| Amplitude jitter | Uniform scaling ∈ [0.95, 1.05] | Sensor sensitivity and mounting torque variation across installations |
| Additive Gaussian noise | σ ≈ 0.01 (~30 dB SNR) | Background plant acoustic and electrical noise |

**Forbidden augmentations:**
- Time-reversal: violates causal impact-decay physics (fault impact has sharp onset, slow ring-down).
- Arbitrary pitch/speed scaling: breaks the kinematic link between known RPM and characteristic frequencies.
- Cross-class MixUp: creates physically impossible hybrid bearing states.

---

### C6. Candidate Bearing-Disjoint Split Design

> [!IMPORTANT]
> This is a CANDIDATE split design for review. It must not be written to disk or finalized until approved.

**Proposed allocation (10 / 3 / 3 bearings):**

| Partition | Healthy | Outer-Ring | Inner-Ring | Total |
|---|---|---|---|---|
| **Train** | K001, K002, K003, K004 | KA04, KA16, KA30 | KI04, KI16, KI18 | 10 |
| **Validation** | K005 | KA15 | KI21 | 3 |
| **Test** | K006 | KA22 | KI14 | 3 |

**Class balance in each partition:**
- Train: 4 healthy, 3 outer, 3 inner (slight healthy enrichment, acceptable)
- Validation: **1:1:1 exact balance**
- Test: **1:1:1 exact balance**

**Damage-mode coverage in train:**

| Class | Bearings | Damage modes covered |
|---|---|---|
| Outer (train) | KA04, KA16, KA30 | Fatigue pitting (extent 1, 2) + Plastic indentation (extent 1) |
| Inner (train) | KI04, KI16, KI18 | Fatigue pitting (extent 1, 2, 3) — full severity range |

**Validation and test bearings:**

| Bearing | Class | Mode | Reason for eval placement |
|---|---|---|---|
| KA15 | Outer | Plastic indentation, extent 1 | Provides mode diversity in validation |
| KA22 | Outer | Fatigue pitting, extent 1 | Unseen bearing identity |
| KI21 | Inner | Fatigue pitting, extent 1 | Similar severity to KI04, true hold-out |
| KI14 | Inner | Fatigue pitting, extent 1 | Unseen bearing, same mode as KI21 |

**Why KI16 is in training:**
KI16 contains two anomalously long recordings (4.67 s, 4.28 s). Placing an outlier-containing bearing in the training set avoids contaminating evaluation with edge cases that windowing policy must handle, while training the model on a broader range of recording lengths.

**Operating condition distribution:** All 4 conditions are equally distributed within each bearing; since all bearings have all 4 conditions, every partition automatically receives all 4 operating conditions without any special balancing action.

### C7. Operating Conditions Policy

**All four conditions are retained in all partitions.** No condition is held out from the primary evaluation.

A secondary **robustness ablation** is recommended: train on conditions {N15_M07_F10, N15_M01_F10, N15_M07_F04} and test generalization on N09_M07_F10 (900 rpm, the only non-1500 rpm condition). This constitutes a legitimate hold-out domain shift experiment.

Operating condition is **not** a classification target. It should be stored in the manifest as metadata only.

### C8. Window Count and Computational Estimate

**Under the recommended 1.0 s / zero-overlap policy:**

| Partition | Bearings | Recordings | Windows/Recording | Windows Total |
|---|---:|---:|---:|---:|
| Train | 10 | 800 | 3 | **2,400** |
| Validation | 3 | 240 | 3 | **720** |
| Test | 3 | 240 | 3 | **720** |
| **Total** | 16 | 1,280 | 3 | **3,840** |

| Class | Bearings | Total Windows |
|---|---:|---:|
| Healthy | 6 (4+1+1) | 1,440 |
| Outer-ring | 5 (3+1+1) | 1,200 |
| Inner-ring | 5 (3+1+1) | 1,200 |

**Storage:** Each window = 64,000 × 4 bytes (float32) = 256 KB. Total ≈ 3,840 × 256 KB ≈ **983 MB** for pre-extracted `.npy` files. This is well within local disk budgets.

**Training compute:** At batch size 32, 2,400 windows → 75 batches/epoch. With 30 epochs and early stopping (patience 6), maximum 30 × 75 = 2,250 gradient steps — trivially fast even on CPU.

> [!NOTE]
> 0.5 s windows would produce ~8,960 total windows (5,600 / 1,680 / 1,680 split). This increases storage to ~1.1 GB and batches/epoch to ~175, still computationally light. The cost difference is negligible; the 1.0 s choice is preferred for physical/diagnostic reasons, not compute.

### C9. STFT Compatibility Assessment

**Current scaffold parameters: `n_fft = 256`, `hop_length = 64`**

These parameters are **not appropriate** for 64 kHz industrial vibration data and must be updated before training.

#### Problem analysis

| Parameter | Current | Consequence |
|---|---|---|
| n_fft = 256 | Frequency resolution = 64,000/256 = **250 Hz** | BPFO (76.8 Hz), BPFI (123.2 Hz), shaft (25 Hz), cage (9.6 Hz) all collapse into bin 0. The spectrogram is physically blind to all fault frequencies. |
| hop_length = 64 | Time step = 64/64,000 = **1.0 ms** | Generates 1,000 frames/second. For a 1.0 s window: 1,000 time frames × 129 freq bins. This is large, slow, and provides false temporal precision while lacking frequency precision. |

#### Recommended STFT parameters (for 1.0 s window)

| Parameter | Recommended | Justification |
|---|---|---|
| `n_fft = 2048` | Frequency resolution = 64,000/2,048 = **31.25 Hz** | Separates BPFO (76.8 Hz) from BPFI (123.2 Hz) with a 46.4 Hz gap. Resolves shaft harmonics. |
| `hop_length = 512` | Time step = 512/64,000 = **8.0 ms** | Captures fault impulse load zones (~10-15 ms width at 1500 rpm). Spectrogram shape: 1,025 freq × 125 time. |

Spectrogram output shape for 1.0 s window, n_fft=2048, hop=512: **(1025, 125)**.
After 2 MaxPool2d(2) layers in FrequencyEncoder: **(256, 31)** → pooled to embedding via `.mean((-1,-2))`.

#### Recommended STFT parameters (for 0.5 s ablation)

| Parameter | Recommended |
|---|---|
| `n_fft = 1024` | Frequency resolution = 62.5 Hz |
| `hop_length = 256` | Time step = 4.0 ms; Spectrogram shape: (513, 125) |

> [!WARNING]
> The current `n_fft=256` / `hop_length=64` in `configs/config.yaml` will produce a spectrogram that is physically meaningless for 64 kHz bearing vibration data. **`configs/config.yaml` must be updated before any training run, but NOT in this phase.** Update is deferred to Phase 5 (manifest + config update phase) pending approval of the window duration.

### C10. Manifest Schema Design

The manifest will be a CSV file where each row describes one extracted window.

| Column | Type | Description |
|---|---|---|
| `window_id` | string | Globally unique, e.g. `KI16_N15_M01_F10_m04_w00` |
| `path` | string | Relative path to `.npy` file (relative to `data/`) |
| `label` | string | `healthy` / `outer_ring` / `inner_ring` |
| `label_id` | int | 0 / 1 / 2 (alphabetically sorted) |
| `split` | string | `train` / `validation` / `test` |
| `bearing_id` | string | e.g. `KI16` |
| `archive` | string | e.g. `KI16.rar` |
| `mat_path` | string | Relative path to source MAT file |
| `operating_condition` | string | e.g. `N15_M07_F10` |
| `speed_rpm` | int | 1500 or 900 |
| `load_torque_nm` | float | 0.7 or 0.1 |
| `radial_force_n` | int | 1000 or 400 |
| `measurement_id` | int | 1–20 |
| `window_index` | int | 0-based index within the MAT file |
| `start_sample` | int | Inclusive start sample index |
| `end_sample` | int | Exclusive end sample index |
| `start_time_s` | float | Nominal start time in seconds |
| `end_time_s` | float | Nominal end time in seconds |
| `sampling_rate_hz` | int | Always 64000 (nominal) |
| `damage_location` | string | `none` / `outer_ring` / `inner_ring` |
| `damage_mode` | string | `none` / `fatigue` / `plastic_deformation` |
| `damage_extent` | int | 0 (healthy) / 1 / 2 / 3 |

The `path` column is what `ManifestSignalDataset` uses to load `.npy` arrays. All other columns are provenance metadata. This schema is fully compatible with the existing `dataset.py`.

---

## D. Open Decisions Requiring Review

The following items require explicit approval before Phase 5 can begin:

| # | Decision | Options | Impact |
|---|---|---|---|
| **D1** | Window duration | **1.0 s** (recommended) vs 0.5 s | Determines `signal_length`, STFT config, storage, and number of windows |
| **D2** | Bearing-disjoint partition | **Approve proposed 10/3/3 split** or modify assignments | Locks which bearings are test/val — cannot be changed after training starts |
| **D3** | Primary metric | **Macro-F1** (recommended) vs macro-AUROC vs accuracy | Must be chosen before defining the "beat the baseline" criterion |
| **D4** | Baseline architecture | Suggest: **1D-ResNet-18** (temporal only) vs simple 1D-CNN | Organizer does not specify; team must establish and beat it |
| **D5** | STFT parameters | **n_fft=2048, hop=512** (recommended) vs alternatives | Cannot be changed after STFT branch training |
| **D6** | Random-crop training augmentation | Enable or keep fixed windowing | Minor accuracy vs rigor trade-off |

> [!CAUTION]
> Decisions D2 (split) and D4 (baseline) must be finalized and committed before any training run. Changing the test split or baseline after seeing training results is forbidden under the organizer rules and academic integrity policy.

---

## Summary

| Item | Recommendation |
|---|---|
| **Window length** | 1.0 s = 64,000 samples |
| **Stride / overlap** | 64,000 samples / 0% |
| **Time-axis policy** | Sample-index windowing; no resampling |
| **Normalization** | Global Z-score from training set only |
| **Augmentation** | Disabled for baseline; 3 candidates for ablation |
| **Bearing split** | 10 train / 3 val / 3 test (exact 1:1:1 balance in val and test) |
| **Operating conditions** | All 4 in all partitions; no condition is a target |
| **Windows total** | ~3,840 (2,400 / 720 / 720) |
| **Primary metric** | Macro-F1 (recommended; not yet organizer-mandated) |
| **STFT (n_fft / hop)** | Recommend changing to 2048 / 512 (from unphysical 256 / 64) |
| **Config changes** | None in this phase — deferred to Phase 5 pending approval |
| **Manifest** | Not generated — deferred to Phase 5 pending approval |

---

**PADERBORN DATASET DESIGN COMPLETE — WAITING FOR REVIEW**
