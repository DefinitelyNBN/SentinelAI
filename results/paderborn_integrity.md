# Paderborn Archive Verification and Dataset Integrity

## Scope

All 16 approved archives were found under `data/raw/Paderborn/`, listed successfully with `bsdtar`, and extracted beneath `data/raw/Paderborn/extracted/`. The RAR originals were not modified. This phase loaded and inspected all 1,280 MAT files; it did not train, split, preprocess, or alter `configs/config.yaml`.

| Bearing | Archive | MAT files | Conditions | Duration | Vibration | Vibration Hz | Current | Status |
|---|---|---:|---:|---|---|---:|---|---|
| K001 | K001.rar | 80 | 4×20 | 4.0 s | `vibration_1` | 64 kHz nominal | 2 phase currents | Pass* |
| K002 | K002.rar | 80 | 4×20 | 4.0 s | `vibration_1` | 64 kHz nominal | 2 phase currents | Pass* |
| K003 | K003.rar | 80 | 4×20 | 4.0 s | `vibration_1` | 64 kHz nominal | 2 phase currents | Pass* |
| K004 | K004.rar | 80 | 4×20 | 4.0 s | `vibration_1` | 64 kHz nominal | 2 phase currents | Pass* |
| K005 | K005.rar | 80 | 4×20 | 4.0 s | `vibration_1` | 64 kHz nominal | 2 phase currents | Pass* |
| K006 | K006.rar | 80 | 4×20 | 4.0 s | `vibration_1` | 64 kHz nominal | 2 phase currents | Pass* |
| KA04 | KA04.rar | 80 | 4×20 | 4.0 s | `vibration_1` | 64 kHz nominal | 2 phase currents | Pass* |
| KA15 | KA15.rar | 80 | 4×20 | 4.0 s | `vibration_1` | 64 kHz nominal | 2 phase currents | Pass* |
| KA16 | KA16.rar | 80 | 4×20 | 4.0 s | `vibration_1` | 64 kHz nominal | 2 phase currents | Pass* |
| KA22 | KA22.rar | 80 | 4×20 | 4.0 s | `vibration_1` | 64 kHz nominal | 2 phase currents | Pass* |
| KA30 | KA30.rar | 80 | 4×20 | 4.0 s | `vibration_1` | 64 kHz nominal | 2 phase currents | Pass* |
| KI04 | KI04.rar | 80 | 4×20 | 4.0 s | `vibration_1` | 64 kHz nominal | 2 phase currents | Pass* |
| KI14 | KI14.rar | 80 | 4×20 | 4.0 s | `vibration_1` | 64 kHz nominal | 2 phase currents | Pass* |
| KI16 | KI16.rar | 80 | 4×20 | 4.0–4.672 s | `vibration_1` | 64 kHz nominal | 2 phase currents | Pass; review outliers |
| KI18 | KI18.rar | 80 | 4×20 | 4.0 s | `vibration_1` | 64 kHz nominal | 2 phase currents | Pass* |
| KI21 | KI21.rar | 80 | 4×20 | 4.0 s | `vibration_1` | 64 kHz nominal | 2 phase currents | Pass* |

\*Pass means archive/MAT naming/channel integrity passes; stored sample counts remain variable and require an explicit future input-length policy.

## Archive and recording integrity

- All 16 expected archives exist, are readable, and contain 80 MAT recordings each.
- Every bearing has all four expected settings: `N15_M07_F10`, `N09_M07_F10`, `N15_M01_F10`, and `N15_M07_F04`.
- Every bearing has exactly measurements 1–20 for every condition: 80 recordings per bearing and 1,280 total.
- Every MAT file loaded successfully. No byte-identical duplicate MAT recording was found across the full extracted set.
- Filename identity is consistent: `RAR → extracted bearing directory → Nxx_Mxx_Fxx_<bearing>_<measurement>.mat`.

## Actual MAT structure and channel summary

Every inspected file has one top-level MATLAB struct named exactly after the MAT filename stem. Its fields are `Info`, `X`, `Y`, and `Description`. `Info` carries revision and measurement ID; `Description` contains `General`, `Recording`, and `Measurement`; `X` contains the time axes/raster records; `Y` carries the named signals. This schema, including channel names and raster labels, is consistent across all 1,280 files.

| Channel | Raster | Documented nominal rate | Stored count observed | Purpose |
|---|---|---:|---:|---|
| `vibration_1` | `HostService` | 64 kHz | 255,996–299,038 | Primary 1-D Track 1 candidate |
| `phase_current_1`, `phase_current_2` | `HostService` | 64 kHz | same count as vibration within each MAT | Available, not selected for primary input |
| `force`, `speed`, `torque` | `Mech_4kHz` | 4 kHz | about 16,000–18,691 | Operating-condition/support channels |
| `temp_2_bearing_module` | `Temp_1Hz` | 1 Hz | 4–5 | Support channel |

The documentation specifies 64 kHz for vibration/current, but the stored HostService time axes yield effective rates from approximately 63,998.9 to 72,961.5 Hz (median about 64,000.1 Hz). This means 64 kHz is the documented nominal rate, not a safe assumption that every stored vector has 256,001 elements. Vibration and both phase-current vectors have matching stored lengths within every file.

## Duration and shape findings

Most `Description.Measurement.Length` values are four seconds (with harmless floating-point representation near 4.0). Two KI16 recordings explicitly report longer durations:

- `KI16/N15_M01_F10_KI16_4.mat`: 4.672453600733628 s, 299,038 vibration samples
- `KI16/N15_M07_F10_KI16_9.mat`: 4.283332791552322 s, 274,134 vibration samples

Variable sample lengths are an implementation constraint, not evidence that the archives are corrupt. Before later dataset implementation, the team must explicitly review and document a time-axis-aware fixed-length crop/pad or resampling policy. It must be applied without using test-derived statistics.

## Track 1 compatibility and leakage identity

`vibration_1` is a valid 1-D signal in every MAT file, making the selected archives compatible with Track 1. The critical identity hierarchy is:

1. **Bearing ID** — the independent experimental unit, e.g. `KI16`.
2. **MAT recording** — one measurement, e.g. `N15_M01_F10_KI16_4.mat`.
3. **Signal window** — a derived portion of that MAT recording; it is not independent.

Future window metadata must preserve archive, bearing ID, relative MAT path, condition, measurement number, and sample/time offsets. All recordings and windows for a bearing must remain in a single partition; random window splitting is forbidden.

## Readiness

The selected data are **conditionally ready for dataset implementation after review**. No missing MAT files, corrupt load failures, missing vibration channels, missing conditions, naming failures, or duplicate recordings were found. Before proceeding, approve the variable-length/outlier policy and obtain the organizer’s fixed-split and primary-metric requirements.

**PADERBORN INTEGRITY CHECK COMPLETE — WAITING FOR REVIEW**
