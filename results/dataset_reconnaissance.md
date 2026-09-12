# Paderborn K001 Dataset Reconnaissance

## Scope and integrity

This report is based only on `data/raw/Paderborn/K001.rar`, `data/raw/Paderborn/readme_versions.txt`, and the two PDFs embedded in the archive. The archive was listed with `bsdtar` and selected files were extracted only to a temporary directory under `/private/tmp` for read-only inspection. The original archive was not changed. No model, split, processed dataset, labels, metric, or configuration was created.

## A. Dataset structure

`K001.rar` contains one `K001/` directory, 80 MATLAB MAT files, and two PDFs: `K001.pdf` and `measuring_log_K001.pdf`.

| File series | Recordings |
| --- | ---: |
| `N15_M07_F10_K001_1.mat` through `_20.mat` | 20 |
| `N09_M07_F10_K001_1.mat` through `_20.mat` | 20 |
| `N15_M01_F10_K001_1.mat` through `_20.mat` | 20 |
| `N15_M07_F04_K001_1.mat` through `_20.mat` | 20 |

Each inspected MAT file has exactly one top-level 1×1 MATLAB struct named after its filename stem. It contains `Info`, `X`, `Y`, and `Description`. `Info` includes revision and measurement ID. `Description.Measurement` reports a 4-second measurement (`Length=4`; time limit 4.0 seconds).

## B. Filename convention

The observed form is `Nxx_Mxx_Fxx_<bearing_code>_<measurement_number>.mat`. The local measuring log explicitly calls its four combinations load variants: K0=`N15_M07_F10`, K1=`N09_M07_F10`, K2=`N15_M01_F10`, K3=`N15_M07_F04`. The final number runs 1–20 for each combination. `K001` is explicitly called the bearing code in the profile PDF.

The local documentation does **not** define a formal filename legend for `N`, `M`, or `F`. The recorded channels corroborate that N15 is roughly 1500 min⁻¹ and N09 roughly 900 min⁻¹; F10 recordings are roughly 1,000 N and F04 roughly 400 N. M07 recordings have roughly 1.20–1.26 recorded torque while M01 is roughly 0.64, but the exact intended meaning, units, and setpoint mapping of M are not locally documented. Likewise, the local material does not define the semantics of the leading `K` in code variants such as the example `KA01`; it only establishes that the complete string is a bearing code.

## C. Signal structure

The `X` time axes have rasters `Mech_4kHz` (16,001 samples), `HostService` (256,001), and `Temp_1Hz` (5). The `Y` signals are:

| Channel | Raster | Samples |
| --- | --- | ---: |
| `force` | Mech_4kHz | 16,001 |
| `phase_current_1`, `phase_current_2` | HostService | 256,001 each |
| `speed`, `torque` | Mech_4kHz | 16,001 each |
| `temp_2_bearing_module` | Temp_1Hz | 5 |
| `vibration_1` | HostService | 256,001 |

The measuring log explicitly specifies 64 kHz for motor current and vibration, 4 kHz for mechanical parameters (force, torque, speed), and 1 Hz for temperature. Thus `vibration_1` is the directly available vibration signal; current and other sensor signals are also available. No sensor selection has been made.

## D. Bearing and damage metadata

K001 is documented as an IBU 6203 deep-groove ball bearing, with 29.05 mm pitch-circle diameter, eight 6.75 mm rolling elements, nominal pressure angle 0°, and reported lifetime >50 h. The profile gives a broad load range of 1,000–3,000 N and rotational speed range of 1,500–2,000 min⁻¹.

Its health/damage status is **unknown** from the local files. Although the profile is titled “Profile of rolling bearing damage,” it provides no explicit healthy/damaged label, no artificial-versus-real provenance, and no location, type, subtype, or geometry. The standalone `readme_versions.txt` only notes a correction for KI03, whose archive is not present. Therefore no condition taxonomy can be asserted and no categories have been collapsed into classes.

## E. Operating conditions

There are four locally documented N/M/F variants, each with 20 recordings. The measuring log also records a 37-minute preheat at `F=2000 N; n=2000 1/min`, reaching 45°C. It documents the current, force, speed, torque, vibration, and temperature measurement hardware. This does not establish the complete semantics of every filename token.

## F. Potential labels (not selected)

Future labels could be bearing identity, an official condition label, damage provenance, damage location/type/subtype, or operating condition. The present data/documentation does not support choosing any as the Track 1 target.

## G. Recommended grouping strategy

All windows derived from one MAT recording must remain together. For a claim of generalization to unseen physical bearings, the primary split group must be the full bearing identifier: all recordings, conditions, and windows from that bearing go in exactly one partition. Retain measurement number and N/M/F condition as metadata. K001 alone cannot be split at bearing level because it supplies only one bearing identity.

## H. Leakage risks

Random windows from the same four-second file share nearly identical waveform, noise, acquisition, operating-condition, and bearing-specific structure. Splitting them across train/validation/test would leak source identity and inflate scores. Splitting only by window is invalid for bearing generalization. Operating-condition generalization could be evaluated later by holding out documented N/M/F conditions while also preventing bearing overlap, but cannot be evaluated with K001 alone.

## I. Additional material needed

We need additional bearing archives across each intended official category, including their profile PDFs and measuring logs; the official Paderborn naming/damage-taxonomy documentation; and the organizer-approved Track 1 target, fixed split protocol (or sufficient independent bearings to construct one), mandatory baseline, and primary metric.

## J. Remaining unknowns

K001’s condition/provenance; all damage labels; the N/M/F legend; the official task label scheme; the exact split protocol; baseline; metric; and required sensor representation remain unknown. No result can be reported yet.

**RECONNAISSANCE COMPLETE — WAITING FOR REVIEW**
