# Paderborn Dataset Taxonomy Reconnaissance

## Scope

This is documentation reconnaissance only. It uses the official [Data Sets and Download](https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/data-sets-and-download) page, [operating-condition page](https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/operating-conditions), official [2016 benchmark paper](https://mb.uni-paderborn.de/fileadmin-mb/kat/PDF/Veroeffentlichungen/20160703_PHME16_CM_bearing.pdf), official [archive index](https://groups.uni-paderborn.de/kat/BearingDataCenter/), and the local K001 documentation. No archive was downloaded; no split, training, preprocessing, or config change was made.

The official DataCenter states that the dataset has 32 bearing experiments: 6 healthy references, 12 artificially damaged bearings, and 14 bearings with real damage from accelerated lifetime tests. All are type-6203 ball-bearing experiments. The index confirms the archive filenames below.

## Complete 32-bearing table

### Healthy references (6)

| Code | Status / provenance | Location | Damage type / subtype | Source | Archive |
|---|---|---|---|---|---|
| K001 | Healthy / undamaged reference | — | — | Table 7 | K001.rar |
| K002 | Healthy / undamaged reference | — | — | Table 7 | K002.rar |
| K003 | Healthy / undamaged reference | — | — | Table 7 | K003.rar |
| K004 | Healthy / undamaged reference | — | — | Table 7 | K004.rar |
| K005 | Healthy / undamaged reference | — | — | Table 7 | K005.rar |
| K006 | Healthy / undamaged reference | — | — | Table 7 | K006.rar |

### Artificially damaged bearings (12)

Table 4 identifies component, extent, and machining method. It says every artificial damage is a single-point damage, with no repetition or combination; it does not state an ISO main damage mode/submode in this table.

| Code | Location | Type | Subtype / method | Source | Archive |
|---|---|---|---|---|---|
| KA01 | Outer ring | Artificial single point | EDM, extent 1 | Table 4 | KA01.rar |
| KA03 | Outer ring | Artificial single point | Electric engraver, extent 2 | Table 4 | KA03.rar |
| KA05 | Outer ring | Artificial single point | Electric engraver, extent 1 | Table 4 | KA05.rar |
| KA06 | Outer ring | Artificial single point | Electric engraver, extent 2 | Table 4 | KA06.rar |
| KA07 | Outer ring | Artificial single point | Drilling, extent 1 | Table 4 | KA07.rar |
| KA08 | Outer ring | Artificial single point | Drilling, extent 2 | Table 4 | KA08.rar |
| KA09 | Outer ring | Artificial single point | Drilling, extent 2 | Table 4 | KA09.rar |
| KI01 | Inner ring | Artificial single point | EDM, extent 1 | Table 4 | KI01.rar |
| KI03 | Inner ring | Artificial single point | Electric engraver, extent 1 | Table 4; local readme notes profile correction | KI03.rar |
| KI05 | Inner ring | Artificial single point | Electric engraver, extent 1 | Table 4 | KI05.rar |
| KI07 | Inner ring | Artificial single point | Electric engraver, extent 2 | Table 4 | KI07.rar |
| KI08 | Inner ring | Artificial single point | Electric engraver, extent 2 | Table 4 | KI08.rar |

### Real damage from accelerated lifetime tests (14)

| Code | Location | Main mode | Symptom / detailed subtype | Source | Archive |
|---|---|---|---|---|---|
| KA04 | Outer ring | Fatigue | Pitting; single, extent 1, single point | Table 5 | KA04.rar |
| KA15 | Outer ring | Plastic deformation | Indentations; single, extent 1, single point | Table 5 | KA15.rar |
| KA16 | Outer ring | Fatigue | Pitting; repetitive/random, extent 2, single point | Table 5 | KA16.rar |
| KA22 | Outer ring | Fatigue | Pitting; single, extent 1, single point | Table 5 | KA22.rar |
| KA30 | Outer ring | Plastic deformation | Indentations; repetitive/random, extent 1, distributed | Table 5 | KA30.rar |
| KB23 | Inner ring (+ outer ring) | Fatigue | Pitting; multiple/random, extent 2, single point | Table 5 | KB23.rar |
| KB24 | Inner ring (+ outer ring) | Fatigue | Pitting; multiple/no repetition, extent 3, distributed | Table 5 | KB24.rar |
| KB27 | Outer + inner ring | Plastic deformation | Indentations; multiple/random, extent 1, distributed | Table 5 | KB27.rar |
| KI04 | Inner ring | Fatigue | Pitting; multiple/no repetition, extent 1, single point | Table 5 | KI04.rar |
| KI14 | Inner ring | Fatigue | Pitting; multiple/no repetition, extent 1, single point | Table 5 | KI14.rar |
| KI16 | Inner ring | Fatigue | Pitting; single, extent 3, single point | Table 5 | KI16.rar |
| KI17 | Inner ring | Fatigue | Pitting; repetitive/random, extent 1, single point | Table 5 | KI17.rar |
| KI18 | Inner ring | Fatigue | Pitting; single, extent 2, single point | Table 5 | KI18.rar |
| KI21 | Inner ring | Fatigue | Pitting; single, extent 1, single point | Table 5 | KI21.rar |

The code letters alone must not be used as labels: notably, `KA04`, `KA15`, `KA16`, `KA22`, and `KA30` are real-damage entries despite their `KA` prefix.

## Operating-condition taxonomy

| Code | Speed | Load torque | Radial force |
|---|---:|---:|---:|
| N15_M07_F10 | 1500 rpm | 0.7 Nm | 1000 N |
| N09_M07_F10 | 900 rpm | 0.7 Nm | 1000 N |
| N15_M01_F10 | 1500 rpm | 0.1 Nm | 1000 N |
| N15_M07_F04 | 1500 rpm | 0.7 Nm | 400 N |

The official documentation specifies 20 four-second measurements at each setting and roughly 45–50 °C. The local K001 inspection independently confirms the 4-second duration, 64 kHz vibration/current sampling, and 4 kHz mechanical signals.

## Classification formulations

| Formulation | Independent bearings/class | Meaning and risk | SentinelAI recommendation |
|---|---|---|---|
| Healthy vs damaged | Healthy 6; damaged 26 | Broad detection; damaged label is heterogeneous and imbalanced. Bearing/window leakage can inflate results. | Valid baseline, not primary. |
| Healthy vs artificial vs real | 6 / 12 / 14 | Measures provenance as much as condition; machining may form a shortcut. | Do not use as primary. |
| Healthy vs real outer-ring vs real inner-ring | 6 / 5 / 5 | Clear physical classes; real damage only. Exclude 3 combined-location real bearings. Inner-ring detection is meaningfully difficult due to rotational modulation. | **Primary recommendation.** |
| Real fatigue vs plastic deformation | 11 / 3 | Official main-mode classification, but plastic-deformation group is too small and location-confounded for robust 3-way bearing splits. | Not primary. |
| Operating-condition class | Up to 32 per condition if all archives include all settings | Test-rig condition rather than bearing condition; easy/confounded. | Nuisance robustness factor only. |

## Recommended Track 1 task

**Classify a 1-D vibration segment into one of three bearing-condition classes: healthy, real outer-ring damage, or real inner-ring damage.**

Use these bearing identities only:

- Healthy: `K001`, `K002`, `K003`, `K004`, `K005`, `K006`
- Real outer-ring: `KA04`, `KA15`, `KA16`, `KA22`, `KA30`
- Real inner-ring: `KI04`, `KI14`, `KI16`, `KI18`, `KI21`

This gives 16 independent bearing IDs, uses real lifetime-test damage rather than artificial-machining provenance, and has direct industrial meaning. It creates a credible temporal/frequency hypothesis: transient impacts and their time-frequency signatures may be complementary for healthy-versus-location classification.

### Leakage-safe grouping requirement

Every MAT file and every derived window from one bearing ID must remain in one partition. The five-ID damage classes can support a 3/1/1 train/validation/test allocation per class; healthy can supply 4/1/1. That is only a proposed structure—not a split—and must be reviewed against the organizer’s fixed-split requirement before implementation. Do not randomly distribute windows, recordings, or operating conditions of a bearing across partitions.

### Archives recommended for later download (not downloaded)

`K002.rar`, `K003.rar`, `K004.rar`, `K005.rar`, `K006.rar`, `KA04.rar`, `KA15.rar`, `KA16.rar`, `KA22.rar`, `KA30.rar`, `KI04.rar`, `KI14.rar`, `KI16.rar`, `KI18.rar`, `KI21.rar`. `K001.rar` is already local.

### Input recommendation

**Vibration only** for the primary task. It is the established signal for bearing diagnosis, and the official paper reports stronger vibration than motor-current classification. This also keeps SentinelAI’s contribution focused: temporal Conv1D and STFT/Conv2D are two representations of the same Track 1 signal, not an untested sensor-fusion claim. Motor current can later be a separately declared low-cost-sensing robustness experiment.

### Metric recommendation

**Recommended metric: macro-F1**, because classes have 6/5/5 bearing IDs and all fault-location classes should contribute equally. This is a recommendation only: the supplied hackathon DOCX still provides no mandatory Track 1 primary metric, so it cannot be represented as organizer-mandated.

**PADERBORN TAXONOMY COMPLETE — WAITING FOR REVIEW**
