# Singlish classification pilot test

Date: 28 October 2026

Frozen Whisper with all 512 embedding values had the highest observed pilot balanced accuracy (98.25%, 7 errors). Feature fusion and weighted voting each made 8 errors. The added x-vector and ECAPA-TDNN models made 14 and 13 errors; the small i-vector model made 56.

## Data and method

- **Data:** 2,000 clips: 1,000 Singapore clips from MNSC and 1,000 non-Singapore clips from Common Voice; 883 speaker groups.
- **Split:** 1,280 training, 320 validation and 400 test clips. The test contains 200 clips per class from 176 speakers. No speakers cross partitions.
- **Audio:** shared mono 16 kHz preparation, 30 dB silence trim, peak normalization, a six-second cap and one-second minimum padding.
- **Selection:** validation data select classifier settings; selected classifier pipelines refit on the 1,600 development clips. The test does not select settings.
- **New models:** the i-vector extractor learns only from training clips; x-vector and ECAPA-TDNN use frozen VoxCeleb encoders. All three use LR heads with the same validation selection rule as the original pilot LR heads.

The ten supervised systems below use the same clip identities, speaker partitions and test row order. BA is balanced accuracy; macro F1 gives both classes equal weight.

## Test results

| Model | Balanced accuracy | Macro F1 | Errors out of 400 |
| --- | ---: | ---: | ---: |
| E1: Logistic Regression (LR) | 93.25% | 0.9325 | 27 |
| E1: Support Vector Machine (SVM) | 95.75% | 0.9575 | 17 |
| E1: Random Forest (RF) | 92.00% | 0.9200 | 32 |
| E1: i-vector LR (16 inputs) | 86.00% | 0.8599 | 56 |
| E2: Frozen Whisper full 512 LR | 98.25% | 0.9825 | 7 |
| E2: Frozen Whisper PCA32 LR | 97.00% | 0.9700 | 12 |
| E2: x-vector LR (512 inputs) | 96.50% | 0.9650 | 14 |
| E2: ECAPA-TDNN LR (192 inputs) | 96.75% | 0.9675 | 13 |
| E4: Feature fusion LR | 98.00% | 0.9800 | 8 |
| E4: Weighted voting | 98.00% | 0.9800 | 8 |

## Findings

The RBF SVM is the strongest handcrafted acoustic baseline at 95.75% BA. Both frozen speaker encoders score above it in this split, but neither exceeds frozen full512 Whisper. The small 16-dimensional i-vector extractor is substantially weaker; this result does not represent a large production i-vector system.

Compared with Whisper, x-vector is -1.75 percentage points (paired 95% speaker interval [-3.72, +0.07]) and ECAPA-TDNN is -1.50 points ([-3.60, +0.51]). These intervals include zero, so the pilot does not establish a clear difference for either comparison. It also does not establish that the systems are equivalent.

## Pilot decisions

| Experiment | Proceed with | Drop | Reason |
| --- | --- | --- | --- |
| E1 | Logistic Regression (LR), Support Vector Machine (SVM), Random Forest (RF) | i-vector LR (16 inputs) | i-vector BA is 86.00%, below the three acoustic baselines (92.00%–95.75%). |
| E2 | Frozen Whisper full 512 LR and Frozen Whisper PCA32 LR | x-vector LR (512 inputs), ECAPA-TDNN LR (192 inputs) | Continue with Frozen Whisper only. Full 512 has the highest observed pilot BA (98.25%). |
| E4 | Feature fusion LR | Weighted voting | Both score 98.00% BA. We choose feature fusion because we want to train a model on combined features. |

## Limits

Source and class are linked, so models may use recording or prompt differences as well as accent. This is one fixed split and the test was inspected before the three extensions were added. Some original acoustic controls use older Python and library versions; versions are recorded in each model report. Comparisons therefore assess complete pipelines, rather than isolate architecture. Bootstrap intervals exclude training and selection uncertainty.

## Saved files

- [Executed notebooks](notebooks/) — nine supervised notebooks for the ten reported systems, with cell outputs and inline charts, including the 2,000-clip Whisper pilot rather than the later 40,000-clip study.
- [Results table](evidence/matched_2000_seed42/comparison_with_embeddings.csv) and [comparison chart](evidence/matched_2000_seed42/comparison_with_embeddings.png).
- [Model results](evidence/matched_2000_seed42/) — reports and test predictions for the ten compared systems, paired comparison intervals, and a shared split manifest. Validation selections are included in each model report; error rows can be filtered from predictions.
- [Extension verification](evidence/matched_2000_seed42/embedding_verification.json) and [original pilot verification](evidence/matched_2000_seed42/verification.json). All nine supervised notebooks have executed code cells and no saved cell errors; test metrics were checked against saved predictions.
- [Archive record](archive_manifest.json) — source paths and verified SHA256 hashes for the retained copied files.

This is a results snapshot. Raw audio, feature caches, fitted model binaries and pretrained checkpoints remain in the source project at `/home/mh/nusproj/singlish-whisper/`. Full run scripts and auxiliary outputs also remain there. Notebook reruns use that source project and its recorded environments.
