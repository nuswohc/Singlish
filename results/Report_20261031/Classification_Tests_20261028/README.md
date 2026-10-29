# Final classification tests

Read each report first. Each folder contains the completed notebook with embedded plots, one CSV of detailed predictions or assignments, and one run record with settings, metrics, provenance, and hashes. E2 also includes its implementation module. Large audio files, models, and feature caches stay in the working experiment project.

## Completed final controls and supporting analysis

| Study | Report | Main finding |
|---|---|---|
| E2 Frozen Whisper controls on the final 40k dataset | [E2 report](E2_Whisper_Only_20261028/report.md) | Full512 Whisper BA 98.73%; original fusion BA 98.68%. Fusion minus Whisper: -0.045 percentage points, paired 95% speaker interval [-0.505, +0.421]. A gain over full Whisper is not established. |
| E3 Development-only unsupervised analysis | [E3 report](E3_Clustering_20261028/report.md) | 33,881 clips from 8,443 speakers. PCA50 retains 59.68% variance. Silhouette 0.1272; accent ARI 0.1819. Clusters overlap with the accent labels; source and speaker limitations remain. |

E2 tests 6,119 clips from the original study split. Those clips were evaluated earlier, so this is a retrospective control. E3 excludes all study test clips from fitting, plots, and metrics. Its earlier all-clip exploration remains documented. E3 is descriptive analysis, not classification accuracy or independent validation.

## Existing final model evidence

These records are already stored in results and are linked here to avoid duplicate copies.

| Evidence | Location |
|---|---|
| Final acoustic models, 39/91 baseline comparison, and 65-feature selection | [24 October E1](../../E1_Feature_Engineering_20261024/) |
| Final acoustic and Whisper feature fusion, calibration, and study predictions | [24 October E4](../../E4_Hybrid_Ensemble_20261024/) |
| Fixed 9,000-clip benchmark and preparation checks for the retained release | [24 October benchmark](../../Perfomance_Test_20261024/Performance_Test_Report.md) |
| Later release comparison on the same benchmark | [25 October benchmark](../../Perfomance_Test_20261025/Performance_Test_Report.md) |
| Non-English stress test | [25 October outlier test](../../Outlier_Test_20261025/Outlier_Test_Report.md) |

## Pending work

- A1: refit acoustic feature selection on original training speakers; tune acoustic and fusion with the same C grid as Whisper; report paired test differences. Reuse the completed Whisper heads and embeddings.
- B3: collect a manual live-test record across speakers, devices, and repeated recordings.
- A2 record completion and nested selection remain optional. B1's unstable reserved Singapore clips were discarded; that test is closed.

Add a folder only after a study completes. Keep the same small set of files. Archive notebooks record the completed run; use the working paths and rerun commands in run_record.json to reproduce it.
