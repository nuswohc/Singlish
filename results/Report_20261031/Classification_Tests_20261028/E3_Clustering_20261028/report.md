# E3 Development-only unsupervised study

Completed on 28 October 2026. This is a descriptive analysis, not a classification test.

Used 33,881 development clips from 8,443 speakers. All 6,119 study test clips were excluded from fitting, plots, and metrics. Cache and manifest checks passed; there is no speaker overlap with the excluded test set.

The earlier all-40,000-clip notebook is archived. Its exploration included the study test set. This revision does not restore test independence.

## Procedure

Verified frozen Whisper-base.en embeddings → standard scaling → PCA50 → K-means with two clusters. All fitting uses embeddings only. PCA uses the full SVD solver. K-means uses ten initializations and seed 42. The PCA width and cluster count were not selected from accent-label agreement. UMAP uses the same development PCA features only for display.

PCA50 is an exploratory setting inherited from the earlier E3 run. It is separate from fusion PCA32.

## Primary clip results

- PCA variance retained: **59.68%**.
- Silhouette: **0.1272**, using 5,000 fixed sampled development clips.
- Accent adjusted Rand index after fitting: **0.1819**.

| Cluster | Non-Singapore | Singapore | Total | Singapore share |
|---|---:|---:|---:|---:|
| 0 | 16,308 | 9,108 | 25,416 | 35.84% |
| 1 | 607 | 7,858 | 8,465 | 92.83% |

| Source | Cluster 0 clips | Cluster 1 clips | Distinct speakers |
|---|---:|---:|---:|
| common_voice | 15,621 | 337 | 7,107 |
| mnsc | 9,108 | 7,858 | 1,183 |
| speech_accent_archive | 229 | 50 | 44 |
| speechocean762 | 458 | 220 | 109 |

1,135 speakers have clips in both clusters. Distinct speakers can therefore appear in both cluster totals.

## Sensitivity checks

Additional seeds 7 and 19 compare clustering with the primary seed, using no accent labels. The minimum ARI to the primary partition is **0.9999**. This checks starting conditions, not new-speaker uncertainty.

The separate speaker-mean analysis uses 8,443 observations. Its silhouette is 0.0937; its accent ARI is -0.0614. Aggregation changes both representation and observation weights. The class speaker counts remain unequal.

## Interpretation and limits

Cluster 1 contains 92.83% Singapore clips, but includes only 46.32% of all development Singapore clips. The other Singapore clips are in the mixed cluster. The silhouette (0.1272), accent ARI (0.1819), and label counts show overlap. The two imposed clusters do not reproduce the two accent labels.

The tested initializations give almost identical partitions (minimum ARI to the primary run: 0.9999). The clip-level accent alignment does not persist in the speaker-mean check (accent ARI -0.0614). Aggregation changes representation and observation weights together, so this does not identify a cause.

K-means imposes two groups. Inspect the label table for overlap and the source table for source effects. Do not describe either cluster as a validated Singapore-accent class. All Singapore clips are MNSC, so the data cannot distinguish accent from recording source. ARI and silhouette are not classification accuracy or evidence of performance on new sources. The analysis does not select or change the supervised classifier.

## Files in this folder

- [notebook.ipynb](notebook.ipynb): completed notebook, including label tables, source tables, sensitivity checks, and all plots.
- [assignments.csv](assignments.csv): 33,881 clip assignments and 8,443 speaker-mean assignments. The analysis_unit column separates the two analyses. Speaker-mean rows have no clip_id; clips_averaged gives the number of clips used. Label 0 means non-Singapore and label 1 means Singapore. Cluster numbers have no fixed accent meaning and are local to each analysis.
- [run_record.json](run_record.json): settings, metrics, source and label counts, input audit, versions, source paths, and file hashes.

The archive is a record of the completed run. Rerun from the working project with the command recorded in run_record.json. Fitted transforms, full coordinates, and the earlier all-clip run remain in that project.
