# Frozen Whisper on the final fusion dataset

Completed on 28 October 2026. The encoder weights stayed fixed. Only the SVM heads were trained.

Data: 40,000 original prepared clips. Train: 27,797; validation: 6,084; test: 6,119.
Test: 3,085 non-Singapore clips and 3,034 Singapore clips from 1,482 speakers. Speaker overlap is zero.
All 40,000 decoded audio hashes match the fixed reference. No extra trimming or volume change was applied.
Embeddings: verified original fusion cache. Encoder: `openai/whisper-base.en` at `911407f4214e0e1d82085af863093ec0b66f9cd6`.

| Model | Validation C | Test BA | Macro F1 | ROC AUC | FP | FN | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| Whisper full512 SVM | 10 | 98.73% | 0.9873 | 0.999089 | 45 | 33 | 78 |
| Whisper PCA32 SVM | 1 | 95.43% | 0.9542 | 0.989560 | 148 | 132 | 280 |
| SVM65 | 1 | 96.69% | 0.9668 | 0.993857 | 141 | 62 | 203 |
| Fusion SVM | 10 | 98.68% | 0.9868 | 0.998970 | 62 | 19 | 81 |

Fusion minus full Whisper: **-0.045 percentage points** of balanced accuracy.
Paired 95% speaker interval: **[-0.505, +0.421] percentage points**.
The paired interval includes zero. This test does not establish a fusion gain over full Whisper.

## Test rules

- The SVM C grid is 0.01, 0.1, 1, 10. Validation balanced accuracy selects C. Ties use macro F1, then smaller C.
- Standard scaling, PCA and the SVM are fitted inside each fitting partition.
- Full Whisper uses 512 inputs. The PCA control uses 32 scaled components, as in the fusion branch.
- Chosen settings are fixed before testing. The final heads refit on training plus validation clips.
- `predict()` supplies class labels. Decision scores supply ROC AUC. Scores are not calibrated probabilities.
- Acoustic65 and Fusion SVM predictions come from the saved 24 October uncalibrated experiment.
- All four models use identical test clip IDs, labels and speakers.
- The 2,000 bootstrap draws resample complete speakers within each class. All models share each draw.

## Limits

The test set was already used in the earlier experiments. This is a retrospective control, not a new untouched test set.
All positive clips come from MNSC. Negative clips come from other sources. These results do not isolate accent from recording source.
The speaker folds use C selected on the fixed validation set. They are stability checks, not nested model selection.
The saved fusion model uses the original selected65 acoustic inputs. This run does not repeat acoustic feature selection inside folds.
The bootstrap intervals describe this split and these fitted models. They do not include training or selection uncertainty.
A gain from fusion does not prove that acoustic features add accent information independent of recording conditions.

## Files in this folder

- [notebook.ipynb](notebook.ipynb): completed notebook, including tables and plots.
- [frozen_whisper_40k.py](frozen_whisper_40k.py): the study implementation used by the notebook.
- [predictions.csv](predictions.csv): all four models on the same 6,119 test clips. There are 24,476 rows. Labels and decision scores are uncalibrated.
- [run_record.json](run_record.json): settings, metrics, paired intervals, fold scores, provenance, versions, source paths, and file hashes.

The archive is a record of the completed run. Rerun from the working project with the command recorded in run_record.json. Audio, fitted models, and feature caches remain in that project.
