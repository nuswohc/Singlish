# A2 saved result

Open [the result report](report.md) and [the completed notebook](../notebooks/E4_02_5_frozen_full512_vs_fusion_calibrated.ipynb).

**Fusion has a positive calibrated BA difference on this retrospective test.**

The run uses training-only A1 acoustic columns and five shared calibration folds. It is a separate refit from the 28 October model. Do not mix their scores.

Large CSV files are compressed for storage: `test_predictions.csv.gz` and `calibration_folds.csv.gz`. Decompressed bytes match the original files named in the report. `pandas.read_csv` reads these gzip files directly.

Fitted models and resume files remain in `/home/mh/nusproj/singlish-whisper/outputs/e4_02_5_frozen_full512_vs_fusion_calibrated_20261029`. The archive keeps the completed notebook, report, chart, metrics, predictions, fold assignments and hash records. It does not copy feature caches or model binaries.

The archive chart uses the same values as the notebook, with the BA axis ending at 100% and labels outside the plot. `run_record.json` records source output hashes; `archive_manifest.json` records archive hashes and this presentation change.
