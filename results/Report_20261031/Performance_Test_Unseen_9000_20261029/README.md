# New 9,000-clip performance test

**Complete.** [Read the performance report](Performance_Test_Report.md). Fusion SVM scores **99.442% balanced accuracy with 54 errors**; independent verification passed.

This evaluation uses recordings absent from the complete 40,000-clip study and the earlier 9,000-clip benchmark. Every speaker used in classifier training or validation is excluded. Speakers in the original study test are permitted and their overlap is reported explicitly. Singapore recordings come only from MNSC PART1 and PART2.

Preparation and model identities are fixed in `protocol.json` and `model_lock.json` before prediction. No model is retrained or recalibrated; the probability threshold remains 0.50. Model predictions are never used to admit or replace a recording.

The groups are 3,000 MNSC PART1, 3,000 MNSC PART2 and 3,000 Common Voice recordings. Common Voice negatives retain the project's unambiguous accent mapping and exclude Singaporean and conflicting speaker labels. Sampling limits clips per speaker and uses a fixed seed; it is not a population-representative sample.

## Reproduction

Use the project's `/home/mh/miniconda3/envs/singlish-py314/bin/python` environment. Run from the project root:

```bash
python results/Report_20261031/Performance_Test_Unseen_9000_20261029/prepare_mnsc.py --part 1 --workers 18
python results/Report_20261031/Performance_Test_Unseen_9000_20261029/prepare_mnsc.py --part 2 --workers 18
python results/Report_20261031/Performance_Test_Unseen_9000_20261029/prepare_cv.py
python results/Report_20261031/Performance_Test_Unseen_9000_20261029/freeze_and_extract.py
python results/Report_20261031/Performance_Test_Unseen_9000_20261029/evaluate.py
python results/Report_20261031/Performance_Test_Unseen_9000_20261029/verify_results.py
python results/Report_20261031/Performance_Test_Unseen_9000_20261029/write_report.py
```

Preparation requires the pinned MNSC Hugging Face source, the local Common Voice 26 archive and prepared reserve, and the original exclusion manifests. Paths are recorded in the scripts and preparation records. Feature extraction requires the pinned frozen Whisper-base.en checkpoint and CUDA; model evaluation requires the saved artifacts in `model_contracts.json`. Raw audio and feature caches are under `outputs/performance_test_unseen_9000_20261029/` and are excluded from Git.

The final report, manifests, overlap audit, metrics, predictions and run records are retained here. The old benchmark and report are preserved.

## Interpretation

“Unseen” means speakers unseen during classifier fitting, with new recording identities. Original test speakers can recur; this is not an independent test of entirely new speakers relative to all previous evaluation. Source/class confounding remains. The paired speaker intervals condition on the four fixed models and exclude fitting and model-selection uncertainty. The model identities differ in acoustic feature selection: the Acoustic SVM and PCA Fusion SVM are the 24 October fits; Whisper SVM and Fusion SVM are the 29 October fits. Thus comparisons involving PCA do not isolate compression alone.

The saved `overlap_audit.json` and independent `verification.json` both passed. The report includes source-wise errors and original-test speaker overlap. `preparation_compatibility.json` checks that the prepared-reserve and new raw-audio processing functions agree.

The original run records retain the paths and code hashes recorded at evaluation time. `common.py` was adjusted after consolidation to locate the project root at its new depth.
