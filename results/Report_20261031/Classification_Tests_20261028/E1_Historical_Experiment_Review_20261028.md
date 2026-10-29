# Review of historical E1 experiments

Checked on 28 October 2026. Reviewed the five E1 folders and their 36 top-level notebooks, saved study objects, metadata, split files, and predictions. No training, encoder extraction, audio preparation, or model scoring was rerun. New numerical work below consists only of paired summaries and bootstrap intervals calculated from existing saved class predictions.

## What is worth mentioning

Add the matched 30k-to-40k comparison and the normalized-versus-unnormalized retraining comparison as short supporting studies. Add speaker weighting as an evaluation sensitivity check. Keep the controlled input-preparation check and calibration trade-off already covered in the Word report. The older fine-tuned encoder fusion result is preliminary context for E4.02.2, not a replacement for it.

## 1. Data expansion on a fixed test subset — high value

All 30,000 original clips remain in the 40,000-clip dataset with unchanged labels, source, speaker groups, sampling rate, and train/validation/test assignments. All 30,000 encoded audio hashes match. The shared cached acoustic values agree within absolute tolerance 1e-4 and relative tolerance 1e-5; the largest absolute difference is recorded below. These are separate extraction runs, so they are not bit-identical. No original test clip moved into development.

Feature-cache maximum absolute differences: {'39': 9.1552734375e-05, '91': 9.1552734375e-05}.

Original fitting clips: 21,000 train + 4,500 validation = 25,500 development. Expanded fitting clips: 27,797 train + 6,084 validation = 33,881 development. Compare both saved final models on the same original 4,500 test clips below.

| Model | Clips | Speakers | 30k BA | 40k BA | Change (pp) | 95% paired interval (pp) | 30k FP/FN | 40k FP/FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SVM65 | 4500 | 1285 | 96.244% | 96.244% | +0.000 | [-0.397, +0.398] | 129/40 | 121/48 |
| Fusion SVM | 4500 | 1285 | 98.467% | 98.489% | +0.022 | [-0.199, +0.256] | 53/16 | 52/16 |

The data expansion adds 5,000 MNSC PART2 positives and 5,000 Common Voice negatives. Total speakers increase from 8,596 to 9,925. MNSC PART1 remains 7,500 clips, PART2 increases from 7,500 to 12,500, and Common Voice increases from 13,871 to 18,871. SpeechOcean762 (800) and Speech Accent Archive (329) do not expand.

Interpretation: SVM65 BA is unchanged, with eight fewer false positives and eight more false negatives. Fusion improves by only one correct clip. Both paired intervals include zero; these results do not establish a gain on the original test subset. This is a retrospective comparison of the full training/selection procedure with more data from two sources. SVM feature identities are reselected; both runs retain 65, but not the same 65 columns. Fusion still uses acoustic65 + Whisper PCA32, the same base.en revision and pooling, and C=10; acoustic SVM uses C=1. More data, source coverage, speaker coverage, and reselected features change together. This is not a multi-seed learning curve or an isolated effect of sample count. The reported whole-test SVM65 increase (96.24% on 4,500 clips to 96.69% on 6,119 clips) is not evidence of a training-size benefit: the test population also expanded. Use the matched original test subset above for that question.

## 2. Training with and without gain adjustment — high value

V4 and V5 use the same 40,000 clip identities, labels, speaker groups, and splits. Both retrain their classifiers. All 6,119 test clips and 1,482 speakers are matched. V4 uses upstream RMS normalization (-20 dBFS, peak limit 0.99) and PCM16 FLAC. V5 prepares source audio without loudness normalization or peak scaling and saves float32 WAV. Both use mono 16 kHz and trimming; selection provenance is retained in V5 data/selection_verification.json.

| Model | Normalized BA | No gain adjustment BA | Change (pp) | 95% paired interval (pp) | Normalized FP/FN | No gain FP/FN |
| --- | --- | --- | --- | --- | --- | --- |
| SVM65 | 96.693% | 97.361% | +0.668 | [+0.215, +1.136] | 141/62 | 114/48 |
| Fusion SVM | 98.682% | 98.859% | +0.177 | [-0.162, +0.549] | 62/19 | 48/22 |

Interpretation: the unnormalized float32 preparation gives higher observed internal test BA in these saved runs. The acoustic paired interval excludes zero for this retrospective protocol comparison; the fusion interval includes zero. Do not claim a demonstrated fusion gain. This does not show that normalization is unnecessary for the deployed V4 model. Waveform representation/quantization, preparation implementation, fitting, acoustic feature identities, and fitted scaling/PCA also change. Call it a preparation-protocol retraining comparison, not a pure gain-only ablation. The normalized deployment benchmark favors the 24 October release and does not evaluate V5 on its own matched input contract.

## 3. Fixed-model input-preparation diagnostic — high value, already covered

The same frozen calibrated SVM65 at threshold 0.50 scores the same 3,000 external PART2 recordings. No refitting or feature-count change.

| variant | clips | recall_pct | misses | raw_svm_recall_pct |
| --- | --- | --- | --- | --- |
| original_raw | 3000 | 53.6 | 1392 | 57.766666666666666 |
| trim_only | 3000 | 56.73333333333333 | 1298 | 61.1 |
| normalize_only | 3000 | 97.2 | 84 | 97.93333333333332 |
| trim_and_normalize | 3000 | 97.2 | 84 | 97.93333333333332 |

Interpretation: matching the training preparation recovers Singapore recall from 53.60% to 97.20%; trimming alone gives 56.73%. Normalization includes peak limiting and PCM16 quantization. This diagnoses input-pipeline mismatch. These are all-positive recordings, so recall is measurable, while BA, specificity and ROC AUC are not. The speakers were already represented in the study test, so this is a diagnostic, not a new independent benchmark.

This and section 2 are compatible: a model fitted on normalized audio can fail on raw inputs, while a separately fitted raw-audio model can score well on raw inputs.

## 4. Equal-speaker evaluation — useful short addition

Use the same predictions. Give every speaker equal total evaluation weight by weighting each clip inversely by that speaker's clip count. This is a metric sensitivity check; it does not change fitting or make clips independent.

| model | weighting | balanced_accuracy |
| --- | --- | --- |
| SVM65 | Equal clips | 96.693% |
| SVM65 | Equal speakers | 97.215% |
| Fusion SVM | Equal clips | 98.682% |
| Fusion SVM | Equal speakers | 98.919% |

Interpretation: fusion remains above acoustic SVM65 under both evaluation weightings. Aggregate performance is not explained only by speakers who supplied many clips. This does not remove source confounding. These speaker-weighted metrics differ from the whole-speaker bootstrap, which measures uncertainty for clip-based scores.

## 5. Calibration and error trade-off — useful, already covered

Final normalized 40k SVM65: compare the original SVM decision boundary with sigmoid probability >= 0.50 on the same test clips.

| system | balanced_accuracy | singapore_recall | false_positive | false_negative | brier_score | log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| Original SVM | 96.693% | 97.956% | 141 | 62 | not a probability | not a probability |
| Calibrated SVM at 0.50 | 96.786% | 97.462% | 120 | 77 | 0.0260454990096302 | 0.0947878713156276 |

Calibration removes 21 false positives but adds 15 false negatives. BA increases by only 0.093 percentage points. Do not describe this as a general accuracy breakthrough. Brier score and log loss measure the probability outputs; raw SVM margins have no directly comparable probability score. Calibration uses earlier development-selected features, so it is not a fully nested test of selection plus calibration.

## 6. Earlier fine-tuned transcription encoder fusion — preliminary context

| Clips | Speakers | Frozen base.en fusion BA | Fine-tuned small fusion BA | Change (pp) | 95% paired interval (pp) | Frozen FP/FN | Fine-tuned FP/FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1736 | 684 | 97.415% | 98.741% | +1.327 | [+0.663, +2.019] | 39/9 | 13/9 |

The retained test has 1,736 clips. The acoustic35 inputs and selected C values match between runs. The older fine-tuned run uses the exact same model.safetensors hash as the deployed 15k transcription checkpoint: 2396a2b2660500753d81c7a6d941723e5c8cb1d60fbae2daf22b004c8c3312f6.

Limit: pretrained base.en (512 values) changes to fine-tuned multilingual small (768 values); both are reduced to PCA32. Extraction precision and pooling implementation also differ (the fine-tuned cache records bfloat16 and mean_real_steps_20ms). There is no matched pretrained-small control and no Whisper-only control. ASR training/checkpoint-selection exposure is unresolved. Historical acoustic columns were selected using development data. This supports testing the idea; it does not isolate fine-tuning or establish an independent held-out gain. E4.02.2 now supplies the missing matched small/full768 SVM comparison on the final 40k data.

## 7. Other retained checks — appendix or omit

- Basic full39 versus extended full91: retain the final 40k result already in the Word report (SVM 95.30% versus 96.53% test BA). This is a useful controlled feature-block comparison, not a ranked 39-of-91 experiment.
- Extra downstream processing on the early 11,486-clip data: additional trim/peak normalization gives full91 SVM test BA 93.12%, versus 94.44% when decoding the upstream-prepared audio directly. Same dataset identity and split, but several waveform operations change. Mention only as a reason to keep one preparation contract; do not call it an isolated loudness-normalization result.
- Earlier SVM50 PART1/PART2 evaluations: positive-only recall checks for older models. Most speakers were already in the earlier held-out test. Keep in historical appendix if needed; do not combine these with final SVM65 results or call them fully new speaker tests.
- 500 phrase-specific additions: preparation metadata says models_retrained=false, while a later results_91_svm study file exists. The saved metadata and result chronology need reconciliation. The test set also gained 75 clips. Do not claim an improvement due to the phrase additions from these records.
- Bounded C/gamma search notebook: no executed outputs are retained in the top-level notebook. Treat it as a plan unless completed result files can be linked.
- Do not revive the discarded 554 Common Voice Singaporean clips as evidence. No valid result is available for that cancelled study.

## Folder identity and duplicates

- E1_Feature_Engineering contains the 2,000-clip prototype, early 11,486-clip studies, extra-data variants, and older fusion/evaluation studies.
- V2 and V3 retain the same 30k study: study_results.joblib, speaker_splits.csv, and fusion test_predictions.csv hashes match exactly. These are not independent replications. V4 also retains the same 30k study identity and acoustic result cache.
- V4 outputs/e1_feature_study_accent_40000 is the final normalized 40k study. V5 is the alternate no-gain float32 preparation of those same selected clips.
- The copied E1_feature_engineering_results_report_20261019.md in V2/V3 still describes the old 11,486-clip experiment. It is not evidence for the local 30k results. Prefer identity-linked saved results and notebook outputs.

## Interval method and limits

New intervals in sections 1, 2 and 6 use 2,000 paired whole-speaker bootstrap draws, stratified by class, seed 42. A speaker has one class in these datasets. All clips of a drawn speaker are retained; the same speaker multiplicities apply to both models. Scores remain clip-based. These intervals condition on the fixed saved models and exclude training, feature-selection and tuning uncertainty. They are retrospective descriptive analyses; no multiple-comparison correction or equivalence margin was applied. An interval including zero does not establish equivalence.

## Reproducible sources

All numerical summaries above come from existing saved files. The following SHA256 list identifies the directly read sources. The review script used for the derived paired summaries is /tmp/review_e1_history.py; the algorithm and parameters are stated above.

- [E1_Feature_Engineering/outputs/svm35_fusion_comparison_finetuned_whisper/model_metadata.json](/home/mh/nusproj/E1_Feature_Engineering/outputs/svm35_fusion_comparison_finetuned_whisper/model_metadata.json) — SHA256 `363fe4dc3b017e875a162bf738490d63e00604738d116af980904af4c5a896be`
- [E1_Feature_Engineering/outputs/svm35_fusion_comparison_finetuned_whisper/test_predictions.csv](/home/mh/nusproj/E1_Feature_Engineering/outputs/svm35_fusion_comparison_finetuned_whisper/test_predictions.csv) — SHA256 `401fe57816052f9fde1ff83d2310d949ab51b90993ff8e0581005088ecb4aa79`
- [E1_Feature_Engineering/outputs/svm35_fusion_comparison_upstream/model_metadata.json](/home/mh/nusproj/E1_Feature_Engineering/outputs/svm35_fusion_comparison_upstream/model_metadata.json) — SHA256 `667fa8166d6e8d86d957ca5576397d6b0735a19e802634b5a1710c0d242d2e0e`
- [E1_Feature_Engineering/outputs/svm35_fusion_comparison_upstream/test_predictions.csv](/home/mh/nusproj/E1_Feature_Engineering/outputs/svm35_fusion_comparison_upstream/test_predictions.csv) — SHA256 `587efb39afb0bdbb480e755317980106cc1f753c58284a3eb522d27011e6779c`
- [E1_Feature_EngineeringV2/outputs/e1_feature_study_accent_30000/results_91/study_results.joblib](/home/mh/nusproj/E1_Feature_EngineeringV2/outputs/e1_feature_study_accent_30000/results_91/study_results.joblib) — SHA256 `2128215cd6a569c750d3ca068c831117b2eed5c5774b1bcb482a4092a9bb8040`
- [E1_Feature_EngineeringV2/outputs/e1_feature_study_accent_30000/speaker_splits.csv](/home/mh/nusproj/E1_Feature_EngineeringV2/outputs/e1_feature_study_accent_30000/speaker_splits.csv) — SHA256 `2df542ea4445f23fabb0fa7e0c9eb19cb5e2ce18354931a79dae09191f202db1`
- [E1_Feature_EngineeringV2/outputs/svm65_fusion_comparison_30000/test_predictions.csv](/home/mh/nusproj/E1_Feature_EngineeringV2/outputs/svm65_fusion_comparison_30000/test_predictions.csv) — SHA256 `f9357b2707c682c1f0193818efec5c71a7fab52f12edabe3a949ec8fe286cc3f`
- [E1_Feature_EngineeringV3/outputs/e1_feature_study_accent_30000/acoustic_features_39.npz](/home/mh/nusproj/E1_Feature_EngineeringV3/outputs/e1_feature_study_accent_30000/acoustic_features_39.npz) — SHA256 `1127e04a158990e9c324a2bfa692d9e2f5871730f10fd15bbde545eeffe7731f`
- [E1_Feature_EngineeringV3/outputs/e1_feature_study_accent_30000/acoustic_features_91.npz](/home/mh/nusproj/E1_Feature_EngineeringV3/outputs/e1_feature_study_accent_30000/acoustic_features_91.npz) — SHA256 `d65efe8a84a8df3d423252432a4dc73397c5a8ef3e0e7e9ed14b2aec0bce07ad`
- [E1_Feature_EngineeringV3/outputs/e1_feature_study_accent_30000/current_cache_metadata.json](/home/mh/nusproj/E1_Feature_EngineeringV3/outputs/e1_feature_study_accent_30000/current_cache_metadata.json) — SHA256 `441bc058cfc2919fc4a1415d18d830d27b5c368c15d015a3f3f4ec71f37c21df`
- [E1_Feature_EngineeringV3/outputs/e1_feature_study_accent_30000/results_91/study_results.joblib](/home/mh/nusproj/E1_Feature_EngineeringV3/outputs/e1_feature_study_accent_30000/results_91/study_results.joblib) — SHA256 `2128215cd6a569c750d3ca068c831117b2eed5c5774b1bcb482a4092a9bb8040`
- [E1_Feature_EngineeringV3/outputs/e1_feature_study_accent_30000/speaker_splits.csv](/home/mh/nusproj/E1_Feature_EngineeringV3/outputs/e1_feature_study_accent_30000/speaker_splits.csv) — SHA256 `2df542ea4445f23fabb0fa7e0c9eb19cb5e2ce18354931a79dae09191f202db1`
- [E1_Feature_EngineeringV3/outputs/svm65_fusion_comparison_30000/test_predictions.csv](/home/mh/nusproj/E1_Feature_EngineeringV3/outputs/svm65_fusion_comparison_30000/test_predictions.csv) — SHA256 `f9357b2707c682c1f0193818efec5c71a7fab52f12edabe3a949ec8fe286cc3f`
- [E1_Feature_EngineeringV4/outputs/e1_feature_study_accent_40000/acoustic_features_39.npz](/home/mh/nusproj/E1_Feature_EngineeringV4/outputs/e1_feature_study_accent_40000/acoustic_features_39.npz) — SHA256 `fabbcecf8f5f6ee14a6cbcc664d72f9f10110e57596ff891da6570bc04d09468`
- [E1_Feature_EngineeringV4/outputs/e1_feature_study_accent_40000/acoustic_features_91.npz](/home/mh/nusproj/E1_Feature_EngineeringV4/outputs/e1_feature_study_accent_40000/acoustic_features_91.npz) — SHA256 `5bd24a1f3a26b05a02694ed4a1d4e3fc0b72524eaea45d1429276f0f71f2438d`
- [E1_Feature_EngineeringV4/outputs/e1_feature_study_accent_40000/current_cache_metadata.json](/home/mh/nusproj/E1_Feature_EngineeringV4/outputs/e1_feature_study_accent_40000/current_cache_metadata.json) — SHA256 `3003723da7bef6ef5e44fe03a526559ad349d35844f73bd65a21f10ac834e741`
- [E1_Feature_EngineeringV4/outputs/e1_feature_study_accent_40000/speaker_splits.csv](/home/mh/nusproj/E1_Feature_EngineeringV4/outputs/e1_feature_study_accent_40000/speaker_splits.csv) — SHA256 `428937b39eb646a41e46b8ee3cdeb86652069802a675575892e6f17e289b8581`
- [E1_Feature_EngineeringV4/outputs/e1_feature_study_accent_40000/svm_calibrated_probability_testing/test_summary.csv](/home/mh/nusproj/E1_Feature_EngineeringV4/outputs/e1_feature_study_accent_40000/svm_calibrated_probability_testing/test_summary.csv) — SHA256 `110a3f9bd3fcc5b6bcc8c77d168e5fb4477208fd76ea7d2de03f944597cf4505`
- [E1_Feature_EngineeringV4/outputs/part2_recall_review_20261024/preprocessing_summary.csv](/home/mh/nusproj/E1_Feature_EngineeringV4/outputs/part2_recall_review_20261024/preprocessing_summary.csv) — SHA256 `100f8ccec72f7e7e606b4e61e73234c203f9182fe5522749a9eb9633d85f0781`
- [E1_Feature_EngineeringV4/outputs/svm65_fusion_comparison_40000/model_metadata.json](/home/mh/nusproj/E1_Feature_EngineeringV4/outputs/svm65_fusion_comparison_40000/model_metadata.json) — SHA256 `4b2922add8ba9298b9cefb4edbd14b0a27ab345c837b63640f066836dea27f5c`
- [E1_Feature_EngineeringV4/outputs/svm65_fusion_comparison_40000/speaker_weight_comparison.csv](/home/mh/nusproj/E1_Feature_EngineeringV4/outputs/svm65_fusion_comparison_40000/speaker_weight_comparison.csv) — SHA256 `c8bdde3eb5e43d990e15f253fa2dd2bbf5621b354a4ffe1ffe7645f3cabdd053`
- [E1_Feature_EngineeringV4/outputs/svm65_fusion_comparison_40000/test_predictions.csv](/home/mh/nusproj/E1_Feature_EngineeringV4/outputs/svm65_fusion_comparison_40000/test_predictions.csv) — SHA256 `2ee1118712adc9ffd5588ea03e3eeec0f75b3c641fb71374e5672ab101baa756`
- [E1_Feature_EngineeringV5/outputs/e1_feature_study_accent_40000_no_loudness_norm/speaker_splits.csv](/home/mh/nusproj/E1_Feature_EngineeringV5/outputs/e1_feature_study_accent_40000_no_loudness_norm/speaker_splits.csv) — SHA256 `428937b39eb646a41e46b8ee3cdeb86652069802a675575892e6f17e289b8581`
- [E1_Feature_EngineeringV5/outputs/whisper_svm_fusion_comparison_40000_no_loudness_norm/model_metadata.json](/home/mh/nusproj/E1_Feature_EngineeringV5/outputs/whisper_svm_fusion_comparison_40000_no_loudness_norm/model_metadata.json) — SHA256 `cbef1daa8075917db786c71ab530ba77cb9ce5757256ef69f4b6e640c62c2952`
- [E1_Feature_EngineeringV5/outputs/whisper_svm_fusion_comparison_40000_no_loudness_norm/test_predictions.csv](/home/mh/nusproj/E1_Feature_EngineeringV5/outputs/whisper_svm_fusion_comparison_40000_no_loudness_norm/test_predictions.csv) — SHA256 `97bb23ddb8aeb962c4a1e8aeb4e7d4c7c48b25933bc3bbc317bade9ed802ad61`
- [Singlish/mvp/models/whisper-small-mnsc-sg-15k-heldout/model.safetensors](/home/mh/nusproj/Singlish/mvp/models/whisper-small-mnsc-sg-15k-heldout/model.safetensors) — SHA256 `2396a2b2660500753d81c7a6d941723e5c8cb1d60fbae2daf22b004c8c3312f6`
