# A1: matched acoustic, frozen Whisper, and fusion comparison

Status: complete. This is a retrospective comparison on an already inspected test set.

## Procedure

Data: final 24 October preparation, 40,000 clips. Train 27,797; validation 6,084; development 33,881; test 6,119. Speaker overlap across train, validation, and test: zero.

Whisper: frozen openai/whisper-base.en, revision 911407f4214e0e1d82085af863093ec0b66f9cd6. Reuse the verified final feature cache and completed E2 controls. No new extraction or Whisper head training.

Rank all 91 acoustic inputs using a C=1 full91 SVM on original training speakers only. Permutation importance: balanced accuracy, five repeats, seed 42. Select 65 inputs, then keep those columns fixed. Do not use historical development-ranked columns.

Tune acoustic SVM65 and fusion over C=0.01, 0.1, 1, 10. Select validation BA, then macro F1, then smaller C. Match the completed Whisper control search. Fit scaling and PCA within each fitting partition.

Fusion: standardize acoustic65 and divide by sqrt(65); standardize Whisper512, reduce to PCA32, standardize components, and divide by sqrt(32). Concatenate to 97 processed inputs and fit one RBF SVM. Refit both selected pipelines on development data. No probability calibration.

## Selected validation results

| model | C | balanced_accuracy | macro_f1 | origin |
| --- | --- | --- | --- | --- |
| Acoustic SVM65 | 1.0 | 0.9711696755014704 | 0.9710712132116945 | new training-only acoustic selection |
| Fusion SVM65+32 | 10.0 | 0.987224551581901 | 0.9871793985108659 | new training-only acoustic selection |
| Whisper full512 SVM | 10.0 | 0.9873632612673386 | 0.9873434338639522 | completed E2 control |
| Whisper PCA32 SVM | 1.0 | 0.9486190037780692 | 0.9485535817790895 | completed E2 control |

## Test results

| model | C | balanced_accuracy | macro_f1 | roc_auc | false_positive | false_negative | errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Acoustic SVM65 | 1.0 | 96.56% | 0.9655 | 0.9945 | 155 | 56 | 211 |
| Whisper full512 SVM | 10.0 | 98.73% | 0.9873 | 0.9991 | 45 | 33 | 78 |
| Whisper PCA32 SVM | 1.0 | 95.43% | 0.9542 | 0.9896 | 148 | 132 | 280 |
| Fusion SVM65+32 | 10.0 | 98.68% | 0.9868 | 0.9991 | 62 | 19 | 81 |

## Paired differences

Values below are percentage-point differences. Use 2,000 paired whole-speaker bootstrap draws, stratified by class, seed 42. Primary: fusion minus full512 Whisper. Secondary comparisons are descriptive. No multiple-comparison correction.

| reference | metric | difference_pp | low_pp | high_pp |
| --- | --- | --- | --- | --- |
| Whisper full512 SVM | balanced_accuracy | -0.045 | -0.490 | +0.412 |
| Whisper full512 SVM | macro_f1 | -0.049 | -0.500 | +0.404 |
| Acoustic SVM65 | balanced_accuracy | +2.117 | +1.538 | +2.727 |
| Acoustic SVM65 | macro_f1 | +2.125 | +1.540 | +2.763 |
| Whisper PCA32 SVM | balanced_accuracy | +3.256 | +2.349 | +4.328 |
| Whisper PCA32 SVM | macro_f1 | +3.252 | +2.330 | +4.331 |

Fusion minus Whisper full512 SVM: -0.045 percentage points; 95% interval [-0.490, +0.412]. The interval includes zero. A fusion gain is not established. This does not establish equivalence.
Fusion minus Acoustic SVM65: +2.117 percentage points; 95% interval [+1.538, +2.727]. The interval supports a fusion gain on this retrospective test.
Fusion minus Whisper PCA32 SVM: +3.256 percentage points; 95% interval [+2.349, +4.328]. The interval supports a fusion gain on this retrospective test.

## Source scores

| model | source | label | clips | speakers | errors | metric | value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Whisper full512 SVM | common_voice | 0 | 2913 | 1241 | 25 | non-Singapore specificity | 99.14% |
| Whisper full512 SVM | mnsc | 1 | 3034 | 215 | 33 | Singapore recall | 98.91% |
| Whisper full512 SVM | speech_accent_archive | 0 | 50 | 8 | 1 | non-Singapore specificity | 98.00% |
| Whisper full512 SVM | speechocean762 | 0 | 122 | 18 | 19 | non-Singapore specificity | 84.43% |
| Whisper PCA32 SVM | common_voice | 0 | 2913 | 1241 | 96 | non-Singapore specificity | 96.70% |
| Whisper PCA32 SVM | mnsc | 1 | 3034 | 215 | 132 | Singapore recall | 95.65% |
| Whisper PCA32 SVM | speech_accent_archive | 0 | 50 | 8 | 3 | non-Singapore specificity | 94.00% |
| Whisper PCA32 SVM | speechocean762 | 0 | 122 | 18 | 49 | non-Singapore specificity | 59.84% |
| Acoustic SVM65 | common_voice | 0 | 2913 | 1241 | 83 | non-Singapore specificity | 97.15% |
| Acoustic SVM65 | mnsc | 1 | 3034 | 215 | 56 | Singapore recall | 98.15% |
| Acoustic SVM65 | speech_accent_archive | 0 | 50 | 8 | 22 | non-Singapore specificity | 56.00% |
| Acoustic SVM65 | speechocean762 | 0 | 122 | 18 | 50 | non-Singapore specificity | 59.02% |
| Fusion SVM65+32 | common_voice | 0 | 2913 | 1241 | 34 | non-Singapore specificity | 98.83% |
| Fusion SVM65+32 | mnsc | 1 | 3034 | 215 | 19 | Singapore recall | 99.37% |
| Fusion SVM65+32 | speech_accent_archive | 0 | 50 | 8 | 6 | non-Singapore specificity | 88.00% |
| Fusion SVM65+32 | speechocean762 | 0 | 122 | 18 | 22 | non-Singapore specificity | 81.97% |

All Singapore positives are MNSC. Source and class are confounded. These scores do not isolate accent from recording or corpus cues.

## Time

Ranking-model fit: 11.0 s. Permutation ranking: 4587.7 s. These are one-time selection costs.

| model | final_fit_seconds | test_predict_and_score_seconds | origin |
| --- | --- | --- | --- |
| Acoustic SVM65 | 14.067511042998376 | 3.541094938002061 | current run; cached input features |
| Fusion SVM65+32 | 14.660307345999172 | 3.3605001360010647 | current run; cached input features |
| Whisper full512 SVM | 50.49780748800004 | 27.010955352001474 | completed E2 run; cached input features |
| Whisper PCA32 SVM | 11.326246014999924 | 2.914207867999721 | completed E2 run; cached input features |

Fit and prediction times use cached features. Prediction time includes separate predict and decision-score calls. E2 times come from a separate completed run. These times are descriptive, not an end-to-end deployment benchmark.

## Limits

The test set was inspected in earlier work. Count65, PCA32, branch weights, and the model family are prior project choices. Training-set permutation importance is not an independent feature-importance test. Correlated inputs can share importance. Reduced input count alone does not prove less overfitting. Bootstrap intervals condition on fixed fitted models and do not include training or selection uncertainty. An interval containing zero does not establish equivalence. Development speaker folds were not repeated here.

## Error examples for audio review

Up to five errors per model, ordered by absolute SVM margin. Decision scores are not probabilities.

| model | clip_id | source | label | prediction | decision_score |
| --- | --- | --- | --- | --- | --- |
| Whisper PCA32 SVM | speechocean762::so762_012930040 | speechocean762 | 0 | 1 | 3.264583884370856 |
| Whisper PCA32 SVM | speechocean762::so762_013720189 | speechocean762 | 0 | 1 | 3.239443346524466 |
| Whisper PCA32 SVM | mnsc::mnsc_part1-train_00002_01341 | mnsc | 1 | 0 | -3.160818923123352 |
| Whisper PCA32 SVM | mnsc::mnsc_part1-train_00194_00628 | mnsc | 1 | 0 | -2.6324215950893457 |
| Acoustic SVM65 | speech_accent_archive::saa_korean1_c04 | speech_accent_archive | 0 | 1 | 2.5621599314276766 |
| Whisper PCA32 SVM | speechocean762::so762_012930225 | speechocean762 | 0 | 1 | 2.489282134723413 |
| Acoustic SVM65 | speech_accent_archive::saa_korean1_c07 | speech_accent_archive | 0 | 1 | 2.1901300638125 |
| Acoustic SVM65 | speech_accent_archive::saa_korean1_c05 | speech_accent_archive | 0 | 1 | 2.032984178212848 |
| Acoustic SVM65 | common_voice::common_voice_en_26958478 | common_voice | 0 | 1 | 1.9869717289833984 |
| Acoustic SVM65 | speech_accent_archive::saa_korean1_c02 | speech_accent_archive | 0 | 1 | 1.8971148361243373 |
| Fusion SVM65+32 | mnsc::mnsc_part1-train_00194_00628 | mnsc | 1 | 0 | -1.573488357493186 |
| Fusion SVM65+32 | speech_accent_archive::saa_korean1_c04 | speech_accent_archive | 0 | 1 | 1.5291232055913297 |
| Fusion SVM65+32 | speechocean762::so762_026210141 | speechocean762 | 0 | 1 | 1.2783165506322507 |
| Fusion SVM65+32 | speechocean762::so762_004610177 | speechocean762 | 0 | 1 | 1.2096803606290103 |
| Whisper full512 SVM | speechocean762::so762_021290282 | speechocean762 | 0 | 1 | 1.111493258850518 |
| Whisper full512 SVM | speechocean762::so762_004610230 | speechocean762 | 0 | 1 | 1.0890288614259187 |
| Fusion SVM65+32 | speech_accent_archive::saa_korean1_c03 | speech_accent_archive | 0 | 1 | 1.068087525534984 |
| Whisper full512 SVM | mnsc::mnsc_part1-train_00002_00802 | mnsc | 1 | 0 | -1.0361033855689346 |
| Whisper full512 SVM | speechocean762::so762_012930114 | speechocean762 | 0 | 1 | 1.0152999712261552 |
| Whisper full512 SVM | mnsc::mnsc_part1-train_00002_01341 | mnsc | 1 | 0 | -0.8926697525545841 |

## Files

- study.joblib: training-only ranking, selected columns, C trials, and fitted acoustic/fusion pipelines.
- validation_trials.csv: all four C values for all four models; selected rows are marked.
- predictions.csv: validation and test predictions for all four models, with IDs, speakers, labels, sources, and decision scores.
- test_scores.csv: test metrics and speaker intervals.
- run_record.json: settings, input hashes, audio audit, full ranking, selected C, metrics, paired differences, source scores, and time.
- report.md: this report.

Save the executed notebook to retain the charts. Add the completed study to the central final-test archive after review.
