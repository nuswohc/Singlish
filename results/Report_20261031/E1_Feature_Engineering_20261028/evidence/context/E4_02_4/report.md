# E4.02.4: fine-tuned Whisper feature ablation

Status: complete. Only PCA32-alone and acoustic65+full768 classifiers were fitted here.

The full768 and acoustic65+PCA32 predictions, C selections, and scores were imported unchanged from E4.02.3. Cached encoder features and the training-selected acoustic65 columns were reused. No audio or encoder extraction was repeated.

## Test results

| model | action | processed_features | accuracy | balanced_accuracy | macro_f1 | false_positive | false_negative | errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fine-tuned small SVM768 | reused E4.02.3 | 768 | 99.395% | 99.395% | 0.9939528236213468 | 18 | 19 | 37 |
| Fine-tuned small PCA32 SVM | trained here | 32 | 97.647% | 97.649% | 0.9764659729660605 | 80 | 64 | 144 |
| Fine-tuned small fusion65+32 | reused E4.02.3 | 97 | 99.052% | 99.056% | 0.990521215371516 | 44 | 14 | 58 |
| Fine-tuned small fusion65+768 | trained here | 833 | 99.526% | 99.528% | 0.9952604994569436 | 22 | 7 | 29 |

## Paired balanced-accuracy comparisons

Fine-tuned small PCA32 SVM minus Fine-tuned small SVM768: -1.746 percentage points; 95% interval [-2.487, -1.113]. The interval supports a lower score on this retrospective test.
Fine-tuned small fusion65+32 minus Fine-tuned small PCA32 SVM: +1.407 percentage points; 95% interval [+0.653, +2.266]. The interval supports a gain on this retrospective test, subject to the ASR exposure audit.
Fine-tuned small fusion65+768 minus Fine-tuned small SVM768: +0.133 percentage points; 95% interval [-0.123, +0.417]. A gain is not established. This does not establish equivalence.
Fine-tuned small fusion65+768 minus Fine-tuned small fusion65+32: +0.472 percentage points; 95% interval [+0.233, +0.736]. The interval supports a gain on this retrospective test, subject to the ASR exposure audit.

## Interpretation limits

Exploratory retrospective ablation after inspecting earlier test results. The test set did not select C. The 2,000 paired whole-speaker bootstrap intervals condition on fitted models and exclude training/selection uncertainty. No multiple-comparison correction. Full fusion changes PCA compression and Whisper branch scaling; feature count alone is not isolated.

ASR exposure audit: unresolved: exact 15k ASR exposure manifest not supplied.

All Singapore positives come from MNSC, so source and accent remain confounded. No independent speaker-held-out gain can be claimed until all ASR training and checkpoint-selection exposure is checked.

## New validation trials

| model | C | balanced_accuracy | macro_f1 | errors |
| --- | --- | --- | --- | --- |
| Fine-tuned small PCA32 SVM | 0.01 | 0.9519092385837931 | 0.9518408824389715 | 293 |
| Fine-tuned small PCA32 SVM | 0.1 | 0.9645027487428861 | 0.9644951844102847 | 216 |
| Fine-tuned small PCA32 SVM | 1.0 | 0.9676461685056281 | 0.967619251144459 | 197 |
| Fine-tuned small PCA32 SVM | 10.0 | 0.9612557425107306 | 0.961209525039403 | 236 |
| Fine-tuned small fusion65+768 | 0.01 | 0.964874730618442 | 0.9648256356789393 | 214 |
| Fine-tuned small fusion65+768 | 0.1 | 0.9899905858973912 | 0.9899733306762981 | 61 |
| Fine-tuned small fusion65+768 | 1.0 | 0.9957316106458576 | 0.9957262514136948 | 26 |
| Fine-tuned small fusion65+768 | 10.0 | 0.9958973381897753 | 0.9958906364626015 | 25 |

## Source diagnostics

| model | source | label | clips | speakers | errors | metric | value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fine-tuned small SVM768 | common_voice | 0 | 2913 | 1241 | 13 | non-Singapore specificity | 0.9955372468245794 |
| Fine-tuned small SVM768 | mnsc | 1 | 3034 | 215 | 19 | Singapore recall | 0.9937376400791035 |
| Fine-tuned small SVM768 | speech_accent_archive | 0 | 50 | 8 | 0 | non-Singapore specificity | 1.0 |
| Fine-tuned small SVM768 | speechocean762 | 0 | 122 | 18 | 5 | non-Singapore specificity | 0.9590163934426229 |
| Fine-tuned small fusion65+32 | common_voice | 0 | 2913 | 1241 | 29 | non-Singapore specificity | 0.9900446275317542 |
| Fine-tuned small fusion65+32 | mnsc | 1 | 3034 | 215 | 14 | Singapore recall | 0.995385629531971 |
| Fine-tuned small fusion65+32 | speech_accent_archive | 0 | 50 | 8 | 3 | non-Singapore specificity | 0.94 |
| Fine-tuned small fusion65+32 | speechocean762 | 0 | 122 | 18 | 12 | non-Singapore specificity | 0.9016393442622951 |
| Fine-tuned small PCA32 SVM | common_voice | 0 | 2913 | 1241 | 63 | non-Singapore specificity | 0.9783728115345005 |
| Fine-tuned small PCA32 SVM | mnsc | 1 | 3034 | 215 | 64 | Singapore recall | 0.978905735003296 |
| Fine-tuned small PCA32 SVM | speech_accent_archive | 0 | 50 | 8 | 1 | non-Singapore specificity | 0.98 |
| Fine-tuned small PCA32 SVM | speechocean762 | 0 | 122 | 18 | 16 | non-Singapore specificity | 0.8688524590163934 |
| Fine-tuned small fusion65+768 | common_voice | 0 | 2913 | 1241 | 14 | non-Singapore specificity | 0.9951939581187779 |
| Fine-tuned small fusion65+768 | mnsc | 1 | 3034 | 215 | 7 | Singapore recall | 0.9976928147659855 |
| Fine-tuned small fusion65+768 | speech_accent_archive | 0 | 50 | 8 | 2 | non-Singapore specificity | 0.96 |
| Fine-tuned small fusion65+768 | speechocean762 | 0 | 122 | 18 | 6 | non-Singapore specificity | 0.9508196721311475 |

## New fitting time

| model | final_fit_seconds | test_predict_and_score_seconds | origin |
| --- | --- | --- | --- |
| Fine-tuned small PCA32 SVM | 12.1194265310005 | 2.746969634999914 | new comparison; cached input features |
| Fine-tuned small fusion65+768 | 65.98926423900048 | 27.38274684299995 | new comparison; cached input features |

Source: /home/mh/nusproj/singlish-whisper/outputs/e4_02_3_fusion_enabled_40000. All source input hashes were rechecked after execution.

