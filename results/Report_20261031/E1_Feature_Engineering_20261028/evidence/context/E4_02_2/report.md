# E4.02.2: fine-tuned transcription encoder + SVM

Status: numerical comparison complete. Retrospective test; ASR exposure audit requires review.

Primary: fine-tuned small SVM768 minus frozen pretrained small SVM768. Secondary: comparison with completed frozen base.en SVM512. Only the matched small comparison controls encoder architecture and width.

Use the exact final 24 October 40k prepared audio and speaker splits. Fit pipelines on training data; select C=0.01/0.1/1/10 by validation balanced accuracy, then macro F1, then smaller C. Refit on development. Test does not select C. Encoders are fixed during extraction. No transcript decoding or new encoder training.

## ASR exposure audit

unresolved: exact 15k ASR exposure manifest not supplied

Do not interpret a numerical gain as an independent speaker-held-out gain until the exact 15k training and checkpoint-selection exposure, all conversation speakers, and cross-part ID mappings are checked. Different MNSC parts alone do not establish independence.

```json
{
  "status": "unresolved: exact 15k ASR exposure manifest not supplied",
  "manual_completeness_and_cross_part_speaker_mapping_check_required": true
}
```

## Test scores

| model | accuracy | balanced_accuracy | macro_f1 | roc_auc | sg_recall | non_sg_fpr | errors | false_positive | false_negative | ba_ci95_low | ba_ci95_high | f1_ci95_low | f1_ci95_high | C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Whisper full512 SVM | 0.9872528190880863 | 0.9872682798622633 | 0.9872523012427138 | 0.9990893055367104 | 0.9891232696110744 | 0.014586709886547812 | 78 | 45 | 33 | 0.9825244250702615 | 0.9915761395142995 | 0.9823947193732667 | 0.9915574700860196 | 10.0 |
| Whisper PCA32 SVM | 0.9542408890341559 | 0.9542595051864926 | 0.9542393918784509 | 0.9895603473972451 | 0.956493078444298 | 0.047974068071312805 | 280 | 148 | 132 | 0.9430143191922988 | 0.964187741902931 | 0.9427697746729848 | 0.9641555780947657 | 1.0 |
| Frozen small SVM768 | 0.9934629841477366 | 0.993443459271423 | 0.9934622464229865 | 0.9998112157300995 | 0.9911008569545154 | 0.004213938411669368 | 40 | 13 | 27 | 0.9895878676059171 | 0.9963194839971817 | 0.9896069317271428 | 0.9963067869262747 | 10.0 |
| Fine-tuned small SVM768 | 0.9939532603366563 | 0.9939514780622422 | 0.9939528236213468 | 0.999762817725422 | 0.9937376400791035 | 0.005834683954619124 | 37 | 18 | 19 | 0.9910722779245995 | 0.9962891903730263 | 0.9910639760605774 | 0.9962813855833994 | 10.0 |

## Paired differences

| candidate | reference | metric | difference | ci95_low | ci95_high | candidate_only_correct | reference_only_correct | difference_pp | low_pp | high_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fine-tuned small SVM768 | Frozen small SVM768 | balanced_accuracy | 0.0005080187908191869 | -0.0014772958999338642 | 0.00262423224390661 | 17 | 14 | 0.05080187908191869 | -0.14772958999338642 | 0.262423224390661 |
| Fine-tuned small SVM768 | Frozen small SVM768 | macro_f1 | 0.0004905771983603202 | -0.0014877751139145718 | 0.0026091819536419644 | 17 | 14 | 0.04905771983603202 | -0.14877751139145717 | 0.2609181953641964 |
| Fine-tuned small SVM768 | Whisper full512 SVM | balanced_accuracy | 0.006683198199978868 | 0.003504652913046011 | 0.010015663289978771 | 59 | 18 | 0.6683198199978868 | 0.3504652913046011 | 1.0015663289978771 |
| Fine-tuned small SVM768 | Whisper full512 SVM | macro_f1 | 0.006700522378633056 | 0.003561678125870571 | 0.010056123397044127 | 59 | 18 | 0.6700522378633056 | 0.3561678125870571 | 1.0056123397044128 |

Fine-tuned small SVM768 minus Frozen small SVM768: +0.051 percentage points; 95% interval [-0.148, +0.262]. A gain is not established. This does not establish equivalence.
Fine-tuned small SVM768 minus Whisper full512 SVM: +0.668 percentage points; 95% interval [+0.350, +1.002]. The interval supports a gain on this retrospective test, subject to the ASR exposure audit.

Primary interval: 2,000 whole-speaker paired bootstrap samples, stratified by class. Intervals condition on fixed models and exclude training/selection uncertainty. Secondary comparisons are descriptive; no multiple-comparison correction.

## Source scores

| model | source | label | clips | speakers | errors | metric | value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Whisper full512 SVM | common_voice | 0 | 2913 | 1241 | 25 | non-Singapore specificity | 0.9914177823549605 |
| Whisper full512 SVM | mnsc | 1 | 3034 | 215 | 33 | Singapore recall | 0.9891232696110744 |
| Whisper full512 SVM | speech_accent_archive | 0 | 50 | 8 | 1 | non-Singapore specificity | 0.98 |
| Whisper full512 SVM | speechocean762 | 0 | 122 | 18 | 19 | non-Singapore specificity | 0.8442622950819673 |
| Whisper PCA32 SVM | common_voice | 0 | 2913 | 1241 | 96 | non-Singapore specificity | 0.9670442842430484 |
| Whisper PCA32 SVM | mnsc | 1 | 3034 | 215 | 132 | Singapore recall | 0.956493078444298 |
| Whisper PCA32 SVM | speech_accent_archive | 0 | 50 | 8 | 3 | non-Singapore specificity | 0.94 |
| Whisper PCA32 SVM | speechocean762 | 0 | 122 | 18 | 49 | non-Singapore specificity | 0.5983606557377049 |
| Frozen small SVM768 | common_voice | 0 | 2913 | 1241 | 8 | non-Singapore specificity | 0.9972536903535874 |
| Frozen small SVM768 | mnsc | 1 | 3034 | 215 | 27 | Singapore recall | 0.9911008569545154 |
| Frozen small SVM768 | speech_accent_archive | 0 | 50 | 8 | 0 | non-Singapore specificity | 1.0 |
| Frozen small SVM768 | speechocean762 | 0 | 122 | 18 | 5 | non-Singapore specificity | 0.9590163934426229 |
| Fine-tuned small SVM768 | common_voice | 0 | 2913 | 1241 | 13 | non-Singapore specificity | 0.9955372468245794 |
| Fine-tuned small SVM768 | mnsc | 1 | 3034 | 215 | 19 | Singapore recall | 0.9937376400791035 |
| Fine-tuned small SVM768 | speech_accent_archive | 0 | 50 | 8 | 0 | non-Singapore specificity | 1.0 |
| Fine-tuned small SVM768 | speechocean762 | 0 | 122 | 18 | 5 | non-Singapore specificity | 0.9590163934426229 |

All positive clips are MNSC; source and class are confounded. These results do not isolate accent cues from corpus cues.

## Time

| model | final_fit_seconds | test_predict_and_score_seconds | origin |
| --- | --- | --- | --- |
| Frozen small SVM768 | 60.12391360399852 | 27.959959643998445 | current run; cached input features |
| Fine-tuned small SVM768 | 65.69218996199925 | 29.707041387999197 | current run; cached input features |
| Whisper full512 SVM | 50.49780748800004 | 27.010955352001474 | completed E2 run; cached input features |
| Whisper PCA32 SVM | 11.326246014999924 | 2.914207867999721 | completed E2 run; cached input features |

Classifier times use cached features and include separate predict and decision-score calls. E2 times were measured in a separate run. Extraction times below include model loading and file I/O. These are descriptive, not a deployment benchmark.

```json
{
  "pretrained_small": {
    "shards": [
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0000.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0001.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0002.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0003.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0004.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0005.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0006.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0007.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0008.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0009.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0010.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0011.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0012.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0013.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0014.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0015.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0016.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0017.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0018.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0019.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0020.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0021.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0022.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0023.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0024.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0025.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0026.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0027.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0028.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0029.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0030.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0031.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0032.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0033.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0034.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0035.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0036.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0037.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0038.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0039.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0040.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0041.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0042.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0043.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0044.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0045.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0046.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0047.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0048.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0049.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0050.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0051.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0052.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0053.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0054.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0055.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0056.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0057.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0058.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0059.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0060.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0061.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0062.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0063.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0064.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0065.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0066.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0067.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0068.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0069.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0070.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0071.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0072.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0073.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0074.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0075.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0000.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0001.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0002.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0003.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0004.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0005.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0006.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0007.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0008.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0009.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0010.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0011.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0012.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0013.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0014.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0015.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0016.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0017.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0018.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0019.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0020.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0021.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0022.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0023.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0024.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0025.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0026.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0027.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0028.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0029.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0030.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0031.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0032.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0033.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0034.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0035.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0036.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0037.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0038.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0039.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0040.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0041.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0042.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0043.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0044.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0045.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0046.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0047.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0048.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0049.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0050.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0051.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0052.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0053.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0054.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0055.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0056.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0057.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0058.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0059.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0060.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0061.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0062.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0063.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0064.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0065.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0066.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0067.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0068.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0069.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0070.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0071.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0072.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0073.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0074.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0075.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0076.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0077.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0078.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0079.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/saa_subset/part-0000.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/saa_subset/part-0001.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/speechocean_subset/part-0000.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/speechocean_subset/part-0001.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/speechocean_subset/part-0002.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/speechocean_subset/part-0003.parquet"
    ],
    "seconds": 857.511443913023,
    "complete": true,
    "sha256": "edac0ebcb13b413a63cdc7d772d754f95464b1164315dfaefca68d2266d9a206"
  },
  "finetuned_small": {
    "shards": [
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0000.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0001.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0002.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0003.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0004.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0005.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0006.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0007.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0008.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0009.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0010.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0011.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0012.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0013.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0014.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0015.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0016.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0017.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0018.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0019.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0020.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0021.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0022.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0023.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0024.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0025.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0026.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0027.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0028.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0029.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0030.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0031.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0032.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0033.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0034.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0035.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0036.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0037.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0038.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0039.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0040.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0041.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0042.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0043.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0044.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0045.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0046.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0047.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0048.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0049.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0050.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0051.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0052.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0053.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0054.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0055.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0056.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0057.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0058.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0059.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0060.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0061.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0062.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0063.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0064.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0065.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0066.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0067.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0068.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0069.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0070.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0071.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0072.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0073.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0074.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/cv_subset/part-0075.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0000.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0001.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0002.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0003.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0004.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0005.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0006.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0007.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0008.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0009.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0010.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0011.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0012.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0013.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0014.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0015.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0016.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0017.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0018.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0019.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0020.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0021.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0022.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0023.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0024.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0025.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0026.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0027.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0028.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0029.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0030.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0031.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0032.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0033.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0034.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0035.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0036.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0037.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0038.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0039.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0040.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0041.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0042.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0043.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0044.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0045.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0046.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0047.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0048.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0049.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0050.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0051.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0052.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0053.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0054.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0055.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0056.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0057.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0058.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0059.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0060.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0061.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0062.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0063.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0064.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0065.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0066.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0067.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0068.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0069.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0070.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0071.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0072.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0073.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0074.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0075.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0076.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0077.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0078.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/mnsc_subset/part-0079.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/saa_subset/part-0000.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/saa_subset/part-0001.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/speechocean_subset/part-0000.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/speechocean_subset/part-0001.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/speechocean_subset/part-0002.parquet",
      "/home/mh/nusproj/singlish-whisper/data/accent_dataset_3to6s_40k/speechocean_subset/part-0003.parquet"
    ],
    "seconds": 779.8905884550113,
    "complete": true,
    "sha256": "c5e4e0a84b7f9a8ba470dec1fad3dcf24deb1c641da6299821070f89b3ee0f26"
  }
}
```

## Error examples

Highest absolute SVM margins among errors; margins are not probabilities.

| model | clip_id | source | label | prediction | decision_score |
| --- | --- | --- | --- | --- | --- |
| Whisper PCA32 SVM | speechocean762::so762_012930040 | speechocean762 | 0 | 1 | 3.264583884370856 |
| Whisper PCA32 SVM | speechocean762::so762_013720189 | speechocean762 | 0 | 1 | 3.239443346524466 |
| Whisper PCA32 SVM | mnsc::mnsc_part1-train_00002_01341 | mnsc | 1 | 0 | -3.160818923123352 |
| Whisper PCA32 SVM | mnsc::mnsc_part1-train_00194_00628 | mnsc | 1 | 0 | -2.6324215950893457 |
| Whisper PCA32 SVM | speechocean762::so762_012930225 | speechocean762 | 0 | 1 | 2.489282134723413 |
| Whisper full512 SVM | speechocean762::so762_021290282 | speechocean762 | 0 | 1 | 1.111493258850518 |
| Whisper full512 SVM | speechocean762::so762_004610230 | speechocean762 | 0 | 1 | 1.0890288614259187 |
| Whisper full512 SVM | mnsc::mnsc_part1-train_00002_00802 | mnsc | 1 | 0 | -1.0361033855689346 |
| Whisper full512 SVM | speechocean762::so762_012930114 | speechocean762 | 0 | 1 | 1.0152999712261552 |
| Fine-tuned small SVM768 | mnsc::mnsc_part1-train_00194_00628 | mnsc | 1 | 0 | -0.9918115358226613 |
| Fine-tuned small SVM768 | mnsc::mnsc_part1-train_00002_00802 | mnsc | 1 | 0 | -0.9563345687503002 |
| Frozen small SVM768 | mnsc::mnsc_part1-train_00002_00802 | mnsc | 1 | 0 | -0.9479552957020199 |
| Whisper full512 SVM | mnsc::mnsc_part1-train_00002_01341 | mnsc | 1 | 0 | -0.8926697525545841 |
| Fine-tuned small SVM768 | speechocean762::so762_012930114 | speechocean762 | 0 | 1 | 0.860006723328355 |
| Frozen small SVM768 | common_voice::common_voice_en_42816877 | common_voice | 0 | 1 | 0.6593106012373346 |
| Fine-tuned small SVM768 | common_voice::common_voice_en_22437930 | common_voice | 0 | 1 | 0.5379347707741136 |
| Frozen small SVM768 | mnsc::mnsc_part1-train_00000_01454 | mnsc | 1 | 0 | -0.5079892361466722 |
| Frozen small SVM768 | common_voice::common_voice_en_22437930 | common_voice | 0 | 1 | 0.5077250056198832 |
| Fine-tuned small SVM768 | common_voice::common_voice_en_22423410 | common_voice | 0 | 1 | 0.5070412415470043 |
| Frozen small SVM768 | speechocean762::so762_012930097 | speechocean762 | 0 | 1 | 0.4343190607558647 |

## Files

Two *_features.npy files hold raw encoder embeddings in audited dataset order. study.joblib holds extraction progress, C trials, and fitted pipelines. validation_trials.csv holds all C candidates. predictions.csv holds selected validation and fixed test predictions. test_scores.csv holds scores and intervals. run_record.json holds provenance and paired differences. report.md is this report. Save this notebook for charts.
