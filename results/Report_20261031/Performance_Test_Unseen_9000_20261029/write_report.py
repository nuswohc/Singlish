"""Write the report from completed metrics and audits, without changing model choices."""
import json
import pandas as pd
from common import OUT,GROUPS
from evaluate import DISPLAY,NAMES

def main():
 audit=json.loads((OUT/'overlap_audit.json').read_text());run=json.loads((OUT/'run_record.json').read_text());assert run['status']=='complete'
 metrics=pd.read_csv(OUT/'metrics.csv');overall=metrics.loc[metrics.dataset.eq('all')].set_index('model');pairs=pd.read_csv(OUT/'paired_comparisons.csv')
 p=pairs.loc[pairs.reference.eq('Frozen Whisper (512)')&pairs.metric.eq('balanced_accuracy')].iloc[0]
 cv=json.loads((OUT/'common_voice_preparation.json').read_text());fusion=overall.loc['512+65'];whisper=overall.loc['Frozen Whisper (512)'];diff=int(whisper.errors-fusion.errors);error_change=f'{abs(diff)} '+('fewer' if diff>=0 else 'more')
 lines=['# Performance test on 9,000 new recordings','','29 October 2026 | Four frozen calibrated models | Probability threshold 0.50','',
 f'Fusion SVM achieves **{fusion.balanced_accuracy*100:.3f}% balanced accuracy with {int(fusion.errors)} errors**. Whisper SVM achieves **{whisper.balanced_accuracy*100:.3f}% with {int(whisper.errors)} errors**. Fusion makes {error_change} errors; its paired balanced-accuracy difference is **{p.difference*100:+.3f} percentage points [{p.ci95_low*100:+.3f}, {p.ci95_high*100:+.3f}]**.','',
 '## What unseen means here','',
 'No selected speaker was used in classifier training or validation, including the development refits and calibration. Every recording is absent from the complete 40,000-clip study and the earlier 9,000-clip benchmark according to source identities and the checks below. Original study-test speakers are allowed. Thus the recordings and fitting-speaker exclusions are new, but the test is not independent of all earlier evaluation.','',
 '| Group | Clips | Speakers | Speakers also in original test | Clips from original test speakers |', '| --- | ---: | ---: | ---: | ---: |']
 for g in audit['groups']:lines.append(f"| {g['dataset']} | {g['clips']:,} | {g['speakers']:,} | {g['original_test_speakers']:,} | {g['clips_from_original_test_speakers']:,} |")
 lines+=['',f"There are {audit['speakers']:,} distinct speakers overall. Speakers shared between PART1 and PART2 count only once in this total and are resampled together in the confidence intervals. Class counts are 6,000 Singapore and 3,000 non-Singapore; balanced accuracy weights the two classes equally.",'','## Overlap verification','',
 'All three groups passed with **zero** training/validation speaker overlap, zero study or old-benchmark clip-ID overlap, zero recorded raw/PCM hash overlap, zero MNSC source-file/row overlap, and zero existing normalized speaker/transcript matches. New prepared PCM hashes are unique across all 9,000 clips. Checks use all 40k partitions when excluding recordings and use train plus validation when excluding fitting speakers.','',
 'Source coordinates prevent renamed MNSC IDs from appearing new. Speaker/transcript exclusions also remove repeated microphone-channel versions of the same utterance. Available hashes identify exact duplicates; no perceptual fingerprinting or independent speaker recognition was performed. Speaker independence relies on the source speaker IDs.','',
 '## Results','', '| Model | Balanced accuracy [95% speaker interval] | Accuracy | Macro F1 | FP | FN | Errors |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
 for n in NAMES:
  r=overall.loc[n];lines.append(f'| {DISPLAY[n]} | {r.balanced_accuracy*100:.3f}% [{r.balanced_accuracy_ci95_low*100:.3f}, {r.balanced_accuracy_ci95_high*100:.3f}] | {r.accuracy*100:.3f}% | {r.macro_f1:.4f} | {int(r.false_positive)} | {int(r.false_negative)} | {int(r.errors)} |')
 lines+=['','![Four saved models on the new evaluation clips](performance_comparison.png)','','FP means a non-Singapore clip incorrectly labelled Singapore; FN means a Singapore clip missed.','', '| Group | Acoustic SVM errors | Whisper SVM errors | PCA Fusion SVM errors | Fusion SVM errors |','| --- | ---: | ---: | ---: | ---: |']
 for g in GROUPS:
  rs=metrics.loc[metrics.dataset.eq(g)].set_index('model');lines.append('| '+g+' | '+' | '.join(str(int(rs.loc[n,'errors'])) for n in NAMES)+' |')
 lines+=['','## Fixed method and sampling','',
 '- The protocol and four saved model checksums were locked before new predictions. No training, calibration fitting, feature selection or threshold tuning used this set.',
 '- Singapore clips come exclusively from pinned MNSC PART1/PART2 training-source shards. Source corpus “Train” is distinct from this project’s training split: candidate speakers were explicitly excluded using the project manifests.',
 '- MNSC source shards and rows use seeded order, then speaker round-robin selection with at most 30 clips per speaker per group. The selected set is screened for cross-group duplicate audio and utterances.',
 '- Common Voice uses a prepared CV26 reserve followed by the local CV26 archive until a 4,500-candidate pool is reached, then seeded speaker round-robin selection with at most ten clips per speaker. Negative accent labels must be unambiguous. Singaporean and conflicting speaker labels are excluded. This convenience candidate pool has no accent quotas.',
 '- All groups use mono 16 kHz audio, 40 dB RMS silence trimming with 100 ms edge padding, 3–6-second duration eligibility after trimming, -20 dBFS RMS normalization with a 0.99 peak limit, and PCM16 encode/decode. Both feature branches receive the same decoded waveform. No new crop is introduced for overlength recordings; they are rejected.',
 '- The frozen Whisper-base.en revision is 911407f4214e0e1d82085af863093ec0b66f9cd6. Encoder pooling averages real-audio frames and excludes padding frames. Extraction uses CUDA bfloat16.',
 '- Acoustic SVM and PCA Fusion SVM are the 24 October development-ranked saved fits. Whisper SVM and Fusion SVM are the 29 October fits, with Fusion using training-ranked acoustic columns. This matches the four-model comparison in the existing report and does not isolate PCA compression.',
 '- Intervals use 2,000 paired whole-speaker bootstrap draws within class, seed 42. The primary comparison is Fusion SVM minus Whisper SVM balanced accuracy. Other comparisons are secondary; no multiple-comparison adjustment is applied.','',
 'Common Voice selected accent counts: '+', '.join(f'{k}: {v:,}' for k,v in cv['accent_counts'].items())+'.','',
 '## Interpretation and limits','']
 if p.ci95_low>0:lines.append('The primary paired interval is above zero, supporting an observed Fusion SVM advantage for these fixed models on this sample.')
 elif p.ci95_high<0:lines.append('The primary paired interval is below zero, supporting an observed Whisper SVM advantage for these fixed models on this sample.')
 else:lines.append('The primary paired interval includes zero; this evaluation does not establish a clear balanced-accuracy difference between Fusion SVM and Whisper SVM.')
 lines+=['','Original test speakers recur, sources remain linked to class, and neither convenience sampling nor these intervals establishes performance on the general speaker population or new recording sources. Intervals exclude model-fitting and selection uncertainty. “Unseen” refers to classifier fitting, not the pretrained Whisper corpus. The new benchmark must not be used for retuning and then presented as untouched.','',
 '## Reproducibility','',
 'See [README.md](README.md) for commands and external source/artifact requirements. `protocol.json` and `model_lock.json` retain pre-prediction decisions. `manifest.csv.gz`, per-group manifests, preparation records and `overlap_audit.json` retain identities and exclusions. `predictions.csv.gz`, `metrics.csv` and `paired_comparisons.csv` retain results. `feature_record.json` and `run_record.json` retain hashes, runtime and saved-model verification. Prepared audio and feature caches remain under `outputs/performance_test_unseen_9000_20261029/`. The earlier benchmark and training reports were not overwritten.','']
 (OUT/'Performance_Test_Report.md').write_text('\n'.join(lines))
 print('Report written')
if __name__=='__main__':main()
