"""Independent checks of final identities and metrics using portable records."""
import csv,gzip,json,collections,hashlib
from pathlib import Path
from common import OUT,ROOT,WORK,sha,write_json,norm

def read(p):
 with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p)) as f:return list(csv.DictReader(f))
def main():
 study=read(ROOT/'accent_classifier_data/manifests/clips_40000.csv.gz');new=read(OUT/'manifest.csv.gz');old=[]
 for p in (ROOT/'results/Perfomance_Test_20261023/manifests').glob('*.csv'):old+=read(p)
 excluded_ids={r['source']+'::'+r['clip_id'] for r in study}|{r['clip_id'] for r in old}
 fit={r['source']+'::'+r['speaker_id'] for r in study if r['split'] in ['train','val','validation']}
 coords={(r['source_shard'],int(float(r['source_row']))) for r in study+old if r['source']=='mnsc'}
 hashes={r[k] for r in study+old for k in ['raw_sha256','decoded_sha256','raw_pcm_sha256','pcm_sha256','audio_sha256'] if r.get(k)}
 utterances={(r['source']+'::'+r['speaker_id'],norm(r['sentence'])) for r in study+old}
 assert len(new)==9000 and collections.Counter(r['dataset'] for r in new)=={'mnsc_part1':3000,'mnsc_part2':3000,'common_voice':3000}
 assert len({r['clip_id'] for r in new})==9000 and len({r['pcm_sha256'] for r in new})==9000
 for r in new:
  assert r['clip_id'] not in excluded_ids and r['speaker_group'] not in fit
  assert (r['speaker_group'],norm(r['sentence'])) not in utterances
  assert all(r.get(k) not in hashes for k in ['raw_sha256','raw_pcm_sha256','pcm_sha256','audio_sha256'])
  if r['source']=='mnsc':assert (r['source_shard'],int(float(r['source_row']))) not in coords
  assert 3<=float(r['duration_sec'])<=6
 truth={r['clip_id']:r for r in new};pred=read(OUT/'predictions.csv.gz');assert len(pred)==36000
 groups=collections.defaultdict(list)
 for r in pred:
  t=truth[r['clip_id']];assert r['label']==t['label'] and r['speaker_group']==t['speaker_group']
  assert int(r['prediction'])==int(float(r['p_singapore'])>=.5)
  groups[r['model']].append(r)
 assert len(groups)==4
 metrics=read(OUT/'metrics.csv');checks=[]
 for model,rs in groups.items():
  assert len(rs)==9000 and len({r['clip_id'] for r in rs})==9000
  for ds in ['all','mnsc_part1','mnsc_part2','common_voice']:
   selected=rs if ds=='all' else [r for r in rs if r['dataset']==ds]
   c=collections.Counter((int(r['label']),int(r['prediction'])) for r in selected);tn,fp,fn,tp=[c[x] for x in [(0,0),(0,1),(1,0),(1,1)]]
   m=next(m for m in metrics if m['model']==model and m['dataset']==ds)
   assert int(m['false_positive'])==fp and int(m['false_negative'])==fn and int(m['errors'])==fp+fn
   assert abs(float(m['accuracy'])-(tn+tp)/len(selected))<1e-12
   if ds=='all':assert abs(float(m['balanced_accuracy'])-.5*(tp/(tp+fn)+tn/(tn+fp)))<1e-12
   checks.append((model,ds))
 for c in json.loads((OUT/'model_contracts.json').read_text()):assert sha(c['artifact'])==c['sha256']
 write_json(OUT/'verification.json',dict(status='passed',clips=9000,predictions=36000,independent_source_id_checks=9000,independent_fit_speaker_checks=9000,independent_source_coordinate_checks=6000,independent_hash_and_utterance_checks=9000,metric_groups_recomputed=len(checks),saved_model_hashes_unchanged=True,manifest_sha256=sha(OUT/'manifest.csv.gz'),predictions_sha256=sha(OUT/'predictions.csv.gz')))
 print('Independent identity, threshold, count, accuracy, BA and model checks passed.')
if __name__=='__main__':main()
