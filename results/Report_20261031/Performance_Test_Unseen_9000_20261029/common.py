"""Shared paths and identity checks for a frozen, speaker-held-out evaluation."""
from pathlib import Path
import csv,gzip,hashlib,io,json,re
import numpy as np
import pandas as pd
import soundfile as sf
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
WORK=ROOT/'outputs/performance_test_unseen_9000_20261029'
OLD=ROOT/'results/Perfomance_Test_20261023/manifests'
REV='fb35c8c24be1056293c511ae47177e915a3ed757'
SEED=20261029
GROUPS=['mnsc_part1','mnsc_part2','common_voice']
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def norm(t):return ' '.join(re.findall(r"[a-z]+(?:'[a-z]+)?",str(t).lower()))
def pcm_hash(blob):
 pcm,sr=sf.read(io.BytesIO(blob),dtype='int16',always_2d=True)
 return hashlib.sha256(str(sr).encode()+pcm.astype('<i2').tobytes()).hexdigest()
def order(value):return hashlib.sha256(f'{SEED}:{value}'.encode()).hexdigest()
def write_json(p,x):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
class Inventory:
 def __init__(self):
  self.paths=[ROOT/'accent_classifier_data/manifests/clips_40000.csv.gz',*sorted(OLD.glob('*.csv'))]
  s=pd.read_csv(self.paths[0],dtype=str,keep_default_na=False)
  self.study_ids=set(s.source+'::'+s.clip_id)
  self.fit_speakers=set((s.source+'::'+s.speaker_id)[s.split.ne('test')])
  self.test_speakers=set((s.source+'::'+s.speaker_id)[s.split.eq('test')])
  self.ids=set(self.study_ids);self.old_ids=set();self.hashes=set();self.utterances=set();self.coords=set()
  for r in s.to_dict('records'):
   self.utterances.add((r['source']+'::'+r['speaker_id'],norm(r['sentence'])))
   for k in ['raw_sha256','decoded_sha256']:
    if r[k]:self.hashes.add(r[k])
   if r['source']=='mnsc':self.coords.add((r['source_shard'],int(float(r['source_row']))))
  for p in self.paths[1:]:
   d=pd.read_csv(p,dtype=str,keep_default_na=False)
   for r in d.to_dict('records'):
    self.ids.add(r['clip_id']);self.old_ids.add(r['clip_id'])
    self.utterances.add((r['speaker_group'],norm(r['sentence'])))
    for k in ['raw_sha256','raw_pcm_sha256','pcm_sha256','audio_sha256']:
     if r.get(k):self.hashes.add(r[k])
    if r['source']=='mnsc':self.coords.add((r['source_shard'],int(float(r['source_row']))))
  self.file_hashes={str(p.relative_to(ROOT)):sha(p) for p in self.paths}
 def eligible(self,r):
  return (r['speaker_group'] not in self.fit_speakers and r['clip_id'] not in self.ids
   and (r['speaker_group'],norm(r['sentence'])) not in self.utterances
   and (r.get('source_shard'),r.get('source_row')) not in self.coords
   and not any(r.get(k) in self.hashes for k in ['raw_sha256','raw_pcm_sha256','pcm_sha256','audio_sha256']))
def select(records,target=3000,cap=30):
 # Seeded speaker round-robin prevents a few prolific speakers dominating.
 buckets={};seen_ids=set()
 for r in records:
  if r['clip_id'] in seen_ids:continue
  seen_ids.add(r['clip_id']);buckets.setdefault(r['speaker_group'],[]).append(r)
 for rows in buckets.values():rows.sort(key=lambda r:order(r['clip_id']))
 speakers=sorted(buckets,key=order);chosen=[];hashes=set();utterances=set();counts={s:0 for s in speakers}
 for roundno in range(max(map(len,buckets.values()),default=0)):
  for s in speakers:
   if roundno>=len(buckets[s]) or counts[s]>=cap:continue
   r=buckets[s][roundno];u=(s,norm(r['sentence']))
   if r['pcm_sha256'] in hashes or u in utterances:continue
   hashes.add(r['pcm_sha256']);utterances.add(u);chosen.append(r);counts[s]+=1
   if len(chosen)==target:return chosen
 return chosen
