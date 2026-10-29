"""Prepare new MNSC PART1/PART2 clips using bounded pinned-source byte ranges."""
import argparse,collections,hashlib,io,struct,sys,tempfile,threading,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np,pandas as pd,pyarrow.parquet as pq,requests,soundfile as sf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from common import ROOT,OUT,WORK,REV,SEED,Inventory,norm,pcm_hash,select,write_json
sys.path.insert(0,str(ROOT/'accent_classifier_data/scripts'))
from cv_utils import preprocess_waveform,encode_flac
LOCAL=threading.local()
def http():
 if not hasattr(LOCAL,'http'):
  LOCAL.http=requests.Session();LOCAL.http.mount('https://',HTTPAdapter(max_retries=Retry(total=4,backoff_factor=1,status_forcelist=[429,500,502,503,504])))
 return LOCAL.http
class Remote:
 def __init__(self,part,shard):
  total=772 if part==1 else 728;self.source=f'ASR-PART{part}-Train/train-{shard:05d}-of-{total:05d}.parquet'
  url=f'https://huggingface.co/datasets/AudioLLMs/Multitask-National-Speech-Corpus-v1-extend/resolve/{REV}/{self.source}'
  r=http().get(url+'?new-eval-footer=20261029',headers={'Range':'bytes=-131072'},timeout=(20,90));r.raise_for_status();assert r.status_code==206
  self.size=int(r.headers['Content-Range'].split('/')[-1]);self.url=r.url;tail=r.content
  assert tail[-4:]==b'PAR1';n=struct.unpack('<I',tail[-8:-4])[0];assert n+8<=len(tail)
  self.meta=pq.read_metadata(io.BytesIO(b'PAR1'+tail[-n-8:]));self.f=tempfile.TemporaryFile();self.f.truncate(self.size);self.f.write(b'PAR1');self.f.seek(self.size-len(tail));self.f.write(tail);self.f.flush()
  self.pq=pq.ParquetFile(self.f,metadata=self.meta)
 def read(self,g,audio=False):
  rg=self.meta.row_group(g);cols=[rg.column(i) for i in range(rg.num_columns) if audio or rg.column(i).path_in_schema.startswith('other_attributes.')]
  starts=[c.dictionary_page_offset or c.data_page_offset for c in cols];start=min(starts);end=max(s+c.total_compressed_size for s,c in zip(starts,cols))-1
  r=http().get(self.url,headers={'Range':f'bytes={start}-{end}'},timeout=(20,90));r.raise_for_status()
  assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{self.size}' and len(r.content)==end-start+1
  self.f.seek(start);self.f.write(r.content);self.f.flush()
  return self.pq.read_row_group(g,columns=None if audio else ['other_attributes']).to_pylist()
 def close(self):self.f.close()
def prepare(part,shard,inv):
 folder=WORK/f'candidates/mnsc_part{part}';folder.mkdir(parents=True,exist_ok=True);path=folder/f'shard_{shard:05d}.parquet'
 if path.exists():return pd.read_parquet(path).to_dict('records')
 src=Remote(part,shard);rng=np.random.default_rng(SEED+part*1000+shard);records=[];reasons=collections.Counter();counts=collections.Counter();seen=set()
 offsets=np.cumsum([0]+[src.meta.row_group(g).num_rows for g in range(src.meta.num_row_groups)])
 try:
  for g in rng.permutation(src.meta.num_row_groups):
   metadata=src.read(int(g));eligible=[]
   for pos in rng.permutation(len(metadata)):
    a=metadata[pos]['other_attributes'];s=str((a.get('speaker') or {}).get('speaker_id') or '');t=a.get('Transcription') or '';row=int(offsets[g])+int(pos)
    identity=f'mnsc::mnsc_part{part}_train_{shard:05d}_{row:05d}'
    r=dict(clip_id=identity,source='mnsc',speaker_id='mnsc_'+s,speaker_group='mnsc::mnsc_'+s,sentence=t,source_shard=src.source,source_row=row)
    if not s or not norm(t) or not inv.eligible(r):reasons['excluded_identity_speaker_or_utterance']+=1;continue
    if counts[s]>=30:continue
    eligible.append((int(pos),s,r))
   if not eligible:continue
   audio=src.read(int(g),True)
   for pos,s,r in eligible:
    key=(r['speaker_group'],norm(r['sentence']))
    if counts[s]>=30 or key in seen:continue
    blob=audio[pos]['context']['bytes'];wave,sr=sf.read(io.BytesIO(blob),dtype='float32')
    if not np.isfinite(wave).all():reasons['nonfinite']+=1;continue
    wave,reason=preprocess_waveform(wave,sr,3,6)
    if reason!='ok':reasons[reason]+=1;continue
    encoded=encode_flac(wave)
    r.update(dataset=f'mnsc_part{part}',label=1,accent_group='Singaporean',split='new_performance',sampling_rate=16000,duration_sec=len(wave)/16000,source_revision=REV,source_row_group=int(g),source_channel=str(audio[pos]['other_attributes'].get('channel')),raw_sha256=hashlib.sha256(blob).hexdigest(),raw_pcm_sha256=pcm_hash(blob),pcm_sha256=pcm_hash(encoded),audio_sha256=hashlib.sha256(encoded).hexdigest(),audio_flac=encoded)
    if not inv.eligible(r):reasons['excluded_audio_hash']+=1;continue
    records.append(r);counts[s]+=1;seen.add(key)
    if len(records)>=300:break
   if len(records)>=300:break
 finally:src.close()
 if records:pd.DataFrame(records).to_parquet(path,index=False)
 write_json(path.with_suffix('.json'),dict(part=part,shard=shard,candidates=len(records),rejections=dict(reasons),exclusion_hashes=inv.file_hashes))
 print(f'PART{part} shard {shard}: {len(records)} prepared',flush=True);return records
def main(part,workers):
 inv=Inventory();records=[];order=np.random.default_rng(SEED+part).permutation(772 if part==1 else 728).tolist();scanned=[]
 for start in range(0,len(order),workers):
  batch=order[start:start+workers]
  with ThreadPoolExecutor(max_workers=workers) as ex:
   for shard,rs in zip(batch,ex.map(lambda sh:prepare(part,sh,inv),batch)):records.extend(rs);scanned.append(shard)
  chosen=select(records,3000,30)
  print(f'PART{part}: {len(records)} candidates; {len(chosen)}/3000 selected capacity',flush=True)
  if len(chosen)==3000 and len(records)>=4200:break
 assert len(chosen)==3000
 dest=WORK/f'prepared/mnsc_part{part}.parquet';dest.parent.mkdir(exist_ok=True)
 pd.DataFrame(chosen).sort_values('clip_id').to_parquet(dest,index=False)
 write_json(OUT/f'mnsc_part{part}_preparation.json',dict(candidates=len(records),selected=3000,speakers=len({r['speaker_group'] for r in chosen}),scanned_shards=scanned,seed=SEED+part,exclusion_hashes=inv.file_hashes))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--part',type=int,choices=[1,2],required=True);p.add_argument('--workers',type=int,default=6);a=p.parse_args();main(a.part,a.workers)
