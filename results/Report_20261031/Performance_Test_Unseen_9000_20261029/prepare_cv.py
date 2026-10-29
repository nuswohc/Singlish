"""Build a new CV26 negative evaluation pool, without consulting classifier scores."""
import collections,hashlib,io,json,sys,tarfile,time
from pathlib import Path
import numpy as np,pandas as pd,soundfile as sf
from common import ROOT,OUT,WORK,Inventory,norm,pcm_hash,select,write_json,sha
sys.path.insert(0,str(ROOT/'accent_classifier_data/scripts'))
from cv_utils import preprocess_waveform,encode_flac
SOURCE=ROOT.parent/'E1_Feature_Engineering/data/non_singaporean_15k_3to6s/_work'
ARCHIVE=ROOT.parent/'singlish-whisper/data/common_voice_download/common-voice-scripted-speech-26-0-english.tar.gz'
GROUPS={'American','British','Australian','Filipino','South Asian','Hong Kong'}
def main():
 inv=Inventory();records=[];ids=set();hashes=set();pairs=set();counts=collections.Counter();reasons=collections.Counter();source_hashes={}
 folder=WORK/'candidates/common_voice';folder.mkdir(parents=True,exist_ok=True)
 meta_path=SOURCE/'cv26_validated_groups.parquet';meta=pd.read_parquet(meta_path)
 # Exclude ambiguous or Singapore-labelled speakers even if a separate row is negative.
 conflict=set(meta.groupby('speaker_id').accent_group.nunique().loc[lambda x:x>1].index)|set(meta.loc[meta.accent_group.eq('Singaporean'),'speaker_id'])
 def accept(r):
  pair=(r['speaker_group'],norm(r['sentence']))
  if r['accent_group'] not in GROUPS or r['speaker_id'] in conflict or not inv.eligible(r):return False
  if r['clip_id'] in ids or r['pcm_sha256'] in hashes or pair in pairs or counts[r['speaker_group']]>=10:return False
  records.append(r);ids.add(r['clip_id']);hashes.add(r['pcm_sha256']);pairs.add(pair);counts[r['speaker_group']]+=1;return True
 for p in sorted((SOURCE/'prepared_candidates').glob('*.parquet')):
  d=pd.read_parquet(p)
  for r in d.to_dict('records'):
   r['clip_id']='common_voice::'+r['clip_id'].removeprefix('common_voice::')
   r['dataset']='common_voice';r['label']=0;r['split']='new_performance';r['provenance_kind']='prepared_reserve';r['candidate_file']=str(p)
   if accept(r):
    assert pcm_hash(r['audio_flac'])==r['pcm_sha256']
    source_hashes[str(p)]=sha(p)
 print('Eligible prepared reserve:',len(records),'speakers',len(counts),flush=True)
 meta=meta.loc[meta.accent_group.isin(GROUPS)&~meta.speaker_id.isin(conflict)&meta.raw_duration_sec.ge(3)].drop_duplicates('path')
 wanted={}
 for r in meta.to_dict('records'):
  r.update(clip_id='common_voice::'+Path(r['path']).stem,source='common_voice',speaker_group='common_voice::'+r['speaker_id'])
  if inv.eligible(r) and r['clip_id'] not in ids:wanted[r['path']]=r
 print('Eligible archive metadata:',len(wanted),flush=True)
 # Reuse committed candidate batches if a previous scan was interrupted.
 for p in sorted(folder.glob('archive_batch_*.parquet')):
  for r in pd.read_parquet(p).to_dict('records'):accept(r)
 buffer=[];batch_no=len(list(folder.glob('archive_batch_*.parquet')));seen_members=0;t=time.monotonic()
 def flush():
  nonlocal batch_no
  if buffer:
   pd.DataFrame(buffer).to_parquet(folder/f'archive_batch_{batch_no:04d}.parquet',index=False);buffer.clear();batch_no+=1
 if len(records)<4500:
  with tarfile.open(ARCHIVE,'r|gz') as archive:
   for member in archive:
    if not member.isfile() or not member.name.endswith('.mp3'):continue
    seen_members+=1;basename=Path(member.name).name
    r=wanted.get(basename)
    if r is not None and r['clip_id'] not in ids and counts[r['speaker_group']]<10:
     blob=archive.extractfile(member).read();raw_sha=hashlib.sha256(blob).hexdigest()
     if raw_sha in inv.hashes:reasons['duplicate_raw']+=1;continue
     wave,sr=sf.read(io.BytesIO(blob),dtype='float32')
     if not np.isfinite(wave).all():reasons['nonfinite']+=1;continue
     prepared,reason=preprocess_waveform(wave,sr,3,6)
     if reason!='ok':reasons[reason]+=1;continue
     encoded=encode_flac(prepared)
     r={**r,'dataset':'common_voice','label':0,'split':'new_performance','sampling_rate':16000,'duration_sec':len(prepared)/16000,'raw_sha256':raw_sha,'raw_pcm_sha256':pcm_hash(blob),'pcm_sha256':pcm_hash(encoded),'audio_sha256':hashlib.sha256(encoded).hexdigest(),'audio_flac':encoded,'source_version':'Common Voice English 26.0','source_member':member.name,'provenance_kind':'raw_archive','candidate_file':str(ARCHIVE)}
     if accept(r):buffer.append(r)
     if len(buffer)>=200:flush();print(f'CV candidates {len(records)}; archive members {seen_members}; {time.monotonic()-t:.0f}s',flush=True)
    if seen_members%50000==0:print(f'Archive scanned {seen_members}; candidates {len(records)}; {time.monotonic()-t:.0f}s',flush=True)
    if len(records)>=4500:break
 flush();chosen=select(records,3000,10);assert len(chosen)==3000
 dest=WORK/'prepared/common_voice.parquet';dest.parent.mkdir(exist_ok=True)
 pd.DataFrame(chosen).sort_values('clip_id').to_parquet(dest,index=False)
 write_json(OUT/'common_voice_preparation.json',dict(candidates=len(records),selected=3000,speakers=len({r['speaker_group'] for r in chosen}),accent_counts=dict(collections.Counter(r['accent_group'] for r in chosen)),archive_members_scanned=seen_members,rejections=dict(reasons),metadata_sha256=sha(meta_path),prepared_reserve_hashes=source_hashes,archive_path=str(ARCHIVE),archive_size=ARCHIVE.stat().st_size,exclusion_hashes=inv.file_hashes,selection='Seeded speaker round-robin; archive-order convenience candidate pool; no model scores'))
 print('Common Voice complete',len(chosen),flush=True)
if __name__=='__main__':main()
