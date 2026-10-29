"""Freeze independently audited manifests before feature extraction or prediction."""
import argparse,collections,hashlib,io,json,multiprocessing,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np,pandas as pd,soundfile as sf
from common import ROOT,OUT,WORK,GROUPS,Inventory,pcm_hash,norm,select,write_json,sha
sys.path.insert(0,str(ROOT))
from mvp.features import extract_features

def feature(item):
 identity,blob,digest=item
 assert pcm_hash(blob)==digest
 wave,sr=sf.read(io.BytesIO(blob),dtype='float32');assert sr==16000 and wave.ndim==1 and 48000<=len(wave)<=96000
 assert np.isfinite(wave).all()
 return identity,extract_features(wave),wave

def freeze():
 inv=Inventory();frames=[];seen_hashes=set();seen_raw=set();seen_utterances=set();audit=[]
 for group in GROUPS:
  d=pd.read_parquet(WORK/f'prepared/{group}.parquet')
  rows=d.to_dict('records')
  # Refill any cross-group duplicate from the same precomputed candidate pool.
  if group=='mnsc_part2':
   rows=[]
   for p in sorted((WORK/'candidates/mnsc_part2').glob('shard_*.parquet')):rows+=pd.read_parquet(p).to_dict('records')
   rows=[r for r in rows if r['pcm_sha256'] not in seen_hashes and r['raw_sha256'] not in seen_raw and (r['speaker_group'],norm(r['sentence'])) not in seen_utterances]
   rows=select(rows,3000,30)
  assert len(rows)==3000
  for r in rows:
   assert inv.eligible(r),r['clip_id']
   assert r['pcm_sha256'] not in seen_hashes and r['raw_sha256'] not in seen_raw
   assert (r['speaker_group'],norm(r['sentence'])) not in seen_utterances
   assert pcm_hash(r['audio_flac'])==r['pcm_sha256']
   seen_hashes.add(r['pcm_sha256']);seen_raw.add(r['raw_sha256']);seen_utterances.add((r['speaker_group'],norm(r['sentence'])))
  d=pd.DataFrame(rows).sort_values('clip_id').reset_index(drop=True)
  d['original_test_speaker']=d.speaker_group.isin(inv.test_speakers)
  d.to_parquet(WORK/f'prepared/{group}_frozen.parquet',index=False)
  meta=d.drop(columns='audio_flac');meta.to_csv(OUT/f'{group}_manifest.csv.gz',index=False,compression={'method':'gzip','mtime':0})
  frames.append(meta)
  audit.append(dict(dataset=group,clips=len(d),speakers=d.speaker_group.nunique(),training_validation_speaker_overlap=len(set(d.speaker_group)&inv.fit_speakers),clips_from_training_validation_speakers=int(d.speaker_group.isin(inv.fit_speakers).sum()),study_clip_id_overlap=int(d.clip_id.isin(inv.study_ids).sum()),old_benchmark_clip_id_overlap=int(d.clip_id.isin(inv.old_ids).sum()),recorded_hash_overlap=sum(any(r.get(k) in inv.hashes for k in ['raw_sha256','raw_pcm_sha256','pcm_sha256','audio_sha256']) for r in rows),source_coordinate_overlap=sum((r.get('source_shard'),r.get('source_row')) in inv.coords for r in rows),speaker_transcript_overlap=sum((r['speaker_group'],norm(r['sentence'])) in inv.utterances for r in rows),original_test_speakers=len(set(d.speaker_group)&inv.test_speakers),clips_from_original_test_speakers=int(d.original_test_speaker.sum()),max_clips_per_speaker=int(d.groupby('speaker_group').size().max()),duration_min=float(d.duration_sec.min()),duration_max=float(d.duration_sec.max()),manifest_sha256=sha(OUT/f'{group}_manifest.csv.gz')))
 truth=pd.concat(frames,ignore_index=True);assert len(truth)==9000 and truth.clip_id.is_unique and truth.pcm_sha256.is_unique
 assert truth.groupby('speaker_group').label.nunique().eq(1).all()
 truth.to_csv(OUT/'manifest.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 write_json(OUT/'overlap_audit.json',dict(status='passed',clips=9000,speakers=truth.speaker_group.nunique(),groups=audit,study_and_old_benchmark_hashes=inv.file_hashes,manifest_sha256=sha(OUT/'manifest.csv.gz'),protocol_sha256=sha(OUT/'protocol.json'),scope='Clip IDs, source coordinates, exact available hashes and normalized speaker-transcript pairs; not perceptual audio fingerprinting. Original test speakers allowed.'))
 print('FROZEN: all exclusions passed',json.dumps(audit),flush=True)
 return truth

def main(groups=None,features_only=False):
 if not features_only: freeze()
 t=time.monotonic();revision='911407f4214e0e1d82085af863093ec0b66f9cd6'
 for group in (groups or GROUPS):
  cache=WORK/f'{group}_features.npz';meta_sha=sha(OUT/f'{group}_manifest.csv.gz')
  if cache.exists():
   with np.load(cache,allow_pickle=False) as z:assert z['manifest_sha256'].item()==meta_sha
   continue
  d=pd.read_parquet(WORK/f'prepared/{group}_frozen.parquet');items=list(zip(d.clip_id,d.audio_flac,d.pcm_sha256));processed=[]
  with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
   for i,r in enumerate(pool.map(feature,items,chunksize=10),1):
    processed.append(r)
    if i%500==0:print(group,'acoustic',i,'elapsed',round(time.monotonic()-t),flush=True)
  acoustic=np.stack([r[1] for r in processed]);waves=[r[2] for r in processed]
  # Import CUDA after CPU extraction. All models receive identical decoded PCM.
  import torch
  from transformers import WhisperFeatureExtractor,WhisperModel
  torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.set_float32_matmul_precision('high')
  assert torch.cuda.is_available();revision='911407f4214e0e1d82085af863093ec0b66f9cd6'
  extractor=WhisperFeatureExtractor.from_pretrained('openai/whisper-base.en',revision=revision,local_files_only=True)
  model=WhisperModel.from_pretrained('openai/whisper-base.en',revision=revision,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa')
  encoder=model.get_encoder().eval().to('cuda');encoder.requires_grad_(False);del model;vectors=[]
  with torch.inference_mode():
   for start in range(0,len(waves),64):
    batch=waves[start:start+64]
    x=extractor(batch,sampling_rate=16000,do_normalize=False,device='cuda',return_tensors='pt').input_features.to(device='cuda',dtype=torch.bfloat16)
    hidden=encoder(input_features=x).last_hidden_state.float();lengths=torch.tensor([(len(w)+319)//320 for w in batch],device='cuda').clamp(1,hidden.shape[1])
    mask=torch.arange(hidden.shape[1],device='cuda')[None,:]<lengths[:,None]
    vectors.append(((hidden*mask[:,:,None]).sum(1)/lengths[:,None]).cpu().numpy())
    if start%640==0:print(group,'Whisper',start+len(batch),flush=True)
  whisper=np.vstack(vectors);assert acoustic.shape==(3000,91) and whisper.shape==(3000,512)
  assert np.isfinite(acoustic).all() and np.isfinite(whisper).all()
  np.savez_compressed(cache,clip_ids=d.clip_id.to_numpy(dtype=str),manifest_sha256=np.asarray(meta_sha),acoustic=acoustic,whisper=whisper)
  del encoder,hidden,x;torch.cuda.empty_cache()
  print(group,'features saved',flush=True)
 if not all((WORK/f'{g}_features.npz').exists() for g in GROUPS): return
 write_json(OUT/'feature_record.json',dict(clips=9000,feature_shapes={'acoustic':[9000,91],'whisper':[9000,512]},whisper_revision=revision,device='CUDA bfloat16',pooling='Mean real 20 ms encoder frames; ceil(samples/320)',feature_code_sha256=sha(ROOT/'mvp/features.py'),cache_hashes={g:sha(WORK/f'{g}_features.npz') for g in GROUPS},seconds=time.monotonic()-t))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--group',choices=GROUPS);p.add_argument('--features-only',action='store_true');a=p.parse_args();main([a.group] if a.group else None,a.features_only)
