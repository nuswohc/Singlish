"""Evaluate predeclared saved models once on the frozen new 9,000-clip set."""
import hashlib,importlib.metadata,json,sys,time
from datetime import datetime,timezone
import joblib,numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,roc_auc_score,brier_score_loss,log_loss
from threadpoolctl import threadpool_limits
from common import ROOT,OUT,WORK,GROUPS,sha,write_json
NAMES=['SVM65','Frozen Whisper (512)','PCA32+65','512+65']
DISPLAY={'SVM65':'Acoustic SVM','Frozen Whisper (512)':'Whisper SVM','PCA32+65':'PCA Fusion SVM','512+65':'Fusion SVM'}
def metrics(f):
 y=f.label.to_numpy();p=f.prediction.to_numpy();prob=f.p_singapore.to_numpy();tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel();both=len(set(y))==2
 return dict(clips=len(f),speakers=f.speaker_group.nunique(),accuracy=accuracy_score(y,p),balanced_accuracy=balanced_accuracy_score(y,p) if both else np.nan,macro_f1=f1_score(y,p,average='macro') if both else np.nan,false_positive=int(fp),false_negative=int(fn),errors=int(fp+fn),singapore_recall=tp/(tp+fn) if tp+fn else np.nan,non_singapore_specificity=tn/(tn+fp) if tn+fp else np.nan,roc_auc=roc_auc_score(y,prob) if both else np.nan,brier_score=brier_score_loss(y,prob),log_loss=log_loss(y,prob,labels=[0,1]))
def main():
 lock=json.loads((OUT/'model_lock.json').read_text())
 assert lock['protocol_sha256']==sha(OUT/'protocol.json') and lock['contracts_sha256']==sha(OUT/'model_contracts.json')
 audit=json.loads((OUT/'overlap_audit.json').read_text());assert audit['status']=='passed'
 assert audit['manifest_sha256']==sha(OUT/'manifest.csv.gz') and audit['protocol_sha256']==sha(OUT/'protocol.json')
 for g in audit['groups']:
  for k in ['training_validation_speaker_overlap','clips_from_training_validation_speakers','study_clip_id_overlap','old_benchmark_clip_id_overlap','recorded_hash_overlap','source_coordinate_overlap','speaker_transcript_overlap']:assert g[k]==0,(g,k)
 frames=[];ab=[];wb=[]
 record=json.loads((OUT/'feature_record.json').read_text())
 for group in GROUPS:
  d=pd.read_csv(OUT/f'{group}_manifest.csv.gz',keep_default_na=False);cache=WORK/f'{group}_features.npz'
  assert sha(cache)==record['cache_hashes'][group]
  with np.load(cache,allow_pickle=False) as z:
   assert z['manifest_sha256'].item()==sha(OUT/f'{group}_manifest.csv.gz');np.testing.assert_array_equal(z['clip_ids'],d.clip_id)
   ab.append(z['acoustic'].copy());wb.append(z['whisper'].copy())
  frames.append(d[['clip_id','speaker_group','label','dataset','accent_group','original_test_speaker']])
 truth=pd.concat(frames,ignore_index=True);acoustic=np.concatenate(ab);whisper=np.concatenate(wb);assert len(truth)==9000 and truth.clip_id.is_unique
 contracts=json.loads((OUT/'model_contracts.json').read_text());pred=[];times={}
 for c in contracts:
  assert sha(c['artifact'])==c['sha256'];model=joblib.load(c['artifact']);np.testing.assert_array_equal(model.classes_,[0,1])
  name=c['model'];cols=c['acoustic_columns']
  x=whisper if not cols else acoustic[:,cols] if name=='SVM65' else np.column_stack([acoustic[:,cols],whisper])
  assert x.shape[1]==c['raw_inputs'];t=time.monotonic()
  with threadpool_limits(limits=4):prob=model.predict_proba(x)[:,1]
  times[name]=time.monotonic()-t;f=truth.copy();f['model']=name;f['prediction']=(prob>=.5).astype(int);f['p_singapore']=prob;pred.append(f)
  print('Scored',name,'errors',int((f.label!=f.prediction).sum()),flush=True)
 predictions=pd.concat(pred,ignore_index=True);predictions.to_csv(OUT/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 sp=truth.groupby('speaker_group',sort=True).label.first();pos=sp.index.get_indexer(truth.speaker_group);weights=np.zeros((2000,len(sp)),dtype=np.int32);rng=np.random.default_rng(42)
 for label in [0,1]:
  ids=np.flatnonzero(sp.to_numpy()==label);draw=ids[rng.integers(0,len(ids),size=(2000,len(ids)))];np.add.at(weights,(np.arange(2000)[:,None],draw),1)
 rows=[];samples={}
 for name,f in zip(NAMES,pred):
  assert f.model.eq(name).all();row=dict(dataset='all',model=name,**metrics(f));cm=np.zeros((len(sp),4),dtype=np.int64);np.add.at(cm,(pos,2*f.label.to_numpy()+f.prediction.to_numpy()),1)
  tn,fp,fn,tp=(weights@cm).T;samples[name]={'balanced_accuracy':.5*(tp/(tp+fn)+tn/(tn+fp)),'macro_f1':.5*(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))}
  for k,v in samples[name].items():row[k+'_ci95_low'],row[k+'_ci95_high']=np.quantile(v,[.025,.975])
  rows.append(row)
  for dataset,g in f.groupby('dataset',sort=False):rows.append(dict(dataset=dataset,model=name,**metrics(g)))
  for accent,g in f.loc[f.dataset.eq('common_voice')].groupby('accent_group'):rows.append(dict(dataset='cv_accent_'+accent,model=name,**metrics(g)))
 result=pd.DataFrame(rows);result.to_csv(OUT/'metrics.csv',index=False);overall=result.loc[result.dataset.eq('all')].set_index('model');paired=[]
 for ref in NAMES[:-1]:
  for metric in ['balanced_accuracy','macro_f1']:
   lo,hi=np.quantile(samples[NAMES[3]][metric]-samples[ref][metric],[.025,.975]);paired.append(dict(candidate=NAMES[3],reference=ref,metric=metric,difference=overall.loc[NAMES[3],metric]-overall.loc[ref,metric],ci95_low=lo,ci95_high=hi))
 pd.DataFrame(paired).to_csv(OUT/'paired_comparisons.csv',index=False)
 for c in contracts:assert sha(c['artifact'])==c['sha256']
 write_json(OUT/'run_record.json',dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),clips=9000,speakers=truth.speaker_group.nunique(),threshold=.5,models_refitted=False,models_deployed=False,model_artifacts_unchanged=True,manifest_sha256=sha(OUT/'manifest.csv.gz'),protocol_sha256=sha(OUT/'protocol.json'),prediction_seconds=times,prediction_seconds_scope='One full-set call; excludes feature extraction; not service latency',code_hashes={p.name:sha(p) for p in OUT.glob('*.py')},packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn','joblib','torch','transformers','librosa','soundfile']}))
 # A compact standalone figure for the report.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(9,4));colours=['#8897a5','#254b66','#c27d20','#16818a']
 for i,n in enumerate(NAMES):
  r=overall.loc[n];v=r.balanced_accuracy*100;lo=r.balanced_accuracy_ci95_low*100;hi=r.balanced_accuracy_ci95_high*100
  ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=4,color=colours[i]);ax.text(1.02,i,f'{v:.3f}% | {int(r.errors)} errors',transform=ax.get_yaxis_transform(),va='center')
 ax.set(yticks=range(4),yticklabels=[DISPLAY[n] for n in NAMES],ylim=(3.5,-.5),xlabel='Balanced accuracy (%) with 95% speaker interval',xlim=(np.floor(overall.balanced_accuracy_ci95_low.min()*200)/2,100));ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
 fig.subplots_adjust(left=.22,right=.72,bottom=.18);fig.savefig(OUT/'performance_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
 print(overall[['balanced_accuracy','errors']].to_string(),flush=True)
if __name__=='__main__':main()
