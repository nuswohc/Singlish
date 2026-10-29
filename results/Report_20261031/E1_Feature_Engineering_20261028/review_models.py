"""Review the saved 24 October PCA32 fusion and 28 October full512 fusion.
No training, threshold selection, deployment or source-file changes.
Run with /home/mh/miniconda3/envs/singlish-py314/bin/python review_models.py
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='4'
from pathlib import Path
from time import perf_counter
from datetime import datetime,timezone
import hashlib,json,platform,sys
import joblib,numpy as np,pandas as pd,sklearn
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score,brier_score_loss,log_loss,confusion_matrix
from threadpoolctl import threadpool_limits,threadpool_info
ROOT=Path('/home/mh/nusproj/Singlish')
STUDY=ROOT.parent/'E1_Feature_EngineeringV4'
HERE=Path(__file__).resolve().parent
OUT=HERE/'comparison';OUT.mkdir(exist_ok=True)
OLD=STUDY/'outputs/svm65_fusion_comparison_40000'
NEW=STUDY/'outputs/svm65_full_whisper_fusion_comparison_40000'
LABELS=['24 Oct PCA32+65','28 Oct full512+65']
paths=[ROOT/'mvp/models/fusion_svm_c_261024.joblib',NEW/'calibrated_probability_testing/fusion_svm_calibrated.joblib']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source_hashes={}
def protect(p):
    p=Path(p);source_hashes[str(p)]=sha(p);return p

def readjson(p):return json.loads(protect(p).read_text())
def csv(p,**kw):return pd.read_csv(protect(p),keep_default_na=False,**kw)
def writejson(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def score(frame,pred,prob=None,scores=None):
    y=frame.true_label.to_numpy();pred=np.asarray(pred)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    result=dict(clips=len(y),speakers=frame.speaker_group.nunique(),accuracy=float(accuracy_score(y,pred)),
        false_positive=int(fp),false_negative=int(fn),errors=int(fp+fn))
    if len(set(y))==2:
        result.update(balanced_accuracy=float(balanced_accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro')),
            singapore_recall=float(tp/(tp+fn)),non_singapore_specificity=float(tn/(tn+fp)))
        ranking=prob if prob is not None else scores
        if ranking is not None:result['roc_auc']=float(roc_auc_score(y,ranking))
    else:
        result['singapore_recall' if y[0]==1 else 'non_singapore_specificity']=float(accuracy_score(y,pred))
    if prob is not None:
        result['brier_score']=float(brier_score_loss(y,prob))
        result['log_loss']=float(log_loss(y,prob,labels=[0,1]))
    return result

def paired(frame,old_pred,new_pred,old_p=None,new_p=None):
    # Sample whole speakers within each class. Retain all clips of each sampled speaker.
    assert frame.groupby('speaker_group').true_label.nunique().eq(1).all()
    groups=frame.groupby('speaker_group',sort=True).true_label.first()
    positions=groups.index.get_indexer(frame.speaker_group)
    old=np.asarray(old_pred)==frame.true_label.to_numpy();new=np.asarray(new_pred)==frame.true_label.to_numpy()
    G=len(groups)
    counts=np.bincount(positions,minlength=G)
    delta=np.bincount(positions,weights=new.astype(float)-old.astype(float),minlength=G)
    rng=np.random.default_rng(42);draws=np.zeros(2000);estimate=0.
    for label in [0,1]:
        rows=np.flatnonzero(groups.to_numpy()==label)
        ind=rng.integers(0,len(rows),size=(2000,len(rows)))
        draws+=.5*delta[rows][ind].sum(axis=1)/counts[rows][ind].sum(axis=1)
        estimate+=.5*delta[rows].sum()/counts[rows].sum()
    low,high=np.quantile(draws,[.025,.975])
    result=dict(ba_gain_pp=100*estimate,ba_gain_ci95_pp=[float(100*low),float(100*high)],
        candidate_only_correct=int((new&~old).sum()),release_only_correct=int((old&~new).sum()),
        prediction_disagreements=int((new!=old).sum()),bootstrap_repeats=2000,seed=42,
        method='paired class-stratified whole-speaker bootstrap; clip-balanced BA; fixed fitted models')
    if old_p is not None:
        y=frame.true_label.to_numpy()
        result['brier_change']=float(np.mean((new_p-y)**2-(old_p-y)**2))
        result['log_loss_change']=float(log_loss(y,new_p)-log_loss(y,old_p))
    return result

started=perf_counter()
metadata=[readjson(x/'model_metadata.json') for x in [OLD,NEW]]
cal_meta=[readjson(ROOT/'mvp/models/fusion_svm_c_261024_metadata.json'),readjson(NEW/'calibrated_probability_testing/fusion_svm_calibrated_metadata.json')]
assert metadata[0]['dataset_id']==metadata[1]['dataset_id']
assert metadata[0]['feature_columns_in_91']==metadata[1]['feature_columns_in_91']
assert metadata[0]['whisper_provenance']==metadata[1]['whisper_provenance']
assert sha(OLD/'speaker_splits.csv')==sha(NEW/'speaker_splits.csv')
release_info=readjson(ROOT/'results/Perfomance_Test_20261024/metadata.json')
assert sha(paths[0])==next(x['sha256'] for x in release_info['models'] if x['id']=='fusion65_261024')
assert sha(paths[0])==sha(OLD/'calibrated_probability_testing/fusion_svm_calibrated.joblib')
models=[joblib.load(protect(p)) for p in paths]
bases=[m.calibrated_classifiers_[0].estimator for m in models]
for m,info in zip(models,cal_meta):
    assert m.e4_probability_metadata_==info
    assert m.n_features_in_==577 and m.classes_.tolist()==[0,1]
for width,b in zip([97,577],bases):assert b.named_steps['classifier'].n_features_in_==width
columns=metadata[0]['feature_columns_in_91']
split=csv(NEW/'speaker_splits.csv',dtype={'clip_id':str,'speaker_group':str})
assert len(split)==40000 and split.clip_id.is_unique
assert split.groupby('speaker_group').split.nunique().eq(1).all()
assert split.groupby('speaker_group').label.nunique().eq(1).all()
test_idx=np.flatnonzero(split.split.eq('test'));assert len(test_idx)==6119
ac_path=protect(STUDY/'outputs/e1_feature_study_accent_40000/acoustic_features_91.npz')
w_path=protect(OLD/'feature_cache'/('whisper_'+metadata[0]['whisper_provenance']['cache_id']+'.npz'))
with np.load(ac_path,allow_pickle=False) as z:
    np.testing.assert_array_equal(z['clip_ids'],split.clip_id);np.testing.assert_array_equal(z['labels'],split.label)
    np.testing.assert_array_equal(z['groups'],split.speaker_group)
    X91=z['features'].copy();assert z['feature_names'][columns].tolist()==metadata[0]['feature_names']
with np.load(w_path,allow_pickle=False) as z:
    np.testing.assert_array_equal(z['clip_ids'],split.clip_id);np.testing.assert_array_equal(z['labels'],split.label)
    np.testing.assert_array_equal(z['groups'],split.speaker_group)
    W=z['features'].copy()
X=np.column_stack([X91[test_idx][:,columns].astype(np.float32),W[test_idx]])
truth=split.iloc[test_idx][['clip_id','speaker_group','source','label']].rename(columns={'label':'true_label'}).reset_index(drop=True)
assert np.isfinite(X).all() and X.shape==(6119,577)
paired_test=truth.copy();test_metrics=[];source_scores=[];replay=[];model_contract=[]
with threadpool_limits(limits=4):
    for i,(label,m,base,folder) in enumerate(zip(LABELS,models,bases,[OLD,NEW])):
        saved=csv(folder/'calibrated_probability_testing/test_probability_predictions.csv').set_index('clip_id').loc[truth.clip_id]
        np.testing.assert_array_equal(saved.true_label,truth.true_label);np.testing.assert_array_equal(saved.speaker_group,truth.speaker_group)
        t=perf_counter();p=m.predict_proba(X)[:,1];elapsed=perf_counter()-t
        pred=(p>=.5).astype(int)
        np.testing.assert_allclose(p,saved.p_singapore,rtol=1e-8,atol=1e-6)
        np.testing.assert_array_equal(pred,saved.predicted_class)
        # In sigmoid calibration, raw scores can be recovered from the monotone mapping,
        # but replay the estimator explicitly to verify the saved raw classifications.
        raw_scores=base.decision_function(X);raw_pred=(raw_scores>=0).astype(int)
        np.testing.assert_allclose(raw_scores,saved.original_decision_score,rtol=1e-8,atol=1e-5)
        np.testing.assert_array_equal(raw_pred,saved.original_svm_prediction)
        for mode,prediction,prob,scores in [('calibrated',pred,p,None),('raw',raw_pred,None,raw_scores)]:
            test_metrics.append(dict(model=label,mode=mode,**score(truth,prediction,prob,scores)))
        prefix='release' if i==0 else 'candidate'
        paired_test[prefix+'_prediction']=pred;paired_test[prefix+'_p_singapore']=p;paired_test[prefix+'_raw_prediction']=raw_pred
        for src,idx in truth.groupby('source').groups.items():
            source_scores.append(dict(model=label,source=src,**score(truth.iloc[idx],pred[idx],p[idx])))
        replay.append(dict(model=label,probability_max_abs_difference=float(np.max(np.abs(p-saved.p_singapore))),class_differences=int(np.sum(pred!=saved.predicted_class)),raw_score_max_abs_difference=float(np.max(np.abs(raw_scores-saved.original_decision_score))),test_predict_proba_seconds=elapsed))
        svm=base.named_steps['classifier']
        model_contract.append(dict(model=label,artifact=str(paths[i]),artifact_sha256=sha(paths[i]),artifact_bytes=paths[i].stat().st_size,input_width=m.n_features_in_,processed_width=svm.n_features_in_,C=float(svm.C),support_vectors=int(svm.n_support_.sum()),support_vector_bytes=int(svm.support_vectors_.nbytes),calibration_folds=5,branch_weights=base.named_steps['preprocess'].transformer_weights))
        print('Internal replay:',label,'passed;',round(elapsed,3),'seconds',flush=True)
    test_pair=paired(truth,paired_test.release_prediction,paired_test.candidate_prediction,paired_test.release_p_singapore.to_numpy(),paired_test.candidate_p_singapore.to_numpy())
    test_pair['raw_models']=paired(truth,paired_test.release_raw_prediction,paired_test.candidate_raw_prediction)
    # Reference E2 control from the completed A1 comparison (raw SVM threshold zero).
    e2_path=ROOT.parent/'singlish-whisper/outputs/a1_tuned_acoustic_whisper_fusion_40000/predictions.csv'
    e2=csv(e2_path);e2=e2.loc[e2.model.eq('Whisper full512 SVM')].set_index('clip_id').loc[truth.clip_id]
    np.testing.assert_array_equal(e2.label,truth.true_label)
    test_pair['raw_candidate_vs_frozen_whisper_only']=paired(truth,e2.prediction.to_numpy(),paired_test.candidate_raw_prediction)
    # Fixed benchmark: preserve the previous metadata, cache hashes and preparation.
    historical=csv(ROOT/'results/Perfomance_Test_20261024/predictions.csv')
    prior=readjson(ROOT/'results/Perfomance_Test_20261023/metadata.json')
    fit_speakers=set(split.loc[split.split.isin(['train','validation']),'speaker_group'])
    fit_ids=set(split.loc[split.split.isin(['train','validation']),'clip_id'])
    benchmark=[];overlap={}
    for dataset,settings in prior['datasets'].items():
        manifest_path=ROOT/'results/Perfomance_Test_20261023/manifests'/(dataset+'.csv')
        assert sha(manifest_path)==settings['manifest_sha256']
        manifest=csv(manifest_path).sort_values('clip_id').reset_index(drop=True)
        cache_path=protect(ROOT/'tmp/release-comparison-portable-cache'/(dataset+'.npz'))
        assert sha(cache_path)==release_info['feature_cache_sha256'][dataset]
        with np.load(cache_path,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['clip_ids'],manifest.clip_id)
            ac,wh=z['acoustic'].copy(),z['whisper'].copy()
        if dataset=='mnsc_part2_unseen_3000':
            prep_path=protect(ROOT/release_info['preprocessing_correction']['cache_path'])
            assert sha(prep_path)==release_info['preprocessing_correction']['cache_sha256']
            with np.load(prep_path,allow_pickle=False) as z:
                ids=z['clip_ids'].tolist();order=[ids.index(x) for x in manifest.clip_id]
                ac=z['trim_and_normalize'][order];wh=z['whisper_prepared'][order]
        values=np.column_stack([ac[:,columns].astype(np.float32),wh]);assert values.shape==(3000,577)
        frame=manifest[['clip_id','speaker_group']].copy();frame['source']=dataset;frame['true_label']=int(settings['label'])
        frame['fit_speaker_independent']=~frame.speaker_group.isin(fit_speakers)
        previous=historical.loc[historical.dataset.eq(dataset)].set_index('clip_id').loc[frame.clip_id]
        for i,(label,m) in enumerate(zip(LABELS,models)):
            p=m.predict_proba(values)[:,1];pred=(p>=.5).astype(int)
            if i==0:
                np.testing.assert_allclose(p,previous.fusion65_261024_p_singapore,rtol=1e-8,atol=1e-6)
                np.testing.assert_array_equal(pred,previous.fusion65_261024)
            prefix='release' if i==0 else 'candidate'
            frame[prefix+'_prediction']=pred;frame[prefix+'_p_singapore']=p
        overlap[dataset]=dict(clips=len(frame),speakers=frame.speaker_group.nunique(),fit_speaker_clips=int((~frame.fit_speaker_independent).sum()),fit_clip_id_overlap=len(set(frame.clip_id)&fit_ids),all_study_clip_id_overlap=len(set(frame.clip_id)&set(split.clip_id)),speakers_by_split={s:len(set(frame.speaker_group)&set(split.loc[split.split.eq(s),'speaker_group'])) for s in ['train','validation','test']})
        benchmark.append(frame);print('Benchmark:',dataset,'passed',flush=True)
    bench=pd.concat(benchmark,ignore_index=True);bench_scores=[];bench_pairs={}
    for subset,frame in [('all_9000',bench),('fit_speaker_independent',bench.loc[bench.fit_speaker_independent])]:
        bench_pairs[subset]=paired(frame,frame.release_prediction,frame.candidate_prediction,frame.release_p_singapore.to_numpy(),frame.candidate_p_singapore.to_numpy())
        for src,f in [('all',frame)]+list(frame.groupby('source')):
            for prefix,label in zip(['release','candidate'],LABELS):
                bench_scores.append(dict(subset=subset,source=src,model=label,**score(f,f[prefix+'_prediction'],f[prefix+'_p_singapore'])))
    # Latency: cached raw features; warm pipelines, alternating order, fixed thread count.
    sample_idx=np.linspace(0,len(X)-1,256,dtype=int);batch=X[sample_idx]
    for m in models:m.predict_proba(batch[:4])
    timings=[]
    for repeat in range(5):
        for i in ([0,1] if repeat%2==0 else [1,0]):
            t=perf_counter();models[i].predict_proba(batch);elapsed=perf_counter()-t
            timings.append(dict(model=LABELS[i],mode='batch_256',repeat=repeat,clips=256,seconds=elapsed))
    for j,idx in enumerate(sample_idx[:60]):
        for i in ([0,1] if j%2==0 else [1,0]):
            t=perf_counter();models[i].predict_proba(X[idx:idx+1]);elapsed=perf_counter()-t
            timings.append(dict(model=LABELS[i],mode='single_clip',repeat=j,clips=1,seconds=elapsed))
    timing_frame=pd.DataFrame(timings);timing_summary=[]
    for (label,mode),part in timing_frame.groupby(['model','mode'],sort=False):
        ms=1000*part.seconds.to_numpy();timing_summary.append(dict(model=label,mode=mode,measurements=len(part),median_ms=float(np.median(ms)),p95_ms=float(np.quantile(ms,.95)),median_clips_per_second=float((part.clips/part.seconds).median())))
paired_test.to_csv(OUT/'internal_predictions.csv',index=False)
pd.DataFrame(test_metrics).to_csv(OUT/'internal_metrics.csv',index=False)
pd.DataFrame(source_scores).to_csv(OUT/'internal_source_metrics.csv',index=False)
bench.to_csv(OUT/'benchmark_predictions.csv',index=False)
pd.DataFrame(bench_scores).to_csv(OUT/'benchmark_metrics.csv',index=False)
timing_frame.to_csv(OUT/'latency_samples.csv',index=False)
pd.DataFrame(timing_summary).to_csv(OUT/'latency_summary.csv',index=False)
writejson(OUT/'paired_comparisons.json',dict(internal=test_pair,benchmark=bench_pairs))
writejson(OUT/'model_contracts.json',model_contract)
writejson(OUT/'benchmark_overlap.json',overlap)
# Check the existing MVP loader without changing files or the service environment.
for runtime_path in ['mvp/models.py','mvp/models/manifest.json']:
    protect(ROOT/runtime_path)
sys.path.insert(0,str(ROOT))
from mvp.models import EnsembleModel
previous_override=os.environ.get('ENSEMBLE_MODEL_PATH')
os.environ['ENSEMBLE_MODEL_PATH']=str(paths[1])
try:
    candidate_runtime=EnsembleModel()
    result=candidate_runtime.predict(X91[test_idx[0]],W[test_idx[0]])
    assert result['class_id']==int(paired_test.candidate_prediction.iloc[0])
    expected_confidence=float(paired_test.candidate_p_singapore.iloc[0])
    if result['class_id']==0:expected_confidence=1-expected_confidence
    assert abs(result['probability']-expected_confidence)<1e-12
finally:
    if previous_override is None:os.environ.pop('ENSEMBLE_MODEL_PATH',None)
    else:os.environ['ENSEMBLE_MODEL_PATH']=previous_override
runtime_compatibility=dict(status='passed',loader='mvp.models.EnsembleModel',
    file_changes=False,service_changes=False,scope='Artifact, sidecar, feature contract and one cached prediction only; not an end-to-end service test')

for p,expected in source_hashes.items():assert sha(p)==expected,p
record=dict(completed_utc=datetime.now(timezone.utc).isoformat(),status='passed',source_files_unchanged=True,
    internal_replay=replay,runtime_compatibility=runtime_compatibility,benchmark_release_prediction_differences=0,source_hashes=source_hashes,
    environment=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,processor=platform.processor(),platform=platform.platform(),cpu_count=os.cpu_count(),blas=threadpool_info()),
    timing_method='Warm cached-feature predict_proba; four BLAS threads; alternating order; five 256-clip batches and 60 single requests per model. No audio preparation, Whisper extraction, networking or concurrency included.',
    seconds=perf_counter()-started,script_sha256=sha(__file__),threshold=.5,models_refitted=False)
writejson(OUT/'verification.json',record)
print('PASS: replay, paired comparison, fixed benchmark and matched head timing complete.',flush=True)
print(json.dumps(dict(internal=test_pair,benchmark=bench_pairs,timing=timing_summary),indent=2),flush=True)
