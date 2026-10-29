"""Evaluate four saved classifiers on the same prepared 9,000-clip set.

No fitting, feature extraction, threshold selection or deployment. The two
Whisper/full512 heads are the 29 October fits used in the training report.
"""
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import hashlib
import importlib.metadata
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score, brier_score_loss, log_loss
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
UPSTREAM = ROOT.parent/'E1_Feature_EngineeringV4'
LATEST = ROOT.parent/'singlish-whisper/outputs/e4_02_5_frozen_full512_vs_fusion_calibrated_20261029'
SEED, REPEATS = 42, 2000
NAMES = ['SVM65', 'Frozen Whisper (512)', 'PCA32+65', '512+65']
HASHES = {}

def protect(path, expected=None):
    path = Path(path)
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    if expected is not None:
        assert value == expected, path
    HASHES[str(path)] = value
    return path

def read_json(path):
    return json.loads(protect(path).read_text())

baseline = read_json(ROOT/'results/Perfomance_Test_20261023/metadata.json')
perf24 = read_json(ROOT/'results/Perfomance_Test_20261024/metadata.json')
run29 = read_json(LATEST/'run_record.json')
split_path = protect(UPSTREAM/perf24['study_splits'], perf24['study_splits_sha256'])
study = pd.read_csv(split_path, keep_default_na=False)
assert len(study) == 40000 and study.clip_id.is_unique
assert study.groupby('speaker_group').split.nunique().eq(1).all()
fit = study.loc[study.split.isin(['train', 'validation'])]
fit_ids, fit_speakers = set(fit.clip_id), set(fit.speaker_group)

release_info = [read_json(ROOT/'mvp/models'/f'{stem}_metadata.json')
                for stem in ['svm_65_c_261024', 'fusion_svm_c_261024']]
assert release_info[0]['columns'] == release_info[1]['acoustic_columns_in_91']
columns24 = release_info[0]['columns']
columns29 = run29['selected_columns']
assert len(columns24) == len(columns29) == 65
paths = {
    NAMES[0]: protect(ROOT/'mvp/models/svm_65_c_261024.joblib', perf24['models'][0]['sha256']),
    NAMES[1]: protect(LATEST/'whisper_full512_calibrated.joblib', run29['models']['Whisper full512 SVM']['artifact_sha256']),
    NAMES[2]: protect(ROOT/'mvp/models/fusion_svm_c_261024.joblib', perf24['models'][1]['sha256']),
    NAMES[3]: protect(LATEST/'fusion_full512_acoustic65_calibrated.joblib', run29['models']['Fusion SVM 512+65 (A2)']['artifact_sha256']),
}
models = {name:joblib.load(path) for name,path in paths.items()}
contracts = []
for name,model in models.items():
    np.testing.assert_array_equal(model.classes_, [0,1])
    base = model.calibrated_classifiers_[0].estimator
    assert len(model.calibrated_classifiers_) == 1
    c = base.named_steps['classifier']
    contracts.append(dict(model=name, artifact=str(paths[name]), sha256=HASHES[str(paths[name])],
        fit_date='2026-10-29' if name in [NAMES[1],NAMES[3]] else '2026-10-24',
        raw_inputs=int(model.n_features_in_), svm_inputs=int(c.n_features_in_), C=float(c.C),
        acoustic_ranking='training only' if name==NAMES[3] else ('none' if name==NAMES[1] else 'development'),
        acoustic_columns=[] if name==NAMES[1] else (columns29 if name==NAMES[3] else columns24),
        calibration='sigmoid; five speaker folds; ensemble=False', threshold=0.5))

frames, acoustic_blocks, whisper_blocks = [], [], []
for dataset in ['mnsc_part1_unseen_3000','mnsc_part2_unseen_3000','cv_unseen_3000']:
    settings = baseline['datasets'][dataset]
    manifest_path = protect(ROOT/'results/Perfomance_Test_20261023/manifests'/f'{dataset}.csv', settings['manifest_sha256'])
    manifest = pd.read_csv(manifest_path, keep_default_na=False).sort_values('clip_id').reset_index(drop=True)
    assert len(manifest)==3000 and manifest.clip_id.is_unique
    if dataset == 'mnsc_part2_unseen_3000':
        correction = perf24['preprocessing_correction']
        cache = protect(ROOT/correction['cache_path'], correction['cache_sha256'])
        keys = ['trim_and_normalize','whisper_prepared']
    else:
        cache = protect(ROOT/'tmp/release-comparison-portable-cache'/f'{dataset}.npz', perf24['feature_cache_sha256'][dataset])
        keys = ['acoustic','whisper']
    with np.load(cache, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['clip_ids'], manifest.clip_id)
        assert z['manifest_sha256'].item() == settings['manifest_sha256']
        acoustic,whisper = [z[key].copy() for key in keys]
    assert acoustic.shape==(3000,91) and whisper.shape==(3000,512)
    assert np.isfinite(acoustic).all() and np.isfinite(whisper).all()
    f = manifest[['clip_id','speaker_group','label']].copy()
    f['dataset'] = dataset
    f['fit_clip_id_overlap'] = f.clip_id.isin(fit_ids)
    f['fit_speaker_overlap'] = f.speaker_group.isin(fit_speakers)
    frames.append(f); acoustic_blocks.append(acoustic); whisper_blocks.append(whisper)
truth = pd.concat(frames, ignore_index=True)
acoustic,whisper = np.concatenate(acoustic_blocks),np.concatenate(whisper_blocks)
assert len(truth)==9000 and truth.clip_id.is_unique and truth.speaker_group.nunique()==2684
assert truth.groupby('speaker_group').label.nunique().eq(1).all()
assert truth.fit_clip_id_overlap.sum()==137 and truth.fit_speaker_overlap.sum()==446
X = {NAMES[0]:acoustic[:,columns24], NAMES[1]:whisper,
     NAMES[2]:np.column_stack([acoustic[:,columns24],whisper]),
     NAMES[3]:np.column_stack([acoustic[:,columns29],whisper])}
original = pd.read_csv(protect(ROOT/'results/Perfomance_Test_20261024/predictions.csv'))
original = original.set_index('clip_id').loc[truth.clip_id]
predictions, replay, elapsed = [], {}, {}
with threadpool_limits(limits=4):
    for name,model in models.items():
        t=perf_counter();probability=model.predict_proba(X[name])[:,1];elapsed[name]=perf_counter()-t
        predicted=(probability>=0.5).astype(int)
        f=truth.copy();f['model']=name;f['prediction']=predicted;f['p_singapore']=probability
        predictions.append(f)
        if name in [NAMES[0],NAMES[2]]:
            key='svm65_261024' if name==NAMES[0] else 'fusion65_261024'
            np.testing.assert_array_equal(predicted,original[key])
            difference=float(np.max(np.abs(probability-original[key+'_p_singapore'].to_numpy())))
            assert difference<1e-6
            replay[name]=dict(class_differences=0,max_abs_probability_difference=difference)
        print('Scored',name,flush=True)
predictions=pd.concat(predictions,ignore_index=True)
predictions.to_csv(OUT/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
contracts_path=OUT/'model_contracts.json'
contracts_path.write_text(json.dumps(contracts,indent=2)+'\n')

def scores(f):
    y,p,prob=f.label,f.prediction,f.p_singapore
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    both=y.nunique()==2
    return dict(clips=len(f),speakers=f.speaker_group.nunique(),accuracy=accuracy_score(y,p),
        balanced_accuracy=balanced_accuracy_score(y,p) if both else np.nan,
        macro_f1=f1_score(y,p,average='macro') if both else np.nan,
        false_positive=int(fp),false_negative=int(fn),errors=int(fp+fn),
        singapore_recall=tp/(tp+fn) if tp+fn else np.nan,
        non_singapore_specificity=tn/(tn+fp) if tn+fp else np.nan,
        roc_auc=roc_auc_score(y,prob) if both else np.nan,
        brier_score=brier_score_loss(y,prob),log_loss=log_loss(y,prob,labels=[0,1]))

rows,paired=[] ,[]
subsets={'all_9000':np.ones(len(truth),dtype=bool),
         'exclude_fit_clip_ids':~truth.fit_clip_id_overlap.to_numpy(),
         'exclude_fit_speakers':~truth.fit_speaker_overlap.to_numpy()}
for subset,mask in subsets.items():
    t=truth.loc[mask].reset_index(drop=True)
    sp=t.groupby('speaker_group',sort=True).label.first()
    positions=sp.index.get_indexer(t.speaker_group)
    weights=np.zeros((REPEATS,len(sp)),dtype=np.int32);rng=np.random.default_rng(SEED)
    for label in [0,1]:
        ids=np.flatnonzero(sp.to_numpy()==label)
        draw=ids[rng.integers(0,len(ids),size=(REPEATS,len(ids)))]
        np.add.at(weights,(np.arange(REPEATS)[:,None],draw),1)
    samples={}
    for name in NAMES:
        f=predictions.loc[predictions.model.eq(name)].reset_index(drop=True).loc[mask].reset_index(drop=True)
        np.testing.assert_array_equal(f.clip_id,t.clip_id)
        row=dict(subset=subset,dataset='all',model=name,**scores(f))
        cm=np.zeros((len(sp),4),dtype=np.int64)
        np.add.at(cm,(positions,2*f.label.to_numpy()+f.prediction.to_numpy()),1)
        tn,fp,fn,tp=(weights@cm).T
        samples[name]={'balanced_accuracy':.5*(tp/(tp+fn)+tn/(tn+fp)),
                       'macro_f1':.5*(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))}
        for key,values in samples[name].items():
            lo,hi=np.quantile(values,[.025,.975]);row[key+'_ci95_low']=float(lo);row[key+'_ci95_high']=float(hi)
        rows.append(row)
        for dataset,g in f.groupby('dataset',sort=False):
            rows.append(dict(subset=subset,dataset=dataset,model=name,**scores(g)))
    for reference in NAMES[:-1]:
        for key in ['balanced_accuracy','macro_f1']:
            lo,hi=np.quantile(samples[NAMES[3]][key]-samples[reference][key],[.025,.975])
            candidate=next(r[key] for r in rows if r['subset']==subset and r['dataset']=='all' and r['model']==NAMES[3])
            reference_score=next(r[key] for r in rows if r['subset']==subset and r['dataset']=='all' and r['model']==reference)
            paired.append(dict(subset=subset,candidate=NAMES[3],reference=reference,metric=key,
                difference=candidate-reference_score,ci95_low=float(lo),ci95_high=float(hi)))
metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'metrics.csv',index=False)
pd.DataFrame(paired).to_csv(OUT/'paired_comparisons.csv',index=False)
overlap=[]
for dataset,f in truth.groupby('dataset',sort=False):
    overlap.append(dict(dataset=dataset,clips=len(f),speakers=f.speaker_group.nunique(),
        fit_clip_id_overlap=int(f.fit_clip_id_overlap.sum()),fit_speaker_clips=int(f.fit_speaker_overlap.sum())))
(OUT/'overlap_audit.json').write_text(json.dumps(overlap,indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(8.7,3.8))
overall=metrics.loc[metrics.subset.eq('all_9000') & metrics.dataset.eq('all')].set_index('model')
for i,name in enumerate(NAMES):
    r=overall.loc[name];v=100*r.balanced_accuracy;lo=100*r.balanced_accuracy_ci95_low;hi=100*r.balanced_accuracy_ci95_high
    ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=3,ms=7,color='#007D84' if name=='512+65' else '#777F88')
    ax.text(1.02,i,f'{v:.3f}% · {int(r.errors)} errors',va='center',transform=ax.get_yaxis_transform(),fontsize=11)
lower=np.floor(100*overall.balanced_accuracy_ci95_low.min()*2)/2
ax.set(yticks=range(4),yticklabels=NAMES,xlim=(lower,100),ylim=(3.5,-.5),xlabel='Balanced accuracy (%) · 9,000 prepared clips')
ax.grid(axis='x',color='#E4E8EB');ax.set_axisbelow(True)
fig.subplots_adjust(left=.27,right=.70,bottom=.22,top=.94)
fig.savefig(OUT/'performance_comparison.png',dpi=180,bbox_inches='tight',pad_inches=.16);plt.close(fig)

for path,expected in list(HASHES.items()):
    protect(path,expected)
record=dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
    clips=9000,speakers=2684,threshold=.5,models_refitted=False,models_deployed=False,
    whisper_revision='911407f4214e0e1d82085af863093ec0b66f9cd6',
    feature_protocol='prepared PART1 and Common Voice caches; corrected trim/normalize/PCM16 PART2 acoustic and Whisper cache; all hashes verified',
    calibration='saved sigmoid heads; no calibration fitting on performance data',
    interval_scope='2,000 paired whole-speaker samples within class, seed 42; conditional on fitted models; retrospective test reuse; excludes fitting and selection uncertainty',
    input_hashes=HASHES,source_files_unchanged=True,historical_replay=replay,
    prediction_seconds=elapsed,prediction_seconds_scope='single whole-set call; diagnostic only, not a latency benchmark',
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','joblib','threadpoolctl','matplotlib']})
(OUT/'run_record.json').write_text(json.dumps(record,indent=2)+'\n')
print(overall[['balanced_accuracy','accuracy','macro_f1','false_positive','false_negative','errors']].to_string())
print(pd.DataFrame(paired).loc[lambda f:f.reference.eq(NAMES[1]) & f.metric.eq('balanced_accuracy')].to_string(index=False))
