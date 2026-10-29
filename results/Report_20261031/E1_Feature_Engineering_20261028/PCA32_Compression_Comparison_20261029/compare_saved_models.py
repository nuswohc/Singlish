"""Compare saved PCA32+65 and 512+65 pipelines. No training or model export.

Run with the original singlish-py314 environment. Original caches and model
files remain in their recorded source paths. All generated evidence is local
to this script. Scores use decision_function >= 0, before calibration.
"""
from pathlib import Path
from io import BytesIO
from time import perf_counter
import hashlib
import importlib.metadata
import json
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
SOURCE = Path('/home/mh/nusproj/singlish-whisper/outputs')
PCA_SOURCE = SOURCE / 'a1_tuned_acoustic_whisper_fusion_40000'
FULL_SOURCE = SOURCE / 'e4_02_5_frozen_full512_vs_fusion_calibrated_20261029'
NAMES = ['PCA32+65', '512+65']
SEED, REPEATS = 42, 2000

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

a = json.loads((PCA_SOURCE / 'run_record.json').read_text())
b = json.loads((FULL_SOURCE / 'run_record.json').read_text())
assert a['selected_columns'] == b['selected_columns']
assert a['selected_C']['Fusion SVM65+32'] == b['selected_C']['Fusion SVM 512+65 (A2)'] == 10
assert a['whisper']['resolved_revision'] == b['models']['Fusion SVM 512+65 (A2)']['frozen_whisper_revision']
input_hashes = {}
for record in (a, b):
    for path, expected in record['input_hashes'].items():
        if path.endswith(('.npz', 'speaker_splits.csv')):
            assert sha(path) == expected, path
            input_hashes[path] = expected
for path in [PCA_SOURCE/'run_record.json', PCA_SOURCE/'study.joblib', PCA_SOURCE/'predictions.csv',
             FULL_SOURCE/'run_record.json', FULL_SOURCE/'fusion_full512_acoustic65_calibrated.joblib',
             FULL_SOURCE/'test_predictions.csv']:
    input_hashes[str(path)] = sha(path)

models = {
    NAMES[0]: joblib.load(PCA_SOURCE/'study.joblib')['final']['Fusion SVM65+32']['model'],
    NAMES[1]: joblib.load(FULL_SOURCE/'fusion_full512_acoustic65_calibrated.joblib').calibrated_classifiers_[0].estimator,
}
split_path = next(p for p in input_hashes if p.endswith('speaker_splits.csv'))
acoustic_path = next(p for p in input_hashes if p.endswith('acoustic_features_91.npz'))
manifest = pd.read_csv(split_path, keep_default_na=False)
assert len(manifest) == 40000 and manifest.clip_id.is_unique
assert manifest.groupby('speaker_group').split.nunique().eq(1).all()
assert manifest.groupby('speaker_group').label.nunique().eq(1).all()
blocks = []
for path in [acoustic_path, a['whisper']['cache_path']]:
    with np.load(path, allow_pickle=False) as z:
        for key, values in [('clip_ids', manifest.clip_id), ('labels', manifest.label), ('groups', manifest.speaker_group)]:
            np.testing.assert_array_equal(z[key], values)
        blocks.append(z['features'].copy())
X = np.column_stack([blocks[0][:, a['selected_columns']], blocks[1]])
assert X.shape == (40000, 577) and np.isfinite(X).all()
idx = np.flatnonzero(manifest.split.eq('test'))
truth = manifest.iloc[idx][['clip_id', 'speaker_group', 'source', 'label']].reset_index(drop=True)
X = X[idx]
assert len(truth) == 6119 and truth.speaker_group.nunique() == 1482
saved_pca = pd.read_csv(PCA_SOURCE/'predictions.csv')
saved_pca = saved_pca.loc[saved_pca.model.eq('Fusion SVM65+32') & saved_pca.stage.eq('test')]
saved_full = pd.read_csv(FULL_SOURCE/'test_predictions.csv')
saved_full = saved_full.loc[saved_full.model.eq('Fusion SVM 512+65 (A2)') & saved_full['mode'].eq('raw')]

predictions, rows, replay_max_abs_score_error = [], [], {}
with threadpool_limits(limits=4):
    for name, saved in zip(NAMES, [saved_pca, saved_full]):
        saved = saved.set_index('clip_id').loc[truth.clip_id].reset_index()
        for key in ['clip_id', 'speaker_group', 'source', 'label']:
            np.testing.assert_array_equal(saved[key], truth[key])
        score = models[name].decision_function(X)
        predicted = (score >= 0).astype(int)
        # Float32 PCA transforms can differ slightly with BLAS batch layout.
        np.testing.assert_allclose(score, saved.decision_score, rtol=0, atol=1e-6)
        replay_max_abs_score_error[name] = float(np.max(np.abs(score-saved.decision_score.to_numpy())))
        np.testing.assert_array_equal(predicted, saved.prediction)
        f = truth.copy(); f['model'] = name; f['prediction'] = predicted; f['decision_score'] = score
        predictions.append(f)
        tn, fp, fn, tp = confusion_matrix(f.label, predicted, labels=[0, 1]).ravel()
        blob = BytesIO(); joblib.dump(models[name], blob, compress=3)
        rows.append(dict(model=name, clips=len(f), speakers=f.speaker_group.nunique(),
            balanced_accuracy=balanced_accuracy_score(f.label, predicted), accuracy=accuracy_score(f.label, predicted),
            macro_f1=f1_score(f.label, predicted, average='macro'), roc_auc=roc_auc_score(f.label, score),
            false_positive=int(fp), false_negative=int(fn), errors=int(fp+fn),
            svm_inputs=models[name].named_steps['classifier'].support_vectors_.shape[1],
            support_vectors=int(models[name].named_steps['classifier'].n_support_.sum()),
            compressed_pipeline_bytes=blob.tell()))
metrics = pd.DataFrame(rows)
pd.concat(predictions, ignore_index=True).to_csv(OUT/'test_predictions.csv.gz', index=False, compression={'method':'gzip', 'mtime':0})

# Paired, class-stratified whole-speaker resampling. Keep fitted models fixed.
speakers = truth.groupby('speaker_group', sort=True).label.first()
positions = speakers.index.get_indexer(truth.speaker_group)
weights = np.zeros((REPEATS, len(speakers)), dtype=np.int32)
rng = np.random.default_rng(SEED)
for label in [0, 1]:
    ids = np.flatnonzero(speakers.to_numpy() == label)
    draw = ids[rng.integers(0, len(ids), size=(REPEATS, len(ids)))]
    np.add.at(weights, (np.arange(REPEATS)[:, None], draw), 1)
samples = {}
for name, f in zip(NAMES, predictions):
    cm = np.zeros((len(speakers), 4), dtype=np.int64)
    np.add.at(cm, (positions, 2*f.label.to_numpy()+f.prediction.to_numpy()), 1)
    tn, fp, fn, tp = (weights @ cm).T
    samples[name] = {'balanced_accuracy': .5*(tp/(tp+fn)+tn/(tn+fp)),
                     'macro_f1': .5*(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))}
    for key, values in samples[name].items():
        metrics.loc[metrics.model.eq(name), [key+'_ci95_low', key+'_ci95_high']] = np.quantile(values, [.025, .975])
comparisons = []
for key in ['balanced_accuracy', 'macro_f1']:
    low, high = np.quantile(samples[NAMES[1]][key]-samples[NAMES[0]][key], [.025, .975])
    comparisons.append(dict(comparison='512+65 minus PCA32+65', metric=key,
        difference=float(metrics.loc[1, key]-metrics.loc[0, key]), ci95_low=float(low), ci95_high=float(high)))
pd.DataFrame(comparisons).to_csv(OUT/'paired_comparisons.csv', index=False)

# Warm, alternating head-only measurements on identical cached inputs.
latency = []
with threadpool_limits(limits=4):
    for model in models.values():
        model.decision_function(X[:4]); model.decision_function(X[:256])
    for batch_size, repeats in [(1, 60), (256, 5)]:
        starts = np.linspace(0, len(X)-batch_size, repeats, dtype=int)
        for repeat, start in enumerate(starts):
            x = X[start:start+batch_size]
            for name in (NAMES if repeat % 2 == 0 else NAMES[::-1]):
                t = perf_counter(); models[name].decision_function(x)
                latency.append(dict(model=name, batch_size=batch_size, repeat=repeat,
                                    milliseconds=1000*(perf_counter()-t)))
latency = pd.DataFrame(latency)
latency.to_csv(OUT/'latency_samples.csv', index=False)
for name in NAMES:
    for batch_size in [1, 256]:
        times = latency.loc[latency.model.eq(name) & latency.batch_size.eq(batch_size), 'milliseconds']
        metrics.loc[metrics.model.eq(name), f'batch{batch_size}_median_ms'] = times.median()
        metrics.loc[metrics.model.eq(name), f'batch{batch_size}_observed_p95_ms'] = times.quantile(.95)
metrics.to_csv(OUT/'metrics.csv', index=False)

pca = models[NAMES[0]].named_steps['preprocess'].named_transformers_['whisper'].named_steps['pca']
correct = [f.prediction.eq(f.label).to_numpy() for f in predictions]
record = dict(status='complete', method='saved-model replay; no refitting; decision score >= 0',
    seed=SEED, bootstrap_draws=REPEATS, bootstrap_unit='whole speakers within each class; same draws for both models',
    interval_scope='retrospective; fitted-model conditional; excludes training and selection uncertainty',
    controls=dict(encoder=a['whisper']['model'], revision=a['whisper']['resolved_revision'],
                  selected_columns=a['selected_columns'], acoustic_ranking='training only', C=10,
                  train=27797, validation=6084, development=33881, test=6119, test_speakers=1482),
    pca_variance_retained=float(pca.explained_variance_ratio_.sum()),
    pipeline_difference='PCA32 adds PCA and component scaling; Whisper branch weight 1/sqrt(32) versus 1/sqrt(512); gamma=scale follows transformed input',
    prediction_disagreements=int((predictions[0].prediction != predictions[1].prediction).sum()),
    pca_only_correct=int((correct[0] & ~correct[1]).sum()), full_only_correct=int((correct[1] & ~correct[0]).sum()),
    timing=dict(threads=4, singles=60, batches256=5, order='alternating', warm=True,
                scope='preprocessing plus SVM decision_function only; excludes extraction, loading and service; no concurrent training in this script'),
    serialization='joblib.dump each raw pipeline to BytesIO with compress=3; no calibrated wrapper or encoder',
    python=platform.python_version(), host=platform.platform(),
    packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','joblib','threadpoolctl','matplotlib']},
    input_hashes=input_hashes, code_sha256=sha(__file__), replay_verified=True,
    replay_max_abs_score_error=replay_max_abs_score_error, replay_score_atol=1e-6,
    replay_predictions_exact=True)
assert all(sha(path) == expected for path, expected in input_hashes.items())
(OUT/'run_record.json').write_text(json.dumps(record, indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.7))
panels = [('SVM inputs', 'Values after preprocessing', metrics.svm_inputs),
          ('Test errors', 'Errors / 6,119 clips', metrics.errors),
          ('Classifier size', 'Compressed raw pipeline (MiB)', metrics.compressed_pipeline_bytes/2**20),
          ('One cached clip', 'Median classifier time (ms)', metrics.batch1_median_ms)]
for ax, (title, unit, values) in zip(axes.flat, panels):
    values = values.to_numpy()
    ax.barh([1, 0], values, height=.48, color=['#777F88', '#007D84'])
    for y, value in zip([1, 0], values):
        label = f'{int(value):,}' if unit.startswith(('Values', 'Errors')) else f'{value:.2f}'
        ax.text(value+.03*max(values), y, label, va='center', color='#1C2F3D')
    ax.set(yticks=[1, 0], yticklabels=NAMES, xlim=(0, 1.28*max(values)), ylim=(-.65, 1.65), xlabel=unit)
    ax.set_title(title, loc='left', fontweight='bold', fontsize=12)
    ax.grid(axis='x', color='#E4E8EB'); ax.set_axisbelow(True)
fig.subplots_adjust(left=.15, right=.98, bottom=.12, top=.92, hspace=.75, wspace=.55)
fig.savefig(OUT/'compression_tradeoff.png', dpi=180, bbox_inches='tight', pad_inches=.15)
plt.close(fig)
print(metrics.to_string(index=False))
print(pd.DataFrame(comparisons).to_string(index=False))
print('Replay and source hashes verified; disagreements:', record['prediction_disagreements'])
