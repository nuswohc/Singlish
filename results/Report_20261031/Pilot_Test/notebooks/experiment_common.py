"""Shared, auditable data and evaluation code for the matched E1/E2/E4 runs.

The notebooks contain model definitions and selection decisions. This module
keeps audio preparation, row order, speaker partitions and reporting identical.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import platform
from pathlib import Path

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
SAMPLE_RATE = 16_000
MAX_SECONDS = 6
MODEL_NAME = 'openai/whisper-base.en'
BOOTSTRAPS = 2_000
CLASS_NAMES = ['Non-Singapore', 'Singapore']
FEATURE_NAMES = (
    [f'mfcc_{i+1}_mean' for i in range(13)]
    + [f'mfcc_{i+1}_std' for i in range(13)]
    + ['pitch_mean', 'pitch_std', 'energy_gated_pitch_fraction']
    + [f'{name}_{stat}' for name in ['centroid', 'bandwidth', 'rolloff', 'zcr', 'rms']
       for stat in ['mean', 'std']]
)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def configure(project_root, data_root):
    project_root, data_root = Path(project_root).resolve(), Path(data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f'Audio directory missing: {data_root}')
    output = project_root / 'outputs' / 'matched_2000_seed42'
    output.mkdir(parents=True, exist_ok=True)
    return {'project': project_root, 'data': data_root, 'output': output}


def prepare_manifest(config):
    rows = []
    for name, label in [('non_singapore', 0), ('singapore', 1)]:
        directory = config['data'] / name
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.rglob('*.wav')):
            relative = path.relative_to(config['data'])
            if len(relative.parts) != 3:
                raise ValueError(f'Expected class/speaker/clip.wav: {path}')
            rows.append({'clip_id': relative.as_posix(), 'path': str(path),
                         'label': label, 'group': f'{name}::{relative.parts[1]}',
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    frame = pd.DataFrame(rows).sort_values('clip_id').reset_index(drop=True)
    if frame.empty or set(frame.label) != {0, 1}:
        raise ValueError('Both classes must contain audio')
    duplicates = frame[frame.duplicated('sha256', keep=False)]
    if not duplicates.empty:
        duplicates.to_csv(config['output'] / 'duplicate_audio.csv', index=False)
        raise ValueError('Duplicate audio found. Audit duplicate_audio.csv before splitting.')
    fingerprint = hashlib.sha256(frame[['clip_id', 'label', 'group', 'sha256']]
                                 .to_csv(index=False).encode()).hexdigest()
    frame.to_csv(config['output'] / 'dataset_manifest.csv', index=False)
    print('Clips:', frame.groupby('label').size().to_dict())
    print('Speakers:', frame.groupby('label')['group'].nunique().to_dict())
    print('Dataset SHA256:', fingerprint)
    return frame, fingerprint


def shared_split(config, frame, fingerprint):
    target = config['output'] / 'speaker_split.npz'
    labels, groups = frame.label.to_numpy(), frame.group.to_numpy(dtype=str)
    if target.exists():
        with np.load(target, allow_pickle=False) as saved:
            if str(saved['fingerprint']) != fingerprint or int(saved['seed']) != SEED:
                raise ValueError('The saved split belongs to different data/settings. Use a new run directory.')
            if not np.array_equal(saved['clip_ids'], frame.clip_id.to_numpy(dtype=str)):
                raise ValueError('Saved clip order differs')
            split = {name: saved[name] for name in ['train', 'validation', 'test']}
    else:
        if min(frame.groupby('label')['group'].nunique()) < 7:
            raise ValueError('At least seven speakers per class are required')
        outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
        trainval, test = next(outer.split(np.zeros(len(frame)), labels, groups))
        inner = StratifiedGroupKFold(5, shuffle=True, random_state=SEED + 1)
        train, validation = next(inner.split(np.zeros(len(trainval)), labels[trainval], groups[trainval]))
        split = {'train': trainval[train], 'validation': trainval[validation], 'test': test}
        np.savez_compressed(target, fingerprint=np.array(fingerprint), seed=np.array(SEED),
                            clip_ids=frame.clip_id.to_numpy(dtype=str), **split)
    assert sorted(np.concatenate(list(split.values())).tolist()) == list(range(len(frame)))
    for name, idx in split.items():
        assert set(labels[idx]) == {0, 1}, name
    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not set(groups[split[left]]) & set(groups[split[right]])
    assignment = np.full(len(frame), '', dtype=object)
    summary = {}
    for name, idx in split.items():
        assignment[idx] = name
        summary[name] = {'clips': len(idx), 'speakers': len(set(groups[idx])),
                         'label_counts': {str(v): int(np.sum(labels[idx] == v)) for v in [0, 1]}}
    frame.assign(partition=assignment).to_csv(config['output'] / 'split_manifest.csv', index=False)
    write_json(config['output'] / 'split_summary.json', {
        'seed': SEED, 'fingerprint': fingerprint, 'speaker_overlap': 0,
        'method': 'First outer grouped fold for test; first inner grouped fold for validation. Not full cross-validation.',
        'partitions': summary})
    print(json.dumps(summary, indent=2))
    print('Speaker overlap: 0. This is one fixed grouped split, not five-fold evaluation.')
    return split


def prepare_audio(path):
    waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if not np.isfinite(waveform).all():
        raise ValueError(f'Non-finite samples: {path}')
    waveform, _ = librosa.effects.trim(waveform, top_db=30)
    if not len(waveform) or np.max(np.abs(waveform)) <= 1e-8:
        raise ValueError(f'No audible signal: {path}')
    waveform = waveform / np.max(np.abs(waveform))
    waveform = waveform[:SAMPLE_RATE * MAX_SECONDS]
    real_samples = len(waveform)
    waveform = librosa.util.fix_length(waveform, size=max(real_samples, SAMPLE_RATE))
    return np.ascontiguousarray(waveform, dtype=np.float32), real_samples


def acoustic_features(waveform):
    def stats(values):
        return np.r_[values.mean(axis=1), values.std(axis=1)]
    options = dict(sr=SAMPLE_RATE, n_fft=1024, hop_length=256)
    mfcc = librosa.feature.mfcc(y=waveform, n_mfcc=13, **options)
    rms = librosa.feature.rms(y=waveform, frame_length=1024, hop_length=256)
    pitch = librosa.yin(waveform, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
                        sr=SAMPLE_RATE, frame_length=2048, hop_length=256)
    energy = rms.ravel()[:len(pitch)]
    # This is an energy gate, not a validated voicing detector.
    usable = np.isfinite(pitch) & (energy > max(float(np.median(energy)) * .1, 1e-4))
    selected = pitch[usable]
    pitch_stats = [selected.mean() if len(selected) else 0., selected.std() if len(selected) else 0., usable.mean()]
    blocks = [librosa.feature.spectral_centroid(y=waveform, **options),
              librosa.feature.spectral_bandwidth(y=waveform, **options),
              librosa.feature.spectral_rolloff(y=waveform, **options),
              librosa.feature.zero_crossing_rate(waveform, frame_length=1024, hop_length=256), rms]
    values = np.r_[stats(mfcc), pitch_stats, *[stats(block) for block in blocks]].astype(np.float32)
    if values.shape != (39,) or not np.isfinite(values).all():
        raise ValueError('Invalid acoustic feature vector')
    return values


def load_features(config, frame, fingerprint, whisper=False):
    """No failed clip is silently removed. All notebooks retain the same rows."""
    feature_hash = hashlib.sha256((inspect.getsource(prepare_audio) + inspect.getsource(acoustic_features)).encode()).hexdigest()
    acoustic_key = f'{fingerprint}|{feature_hash}|sr={SAMPLE_RATE}|seconds={MAX_SECONDS}'
    acoustic_path = config['output'] / 'acoustic_features.npz'
    A = None
    if acoustic_path.exists():
        with np.load(acoustic_path, allow_pickle=False) as cache:
            if str(cache['key']) == acoustic_key:
                A = cache['features']
    if A is None:
        acoustic_rows, real_lengths = [], []
        for i, path in enumerate(frame.path):
            waveform, length = prepare_audio(path)
            acoustic_rows.append(acoustic_features(waveform))
            real_lengths.append(length)
            if (i + 1) % 100 == 0:
                print(f'Acoustic extraction: {i+1}/{len(frame)}', flush=True)
        A = np.stack(acoustic_rows)
        np.savez_compressed(acoustic_path, key=np.array(acoustic_key), features=A,
                            real_samples=np.array(real_lengths), feature_names=np.array(FEATURE_NAMES))
    print('Acoustic features:', A.shape)
    if not whisper:
        return A
    whisper_path = config['output'] / 'whisper_embeddings.npz'
    whisper_key = acoustic_key + f'|{MODEL_NAME}|real_audio_steps_v2|float32'
    W = None
    if whisper_path.exists():
        with np.load(whisper_path, allow_pickle=False) as cache:
            if str(cache['key']) == whisper_key:
                W = cache['embeddings']
    if W is None:
        import torch
        from transformers import WhisperFeatureExtractor, WhisperModel
        torch.manual_seed(SEED)
        torch.set_num_threads(4)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Frozen Whisper:', MODEL_NAME, '| device:', device, flush=True)
        extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
        model = WhisperModel.from_pretrained(MODEL_NAME).to(device)
        encoder = model.get_encoder().eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        batch_size, vectors = (16 if device.type == 'cuda' else 2), []
        with torch.inference_mode():
            for start in range(0, len(frame), batch_size):
                prepared = [prepare_audio(path) for path in frame.path.iloc[start:start+batch_size]]
                features = extractor([row[0] for row in prepared], sampling_rate=SAMPLE_RATE,
                                     return_tensors='pt').input_features.to(device)
                hidden = encoder(input_features=features).last_hidden_state
                for row, (_, real_length) in enumerate(prepared):
                    steps = min(hidden.shape[1], max(1, (real_length + 319) // 320))
                    vectors.append(hidden[row, :steps].mean(dim=0).float().cpu().numpy())
                if start % (batch_size * 10) == 0 or start+batch_size >= len(frame):
                    print(f'Whisper extraction: {min(start+batch_size,len(frame))}/{len(frame)}', flush=True)
        W = np.stack(vectors).astype(np.float32)
        np.savez_compressed(whisper_path, key=np.array(whisper_key), embeddings=W,
                            model_name=np.array(MODEL_NAME),
                            model_commit=np.array(getattr(model.config, '_commit_hash', '') or ''),
                            pooling=np.array('mean over real audio encoder steps; no fine-tuning'))
        del model, encoder
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    assert A.shape == (len(frame), 39) and W.shape == (len(frame), 512)
    assert np.isfinite(A).all() and np.isfinite(W).all()
    print('Whisper features:', W.shape)
    return A, W


def selection_metrics(truth, predicted):
    return {'macro_f1': float(f1_score(truth, predicted, average='macro', zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(truth, predicted))}


def choose_candidate(candidates, X, labels, split):
    """Only train/validation rows enter selection; first candidate breaks ties."""
    rows, fitted = [], []
    train, val = split['train'], split['validation']
    for name, candidate in candidates:
        model = clone(candidate).fit(X[train], labels[train])
        scores = selection_metrics(labels[val], model.predict(X[val]))
        rows.append({'candidate': name, **scores})
        fitted.append(model)
    best = max(range(len(rows)), key=lambda i: (rows[i]['macro_f1'], rows[i]['balanced_accuracy']))
    print(pd.DataFrame(rows).to_string(index=False))
    print('Validation choice:', rows[best]['candidate'])
    return fitted[best], {'selected': rows[best], 'trials': rows}


def predict_scores(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, list(model.classes_).index(1)], True
    return model.decision_function(X), False


def test_metrics(truth, predicted, scores, probability=False):
    result = {'accuracy': float(accuracy_score(truth, predicted)),
              **selection_metrics(truth, predicted),
              'roc_auc': float(roc_auc_score(truth, scores)),
              'errors': int(np.sum(truth != predicted)),
              'confusion_matrix': confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
              'confusion_matrix_labels': [0, 1],
              'per_class': classification_report(truth, predicted, labels=[0, 1],
                  target_names=CLASS_NAMES, output_dict=True, zero_division=0)}
    if probability:
        result['brier_score'] = float(brier_score_loss(truth, scores))
    return result


def bootstrap_metrics(truth, predicted, groups, repetitions=BOOTSTRAPS):
    """Resample whole speakers within each class; retain all their test clips."""
    rng = np.random.default_rng(SEED + 100)
    totals = np.zeros((repetitions, 4), dtype=float)
    for label in [0, 1]:
        speakers = np.unique(groups[truth == label])
        counts = []
        for speaker in speakers:
            idx = groups == speaker
            assert len(np.unique(truth[idx])) == 1
            counts.append(confusion_matrix(truth[idx], predicted[idx], labels=[0, 1]).ravel())
        weights = rng.multinomial(len(speakers), np.full(len(speakers), 1 / len(speakers)), size=repetitions)
        totals += weights @ np.array(counts)
    tn, fp, fn, tp = totals.T
    return {'accuracy': (tn+tp) / (tn+fp+fn+tp),
            'balanced_accuracy': .5 * (tn/(tn+fp) + tp/(tp+fn)),
            'macro_f1': .5 * (2*tn/np.maximum(2*tn+fp+fn, 1) + 2*tp/np.maximum(2*tp+fp+fn, 1))}


def intervals(truth, predicted, groups):
    values = bootstrap_metrics(truth, predicted, groups)
    return {name: [float(v) for v in np.quantile(samples, [.025, .975])]
            for name, samples in values.items()}


def paired_intervals(truth, first, second, groups):
    first_boot = bootstrap_metrics(truth, first, groups)
    second_boot = bootstrap_metrics(truth, second, groups)
    return {name: {'difference': float(selection_metrics(truth, first)[name] - selection_metrics(truth, second)[name]),
                   'ci95': [float(v) for v in np.quantile(first_boot[name]-second_boot[name], [.025,.975])]}
            for name in ['macro_f1', 'balanced_accuracy']}


def environment():
    versions = {}
    for name in ['numpy','scipy','scikit-learn','librosa','torch','transformers','nbclient']:
        versions[name] = importlib.metadata.version(name)
    return {'python': platform.python_version(), 'packages': versions}


def save_result(config, name, frame, fingerprint, split, model, predicted, scores,
                validation, probability=False, extra=None):
    output = config['output'] / name
    output.mkdir(parents=True, exist_ok=True)
    idx = split['test']
    truth, groups = frame.label.to_numpy()[idx], frame.group.to_numpy(dtype=str)[idx]
    metrics = test_metrics(truth, predicted, scores, probability)
    metrics['speaker_bootstrap_ci95'] = intervals(truth, predicted, groups)
    predictions = frame.iloc[idx][['clip_id','path','group','label']].copy()
    predictions['prediction'] = predicted
    predictions['score_class_1'] = scores
    predictions['error'] = truth != predicted
    predictions.to_csv(output / 'test_predictions.csv', index=False)
    predictions[predictions.error].to_csv(output / 'test_errors.csv', index=False)
    report = {'model': name, 'dataset_fingerprint': fingerprint, 'seed': SEED,
              'shared_split': str(config['output'] / 'speaker_split.npz'),
              'positive_label': 1, 'label_meaning': {'0':'Common Voice / non-Singapore', '1':'MNSC / Singapore'},
              'validation': validation, 'test': metrics, 'environment': environment(),
              'uncertainty': f'{BOOTSTRAPS} speaker bootstrap samples, stratified by class. Conditional on this fitted model and split.',
              'limitation': 'Class and corpus are confounded. No independent recording-source test was run.',
              **(extra or {})}
    write_json(output / 'report.json', report)
    if model is not None:
        joblib.dump(model, output / 'model.joblib')
    print(json.dumps({key: value for key,value in metrics.items() if key != 'per_class'}, indent=2))
    plot_confusion(metrics['confusion_matrix'], name, output / 'confusion_matrix.png')
    return report


def plot_confusion(matrix, title, path):
    fig, ax = plt.subplots(figsize=(5,4))
    matrix = np.asarray(matrix)
    ax.imshow(matrix, cmap='Blues', vmin=0)
    for (row,col), value in np.ndenumerate(matrix):
        ax.text(col,row,str(value),ha='center',va='center',color='black',
                bbox=dict(facecolor='white',alpha=.7,edgecolor='none'))
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Non-SG','Singapore'],
           yticklabels=['Non-SG','Singapore'],xlabel='Predicted',ylabel='True',title=title)
    fig.tight_layout()
    fig.savefig(path,dpi=160)
    plt.show()
    plt.close(fig)


def comparison_chart(reports, path, title='Matched test: unseen speakers'):
    fig, ax = plt.subplots(figsize=(9,4.8))
    names = list(reports)
    values = np.array([reports[n]['test']['macro_f1'] for n in names])
    ci = np.array([reports[n]['test']['speaker_bootstrap_ci95']['macro_f1'] for n in names])
    # Percentile intervals need not contain the point estimate. Draw intervals directly.
    positions = np.arange(len(names))
    ax.bar(positions, values, color='steelblue', alpha=.8)
    ax.vlines(positions,ci[:,0],ci[:,1],color='black',linewidth=1.5)
    ax.scatter(positions,ci[:,0],marker='_',color='black')
    ax.scatter(positions,ci[:,1],marker='_',color='black')
    for i,value in enumerate(values):
        ax.text(i,value+.025,f'{value:.3f}',ha='center',fontsize=9)
    ax.set(xticks=positions,xticklabels=names,ylim=(0,1.08),ylabel='Macro F1',title=title)
    ax.tick_params(axis='x',labelrotation=15)
    fig.text(.5,.01,'Intervals: 95% speaker bootstrap. Same test clips for every model.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1))
    fig.savefig(path,dpi=160)
    plt.show()
    plt.close(fig)
