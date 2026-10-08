"""Compare serving features/predictions with real held-out study recordings."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from fastapi.testclient import TestClient
from mvp.app import app
from mvp.audio import decode_audio
from mvp.features import extract_features

parser = argparse.ArgumentParser()
parser.add_argument('study_root', type=Path, help='Directory containing data/ and outputs/e1_feature_study/')
parser.add_argument('--output', type=Path)
args = parser.parse_args()
output = args.study_root / 'outputs/e1_feature_study'
with np.load(output / 'acoustic_features_39.npz', allow_pickle=False) as cache:
    features = cache['features'].copy()
    ids = cache['clip_ids'].tolist()
with (output / 'speaker_splits.csv').open() as f:
    rows = list(csv.DictReader(f))
records = []
with TestClient(app) as client:
    for label in ['0','1']:
        candidates = [r for r in rows if r['split'] == 'test' and r['label'] == label][:3]
        assert len(candidates) == 3
        for row in candidates:
            path = args.study_root / 'data' / row['clip_id']
            data = path.read_bytes()
            waveform, metadata, _ = decode_audio(data)
            vector = extract_features(waveform)
            cached = features[ids.index(row['clip_id'])]
            np.testing.assert_allclose(vector, cached, rtol=1e-5, atol=1e-6)
            expected = int(client.app.state.model.pipeline.predict(cached.reshape(1,-1))[0])
            started = perf_counter()
            response = client.post('/api/classify', files={'audio': (path.name,data)})
            elapsed = perf_counter() - started
            assert response.status_code == 200, response.text
            result = response.json()
            assert result['prediction']['class_id'] == expected
            records.append({'clip_id':row['clip_id'], 'true_class':int(label), 'predicted_class':expected,
                'feature_max_abs_difference':float(np.max(np.abs(vector-cached))), 'warm_request_seconds':round(elapsed,4)})
report = {'checked_recordings':len(records),'feature_rtol':1e-5,'feature_atol':1e-6,'records':records}
if args.output:
    args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
