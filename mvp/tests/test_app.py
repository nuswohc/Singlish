from io import BytesIO
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from mvp.app import app
from mvp.audio import MAX_BYTES, decode_audio, AudioError
from mvp.features import extract_features


def wav(seconds=3, rate=16000, channels=1, subtype="PCM_16", silent=False, format="WAV"):
    t = np.arange(int(seconds * rate)) / rate
    y = np.zeros_like(t) if silent else (0.15 * np.sin(2 * np.pi * 180 * t))
    if channels > 1:
        y = np.tile(y[:, None], (1, channels))
    stream = BytesIO()
    sf.write(stream, y, rate, format=format, subtype=subtype)
    return stream.getvalue()

@pytest.fixture(scope="module")
def client():
    # API tests exercise the real saved SVMs without downloading/running Whisper.
    import mvp.embeddings as embeddings
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(embeddings, 'WhisperEmbedding',
                      lambda *args: lambda waveform: np.zeros(512, dtype=np.float32))
        with TestClient(app) as client:
            yield client

@pytest.mark.parametrize("rate,channels,seconds,format", [(16000, 1, 3, "WAV"), (44100, 2, 6, "FLAC")])
def test_inference_matches_pipeline(client, rate, channels, seconds, format):
    data = wav(seconds, rate, channels, format=format)
    response = client.post('/api/classify', files={'audio': ('clip.' + format.lower(), data)})
    assert response.status_code == 200, response.text
    result = response.json()
    y, metadata, warnings = decode_audio(data)
    vector = extract_features(y)
    ml = client.app.state.models.ml
    values = vector[ml.columns].reshape(1, -1)
    probabilities = ml.pipeline.predict_proba(values)[0]
    svm = ml.decision_estimator
    expected = int(svm.predict(values)[0])
    assert result['models']['ml']['prediction']['decision_score'] == pytest.approx(svm.decision_function(values)[0])
    assert result['models']['ml']['prediction']['class_id'] == expected
    assert result['audio'] == metadata
    assert result['warnings'] == warnings
    assert result['models']['ml']['prediction']['probability'] == pytest.approx(probabilities[expected])
    assert result['models']['ml']['version'] == 'svm_35_calibrated'
    assert result['model_version'] == 'fusion_svm'
    assert result['final_model'] == 'ensemble'
    ensemble = client.app.state.models.ensemble.predict(vector, np.zeros(512, dtype=np.float32))
    assert result['prediction'] == ensemble
    assert result['models']['ensemble']['prediction'] == ensemble
    assert result['prediction']['probability'] is None
    assert result['agreement'] == ('agree' if expected == ensemble['class_id'] else 'disagree')
    assert metadata['truncated'] is False
    assert not warnings

@pytest.mark.parametrize("data,status", [(b'',422), (b'not wav',415), (b'RIFF1234WAVEbad',422), (b'fLaCbad',422),
    (wav(silent=True),422), (wav(.2),422), (wav(11),422), (wav(subtype='FLOAT'),415), (b'x'*(MAX_BYTES+1),413)])
def test_invalid_audio(client, data, status):
    response = client.post('/api/classify', files={'audio': ('clip.wav', data)})
    assert response.status_code == status
    assert response.json()['error']['message']


def test_request_contract(client):
    for files in [{}, {'wrong': ('clip.wav', wav())}, [('audio',('a.wav',wav())),('audio',('b.wav',wav()))]]:
        assert client.post('/api/classify', files=files).status_code == 422
    assert client.post('/api/classify', content=b'bad', headers={'Content-Type':'application/json'}).status_code == 422
    assert client.post('/api/classify', content=b'x'*(MAX_BYTES+65537)).status_code == 413
    page = client.get('/')
    assert page.status_code == 200
    assert page.headers['cache-control'] == 'no-cache'
    assert client.get('/static/app.js').status_code == 200
    assert client.get('/api/health').json()['status'] == 'ready'


def test_model_rejects_feature_order(monkeypatch):
    import mvp.models as models
    import joblib
    artifact = {'model': joblib.load(models.ROOT / 'models/svm_35_calibrated.joblib'),
                'columns': list(range(91)), 'feature_names': models.FEATURE_NAMES.copy()}
    artifact['feature_names'] = list(reversed(artifact['feature_names']))
    monkeypatch.setattr(models.joblib, 'load', lambda path: artifact)
    with pytest.raises(ValueError, match='feature names/order'):
        models.AccentModel()


@pytest.mark.parametrize('calibrated', [True, False])
def test_model_override_needs_no_checksum(monkeypatch, tmp_path, calibrated):
    import mvp.models as models
    import shutil
    alternate = tmp_path / 'development-model.joblib'
    if calibrated:
        shutil.copy2(models.ROOT / 'models/svm_35_calibrated.joblib', alternate)
    else:
        # Exercise an uncalibrated export using the shipped model's fitted SVM.
        models.joblib.dump(models.AccentModel().decision_estimator, alternate)
    shutil.copy2(models.ROOT / 'models/svm_35_calibrated_metadata.json',
                 tmp_path / 'development-model_metadata.json')
    monkeypatch.setenv('MODEL_PATH', str(alternate))
    model = models.AccentModel()
    assert model.manifest['model_version'] == 'development-model'
    vector = np.arange(91, dtype=np.float32)
    values = vector[model.columns].reshape(1, -1)
    result = model.predict(vector)
    assert result['class_id'] == model.decision_estimator.predict(values)[0]
    assert result['decision_score'] == pytest.approx(model.decision_estimator.decision_function(values)[0])
    assert (result['probability'] is None) == (not calibrated)
    monkeypatch.setenv('MODEL_VERSION', 'experiment-2')
    assert models.AccentModel().manifest['model_version'] == 'experiment-2'


def test_preprocessing_trims_and_normalizes():
    rate = 16000
    t = np.arange(4 * rate) / rate
    speech = 0.02 * np.sin(2 * np.pi * 180 * t)
    padded = np.pad(speech, (rate, rate))
    stream = BytesIO()
    sf.write(stream, padded, rate, format='WAV', subtype='PCM_16')
    y, metadata, _ = decode_audio(stream.getvalue())
    assert 4 <= metadata['classified_duration_seconds'] <= 4.25
    assert np.sqrt(np.mean(y.astype(np.float64) ** 2)) == pytest.approx(0.1, abs=5e-5)
    assert np.max(np.abs(y)) <= 0.991
    assert extract_features(y).shape == (91,)


def test_very_quiet_audio_rejected():
    stream = BytesIO()
    sf.write(stream, np.full(4 * 16000, 1e-7), 16000, format='WAV', subtype='PCM_32')
    with pytest.raises(AudioError, match='quiet'):
        decode_audio(stream.getvalue())


@pytest.mark.parametrize('score,label,probabilities', [(1.2, 1, [0.89, 0.11]), (-0.4, 0, [0.11, 0.89]), (0., 1, [0.5, 0.5])])
def test_ml_label_uses_raw_svm_and_keeps_probability_for_its_class(monkeypatch, score, label, probabilities):
    from mvp.models import AccentModel
    model = AccentModel()
    monkeypatch.setattr(model.pipeline, 'predict_proba', lambda x: np.asarray([probabilities]))
    monkeypatch.setattr(model.decision_estimator, 'decision_function', lambda x: np.asarray([score]))
    monkeypatch.setattr(model.decision_estimator, 'predict', lambda x: np.asarray([label]))
    result = model.predict(np.zeros(91, dtype=np.float32))
    assert result['class_id'] == label
    assert result['decision_score'] == score
    assert result['probability'] == probabilities[label]
    assert result['display_label'] == ('Singapore Accent' if label else 'Non-Singapore Accent')


def test_flac_and_wav_decode_identically(client):
    data = wav(4)
    y, rate = sf.read(BytesIO(data), dtype='float32')
    stream = BytesIO()
    sf.write(stream, y, rate, format='FLAC', subtype='PCM_16')
    flac = stream.getvalue()
    np.testing.assert_array_equal(decode_audio(data)[0], decode_audio(flac)[0])
    # Content validation is independent of the filename/claimed MIME type.
    result = client.post('/api/classify', files={'audio': ('clip.wav', flac, 'audio/wav')})
    assert result.status_code == 200
    expected = client.post('/api/classify', files={'audio': ('clip.wav', data)})
    assert result.json() == expected.json()


@pytest.mark.parametrize('format,seconds', [('WAV', 8), ('FLAC', 10)])
def test_long_audio_classifies_first_six_seconds(client, format, seconds):
    data = wav(seconds, format=format)
    processed, metadata, warnings = decode_audio(data)
    first_six, _, _ = decode_audio(wav(6, format=format))
    np.testing.assert_array_equal(processed, first_six)
    assert metadata == {'duration_seconds': seconds, 'classified_duration_seconds': 6, 'truncated': True}
    assert warnings == ['Only the first six seconds after silence trimming were classified.']
    response = client.post('/api/classify', files={'audio': ('long.' + format.lower(), data)})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['audio'] == metadata
    assert result['warnings'] == warnings
    assert result['prediction'] == client.app.state.models.ensemble.predict(
        extract_features(first_six), np.zeros(512, dtype=np.float32))


def test_trim_happens_before_crop():
    rate = 16000
    t = np.arange(7 * rate) / rate
    speech = 0.1 * np.sin(2 * np.pi * 180 * t)
    padded = np.pad(speech, (rate, rate))
    stream = BytesIO()
    sf.write(stream, padded, rate, format='WAV', subtype='PCM_16')
    processed, metadata, _ = decode_audio(stream.getvalue())
    assert len(processed) == 6 * rate
    # The retained start includes upstream's ~100 ms trim padding, not the
    # original full second of leading silence. The crop follows this boundary.
    active_start = np.flatnonzero(np.abs(processed) > 1e-3)[0] / rate
    assert 0.09 <= active_start <= 0.13
    assert metadata['duration_seconds'] == 9
    assert metadata['truncated'] is True


def test_selected_columns_match_export_and_are_applied_in_order(monkeypatch):
    import json
    import mvp.models as models
    metadata = json.loads((models.ROOT / 'models/svm_35_calibrated_metadata.json').read_text())
    model = models.AccentModel()
    assert model.columns == metadata['columns']
    assert len(model.columns) == model.pipeline.n_features_in_ == 35
    vector = np.arange(91, dtype=np.float32)
    captured = []
    def probabilities(x):
        captured.append(x.copy())
        return np.asarray([[0.1, 0.9]])
    monkeypatch.setattr(model.pipeline, 'predict_proba', probabilities)
    assert model.predict(vector)['decision_score'] == pytest.approx(
        model.decision_estimator.decision_function(vector[metadata['columns']].reshape(1, 35))[0])
    np.testing.assert_array_equal(captured[0], vector[metadata['columns']].reshape(1, 35))


def test_selected_model_requires_metadata(monkeypatch, tmp_path):
    import shutil
    import mvp.models as models
    path = tmp_path / 'missing_metadata.joblib'
    shutil.copy2(models.ROOT / 'models/svm_35_calibrated.joblib', path)
    monkeypatch.setenv('MODEL_PATH', str(path))
    with pytest.raises(ValueError, match='need columns and feature_names metadata'):
        models.AccentModel()


def test_ensemble_failure_does_not_return_ml_as_final_result(client, monkeypatch):
    def fail(_):
        raise RuntimeError('Encoder unavailable')
    monkeypatch.setattr(client.app.state.models, 'embedding', fail)
    response = client.post('/api/classify', files={'audio': ('clip.wav', wav())})
    assert response.status_code == 500
    assert response.json()['error']['code'] == 'inference_failed'
    assert 'prediction' not in response.json()
