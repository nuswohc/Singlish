from io import BytesIO
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from mvp.app import app
from mvp.audio import MAX_BYTES, decode_audio, AudioError
from mvp.features import extract_features


def wav(seconds=3, rate=16000, channels=1, subtype="PCM_16", silent=False):
    t = np.arange(int(seconds * rate)) / rate
    y = np.zeros_like(t) if silent else (0.15 * np.sin(2 * np.pi * 180 * t))
    if channels > 1:
        y = np.tile(y[:, None], (1, channels))
    stream = BytesIO()
    sf.write(stream, y, rate, format="WAV", subtype=subtype)
    return stream.getvalue()

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client

@pytest.mark.parametrize("rate,channels,seconds", [(16000,1,3), (44100,2,3), (16000,1,.2), (16000,1,8)])
def test_inference_matches_pipeline(client, rate, channels, seconds):
    data = wav(seconds, rate, channels)
    response = client.post('/api/classify', files={'audio': ('clip.wav', data)})
    assert response.status_code == 200, response.text
    result = response.json()
    y, metadata, warnings = decode_audio(data)
    expected = int(client.app.state.model.pipeline.predict(extract_features(y).reshape(1,-1))[0])
    assert result['prediction']['class_id'] == expected
    assert result['audio'] == metadata
    assert result['warnings'] == warnings
    assert 'probability' not in result['prediction']
    assert metadata['truncated'] == (seconds > 6)
    if seconds < 1 or seconds > 6:
        assert warnings

@pytest.mark.parametrize("data,status", [(b'',422), (b'not wav',415), (b'RIFF1234WAVEbad',422),
    (wav(silent=True),422), (wav(11),422), (wav(subtype='FLOAT'),415), (b'x'*(MAX_BYTES+1),413)])
def test_invalid_audio(client, data, status):
    response = client.post('/api/classify', files={'audio': ('clip.wav', data)})
    assert response.status_code == status
    assert response.json()['error']['message']


def test_request_contract(client):
    for files in [{}, {'wrong': ('clip.wav', wav())}, [('audio',('a.wav',wav())),('audio',('b.wav',wav()))]]:
        assert client.post('/api/classify', files=files).status_code == 422
    assert client.post('/api/classify', content=b'bad', headers={'Content-Type':'application/json'}).status_code == 422
    assert client.post('/api/classify', content=b'x'*(MAX_BYTES+65537)).status_code == 413
    assert client.get('/').status_code == 200
    assert client.get('/static/app.js').status_code == 200
    assert client.get('/api/health').json()['status'] == 'ready'


def test_zero_audio_rejected():
    with pytest.raises(AudioError):
        decode_audio(wav(silent=True))


def test_model_rejects_feature_order(monkeypatch):
    import mvp.models as models
    import joblib
    artifact = joblib.load(models.ROOT / 'models/svm_39.joblib')
    artifact['feature_names'] = list(reversed(artifact['feature_names']))
    monkeypatch.setattr(models.joblib, 'load', lambda path: artifact)
    with pytest.raises(ValueError, match='feature names/order'):
        models.AccentModel()


def test_model_rejects_checksum_mismatch(monkeypatch):
    import mvp.models as models
    from pathlib import Path
    original = Path.read_bytes
    monkeypatch.setattr(Path, 'read_bytes', lambda self: b'corrupted' if self.name == 'svm_39.joblib' else original(self))
    with pytest.raises(ValueError, match='checksum'):
        models.AccentModel()


def test_model_rejects_missing_artifact(monkeypatch):
    import mvp.models as models
    from pathlib import Path
    original = Path.read_bytes
    def read(self):
        if self.name == 'svm_39.joblib':
            raise FileNotFoundError('Model is missing')
        return original(self)
    monkeypatch.setattr(Path, 'read_bytes', read)
    with pytest.raises(FileNotFoundError):
        models.AccentModel()
