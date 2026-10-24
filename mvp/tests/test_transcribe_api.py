"""POST /api/transcribe with a stub Transcriber, health fields and graceful degradation."""
from io import BytesIO
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from mvp.app import app
from mvp.audio import MAX_BYTES, read_audio
from mvp.features import trim_silence


def wav(seconds=3, rate=16000, subtype="PCM_16", silent=False):
    # Same signal as tests/test_app.py's helper.
    t = np.arange(int(seconds * rate)) / rate
    y = np.zeros_like(t) if silent else (0.15 * np.sin(2 * np.pi * 180 * t))
    stream = BytesIO()
    sf.write(stream, y, rate, format="WAV", subtype=subtype)
    return stream.getvalue()


INVALID = {"empty": b"", "not_wav": b"not wav", "bad_riff": b"RIFF1234WAVEbad", "bad_flac": b"fLaCbad",
           "silent": wav(silent=True), "too_short": wav(.2), "too_long": wav(11),
           "float_wav": wav(subtype="FLOAT"), "too_big": b"x" * (MAX_BYTES + 1)}


class RecordingTranscriber:
    version, decoding, device = "fake-asr", "temperature fallback", "cpu"
    instances = []

    def __init__(self, lock=None, **kwargs):
        self.lock = lock
        self.waveforms = []
        self.text = "hello from the fake transcriber"
        self.error = None
        RecordingTranscriber.instances.append(self)

    def transcribe(self, waveform):
        self.waveforms.append(waveform)
        if self.error:
            raise self.error
        return self.text


def stub_embedding(patch):
    import mvp.embeddings as embeddings
    patch.setattr(embeddings, "WhisperEmbedding", lambda *args: lambda waveform: np.zeros(512, dtype=np.float32))


@pytest.fixture
def client():
    import mvp.transcription as transcription
    with pytest.MonkeyPatch.context() as patch:
        stub_embedding(patch)
        patch.setattr(transcription, "Transcriber", RecordingTranscriber)
        with TestClient(app) as client:
            yield client


@pytest.fixture
def degraded_client():
    import mvp.transcription as transcription

    def missing(**kwargs):
        raise FileNotFoundError("ASR model not found")

    with pytest.MonkeyPatch.context() as patch:
        stub_embedding(patch)
        patch.setattr(transcription, "Transcriber", missing)
        with TestClient(app) as client:
            yield client


def quiet_wav(seconds, rate=16000, channels=1, format="WAV"):
    t = np.arange(int(seconds * rate)) / rate
    y = 0.02 * np.sin(2 * np.pi * 180 * t)
    if channels > 1:
        y = np.tile(y[:, None], (1, channels))
    stream = BytesIO()
    sf.write(stream, y, rate, format=format, subtype="PCM_16")
    return stream.getvalue()


def test_transcribe_returns_transcript_for_full_upload(client):
    data = quiet_wav(8)
    response = client.post("/api/transcribe", files={"audio": ("clip.wav", data)})
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"transcript", "model_version", "decoding", "audio", "warnings"}
    assert body["transcript"] == "hello from the fake transcriber"
    assert body["model_version"] == "fake-asr"
    assert body["decoding"] == "temperature fallback"
    assert body["warnings"] == []
    fake = client.app.state.transcriber
    received = fake.waveforms[-1]
    expected = trim_silence(read_audio(data)[0])
    assert np.array_equal(received, expected)
    assert len(received) > 6 * 16000
    assert body["audio"] == {"duration_seconds": 8, "transcribed_duration_seconds": round(len(expected) / 16000, 4)}
    # No loudness normalization (the classifier normalizes to -20 dBFS, RMS 0.1).
    assert np.sqrt(np.mean(received.astype(np.float64) ** 2)) == pytest.approx(0.02 / np.sqrt(2), abs=2e-4)


def test_transcribe_resamples_full_stereo_flac(client):
    data = quiet_wav(10, 44100, 2, "FLAC")
    response = client.post("/api/transcribe", files={"audio": ("clip.flac", data)})
    assert response.status_code == 200, response.text
    received = client.app.state.transcriber.waveforms[-1]
    assert np.array_equal(received, trim_silence(read_audio(data)[0]))
    assert len(received) >= 10 * 16000 - 1


def test_transcriber_shares_classifier_lock(client):
    assert client.app.state.transcriber.lock is client.app.state.models.lock


@pytest.mark.parametrize("name", list(INVALID))
def test_validation_matches_classify(client, name):
    files = {"audio": ("clip.wav", INVALID[name])}
    transcribed = client.post("/api/transcribe", files=files)
    classified = client.post("/api/classify", files=files)
    assert transcribed.status_code == classified.status_code
    assert transcribed.json() == classified.json()
    assert transcribed.json()["error"]["message"]
    assert client.app.state.transcriber.waveforms == []


def test_request_contract(client):
    for files in [{}, {"wrong": ("clip.wav", wav())}, [("audio", ("a.wav", wav())), ("audio", ("b.wav", wav()))]]:
        assert client.post("/api/transcribe", files=files).status_code == 422
    assert client.post("/api/transcribe", content=b"bad", headers={"Content-Type": "application/json"}).status_code == 422
    response = client.post("/api/transcribe", content=b"x" * (MAX_BYTES + 65537))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_short_after_trimming_is_rejected_before_transcription(client):
    samples, rate = sf.read(BytesIO(wav(2.5)), dtype="float32")
    stream = BytesIO()
    sf.write(stream, np.pad(samples, (rate, rate)), rate, format="WAV", subtype="PCM_16")
    response = client.post("/api/transcribe", files={"audio": ("padded.wav", stream.getvalue())})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_duration"
    assert client.app.state.transcriber.waveforms == []


def test_transcription_failure_does_not_break_classification(client):
    client.app.state.transcriber.error = RuntimeError("decoder crashed")
    response = client.post("/api/transcribe", files={"audio": ("clip.wav", wav())})
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "transcription_failed",
                                         "message": "Transcription failed. Please try another WAV or FLAC file."}}
    assert client.post("/api/classify", files={"audio": ("clip.wav", wav())}).status_code == 200


def test_empty_transcript_returns_warning(client):
    client.app.state.transcriber.text = ""
    response = client.post("/api/transcribe", files={"audio": ("clip.wav", wav())})
    assert response.status_code == 200
    assert response.json()["transcript"] == ""
    assert response.json()["warnings"] == ["No speech was recognised in this recording."]


def test_health_reports_asr(client):
    health = client.get("/api/health").json()
    assert health["status"] == "ready"
    assert health["asr"] == {"ready": True, "model_version": "fake-asr", "decoding": "temperature fallback",
                             "device": "cpu"}


def test_missing_asr_model_degrades_gracefully(degraded_client):
    health = degraded_client.get("/api/health").json()
    assert health["status"] == "ready"
    assert health["model_version"] == "fusion_svm_calibrated"
    assert health["asr"] == {"ready": False, "model_version": None, "decoding": "temperature fallback",
                             "reason": "model_not_found"}
    response = degraded_client.post("/api/transcribe", files={"audio": ("clip.wav", wav())})
    assert response.status_code == 503
    assert response.json() == {"error": {"code": "transcription_unavailable",
                                         "message": "Transcription is unavailable on this server. Classification still works."}}
    classified = degraded_client.post("/api/classify", files={"audio": ("clip.wav", wav())})
    assert classified.status_code == 200
    assert set(classified.json()) == {"prediction", "final_model", "models", "agreement", "model_version",
                                      "audio", "warnings"}


def test_asr_load_failure_degrades_gracefully():
    import mvp.transcription as transcription

    def broken(**kwargs):
        raise RuntimeError("corrupt weights")

    with pytest.MonkeyPatch.context() as patch:
        stub_embedding(patch)
        patch.setattr(transcription, "Transcriber", broken)
        with TestClient(app) as client:
            assert client.get("/api/health").json()["asr"]["reason"] == "load_failed"
            assert client.post("/api/transcribe", files={"audio": ("clip.wav", wav())}).status_code == 503
