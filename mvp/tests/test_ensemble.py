"""Check the E4 serving contract against the trusted saved pipeline."""
import json
import numpy as np
import pytest
from mvp.models import EnsembleModel


def test_ensemble_concatenates_raw_features_before_saved_preprocessing(monkeypatch):
    model = EnsembleModel()
    vector = np.arange(91, dtype=np.float32)
    embedding = np.linspace(-1, 1, 512, dtype=np.float32)
    expected = np.concatenate([vector[model.columns], embedding]).reshape(1, 577)
    probabilities = model.pipeline.predict_proba(expected)[0]
    label = int(probabilities[1] >= 0.5)
    score = float(model.decision_estimator.decision_function(expected)[0])
    result = model.predict(vector, embedding)
    assert result['class_id'] == label
    assert result['decision_score'] == pytest.approx(score)
    assert result['probability'] == pytest.approx(probabilities[label])
    # Capture probability inputs so an accidental feature reorder cannot pass on labels alone.
    captured = []
    def predict_proba(values):
        captured.append(values)
        return probabilities.reshape(1, 2)
    monkeypatch.setattr(model.pipeline, 'predict_proba', predict_proba)
    model.predict(vector, embedding)
    np.testing.assert_array_equal(captured[0], expected)


@pytest.mark.parametrize('embedding', [np.zeros(32), np.full(512, np.nan)])
def test_ensemble_rejects_invalid_embeddings(embedding):
    with pytest.raises(ValueError, match='512-value'):
        EnsembleModel().predict(np.zeros(91), embedding)


def test_ensemble_rejects_feature_order(monkeypatch, tmp_path):
    from mvp.models import ROOT
    import shutil
    path = tmp_path / 'fusion.joblib'
    shutil.copy2(ROOT / 'models/fusion_svm_calibrated.joblib', path)
    metadata = json.loads((ROOT / 'models/fusion_svm_calibrated_metadata.json').read_text())
    metadata['acoustic_feature_names'].reverse()
    path.with_name('fusion_metadata.json').write_text(json.dumps(metadata))
    monkeypatch.setenv('ENSEMBLE_MODEL_PATH', str(path))
    with pytest.raises(ValueError, match='feature names/order'):
        EnsembleModel()


def test_whisper_pooling_excludes_thirty_second_padding():
    import torch
    from types import SimpleNamespace
    from mvp.embeddings import WhisperEmbedding
    model = WhisperEmbedding.__new__(WhisperEmbedding)
    hidden = torch.full((1, 1500, 512), 999.)
    # 3.001 seconds rounds up to 151 real encoder steps, matching training.
    hidden[0, :151] = torch.arange(151, dtype=torch.float32)[:, None]
    model.extractor = lambda *args, **kwargs: SimpleNamespace(input_features=torch.zeros(1, 80, 3000))
    model.encoder = lambda **kwargs: SimpleNamespace(last_hidden_state=hidden)
    actual = model(np.zeros(48001, dtype=np.float32))
    np.testing.assert_array_equal(actual, np.full(512, 75., dtype=np.float32))


@pytest.mark.parametrize('ml_class,ensemble_class,agreement', [(1, 1, 'agree'), (0, 1, 'disagree')])
def test_agreement_compares_predicted_classes(ml_class, ensemble_class, agreement):
    from threading import Lock
    from types import SimpleNamespace
    from mvp.models import AccentModels
    models = AccentModels.__new__(AccentModels)
    models.ml = SimpleNamespace(manifest={'model_version': 'ml'}, predict=lambda _: {'class_id': ml_class})
    models.ensemble = SimpleNamespace(version='fusion', predict=lambda *_: {'class_id': ensemble_class})
    models.embedding = lambda _: np.zeros(512)
    models.lock = Lock()
    result = models.predict(np.zeros(91), np.zeros(48000))
    assert result['agreement'] == agreement
    assert result['prediction'] == result['models']['ensemble']['prediction']
