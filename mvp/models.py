"""Load the trusted study export and validate its inference contract."""
import hashlib
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parent

class AccentModel:
    def __init__(self):
        self.manifest = json.loads((ROOT / "models/manifest.json").read_text())
        path = ROOT / "models/svm_39.joblib"
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.manifest["sha256"]:
            raise ValueError("Model checksum does not match models/manifest.json")
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            artifact = joblib.load(path)
        if not isinstance(artifact, dict) or not {"model", "columns", "feature_names"} <= artifact.keys():
            raise ValueError("Expected the E1 study export dictionary")
        if artifact["feature_names"] != FEATURE_NAMES or artifact["columns"] != list(range(39)):
            raise ValueError("Model feature names/order do not match librosa39-v1")
        self.pipeline = artifact["model"]
        if not isinstance(self.pipeline, Pipeline) or not isinstance(self.pipeline.named_steps.get("scale"), StandardScaler):
            raise ValueError("Expected a fitted scaler/SVM pipeline")
        classifier = self.pipeline.named_steps.get("classifier")
        if not isinstance(classifier, SVC) or classifier.kernel != "rbf" or classifier.C != 1.0 or classifier.gamma != "scale" or classifier.class_weight != "balanced":
            raise ValueError("Model does not match the selected RBF SVM settings")
        if self.pipeline.n_features_in_ != 39 or not np.array_equal(self.pipeline.classes_, [0, 1]):
            raise ValueError("Invalid fitted input width or class mapping")

    def predict(self, vector):
        if vector.shape != (39,) or not np.isfinite(vector).all():
            raise ValueError("Expected 39 finite acoustic features")
        label = int(self.pipeline.predict(vector.reshape(1, -1))[0])
        return {"class_id": label,
                "label": "singapore_accent" if label == 1 else "non_singapore_accent",
                "display_label": "Singapore accent" if label == 1 else "Non-Singapore accent"}
