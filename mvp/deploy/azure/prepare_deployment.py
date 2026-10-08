"""Prepare an allowlisted container context for Azure without research data or secrets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New, nonexistent directory")
    args = parser.parse_args()
    templates = Path(__file__).resolve().parent
    mvp = templates.parents[1]
    manifest = json.loads((mvp / "models/manifest.json").read_text())
    model = mvp / "models/svm_39.joblib"
    if hashlib.sha256(model.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Model checksum mismatch; refusing to package")
    names = ["__init__.py", "app.py", "audio.py", "features.py", "models.py",
             "requirements.txt", "models/manifest.json", "models/svm_39.joblib",
             "static/index.html", "static/app.js", "static/recorder.js", "static/styles.css"]
    for name in names:
        if not (mvp / name).is_file():
            raise FileNotFoundError(mvp / name)
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for name in names:
        target = destination / "mvp" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mvp / name, target)
    manifest.pop("source", None)
    (destination / "mvp/models/manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    for name in ("Dockerfile", "compose.yaml", "Caddyfile"):
        shutil.copy2(templates / name, destination / name)
    (destination / ".dockerignore").write_text(
        "**/__pycache__/\n**/.venv/\n**/.pytest_cache/\n.git/\n")
    print(f"Prepared {destination}; model SHA-256 verified.")


if __name__ == "__main__":
    main()
