# Singapore English Accent Classification

A FastAPI demo that records microphone audio or accepts WAV/FLAC uploads and
compares two test release candidates:

| Display name | Saved model | Inputs |
|---|---|---|
| **ML** | `svm_65_calibrated.joblib` | 65 selected acoustic features |
| **Ensemble** | `fusion_svm_calibrated.joblib` | The same 65 features + 512 Whisper values |

The page shows the shared prediction when both models agree, or **Inconclusive**
with a retry prompt when they disagree. Both cards show the predicted accent and its
calibrated probability, with **Agree** or **Disagree** based on their class labels.
Both follow the notebooks' rule: Singapore when `p_singapore >= 0.50`, including
ties. Agreement is not a confidence measure.

Sources: [E1 calibration notebook](../results/E1_Feature_Engineering_20261023/E1_svm_calibrated_probability.ipynb)
and [E4 calibration notebook](../results/E4_Hybrid_Ensemble_20261023/E4_03_whisper_svm_calibrated_probability.ipynb).
The [hosted demo](https://singlish-demo-4e04e8d0.eastasia.cloudapp.azure.com/)
may run an earlier release until redeployed.

## Setup

Python 3.12 or later is required; Python 3.14 matches the serving environment.
Use trusted joblib artifacts only, since loading them executes Python code.
Download both models and their matching metadata from
[release mvp-svm65-rc-20261023](https://github.com/nuswohc/Singlish/releases/tag/mvp-svm65-rc-20261023).
From the repository root:

```bash
for file in svm_65_calibrated.joblib svm_65_calibrated_metadata.json fusion_svm_calibrated.joblib fusion_svm_calibrated_metadata.json; do
  curl -fL "https://github.com/nuswohc/Singlish/releases/download/mvp-svm65-rc-20261023/$file" -o "mvp/models/$file"
done
python3 -m venv mvp/.venv
# Linux CPU serving: install this first to avoid unnecessary CUDA dependencies.
mvp/.venv/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
mvp/.venv/bin/python -m pip install -r mvp/requirements.txt
./mvp/run.sh
```

Open **http://127.0.0.1:8000**. Set `PORT=8001` to use another port.
The first startup downloads `openai/whisper-base.en`, pinned to revision
`911407f4214e0e1d82085af863093ec0b66f9cd6`. Subsequent starts can use
`HF_HUB_OFFLINE=1` once the weights are cached. Both classifiers and Whisper must
load before the API is ready. Allow at least 4 GB host RAM; see
[Azure setup](deploy/azure/README.md).

## Using the demo

- **Record:** allow microphone access and read this sentence:
  **“They tried to keep it a secret but he was able to figure out everything”**.
  Stop and preview your recording. Recording stops automatically at 10 seconds;
  **Cancel** preserves previously selected audio.
- **Upload:** choose or drop a PCM WAV or lossless FLAC file.
- Listen to the preview, then select **Classify recording** to run both models.

Use clear English speech from one speaker. Files must be at most 10 MiB and
10 seconds. Clips shorter than three seconds **after silence trimming** are
rejected with a request to retry with a longer recording, before either model
runs. Longer clips use the first six seconds after trimming, with a notice.

Microphone access requires HTTPS or localhost and a supported browser.
Stopping, cancelling or leaving the page releases the microphone. Audio is sent
only on classification, processed per request, and not retained.

## Model behavior

Shared preprocessing matches E1: mono → 16 kHz (soxr HQ) → 40 dB frame-RMS
boundary trimming (25 ms frames, 10 ms hop, 100 ms edge padding) → require three
seconds and crop to six → reject RMS below 1e-5 → normalize to −20 dBFS → limit
peak to 0.99 → PCM16 quantization. Internal pauses remain.

The app extracts 91 acoustic features once and selects each model's 65 metadata
columns in the saved order. Ensemble concatenates the raw selected values with a
512-value frozen Whisper encoder embedding. Whisper uses the same waveform,
standard 30-second padding and a mean over real 20 ms encoder steps only
(`ceil(samples / 320)`). The saved fusion pipeline handles scaling, Whisper PCA
to 32 components, branch weighting, SVM classification and sigmoid calibration.
No fitting occurs in the app. CPU float32 embeddings can differ slightly from
the research GPU embeddings. Inference is serialized to bound memory use.

The classifiers assess accent characteristics, not nationality or identity.
They can be wrong and do not verify English speech or reject all noise/music.
Transcription remains outside the MVP.

## Model swaps

`MODEL_PATH` / `MODEL_VERSION` configure ML; `ENSEMBLE_MODEL_PATH` /
`ENSEMBLE_MODEL_VERSION` configure Ensemble. Relative paths resolve from `mvp/`.
Versions default to filename stems. Restart after replacing files; no checksum
update is required. Keep each model and its matching metadata together.

ML accepts an estimator or a study dictionary containing `model`, `columns` and
`feature_names`. Selected-feature estimators need `<stem>_metadata.json` with
`columns` and `feature_names`; full 91-feature estimators can omit metadata.
Ensemble uses the E4 calibrated metadata schema: `acoustic_columns_in_91`,
`acoustic_feature_names`, `whisper_width`, `positive_label` and
`whisper_provenance`. Its input width is 577. Both calibrated exports contain
one refitted SVM. Uncalibrated ML overrides use the original SVM decision and
return `probability: null`, shown as a dash in the UI.

For deployment, update `model_file` / `ensemble_file` in the manifest and rebuild.
The deployment packager includes both artifacts and their metadata.

## API

- `GET /` — recording/upload page.
- `GET /api/health` — readiness and both model versions.
- `POST /api/classify` — multipart form with exactly one `audio` WAV/FLAC file.

```bash
curl -F 'audio=@recording.wav' http://127.0.0.1:8000/api/classify
```

Success returns `prediction` (the Ensemble result), `final_model: "ensemble"`,
`models.ml`, `models.ensemble`, `agreement`, `model_version`, `audio` and `warnings`.
Each prediction contains `class_id`, `label`, `display_label`, `probability` (of
the predicted class) and the original `decision_score`. Labels now follow the
calibrated 0.50 threshold, so they can differ from the raw score's sign. The UI
shows probabilities. Ensemble probability is now numeric instead of null.

Audio metadata contains `duration_seconds`, `classified_duration_seconds` and
`truncated`; processed duration is not verified speech duration. Errors return
`error.code` and `error.message`: 413 for size limits, 415 for unsupported
formats, 422 for invalid audio/requests (including `invalid_duration`) and 500
for inference failure. There is no silent fallback to ML if Ensemble fails.

## Verification

```bash
mvp/.venv/bin/python -m pip install -r mvp/requirements-dev.txt
mvp/.venv/bin/python -m pytest mvp/tests -q
node --test mvp/tests/test_ui.cjs mvp/tests/test_recorder.cjs
```

API tests use the real saved SVMs with a stub Whisper embedding. They check
saved-pipeline parity, calibrated labels, feature order, trimmed duration,
request limits and error recovery. Recorder tests check microphone cleanup.
For an end-to-end check, upload a real speech clip using the actual Whisper
encoder. The offline E1 verifier also checks acoustic feature parity:

```bash
mvp/.venv/bin/python -m mvp.scripts.verify_study /path/to/E1_Feature_EngineeringV2 --output mvp/verification.json
```

It requires `pyarrow==26.0.0`, which is not a server dependency.
