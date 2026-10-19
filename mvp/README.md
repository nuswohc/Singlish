# Singapore English Accent Classification

A simple FastAPI demo that records microphone audio or accepts WAV/FLAC uploads,
then runs two models:

| Display name | Saved model | Result |
|---|---|---|
| **ML** | E1 calibrated 35-feature RBF SVM | Accent and raw SVM decision score |
| **Ensemble** | E4_02 acoustic + Whisper feature-fusion SVM | Accent and raw SVM decision score |

The page focuses on recording/uploading, previewing audio and comparing results.
Planned transcription features stay outside the MVP interface.

Ensemble supplies the final result. Both predictions appear below it, with
**Agree** or **Disagree** based on their class labels. Both cards show signed
decision scores: positive means Singapore, negative means Non-Singapore, and
near zero means the decision boundary. Score magnitudes are not directly
comparable across models. Ensemble uses feature
concatenation, rather than averaging the two displayed predictions. There is
no separate deep accent classifier. See the [implementation plan](ENSEMBLE_PLAN.md).

[Hosted demo](https://singlish-demo-4e04e8d0.eastasia.cloudapp.azure.com/)
may run an earlier release until redeployed.

## Setup

Python 3.12 or later is required; Python 3.14 matches the serving environment.
Use trusted joblib artifacts only, since loading them executes Python code.

Place these files in `mvp/models/` (binaries are ignored by Git):

- `svm_35_calibrated.joblib` and `svm_35_calibrated_metadata.json` — existing E1 model.
- `fusion_svm.joblib` and `fusion_svm_metadata.json` — E4 Ensemble export and metadata.

Download both models and their matching metadata from
[release mvp-ensemble-20261019](https://github.com/nuswohc/Singlish/releases/tag/mvp-ensemble-20261019).
From the repository root:

```bash
for file in svm_35_calibrated.joblib svm_35_calibrated_metadata.json fusion_svm.joblib fusion_svm_metadata.json; do
  curl -fL "https://github.com/nuswohc/Singlish/releases/download/mvp-ensemble-20261019/$file" -o "mvp/models/$file"
done
python3 -m venv mvp/.venv
# Linux CPU serving: install this first to avoid unnecessary CUDA dependencies.
mvp/.venv/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
mvp/.venv/bin/python -m pip install -r mvp/requirements.txt
./mvp/run.sh
```

Open **http://127.0.0.1:8000**. Set `PORT=8001` to use another port.
The first startup downloads `openai/whisper-base.en` from Hugging Face to its
normal cache. The manifest pins revision `911407f4214e0e1d82085af863093ec0b66f9cd6`.
Subsequent starts can use `HF_HUB_OFFLINE=1` once the weights are cached.
Both classifiers and Whisper must load successfully before the API is ready.
Allow at least 4 GB host RAM for the two-model demo; see [Azure setup](deploy/azure/README.md).

## Using the demo

- **Record:** select **Record audio**, allow microphone access, read
  **“They tried to keep it a secret”** over 3–6 seconds, then stop. Recording
  stops automatically at 10 seconds. **Cancel** preserves previously selected audio.
- **Upload:** choose or drop a PCM WAV or lossless FLAC file.
- Listen to the preview, then select **Classify recording** to run both models.

Microphone access requires HTTPS or localhost and a supported browser.
Stopping, cancelling or leaving the page releases the microphone. Audio is sent
only on classification, processed per request, and not retained.

Use clear English speech from one speaker. Files must be at most 10 MiB and
10 seconds. The backend validates file content, supports mono/stereo and resampling,
and requires at least three seconds after silence trimming. Longer clips use the
first six seconds after trimming, with a notice. Results compare side by side on
desktop; the page stacks on smaller screens.

## Model behavior

The shared preprocessing matches E1: mono → 16 kHz (soxr HQ) → 40 dB frame-RMS
boundary trimming (25 ms frames, 10 ms hop, 100 ms edge padding) → require three
seconds and crop to six → reject RMS below 1e-5 → normalize to −20 dBFS → limit
peak to 0.99 → PCM16 quantization. Internal pauses remain; no padding is applied
at this stage.

The app extracts 91 acoustic features once. ML selects its 35 metadata columns,
then uses the fitted SVM and scaler inside the calibrated export. The UI shows
this original SVM's raw decision score and predicted class, using its zero
boundary. It no longer uses the calibrated probability's 0.5 threshold for
classification. The calibrated probability of the raw predicted class remains
in the API for compatibility; the UI displays no probabilities.

Ensemble concatenates its 35 raw acoustic columns with a 512-value frozen Whisper
encoder embedding. Whisper uses the same waveform, no extra normalization,
standard 30-second input padding and a mean over only real 20 ms encoder steps
(`ceil(samples / 320)`). The saved pipeline handles acoustic scaling, Whisper
scaling/PCA to 32 components, branch weighting and RBF SVM classification.
No fitting occurs in the app. Serving uses CPU float32; the research embeddings
used GPU bfloat16, so small numerical differences are expected.

The E4 export has no probability calibration or `predict_proba`. Its decision
score is signed (positive for Singapore, negative for Non-Singapore), and is
displayed alongside the ML score. Neither score is converted to a percentage. Agreement
compares the two class labels and is not a confidence measure. Inference is
serialized to bound encoder memory use on small hosts.

The classifiers assess accent characteristics, not nationality or identity.
They can be wrong and do not verify English speech or reject all noise/music.
Transcription, context-assisted retranscription, speech detection and audio
quality scoring from Appendix B remain future work.

## Model swaps

To run ML with the uncalibrated 91-feature SVM, start from the repository root:

```bash
MODEL_PATH=models/svm_91.joblib ./mvp/run.sh
```

That optional reference model is available from the earlier
[E1 study release](https://github.com/nuswohc/Singlish/releases/tag/e1-feature-study-20261019).
Relative model paths resolve from `mvp/`, so omit the `mvp/` prefix in
`MODEL_PATH`. This changes the ML comparison model; Ensemble still supplies
the final result. Uncalibrated ML exports return `probability: null` and keep
their raw decision scores. Stop an existing server before restarting.

`MODEL_PATH` / `MODEL_VERSION` configure ML; `ENSEMBLE_MODEL_PATH` /
`ENSEMBLE_MODEL_VERSION` configure Ensemble. Relative paths resolve from `mvp/`.
Versions default to filename stems. Restart after replacing files; no checksum
update is required. Keep each model and matching metadata together.

ML accepts an estimator with `decision_function` and optional `predict_proba`, or
a calibration wrapper containing one fitted SVM (as in the existing E1 export).
Provide `<stem>_metadata.json`, or a study dictionary with `model`, `columns` and
`feature_names`. A full 91-feature estimator can omit the metadata.
Ensemble requires a direct 547-input classifier and
`<stem>_metadata.json` with E4's `feature_columns_in_91`, `feature_names`,
`whisper_width`, `whisper_model` and `positive_label`. New extraction definitions
require code changes. For deployment, update `model_file` / `ensemble_file` in
the manifest and rebuild; both artifacts and their metadata are packaged.

## API

- `GET /` — recording/upload page.
- `GET /api/health` — readiness, final model version and both model versions.
- `POST /api/classify` — multipart form with exactly one `audio` WAV/FLAC file.

```bash
curl -F 'audio=@recording.wav' http://127.0.0.1:8000/api/classify
```

Success returns:

- `prediction`: final Ensemble result (`class_id`, `label`, `display_label`,
  `probability: null`, `decision_score`).
- `final_model: "ensemble"`, `models.ml` and `models.ensemble`: each model's name,
  version and prediction; both have `decision_score`. ML returns a numeric
  `probability` for its raw predicted class when supported, which can be below
  0.5; otherwise it returns null.
- `agreement`: `"agree"` or `"disagree"`; `model_version`: Ensemble version.
- `audio`: `duration_seconds`, `classified_duration_seconds`, `truncated`.
- `warnings`: crop notices, if any.

**API change:** `prediction` now refers to Ensemble and its probability is null.
Consumers needing the existing calibrated probability must use
`models.ml.prediction.probability`. ML now follows its raw SVM decision boundary
rather than the calibrated probability threshold, so borderline labels may
change. Classified duration means processed signal
duration, not verified speech duration.

Errors return `error.code` and `error.message`: 413 for size limits, 415 for
unsupported formats, 422 for invalid audio/requests and 500 for inference failure.
There is no silent fallback to ML when Ensemble fails.

## Verification

```bash
mvp/.venv/bin/python -m pip install -r mvp/requirements-dev.txt
mvp/.venv/bin/python -m pytest mvp/tests -q
node --test mvp/tests/test_ui.cjs mvp/tests/test_recorder.cjs
```

API tests use the real saved SVMs with a stub embedding, and a separate pooling
test checks the real-audio boundary. Tests focus on pipeline parity, audio/request
limits, failure recovery and microphone cleanup, using representative format
combinations instead of a full cross-product. For an end-to-end check, run the server and
upload a real speech clip so the actual Whisper encoder is exercised.
The offline E1 verifier checks acoustic feature parity and the API's ML result via
`python -m mvp.scripts.verify_study /path/to/E1_Feature_Engineering --output mvp/verification.json`
(requires `pyarrow==26.0.0`; it is not a server dependency).
