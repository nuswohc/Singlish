# Azure deployment

The MVP runs on an Azure Linux VM using Docker Compose. The app container runs FastAPI/Uvicorn, and Caddy provides HTTPS and forwards requests to the app on port 8000.

[Open the hosted demo](https://singlish-demo-4e04e8d0.eastasia.cloudapp.azure.com/).

## Prerequisites

- Complete the [MVP setup](../../README.md#setup), including both trusted SVM artifacts and their matching metadata.
- An Azure Linux VM with Docker Engine and Docker Compose v2 installed.
- A public hostname pointing to the VM and inbound TCP ports 80 and 443 enabled.
- SSH access for deployment, restricted to authorised administrators.

Microphone access requires HTTPS for a hosted demo. Use the HTTPS URL when opening the app on desktop or mobile.

The included `cloud-init.yaml` is an Ubuntu VM bootstrap template. It installs Docker, prepares `/opt/singlish` for the `azureuser` account, and configures 2 GB of swap. Adjust the account and swap settings for your VM before provisioning.

The two-model demo needs more memory than the earlier ML-only host. Use at least 2 vCPUs and 4 GB RAM, with sufficient disk space for the PyTorch image and Whisper weights. The previous 1 GB VM / 600 MB container budget is insufficient. Resource availability and costs depend on the subscription and region.

## Prepare the deployment package

From the repository root:

```bash
mvp/.venv/bin/python mvp/deploy/azure/prepare_deployment.py tmp/azure-mvp
```

To include transcription, pass the saved ASR model folder. The packager checks for and copies its six serving files (not `training_args.bin`) to `mvp/models/whisper-small-mnsc-sg-15k-heldout/`, where the app finds it by default. This makes the package and image about 1 GB larger:

```bash
mvp/.venv/bin/python mvp/deploy/azure/prepare_deployment.py tmp/azure-mvp \
  --asr-model ../whisper_fine-tuned_transcription/models/whisper-small-mnsc-sg-15k-heldout
```

Without `--asr-model` the app still classifies, `/api/health` shows `asr.ready: false` and `/api/transcribe` returns 503.

The destination must not already exist. The script copies only serving code (including `transcription.py` and the vendored `asr_decoding.py`), static assets (including the microphone recorder), pinned dependencies, both model artifacts, matching metadata, the ASR model when requested, and deployment configuration. The Docker build installs CPU PyTorch and caches the pinned Whisper base.en weights; it needs network access to PyPI, the PyTorch CPU index and Hugging Face. Runtime uses those cached weights offline. It excludes recordings, credentials, research data, tests and virtual environments.

Edit `tmp/azure-mvp/Caddyfile` to use your VM's public hostname. The deployment package contains the model binary and should remain outside Git; `tmp/` is ignored by the repository.

## Deploy and run

Copy the package contents to `/opt/singlish` on the VM. Replace `VM_HOST` with the VM's hostname or SSH address:

```bash
scp -r tmp/azure-mvp/. azureuser@VM_HOST:/opt/singlish/
ssh azureuser@VM_HOST
```

On the VM:

```bash
cd /opt/singlish
docker compose up -d --build
docker compose ps
```

Caddy obtains a certificate for the configured hostname and redirects HTTP to HTTPS. Only Caddy publishes web ports; port 8000 remains internal to the Docker network. Containers restart automatically after a VM reboot.

The Compose configuration limits app memory to 3 GB and proxy memory to 128 MB. Whisper inference (classification embedding and transcription) runs one request at a time to bound memory. Adjust these limits if you change the VM size or application requirements.

TODO (transcription, not yet verified on Azure):

- Confirm the VM size and `mem_limit: 3g` with the ASR model loaded. Measured locally on Windows CPU: peak working set about 1.8 GB with the ASR model vs 0.65 GB without, and peak committed memory about 3.7 GB vs 2.1 GB. Check `docker stats` on the VM during a transcription.
- Decide where the deployment gets the ~1 GB ASR model from. It is not in the GitHub release, so today it must be copied from the ASR project before packaging.
- Transcription took about 3.3–5.3 s per 9.5 s clip on a laptop CPU; expect it to be slower on a 2 vCPU VM.

## Check the deployment

Open the configured HTTPS URL and check both input methods:

- Upload a valid PCM WAV, preview it, and confirm that both model cards show calibrated probabilities. The final result shows their shared prediction when they agree, or Inconclusive with a retry prompt when they disagree.
- Allow microphone access, record 3–6 seconds, stop, preview, and classify. Check that cancelling a new take preserves previously selected audio and that recording stops at 10 seconds.
- Deny microphone access and confirm that WAV upload remains available.
- After a classification, confirm that the Transcript box shows a transcript (or, for a package built without `--asr-model`, the "Transcription is unavailable" message while the accent result stays visible). `curl -F 'audio=@clip.wav' https://YOUR_DEMO_HOSTNAME/api/transcribe` returns the transcript JSON.

Check readiness with:

```bash
curl https://YOUR_DEMO_HOSTNAME/api/health
```

The `asr` object in the response shows whether transcription is ready.

To view container status and logs on the VM:

```bash
docker compose ps
docker compose logs --tail=100
```

## Updates and migration

Regenerate the package in a new directory, copy the updated files to the VM, and run `docker compose up -d --build` again. The app's source and static assets are baked into the image, so file changes require a rebuild.

To migrate, deploy the same package to another VM, update the hostname and firewall settings, and check readiness and classification before retiring the previous deployment. The MVP retains no recordings or database, so there is no application data to migrate.

The hosted demo runs continuously. VM scheduling is managed separately from Compose. Deallocating a VM stops compute usage, but disks and public IP resources may continue to incur charges.

## Development model swaps

No checksum is required. Replace `mvp/models/svm_65_calibrated.joblib` and its matching
`svm_65_calibrated_metadata.json`, regenerate the package,
and rebuild the app on the VM. For a different filename, update `model_file` in
`mvp/models/manifest.json`; the packager includes that file automatically. Models
must select their inputs from the current 91-feature pool using their matching
column metadata and retain the current preprocessing. The API reports the
filename stem unless `MODEL_VERSION` is set. Compose passes through optional
`MODEL_PATH`, `MODEL_VERSION`, `ENSEMBLE_MODEL_PATH`, `ENSEMBLE_MODEL_VERSION`, `ASR_MODEL_PATH` and `ASR_MODEL_VERSION` values from the VM environment or `.env` file;
a path override must exist inside the container. Both manifest-selected
models are included in a generated package. Use a unique `MODEL_VERSION` when
replacing a file under the same name so deployments are easy to distinguish.

For Ensemble, replace `fusion_svm_calibrated.joblib` and `fusion_svm_calibrated_metadata.json` together,
or update `ensemble_file` in the manifest for a new filename. The metadata must
match the raw acoustic + Whisper input contract described in the MVP README.

Before deploying, back up `/opt/singlish` and tag the current app image. Keep the
Caddy data volumes when updating so the existing HTTPS certificate is preserved.
