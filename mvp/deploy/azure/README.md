# Azure deployment

The MVP runs on an Azure Linux VM using Docker Compose. The app container runs FastAPI/Uvicorn, and Caddy provides HTTPS and forwards requests to the app on port 8000.

[Open the hosted demo](https://singlish-demo-4e04e8d0.eastasia.cloudapp.azure.com/).

## Prerequisites

- Complete the [MVP prerequisites and setup](../../README.md#prerequisites), including the trusted model artifact.
- An Azure Linux VM with Docker Engine and Docker Compose v2 installed.
- A public hostname pointing to the VM and inbound TCP ports 80 and 443 enabled.
- SSH access for deployment, restricted to authorised administrators.

Microphone access requires HTTPS for a hosted demo. Use the HTTPS URL when opening the app on desktop or mobile.

The included `cloud-init.yaml` is an Ubuntu VM bootstrap template. It installs Docker, prepares `/opt/singlish` for the `azureuser` account, and configures 2 GB of swap. Adjust the account and swap settings for your VM before provisioning.

The current SVM demo uses 2 burstable vCPUs, 1 GB RAM, a 32 GB Standard SSD and 2 GB swap. Resource availability and free allowances depend on the subscription and region; check quotas, credits and costs before provisioning.

## Prepare the deployment package

From the repository root:

```bash
mvp/.venv/bin/python mvp/deploy/azure/prepare_deployment.py tmp/azure-mvp
```

The destination must not already exist. The script verifies the model checksum and copies only serving code, static assets (including the microphone recorder), pinned dependencies, the model artifact and deployment configuration. It excludes recordings, credentials, research data, tests and virtual environments.

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

The Compose configuration limits app memory to 600 MB and proxy memory to 128 MB. Adjust these limits if you change the VM size or application requirements.

## Check the deployment

Open the configured HTTPS URL and check both input methods:

- Upload a valid PCM WAV, preview it, and confirm that classification succeeds.
- Allow microphone access, record 3–6 seconds, stop, preview, and classify. Check that cancelling a new take preserves previously selected audio and that recording stops at 10 seconds.
- Deny microphone access and confirm that WAV upload remains available.

Check readiness with:

```bash
curl https://YOUR_DEMO_HOSTNAME/api/health
```

To view container status and logs on the VM:

```bash
docker compose ps
docker compose logs --tail=100
```

## Updates and migration

Regenerate the package in a new directory, copy the updated files to the VM, and run `docker compose up -d --build` again. The app's source and static assets are baked into the image, so file changes require a rebuild.

To migrate, deploy the same package to another VM, update the hostname and firewall settings, and check readiness and classification before retiring the previous deployment. The MVP retains no recordings or database, so there is no application data to migrate.

The hosted demo runs continuously. VM scheduling is managed separately from Compose. Deallocating a VM stops compute usage, but disks and public IP resources may continue to incur charges.
