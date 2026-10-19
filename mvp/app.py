"""Local WAV/FLAC-upload accent classifier."""
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from .audio import AudioError, MAX_BYTES, decode_audio
from .features import extract_features
from .models import AccentModels

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app):
    app.state.models = AccentModels()
    yield

app = FastAPI(title="Singapore Accent MVP", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

@app.exception_handler(AudioError)
async def audio_error(request, error):
    return JSONResponse({"error": {"code": error.code, "message": error.message}}, status_code=error.status)

@app.get("/")
def index():
    # Revalidate the entry page so returning browsers discover new asset URLs.
    return FileResponse(ROOT / "static/index.html", headers={"Cache-Control": "no-cache"})

@app.get("/api/health")
def health(request: Request):
    models = request.app.state.models
    return {"status": "ready", "model_version": models.ensemble.version, "models": models.versions()}


def classify(models, data):
    waveform, metadata, warnings = decode_audio(data)
    return {**models.predict(extract_features(waveform), waveform),
            "audio": metadata, "warnings": warnings}

@app.post("/api/classify")
async def classify_upload(request: Request):
    # Read the body with a hard limit before multipart parsing. Multipart overhead
    # has its own allowance; the actual file still has a strict 10 MiB limit.
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BYTES + 64 * 1024:
            raise AudioError("The upload exceeds the 10 MiB file limit.", "file_too_large", 413)
    request._body = bytes(body)
    try:
        async with request.form(max_files=1, max_fields=0, max_part_size=MAX_BYTES) as form:
            items = form.multi_items()
            if len(items) != 1 or items[0][0] != "audio" or not isinstance(items[0][1], UploadFile):
                raise AudioError("Upload exactly one WAV or FLAC file in the audio field.", "invalid_request")
            upload = items[0][1]
            data = await upload.read(MAX_BYTES + 1)
            return await run_in_threadpool(classify, request.app.state.models, data)
    except AudioError:
        raise
    except HTTPException as error:
        raise AudioError("Upload exactly one WAV or FLAC file as multipart form data.", "invalid_request") from error
    except Exception:
        logger.exception("Accent inference failed")
        return JSONResponse({"error": {"code": "inference_failed", "message": "Classification failed. Please try another WAV or FLAC file."}}, status_code=500)
