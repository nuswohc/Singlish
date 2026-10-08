"""Bounded WAV decoding; no recordings are written to disk."""
from io import BytesIO
import librosa
import numpy as np
import soundfile as sf
from .features import SAMPLE_RATE, MAX_SECONDS, prepare_waveform

MAX_BYTES = 10 * 1024 * 1024
MAX_DURATION = 10

class AudioError(Exception):
    def __init__(self, message, code="invalid_audio", status=422):
        self.message, self.code, self.status = message, code, status
        super().__init__(message)


def decode_audio(data):
    if not data:
        raise AudioError("Choose a nonempty WAV file.")
    if len(data) > MAX_BYTES:
        raise AudioError("The file exceeds the 10 MiB limit.", "file_too_large", 413)
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioError("Upload a PCM WAV file.", "unsupported_format", 415)
    try:
        info = sf.info(BytesIO(data))
        if info.format != "WAV" or info.subtype not in {"PCM_U8", "PCM_16", "PCM_24", "PCM_32"}:
            raise AudioError("Use an uncompressed PCM WAV file.", "unsupported_encoding", 415)
        if info.frames <= 0 or info.samplerate <= 0:
            raise AudioError("The WAV contains no audio.")
        if info.duration > MAX_DURATION:
            raise AudioError("Use a WAV recording of 10 seconds or less.", "audio_too_long")
        # Bound decoder allocations independently of the claimed duration.
        if info.channels > 8 or info.samplerate > 192_000 or info.frames * info.channels > 4_000_000:
            raise AudioError("Use a WAV with at most 8 channels and a sample rate up to 192 kHz.")
        y, _ = librosa.load(BytesIO(data), sr=SAMPLE_RATE, mono=True)
    except AudioError:
        raise
    except (ValueError, RuntimeError, OSError, EOFError) as error:
        raise AudioError("The WAV could not be decoded. Export it again as PCM WAV.") from error
    if not y.size or not np.isfinite(y).all() or not np.any(y):
        raise AudioError("The recording is empty or silent. Upload clear English speech.")
    trimmed, _ = librosa.effects.trim(y, top_db=30)
    if not trimmed.size or not np.any(trimmed):
        raise AudioError("No audible signal remains after trimming.")
    duration = len(trimmed) / SAMPLE_RATE
    warnings = []
    if duration < 1:
        warnings.append("Short sample; please use 3–6 seconds of speech for a more useful result.")
    if duration > MAX_SECONDS:
        warnings.append("Only the first six seconds after boundary trimming were classified.")
    return prepare_waveform(y), {
        "duration_seconds": round(info.duration, 4),
        "classified_duration_seconds": round(min(duration, MAX_SECONDS), 4),
        "truncated": duration > MAX_SECONDS,
    }, warnings
