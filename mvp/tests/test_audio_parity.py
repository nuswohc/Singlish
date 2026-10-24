"""The classifier waveform must stay bit-identical after the audio.py split, and the
transcriber must receive the full trimmed upload."""
from io import BytesIO
import numpy as np
import pytest
import soundfile as sf
import soxr
from mvp.audio import (AudioError, MAX_BYTES, MAX_DURATION, decode_audio,
                       prepare_transcription_audio, read_audio)
from mvp.features import SAMPLE_RATE, MIN_SECONDS, MAX_SECONDS, trim_silence, normalize_loudness


def legacy_decode_audio(data):
    """Verbatim copy of decode_audio before the transcription change."""
    if not data:
        raise AudioError("Choose a nonempty WAV or FLAC file.")
    if len(data) > MAX_BYTES:
        raise AudioError("The file exceeds the 10 MiB limit.", "file_too_large", 413)
    is_wav = data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    if not is_wav and data[:4] != b"fLaC":
        raise AudioError("Upload a PCM WAV or FLAC file.", "unsupported_format", 415)
    try:
        info = sf.info(BytesIO(data))
        if info.format not in {"WAV", "FLAC"} or info.subtype not in {"PCM_U8", "PCM_S8", "PCM_16", "PCM_24", "PCM_32"}:
            raise AudioError("Use an uncompressed PCM WAV or lossless FLAC file.", "unsupported_encoding", 415)
        if info.frames <= 0 or info.samplerate <= 0:
            raise AudioError("The file contains no audio.")
        if info.duration > MAX_DURATION:
            raise AudioError("Use a WAV or FLAC recording of 10 seconds or less.", "audio_too_long")
        # Bound decoder allocations independently of the claimed duration.
        if info.channels > 8 or info.samplerate > 192_000 or info.frames * info.channels > 4_000_000:
            raise AudioError("Use a WAV or FLAC file with at most 8 channels and a sample rate up to 192 kHz.")
        y, rate = sf.read(BytesIO(data), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if rate != SAMPLE_RATE:
            y = soxr.resample(y, rate, SAMPLE_RATE, quality="HQ").astype(np.float32)
    except AudioError:
        raise
    except (ValueError, RuntimeError, OSError, EOFError) as error:
        raise AudioError("The audio could not be decoded. Export it again as PCM WAV or FLAC.") from error
    if not y.size or not np.isfinite(y).all() or not np.any(y):
        raise AudioError("The recording is empty or silent. Upload clear English speech.")
    y = trim_silence(y)
    duration = len(y) / SAMPLE_RATE
    if duration < MIN_SECONDS:
        raise AudioError("The clip is less than 3 seconds after silence trimming. Please retry with a longer recording.", "invalid_duration")
    truncated = duration > MAX_SECONDS
    warnings = []
    if truncated:
        y = y[:SAMPLE_RATE * MAX_SECONDS]
        warnings.append("Only the first six seconds after silence trimming were classified.")
    try:
        y = normalize_loudness(y)
    except ValueError as error:
        raise AudioError(str(error)) from error
    # Training features used decoded PCM16 FLAC. Reproduce that quantization
    # in memory using the same FLAC encoder (no file is retained).
    stream = BytesIO()
    sf.write(stream, y, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    stream.seek(0)
    y, _ = sf.read(stream, dtype="float32")
    return y, {
        "duration_seconds": round(info.duration, 4),
        "classified_duration_seconds": round(min(duration, MAX_SECONDS), 4),
        "truncated": truncated,
    }, warnings


def encode(y, rate, format="WAV", subtype="PCM_16"):
    stream = BytesIO()
    sf.write(stream, y, rate, format=format, subtype=subtype)
    return stream.getvalue()


def speech(seconds, rate=16000, channels=1, amplitude=0.15, pad=0.0, format="WAV", subtype="PCM_16"):
    # A tone with a slow amplitude envelope, so trimming and resampling have work to do.
    t = np.arange(int(seconds * rate)) / rate
    y = amplitude * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t))
    if pad:
        y = np.pad(y, (int(pad * rate), int(pad * rate)))
    if channels > 1:
        y = np.stack([y, 0.5 * y], axis=1)
    return encode(y, rate, format, subtype)


CASES = {
    "3s_16k_wav": speech(3),
    "6s_44k_stereo_flac": speech(6, 44100, 2, format="FLAC"),
    "8s_wav_truncated": speech(8),
    "10s_flac": speech(10, format="FLAC"),
    "4s_padded": speech(4, pad=1.0),
    "pcm24_48k": speech(5, 48000, subtype="PCM_24"),
    "pcm_u8_8k": speech(4, 8000, subtype="PCM_U8"),
    "quiet_tone": speech(4, amplitude=0.02),
}


@pytest.mark.parametrize("name", list(CASES))
def test_classifier_waveform_is_bit_identical(name):
    data = CASES[name]
    new, legacy = decode_audio(data), legacy_decode_audio(data)
    assert new[0].dtype == legacy[0].dtype
    assert np.array_equal(new[0], legacy[0])
    assert new[1] == legacy[1]
    assert new[2] == legacy[2]


ERRORS = {
    "empty": b"",
    "bad_header": b"RIFF1234WAVEbad",
    "float_wav": encode(np.zeros(16000) + 0.1, 16000, subtype="FLOAT"),
    "too_long": speech(11),
    "too_short": speech(2),
    "silent": encode(np.zeros(4 * 16000), 16000),
    "too_quiet": encode(np.full(4 * 16000, 1e-7), 16000, subtype="PCM_32"),
}


@pytest.mark.parametrize("name", list(ERRORS))
def test_classifier_errors_are_identical(name):
    with pytest.raises(AudioError) as new:
        decode_audio(ERRORS[name])
    with pytest.raises(AudioError) as legacy:
        legacy_decode_audio(ERRORS[name])
    assert (new.value.code, new.value.message, new.value.status) == \
        (legacy.value.code, legacy.value.message, legacy.value.status)


@pytest.mark.parametrize("name,seconds", [("8s_wav_truncated", 8), ("10s_flac", 10)])
def test_transcription_gets_full_trimmed_upload(name, seconds):
    data = CASES[name]
    y, metadata = prepare_transcription_audio(*read_audio(data))
    expected = trim_silence(read_audio(data)[0])
    assert y.dtype == np.float32
    assert np.array_equal(y, expected)
    assert len(y) > MAX_SECONDS * SAMPLE_RATE
    assert metadata == {"duration_seconds": seconds,
                        "transcribed_duration_seconds": round(len(y) / SAMPLE_RATE, 4)}


def test_transcription_audio_is_trimmed_but_not_normalized():
    rate = 16000
    t = np.arange(4 * rate) / rate
    data = encode(np.pad(0.02 * np.sin(2 * np.pi * 180 * t), (rate, rate)), rate)
    y, metadata = prepare_transcription_audio(*read_audio(data))
    # ~0.0141 for the tone, slightly lower with the 2 x 100 ms trim padding; the classifier gives 0.1.
    assert 0.013 <= np.sqrt(np.mean(y.astype(np.float64) ** 2)) <= 0.0142
    assert metadata["duration_seconds"] == 6
    assert 4 <= metadata["transcribed_duration_seconds"] <= 4.25


def test_transcription_rejects_short_clip_after_trimming():
    data = speech(2.5, pad=1.0)
    with pytest.raises(AudioError) as error:
        prepare_transcription_audio(*read_audio(data))
    assert error.value.code == "invalid_duration"
    assert error.value.message == "The clip is less than 3 seconds after silence trimming. Please retry with a longer recording."
