"""Exact librosa39-v1 features from the E1 39-feature study."""
import librosa
import numpy as np

SAMPLE_RATE = 16_000
MAX_SECONDS = 6
N_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 256
EXTRACTION_VERSION = "librosa39-v1"

FEATURE_NAMES = (
    [f'mfcc_{i + 1}_mean' for i in range(N_MFCC)]
    + [f'mfcc_{i + 1}_std' for i in range(N_MFCC)]
    + ['f0_mean', 'f0_std', 'voiced_ratio']
    + [
        f'{family}_{stat}'
        for family in ('spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'zcr', 'rms')
        for stat in ('mean', 'std')
    ]
)
assert len(FEATURE_NAMES) == 39


def prepare_waveform(y):
    y, _ = librosa.effects.trim(y, top_db=30)
    if y.size == 0:
        raise ValueError('No audible signal')
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    y = librosa.util.fix_length(y, size=max(len(y), SAMPLE_RATE))
    return y[:SAMPLE_RATE * MAX_SECONDS].astype(np.float32)


def mean_and_std(feature):
    return np.concatenate([np.mean(feature, axis=1), np.std(feature, axis=1)])


def extract_features(y):
    mfcc = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    f0 = librosa.yin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        sr=SAMPLE_RATE, frame_length=2048, hop_length=HOP_LENGTH,
    )
    energy = rms.ravel()[:len(f0)]
    voiced = np.isfinite(f0) & (energy > max(float(np.median(energy)) * 0.1, 1e-4))
    voiced_f0 = f0[voiced]
    pitch_stats = [
        float(np.mean(voiced_f0)) if voiced_f0.size else 0.0,
        float(np.std(voiced_f0)) if voiced_f0.size else 0.0,
        float(np.mean(voiced)),
    ]
    time_features = [
        librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH),
        librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH),
        librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH),
        librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH),
        rms,
    ]
    vector = np.concatenate([mean_and_std(mfcc), pitch_stats] + [mean_and_std(x) for x in time_features])
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    assert vector.shape == (len(FEATURE_NAMES),)
    return vector
