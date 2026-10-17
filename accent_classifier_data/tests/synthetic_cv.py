"""
synthetic_cv.py

Build a small synthetic Common Voice-style archive for tests (no network,
no real data). Everything is generated in memory and written as ONE
archive file into the folder you pass in (use a temp folder).

Layout:
    cv-corpus-<version>-2099-01-01/en/clips/common_voice_en_<n>.mp3
    cv-corpus-<version>-2099-01-01/en/{validated,train,dev,test,clip_durations}.tsv

Order (default, tsv_first=False): train.tsv, then all clips, then the
remaining TSVs -> exercises the scan "read through" path. Within the clips,
the default-negative groups come first and Singaporean / Ambiguous /
Unmapped / Unlabelled clips come last -> exercises the extract early exit.
"""

import io
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf


# (raw accent string, n_speakers, clips_per_speaker, expected group)
SPEAKER_PLAN = [
    ("Malaysian English",                                  4, 4, "Malaysian"),
    ("Hong Kong English",                                  3, 3, "Hong Kong"),
    ("Chinese English",                                    3, 3, "Mainland Chinese"),
    ("Korean",                                             3, 4, "Korean"),
    ("India and South Asia (India, Pakistan, Sri Lanka)",  3, 3, "South Asian"),
    ("Filipino",                                           3, 3, "Filipino"),
    ("United States English|Canadian English",             4, 3, "American"),
    ("Singaporean English",                                2, 3, "Singaporean"),
    ("Malaysian English|Singaporean English",              1, 2, "Ambiguous"),
    ("something unusual",                                  2, 2, "Unmapped"),
    ("",                                                   1, 2, "Unlabelled"),
]
# One extra Malaysian speaker with many clips (tests --max_clips_per_speaker).
HEAVY_SPEAKER = ("Malaysian English", 12, "Malaysian")

LATE_GROUPS = {"Singaporean", "Ambiguous", "Unmapped", "Unlabelled"}

TSV_COLUMNS = ["client_id", "path", "sentence_id", "sentence", "sentence_domain",
               "up_votes", "down_votes", "age", "gender", "accents", "variant",
               "locale", "segment"]


def _expected() -> Dict[str, Dict[str, int]]:
    exp: Dict[str, Dict[str, int]] = {}
    for _, n_spk, n_clips, group in SPEAKER_PLAN:
        e = exp.setdefault(group, {"n_clips": 0, "n_speakers": 0})
        e["n_clips"] += n_spk * n_clips
        e["n_speakers"] += n_spk
    exp[HEAVY_SPEAKER[2]]["n_clips"] += HEAVY_SPEAKER[1]
    exp[HEAVY_SPEAKER[2]]["n_speakers"] += 1
    return exp


EXPECTED = _expected()

# Clip ids with special audio (filled by build_clips; stable for a given seed).
SPECIAL_CLIPS: Dict[str, str] = {}


def _tone(rng: np.random.Generator, sr: int, voiced_sec: float, stereo: bool) -> np.ndarray:
    """0.5 s silence + voiced tone-plus-noise + 0.5 s silence."""
    n_sil = int(0.5 * sr)
    n_voiced = int(voiced_sec * sr)
    t = np.arange(n_voiced) / sr
    f0 = rng.uniform(120, 300)
    amp = rng.uniform(0.05, 0.6)
    voiced = amp * (0.7 * np.sin(2 * np.pi * f0 * t)
                    + 0.3 * np.sin(2 * np.pi * 2.3 * f0 * t)
                    + 0.05 * rng.standard_normal(n_voiced))
    x = np.concatenate([np.zeros(n_sil), voiced, np.zeros(n_sil)]).astype(np.float32)
    if stereo:
        x = np.stack([x, 0.8 * x], axis=1)
    return x


def encode_mp3(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="MP3")
    return buf.getvalue()


def build_clips(seed: int = 0) -> List[dict]:
    """Return clip dicts: client_id, path, accent, group, sr, duration_ms, mp3."""
    rng = np.random.default_rng(seed)
    clips: List[dict] = []
    n = 0
    SPECIAL_CLIPS.clear()

    speakers = []
    for raw, n_spk, n_clips, group in SPEAKER_PLAN:
        for _ in range(n_spk):
            speakers.append((raw, n_clips, group))
    speakers.append((HEAVY_SPEAKER[0], HEAVY_SPEAKER[1], HEAVY_SPEAKER[2]))

    for raw, n_clips, group in speakers:
        client_id = rng.bytes(16).hex() * 2       # 64 hex chars, like CV
        for _ in range(n_clips):
            n += 1
            sr = 48000 if n % 3 else 44100
            voiced = float(rng.uniform(3.0, 6.0))
            stereo = False
            if group == "Malaysian" and "too_short_prefilter" not in SPECIAL_CLIPS:
                voiced, SPECIAL_CLIPS["too_short_prefilter"] = 1.5, f"common_voice_en_{n}"
            elif group == "Hong Kong" and "too_short_trim" not in SPECIAL_CLIPS:
                voiced, SPECIAL_CLIPS["too_short_trim"] = 2.5, f"common_voice_en_{n}"
            elif group == "Korean" and "too_long" not in SPECIAL_CLIPS:
                voiced, SPECIAL_CLIPS["too_long"] = 20.0, f"common_voice_en_{n}"
            elif group == "American" and "stereo" not in SPECIAL_CLIPS:
                stereo, SPECIAL_CLIPS["stereo"] = True, f"common_voice_en_{n}"
            x = _tone(rng, sr, voiced, stereo)
            clips.append({
                "client_id": client_id,
                "path": f"common_voice_en_{n}.mp3",
                "accent": raw,
                "group": group,
                "sr": sr,
                "duration_ms": int(round(1000 * (voiced + 1.0))),
                "mp3": encode_mp3(x, sr),
            })
    return clips


def _tsv(rows: List[List[str]], header: List[str]) -> bytes:
    lines = ["\t".join(header)] + ["\t".join(r) for r in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_tsvs(clips: List[dict], accent_column: str = "accents") -> Dict[str, bytes]:
    header = [accent_column if c == "accents" else c for c in TSV_COLUMNS]
    rows = []
    for i, c in enumerate(clips):
        sentence = f'Clip {i} says "hello" and it\'s fine.'
        rows.append([c["client_id"], c["path"], f"s{i}", sentence, "", "2", "0",
                     "twenties", "", c["accent"], "", "en", ""])
    validated = _tsv(rows, header)
    durations = _tsv([[c["path"], str(c["duration_ms"])] for c in clips],
                     ["clip", "duration[ms]"])
    split_rows = rows[: max(1, len(rows) // 2)]
    return {
        "train.tsv":          _tsv(split_rows, header),
        "validated.tsv":      validated,
        "clip_durations.tsv": durations,
        "dev.tsv":            _tsv(rows[:2], header),
        "test.tsv":           _tsv(rows[2:4], header),
    }


def ordered_members(clips: List[dict], tsvs: Dict[str, bytes], prefix: str,
                    tsv_first: bool) -> List[tuple]:
    early = [c for c in clips if c["group"] not in LATE_GROUPS]
    late = [c for c in clips if c["group"] in LATE_GROUPS]
    clip_members = [(f"{prefix}/clips/{c['path']}", c["mp3"]) for c in early + late]
    tsv_members = [(f"{prefix}/{name}", data) for name, data in tsvs.items()]
    if tsv_first:
        return tsv_members + clip_members
    # train.tsv before the clips, the rest after (read-through path).
    return tsv_members[:1] + clip_members + tsv_members[1:]


def write_archive(members: List[tuple], path: Path, kind: str) -> Path:
    if kind == "zip":
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
            for name, data in members:
                zf.writestr(name, data)
        return path
    mode = {"tar.gz": "w:gz", "tar": "w"}[kind]
    with tarfile.open(path, mode) as tf:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            tf.addfile(info, io.BytesIO(data))
    return path


def make_synthetic_archive(
    out_dir: Path,
    kind: str = "tar.gz",
    tsv_first: bool = False,
    accent_column: str = "accents",
    version: str = "27.0-test",
    seed: int = 0,
) -> Path:
    """Write cv-synthetic.<kind> into out_dir and return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = build_clips(seed)
    tsvs = build_tsvs(clips, accent_column)
    prefix = f"cv-corpus-{version}/en"
    members = ordered_members(clips, tsvs, prefix, tsv_first)
    return write_archive(members, out_dir / f"cv-synthetic.{kind}", kind)
