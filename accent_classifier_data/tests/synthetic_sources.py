"""
synthetic_sources.py

Small synthetic inputs for the speechocean762 and Speech Accent Archive
prepare scripts (no network, no real data). Audio is generated in memory
and written only into the folder the caller passes in (use a temp folder);
the Kaggle-style builders write MP3s because that is the real input format.
"""

import io
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf


def _voiced(rng: np.random.Generator, sr: int, sec: float) -> np.ndarray:
    n = int(sec * sr)
    t = np.arange(n) / sr
    f0 = rng.uniform(120, 300)
    amp = rng.uniform(0.1, 0.5)
    return amp * (0.7 * np.sin(2 * np.pi * f0 * t)
                  + 0.3 * np.sin(2 * np.pi * 2.3 * f0 * t)
                  + 0.05 * rng.standard_normal(n))


def _floor(rng: np.random.Generator, sr: int, sec: float) -> np.ndarray:
    return 1e-4 * rng.standard_normal(int(sec * sr))


def speech_like(rng: np.random.Generator, sr: int,
                segments: Sequence[Tuple[str, float]]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Concatenate ("v", sec) voiced bursts and ("p", sec) pauses.

    Returns float32 audio and the (start, end) sample ranges of the pauses.
    """
    parts, pauses, pos = [], [], 0
    for kind, sec in segments:
        x = _voiced(rng, sr, sec) if kind == "v" else _floor(rng, sr, sec)
        if kind == "p":
            pauses.append((pos, pos + len(x)))
        parts.append(x)
        pos += len(x)
    return np.concatenate(parts).astype(np.float32), pauses


def paragraph_segments(rng: np.random.Generator, total_sec: float = 30.0) -> List[Tuple[str, float]]:
    """~total_sec of 1.5-3 s voiced phrases separated by 0.3-0.7 s pauses."""
    segs: List[Tuple[str, float]] = [("p", 0.5)]
    t = 0.5
    while t < total_sec - 0.5:
        v = float(rng.uniform(1.5, 3.0))
        p = float(rng.uniform(0.3, 0.7))
        segs += [("v", v), ("p", p)]
        t += v + p
    return segs


def _wav(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _mp3(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="MP3")
    return buf.getvalue()


# ============================================================
# speechocean762
# ============================================================

# (speaker, age, n_utterances, hf split)
SO_SPEAKERS = [
    ("0001", 18, 6, "train"), ("0002", 25, 6, "train"), ("0003", 37, 20, "train"),
    ("0004", 8, 6, "train"), ("0005", 17, 6, "train"),
    ("0006", 20, 6, "test"), ("0007", 30, 6, "test"), ("0008", 12, 6, "test"),
]

SO_SCHEMA = pa.schema([
    ("speaker", pa.string()),
    ("gender", pa.string()),
    ("age", pa.int64()),
    ("text", pa.string()),
    ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
])


def make_speechocean_parquets(out_dir: Path, seed: int = 0) -> Dict:
    """Write data/{train,test}-00000-of-00001.parquet; return expectations."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "test": []}
    special = {}
    for spk, age, n_utt, split in SO_SPEAKERS:
        for u in range(n_utt):
            voiced = float(rng.uniform(3.5, 6.0))
            stem = f"{spk}{u:04d}"
            if spk == "0001" and u == 0:
                voiced, special["too_short"] = 1.5, f"so762_{stem}"
            elif spk == "0002" and u == 0:
                voiced, special["too_long"] = 18.0, f"so762_{stem}"
            x, _ = speech_like(rng, 16000, [("p", 0.3), ("v", voiced), ("p", 0.3)])
            rows[split].append({
                "speaker": spk, "gender": "f" if int(spk) % 2 else "m", "age": age,
                "text": f"SENTENCE {u} OF SPEAKER {spk}",
                "audio": {"bytes": _wav(x, 16000), "path": f"{stem}.wav"},
            })
    for split, r in rows.items():
        pq.write_table(pa.Table.from_pylist(r, schema=SO_SCHEMA),
                       str(out_dir / "data" / f"{split}-00000-of-00001.parquet"))
    adults = [s for s, a, _, _ in SO_SPEAKERS if a >= 18]
    children = [s for s, a, _, _ in SO_SPEAKERS if a < 18]
    return {
        "adult_speakers": adults,
        "child_speakers": children,
        "adult_clips": sum(n for _, a, n, _ in SO_SPEAKERS if a >= 18),
        "child_clips": sum(n for _, a, n, _ in SO_SPEAKERS if a < 18),
        "special": special,
    }


# ============================================================
# Speech Accent Archive
# ============================================================

KAGGLE_COLUMNS = ["age", "age_onset", "birthplace", "filename", "native_language",
                  "sex", "speakerid", "country", "file_missing?"]
KAGGLE_KOREAN = [f"korean{i}" for i in range(1, 6)]


def _kaggle_members(seed: int) -> List[Tuple[str, bytes]]:
    rng = np.random.default_rng(seed)
    lines = [",".join(KAGGLE_COLUMNS)]
    members: List[Tuple[str, bytes]] = []
    speakers = [(k, "korean", "FALSE") for k in KAGGLE_KOREAN]
    speakers += [("spanish1", "spanish", "FALSE"), ("korean6", "korean", "TRUE")]
    for i, (name, lang, missing) in enumerate(speakers):
        lines.append(f"30,10,\"seoul, south korea\",{name},{lang},female,{i + 1},south korea,{missing}")
        if missing == "TRUE":
            continue
        x, _ = speech_like(rng, 44100, paragraph_segments(rng, 30.0))
        members.append((f"recordings/recordings/{name}.mp3", _mp3(x, 44100)))
    members.insert(0, ("speakers_all.csv", ("\n".join(lines) + "\n").encode("utf-8")))
    return members


def make_saa_kaggle_zip(path: Path, seed: int = 0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in _kaggle_members(seed):
            zf.writestr(name, data)
    return path


def make_saa_kaggle_dir(path: Path, seed: int = 0) -> Path:
    path = Path(path)
    for name, data in _kaggle_members(seed):
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return path


# ============================================================
# MNSC (Hub layout: <config>/train-NNNNN-of-MMMMM.parquet)
# ============================================================

MNSC_SPEAKERS = [f"S{i:02d}" for i in range(10)]
MNSC_SPECIAL_ANSWER = "<Speaker1>: [ah] Take the {MRT} (uh) to #Sentosa# lah"
MNSC_SPECIAL_SENTENCE = "ah take the mrt uh to sentosa lah"
_CONTEXT = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
MNSC_PART1_SCHEMA = pa.schema([
    ("context", _CONTEXT),
    ("instruction", pa.string()),
    ("answer", pa.string()),
    ("other_attributes", pa.struct([
        ("id", pa.string()), ("channel", pa.string()),
        ("speaker", pa.struct([("gender", pa.string()), ("speaker_id", pa.string())])),
    ])),
])
MNSC_PART3_SCHEMA = pa.schema([
    ("context", _CONTEXT),
    ("instruction", pa.string()),
    ("answer", pa.string()),
    ("other_attributes", pa.string()),
])


def _mnsc_row(rng, sr: int, voiced: float, answer: str, spk: str, uid: str) -> dict:
    x, _ = speech_like(rng, sr, [("p", 0.3), ("v", voiced), ("p", 0.3)])
    return {"context": {"bytes": _wav(x, sr), "path": None},
            "instruction": "Transcribe the audio.", "answer": answer,
            "other_attributes": {"id": uid, "channel": "0",
                                 "speaker": {"gender": "F", "speaker_id": spk}}}


def make_mnsc_source(out_dir: Path, seed: int = 0) -> Dict:
    """Write ASR-PART1-Train (3 shards), ASR-PART3-Train, ASR-PART1-Test."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    special: Dict[str, str] = {}
    for shard in range(3):
        sr = 44100 if shard == 2 else 16000
        rows = []
        for spk in MNSC_SPEAKERS:
            for u in range(3):
                rows.append(_mnsc_row(rng, sr, float(rng.uniform(3.5, 6.0)),
                                      f"Sentence {shard} {u} of {spk}", spk, f"{spk}-{shard}-{u}"))
        if shard == 0:
            n = len(rows)
            rows.append(_mnsc_row(rng, sr, 1.5, "a very short one", "S00", "short"))
            special["too_short"] = f"mnsc_part1-train_00000_{n:05d}"
            rows.append(_mnsc_row(rng, sr, 18.0, "a very long one", "S01", "long"))
            special["too_long"] = f"mnsc_part1-train_00000_{n + 1:05d}"
            rows.append(_mnsc_row(rng, sr, 4.0, "no speaker here", "", "noid"))
            special["no_id"] = f"mnsc_part1-train_00000_{n + 2:05d}"
            rows.append(_mnsc_row(rng, sr, 4.0, "", "S02", "noanswer"))
            special["no_answer"] = f"mnsc_part1-train_00000_{n + 3:05d}"
            rows.append(_mnsc_row(rng, sr, 4.0, "Sentence 0 0 of S03", "S03", "dup"))
            special["duplicate"] = f"mnsc_part1-train_00000_{n + 4:05d}"
            rows.append(_mnsc_row(rng, sr, 4.0, MNSC_SPECIAL_ANSWER, "S04", "special"))
            special["normalised"] = f"mnsc_part1-train_00000_{n + 5:05d}"
        path = out_dir / "ASR-PART1-Train" / f"train-{shard:05d}-of-00003.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=MNSC_PART1_SCHEMA), str(path))

    rows = []
    for u in range(5):
        voiced = float(rng.uniform(4.0, 5.4))
        x, _ = speech_like(rng, 16000, [("p", 0.3), ("v", voiced), ("p", 0.3)])
        attrs = {"conversation_id": "c1", "start": "10.0", "end": str(10.0 + voiced + 0.6),
                 "speaker1": json.dumps({"speaker_id": "", "gender": "M"}),
                 "speaker2": {"speaker_id": "P3-2", "gender": "F"}}
        rows.append({"context": {"bytes": _wav(x, 16000), "path": None},
                     "instruction": "Transcribe.", "answer": f"<Speaker2>: part three {u}",
                     "other_attributes": json.dumps(attrs)})
    path = out_dir / "ASR-PART3-Train" / "train-00000-of-00001.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=MNSC_PART3_SCHEMA), str(path))

    rows = [_mnsc_row(rng, 16000, 4.0, f"test sentence {u}", f"T{u}", f"t{u}") for u in range(4)]
    path = out_dir / "ASR-PART1-Test" / "train-00000-of-00001.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=MNSC_PART1_SCHEMA), str(path))
    return {"speakers": list(MNSC_SPEAKERS), "part3_speaker": "P3-2",
            "special": special, "normalised_sentence": MNSC_SPECIAL_SENTENCE}


MIRROR_SCHEMA = pa.schema([
    ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("id", pa.string()),
    ("accent", pa.string()),
])


def make_saa_mirror_parquet(path: Path, seed: int = 0) -> Path:
    """data/original-00000-of-00001.parquet: 4 korean, 1 mandarin, 1 spanish."""
    rng = np.random.default_rng(seed)
    path = Path(path)
    (path / "data").mkdir(parents=True, exist_ok=True)
    rows = []
    for rid in ["korean1", "korean2", "korean3", "korean4", "mandarin1", "spanish1"]:
        x, _ = speech_like(rng, 16000, paragraph_segments(rng, 25.0))
        rows.append({"audio": {"bytes": _wav(x, 16000), "path": f"{rid}.wav"},
                     "id": rid, "accent": rid.rstrip("0123456789")})
    pq.write_table(pa.Table.from_pylist(rows, schema=MIRROR_SCHEMA),
                   str(path / "data" / "original-00000-of-00001.parquet"))
    return path
