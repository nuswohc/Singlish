"""
benchmark_stream.py

Rough streaming-throughput benchmark for the pass-1 / pass-2 scripts on a
large synthetic Common Voice-style .tar.gz (real ~39 KB MP3 members, TSVs
at the END so both passes read the whole archive). Used to estimate how
long a pass over the real ~95 GB archive takes on this machine.

Usage (from the project root; use a temp folder, it is deleted afterwards):
    python tests/benchmark_stream.py --out_dir %TEMP%\\cv_bench --target_mb 400
"""

import argparse
import contextlib
import io
import shutil
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cv_utils  # noqa: E402
import extract_common_voice_subset as extract  # noqa: E402
import scan_common_voice_metadata as scan  # noqa: E402

REAL_ARCHIVE_GB = 95.0
GROUPS = ["Malaysian English", "Korean", "Hong Kong English", "Chinese English"]


def build(out_dir: Path, target_mb: int, n_selected: int) -> Path:
    rng = np.random.default_rng(0)
    pool = []
    for i in range(64):
        t = np.arange(32000 * 5) / 32000
        x = (0.3 * np.sin(2 * np.pi * (150 + 5 * i) * t)
             + 0.05 * rng.standard_normal(len(t))).astype(np.float32)
        x = np.concatenate([np.zeros(8000, np.float32), x, np.zeros(8000, np.float32)])
        buf = io.BytesIO()
        sf.write(buf, x, 32000, format="MP3")
        pool.append(buf.getvalue())

    n_members = int(target_mb * 1e6 / np.mean([len(p) for p in pool]))
    step = max(1, n_members // n_selected)
    selected = set(range(step - 1, n_members, step))
    rows, durs = [], []
    path = out_dir / "cv-bench.tar.gz"
    prefix = "cv-corpus-27.0-bench/en"
    with tarfile.open(path, "w:gz", compresslevel=6) as tf:
        for i in range(n_members):
            name = f"common_voice_en_{i}.mp3"
            data = pool[i % len(pool)]
            info = tarfile.TarInfo(f"{prefix}/clips/{name}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
            if i in selected:
                spk = f"spk{i % 40:03d}"
                rows.append(f"{spk}\t{name}\tsentence {i}\t{GROUPS[i % len(GROUPS)]}")
                durs.append(f"{name}\t5500")
        for fname, text in (
            ("validated.tsv", "client_id\tpath\tsentence\taccents\n" + "\n".join(rows) + "\n"),
            ("clip_durations.tsv", "clip\tduration[ms]\n" + "\n".join(durs) + "\n"),
        ):
            b = text.encode("utf-8")
            info = tarfile.TarInfo(f"{prefix}/{fname}")
            info.size = len(b)
            tf.addfile(info, io.BytesIO(b))
    print(f"Built {path.name}: {path.stat().st_size / 1e6:.1f} MB compressed, "
          f"{n_members:,} members, {len(rows)} selectable clips (spread to the end)")
    return path


def timed(label: str, fn, size_mb: float) -> float:
    t0 = time.perf_counter()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = fn()
    dt = time.perf_counter() - t0
    assert code in (0, None), out.getvalue()
    mbps = size_mb / dt
    est_min = REAL_ARCHIVE_GB * 1000 / mbps / 60
    print(f"{label:<44} {dt:7.1f} s  {mbps:7.1f} MB/s  "
          f"-> ~{est_min:5.0f} min for {REAL_ARCHIVE_GB:.0f} GB")
    return mbps


def raw_stream(path: Path) -> int:
    with cv_utils.ArchiveStreamer(path) as s:
        for _ in s:
            pass
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--target_mb", type=int, default=400)
    p.add_argument("--n_selected", type=int, default=400)
    p.add_argument("--keep", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        arc = build(args.out_dir, args.target_mb, args.n_selected)
        size_mb = arc.stat().st_size / 1e6
        data = args.out_dir / "data"
        a = ["--archive", str(arc), "--data_dir", str(data)]
        print()
        timed("ArchiveStreamer only (no reads)", lambda: raw_stream(arc), size_mb)
        timed("pass 1 scan (TSVs at end, read-through)", lambda: scan.main(a), size_mb)
        timed("pass 2 extract (selected spread to end)",
              lambda: extract.main(a + ["--clips_per_accent", "80",
                                        "--max_clips_per_speaker", "15"]), size_mb)
    finally:
        if not args.keep:
            shutil.rmtree(args.out_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
