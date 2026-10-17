"""Unit tests for scripts/cv_utils.py. Run from the project root:
    python -m unittest discover -s tests -v
"""

import io
import os
import sys
import tarfile
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import cv_utils  # noqa: E402


def _make_members():
    return [("top/en/a.tsv", b"x\ty\n1\t2\n"), ("top/en/clips/b.mp3", b"\x00" * 100)]


class ArchiveTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _tar(self, name, mode):
        p = self.dir / name
        with tarfile.open(p, mode) as tf:
            for n, d in _make_members():
                info = tarfile.TarInfo(n)
                info.size = len(d)
                tf.addfile(info, io.BytesIO(d))
        return p

    def _zip(self, name):
        p = self.dir / name
        with zipfile.ZipFile(p, "w") as zf:
            for n, d in _make_members():
                zf.writestr(n, d)
        return p

    def test_detect_types_by_magic(self):
        self.assertEqual(cv_utils.detect_archive_type(self._tar("a.tar.gz", "w:gz")), "tar")
        self.assertEqual(cv_utils.detect_archive_type(self._tar("a.tar", "w")), "tar")
        self.assertEqual(cv_utils.detect_archive_type(self._tar("a.tar.bz2", "w:bz2")), "tar")
        self.assertEqual(cv_utils.detect_archive_type(self._zip("a.zip")), "zip")
        # Extension is ignored: a zip named .tar.gz is still a zip.
        z = self._zip("b.zip")
        renamed = z.rename(self.dir / "misnamed.tar.gz")
        self.assertEqual(cv_utils.detect_archive_type(renamed), "zip")
        bad = self.dir / "bad.bin"
        bad.write_bytes(b"hello world" * 100)
        with self.assertRaises(ValueError):
            cv_utils.detect_archive_type(bad)

    def test_find_latest_archive_ignores_partial(self):
        with self.assertRaises(FileNotFoundError):
            cv_utils.find_latest_archive(self.dir / "missing")
        older = self._tar("older.tar.gz", "w:gz")
        os.utime(older, (time.time() - 100, time.time() - 100))
        (self.dir / "newer.tar.gz.part").write_bytes(b"partial")
        (self.dir / "newer.tar.gz.checksum").write_text("abc")
        self.assertEqual(cv_utils.find_latest_archive(self.dir), older)
        newer = self._zip("newest.zip")
        self.assertEqual(cv_utils.find_latest_archive(self.dir), newer)

    def test_streamer_yields_members(self):
        for p in (self._tar("s.tar.gz", "w:gz"), self._zip("s.zip")):
            with cv_utils.ArchiveStreamer(p) as s:
                got = {e.name: e.read() for e in s}
                self.assertEqual(s.members_seen, 2)
            self.assertEqual(got, dict(_make_members()))


class AudioTests(unittest.TestCase):

    def _signal(self, sr=48000, voiced=4.0, stereo=True, amp=0.3):
        rng = np.random.default_rng(1)
        t = np.arange(int(voiced * sr)) / sr
        v = amp * np.sin(2 * np.pi * 220 * t) + 0.01 * rng.standard_normal(len(t))
        x = np.concatenate([np.zeros(sr), v, np.zeros(sr)]).astype(np.float32)
        return np.stack([x, x], axis=1) if stereo else x

    def test_preprocess_chain(self):
        y, reason = cv_utils.preprocess_waveform(self._signal(), 48000, 3.0, 16.0)
        self.assertEqual(reason, "ok")
        self.assertEqual(y.ndim, 1)
        self.assertEqual(y.dtype, np.float32)
        dur = len(y) / cv_utils.TARGET_SR
        self.assertAlmostEqual(dur, 4.0 + 2 * cv_utils.TRIM_PAD_SEC, delta=0.05)
        rms_db = 20 * np.log10(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
        peak = np.max(np.abs(y))
        self.assertTrue(abs(rms_db - cv_utils.TARGET_RMS_DBFS) < 0.5
                        or abs(peak - cv_utils.PEAK_LIMIT) < 1e-4)
        self.assertLessEqual(peak, cv_utils.PEAK_LIMIT + 1e-6)

    def test_reject_reasons(self):
        short = self._signal(voiced=1.0, stereo=False)
        self.assertEqual(cv_utils.preprocess_waveform(short, 48000, 3.0, 16.0)[1], "too_short")
        long = self._signal(voiced=18.0, stereo=False)
        self.assertEqual(cv_utils.preprocess_waveform(long, 48000, 3.0, 16.0)[1], "too_long")
        silent = np.zeros(5 * 16000, dtype=np.float32)
        self.assertEqual(cv_utils.preprocess_waveform(silent, 16000, 3.0, 16.0)[1], "silent")

    def test_flac_round_trip(self):
        y, _ = cv_utils.preprocess_waveform(self._signal(), 48000, 3.0, 16.0)
        data = cv_utils.encode_flac(y)
        back = cv_utils.decode_flac(data)
        self.assertEqual(len(back), len(y))
        self.assertLess(float(np.max(np.abs(back - y))), 1e-3)

    def test_mp3_decode_in_memory(self):
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, self._signal(), 48000, format="MP3")
        x, sr = cv_utils.decode_audio_bytes(buf.getvalue())
        self.assertEqual(sr, 48000)
        self.assertEqual(x.ndim, 2)


if __name__ == "__main__":
    unittest.main()
