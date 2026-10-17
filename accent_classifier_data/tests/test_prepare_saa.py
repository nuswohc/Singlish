"""Tests for scripts/prepare_speech_accent_archive.py on synthetic inputs."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import cv_utils  # noqa: E402
import prepare_speech_accent_archive as saa  # noqa: E402
import synthetic_sources  # noqa: E402

AUDIO_EXT = (".mp3", ".wav", ".flac", ".ogg", ".opus")
SR = 16000
CHUNK_ARGS = dict(min_sec=3.0, max_sec=16.0, target_min=6.0, target_max=10.0,
                  silence_db=-35.0, min_pause=0.15)
SPLIT = ["--val_fraction", "0.2", "--test_fraction", "0.2"]


def _quiet(fn, argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = fn(argv)
    return code, out.getvalue()


class ChunkingTests(unittest.TestCase):

    def test_chunks_at_pauses(self):
        rng = np.random.default_rng(1)
        y, pauses = synthetic_sources.speech_like(
            rng, SR, synthetic_sources.paragraph_segments(rng, 30.0))
        self.assertGreater(len(y) / SR, 29.0)
        spans = saa.chunk_recording(y, SR, **CHUNK_ARGS)
        self.assertGreaterEqual(len(spans), 3)
        prev_end = 0
        for s, e, _ in spans:
            self.assertGreaterEqual(s, prev_end)
            self.assertTrue(3.0 <= (e - s) / SR <= 16.0, (e - s) / SR)
            prev_end = e
        for _, e, method in spans[:-1]:
            self.assertEqual(method, "pause")
            self.assertTrue(any(ps <= e <= pe for ps, pe in pauses), e)

    def test_fallback_without_pauses(self):
        rng = np.random.default_rng(2)
        y, _ = synthetic_sources.speech_like(rng, SR, [("v", 30.0)])
        spans = saa.chunk_recording(y, SR, **CHUNK_ARGS)
        self.assertGreaterEqual(len(spans), 3)
        for s, e, _ in spans:
            self.assertTrue(3.0 <= (e - s) / SR <= 16.0)
        self.assertIn("fallback", [m for _, _, m in spans])

    def test_short_and_silent(self):
        rng = np.random.default_rng(3)
        y, _ = synthetic_sources.speech_like(rng, SR, [("p", 0.2), ("v", 2.0), ("p", 0.2)])
        self.assertEqual(saa.chunk_recording(y, SR, **CHUNK_ARGS), [])
        self.assertEqual(saa.chunk_recording(np.zeros(SR * 5, np.float32), SR, **CHUNK_ARGS), [])


class SaaKaggleTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.zip = synthetic_sources.make_saa_kaggle_zip(cls.dir / "input" / "saa.zip")
        cls.data = cls.dir / "data"
        code, cls.out = _quiet(saa.main, ["--source_dir", str(cls.zip),
                                          "--data_dir", str(cls.data)] + SPLIT)
        assert code == 0, cls.out
        cls.stats = dict(saa.LAST_STATS)
        cls.df = cv_utils.load_accent_data(sources=["speech_accent_archive"],
                                           data_dir=cls.data, decode_audio=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_language_filter_and_ids(self):
        self.assertEqual(self.stats["mode"], "local_zip")
        self.assertEqual(self.stats["speakers_by_language"], {"korean": 5})
        self.assertEqual(self.stats["missing_audio"], 1)          # korean6
        self.assertEqual(set(self.df["speaker_id"]),
                         {f"saa_{k}" for k in synthetic_sources.KAGGLE_KOREAN})
        self.assertTrue(self.df["clip_id"].is_unique)
        for cid, spk in zip(self.df["clip_id"], self.df["speaker_id"]):
            self.assertRegex(cid, r"^saa_korean\d+_c\d\d$")
            self.assertTrue(cid.startswith(spk + "_c"))
        self.assertIn("SPEAKERS (korean", self.out)

    def test_schema_and_constants(self):
        shards = sorted((self.data / "saa_subset").glob("part-*.parquet"))
        self.assertTrue(shards)
        for s in shards:
            self.assertTrue(pq.read_schema(s).equals(cv_utils.SUBSET_SCHEMA, check_metadata=False))
        self.assertEqual(list(self.df.columns), cv_utils.SUBSET_SCHEMA.names)
        self.assertTrue((self.df["source"] == "speech_accent_archive").all())
        self.assertTrue((self.df["accent_group"] == "Korean").all())
        self.assertTrue((self.df["accent_raw"] == "Korean L1 (Speech Accent Archive)").all())
        self.assertTrue((self.df["sentence"] == "").all())
        self.assertFalse(self.df["is_singaporean"].any())

    def test_durations_and_flac(self):
        self.assertTrue(self.df["duration_sec"].between(3.0, 16.0).all())
        for b, dur in zip(self.df["audio_flac"], self.df["duration_sec"]):
            info = sf.info(io.BytesIO(b))
            self.assertEqual(info.samplerate, 16000)
            self.assertEqual(info.channels, 1)
            self.assertAlmostEqual(info.frames / 16000, dur, delta=0.01)
        # ~30 s recordings -> several chunks per speaker.
        self.assertTrue((self.df.groupby("speaker_id").size() >= 3).all())

    def test_speaker_disjoint_splits(self):
        self.assertTrue((self.df.groupby("speaker_id")["split"].nunique() == 1).all())
        self.assertEqual(set(self.df["split"]), {"train", "val", "test"})

    def test_outputs(self):
        info = json.loads((self.data / "saa_subset_split_info.json").read_text(encoding="utf-8"))
        self.assertTrue(info["elicitation_paragraph"].startswith("Please call Stella"))
        self.assertNotIn(str(self.dir), json.dumps(info))
        self.assertNotIn(str(self.dir).replace("\\", "\\\\"), json.dumps(info))
        self.assertTrue((self.data / "saa_subset_summary.csv").is_file())

    def test_no_audio_files_written(self):
        found = [p for p in self.dir.rglob("*")
                 if p.is_file() and p.suffix.lower() in AUDIO_EXT]
        self.assertEqual(found, [])   # the input zip is .zip, nothing extracted

    def test_unzipped_folder_same_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = synthetic_sources.make_saa_kaggle_dir(Path(tmp) / "input")
            data2 = Path(tmp) / "data"
            code, out = _quiet(saa.main, ["--source_dir", str(folder),
                                          "--data_dir", str(data2)] + SPLIT)
            self.assertEqual(code, 0, out)
            self.assertEqual(saa.LAST_STATS["mode"], "local_dir")
            df2 = cv_utils.load_accent_data(sources=["speech_accent_archive"],
                                            data_dir=data2, decode_audio=False)
            self.assertEqual(sorted(df2["clip_id"]), sorted(self.df["clip_id"]))
            found = [p for p in data2.rglob("*") if p.suffix.lower() in AUDIO_EXT]
            self.assertEqual(found, [])

    def test_mirror_layout_multiple_languages(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = synthetic_sources.make_saa_mirror_parquet(Path(tmp) / "mirror")
            data2 = Path(tmp) / "data"
            code, out = _quiet(saa.main, ["--source_dir", str(src), "--data_dir", str(data2),
                                          "--native_languages", "Korean", "mandarin"] + SPLIT)
            self.assertEqual(code, 0, out)
            self.assertEqual(saa.LAST_STATS["mode"], "local_parquet")
            df = cv_utils.load_accent_data(sources=["speech_accent_archive"],
                                           data_dir=data2, decode_audio=False)
            self.assertEqual(set(df["accent_group"]), {"Korean", "Mainland Chinese"})
            self.assertNotIn("saa_spanish1", set(df["speaker_id"]))

    def test_bad_args_and_empty_source(self):
        code, _ = _quiet(saa.main, ["--source_dir", str(self.zip), "--data_dir", str(self.data),
                                    "--chunk_target_min", "2"])
        self.assertEqual(code, 1)
        with tempfile.TemporaryDirectory() as tmp:
            code, out = _quiet(saa.main, ["--source_dir", tmp,
                                          "--data_dir", str(Path(tmp) / "data")])
            self.assertEqual(code, 1)
            self.assertIn("kaggle.com/datasets/rtatman/speech-accent-archive", out)


if __name__ == "__main__":
    unittest.main()
