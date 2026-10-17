"""Tests for scripts/prepare_speechocean762.py on synthetic parquet files."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import cv_utils  # noqa: E402
import prepare_speechocean762 as so  # noqa: E402
import synthetic_sources  # noqa: E402

AUDIO_EXT = (".mp3", ".wav", ".flac", ".ogg", ".opus")
SMALL = ["--no_download", "--clips_per_accent", "20", "--max_clips_per_speaker", "4",
         "--val_fraction", "0.2", "--test_fraction", "0.2"]


def _quiet(fn, argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = fn(argv)
    return code, out.getvalue()


class SpeechoceanTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.src = cls.dir / "src"
        cls.data = cls.dir / "data"
        cls.expected = synthetic_sources.make_speechocean_parquets(cls.src)
        cls.base = ["--source_dir", str(cls.src), "--data_dir", str(cls.data)]
        code, cls.out = _quiet(so.main, cls.base + SMALL)
        assert code == 0, cls.out
        cls.stats = dict(so.LAST_STATS)
        cls.df = cv_utils.load_accent_data(sources=["speechocean762"],
                                           data_dir=cls.data, decode_audio=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_adult_filter(self):
        exp = self.expected
        self.assertEqual(self.stats["adult_speakers"], len(exp["adult_speakers"]))
        self.assertEqual(self.stats["adult_clips"], exp["adult_clips"])
        self.assertEqual(self.stats["child_clips_excluded"], exp["child_clips"])
        speakers = set(self.df["speaker_id"])
        for child in exp["child_speakers"]:
            self.assertNotIn(f"so762_{child}", speakers)
        self.assertEqual(speakers, {f"so762_{s}" for s in exp["adult_speakers"]})
        self.assertIn("ADULT speakers / clips", self.out)

    def test_min_age_zero_includes_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            data2 = Path(tmp) / "data"
            code, out = _quiet(so.main, ["--source_dir", str(self.src), "--data_dir", str(data2),
                                         "--min_age", "0"] + SMALL[:1])
            self.assertEqual(code, 0, out)
            df = cv_utils.load_accent_data(sources=["speechocean762"], data_dir=data2,
                                           decode_audio=False)
            self.assertTrue({f"so762_{c}" for c in self.expected["child_speakers"]}
                            <= set(df["speaker_id"]))

    def test_schema_and_constants(self):
        shards = sorted((self.data / "speechocean_subset").glob("part-*.parquet"))
        self.assertTrue(shards)
        for s in shards:
            self.assertTrue(pq.read_schema(s).equals(cv_utils.SUBSET_SCHEMA, check_metadata=False))
        self.assertEqual(list(self.df.columns), cv_utils.SUBSET_SCHEMA.names)
        self.assertTrue((self.df["source"] == "speechocean762").all())
        self.assertTrue((self.df["accent_group"] == "Mainland Chinese").all())
        self.assertTrue((self.df["accent_raw"] == "Mandarin L1 (speechocean762)").all())
        self.assertFalse(self.df["is_singaporean"].any())
        self.assertTrue((self.df["sampling_rate"] == 16000).all())
        self.assertTrue(self.df["speaker_id"].str.startswith("so762_").all())
        self.assertTrue(self.df["clip_id"].str.startswith("so762_").all())
        self.assertTrue(self.df["clip_id"].is_unique)
        self.assertTrue(self.df["sentence"].str.startswith("SENTENCE").all())

    def test_durations_and_flac(self):
        self.assertTrue(self.df["duration_sec"].between(3.0, 16.0).all())
        for key in ("too_short", "too_long"):
            self.assertNotIn(self.expected["special"][key], set(self.df["clip_id"]))
        for b, dur in zip(self.df["audio_flac"], self.df["duration_sec"]):
            info = sf.info(io.BytesIO(b))
            self.assertEqual(info.format, "FLAC")
            self.assertEqual(info.samplerate, 16000)
            self.assertEqual(info.channels, 1)
            self.assertAlmostEqual(info.frames / 16000, dur, delta=0.01)

    def test_caps(self):
        self.assertLessEqual(len(self.df), 20)
        self.assertTrue((self.df.groupby("speaker_id").size() <= 4).all())

    def test_speaker_disjoint_splits(self):
        sets = {s: set(x["speaker_id"]) for s, x in self.df.groupby("split")}
        self.assertEqual(set(sets), {"train", "val", "test"})
        self.assertFalse(sets["train"] & sets["val"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["val"] & sets["test"])

    def test_outputs(self):
        summary = pd.read_csv(self.data / "speechocean_subset_summary.csv")
        all_rows = summary[summary["split"] == "ALL"]
        self.assertEqual(int(all_rows["n_clips"].sum()), len(self.df))
        info = json.loads((self.data / "speechocean_subset_split_info.json")
                          .read_text(encoding="utf-8"))
        self.assertNotIn(str(self.dir), json.dumps(info))
        self.assertNotIn(str(self.dir).replace("\\", "\\\\"), json.dumps(info))
        self.assertEqual(info["audio_processing"]["target_sr"], 16000)
        self.assertEqual(info["adult_filter"]["min_age"], 18)

    def test_no_audio_files_on_disk(self):
        found = [p for p in self.dir.rglob("*")
                 if p.is_file() and p.suffix.lower() in AUDIO_EXT]
        self.assertEqual(found, [])

    def test_no_download_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            code, out = _quiet(so.main, ["--source_dir", str(empty), "--data_dir",
                                         str(Path(tmp) / "data"), "--no_download"])
            self.assertEqual(code, 1)
            self.assertIn("huggingface.co/datasets/mispeech/speechocean762", out)

    def test_rerun_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            data2 = Path(tmp) / "data"
            code, out = _quiet(so.main, ["--source_dir", str(self.src),
                                         "--data_dir", str(data2)] + SMALL)
            self.assertEqual(code, 0, out)
            again = cv_utils.load_accent_data(sources=["speechocean762"], data_dir=data2,
                                              decode_audio=False)
            self.assertEqual(dict(zip(again["clip_id"], again["split"])),
                             dict(zip(self.df["clip_id"], self.df["split"])))

    def test_bad_args(self):
        code, _ = _quiet(so.main, self.base + ["--min_age", "-1"])
        self.assertEqual(code, 1)
        code, _ = _quiet(so.main, self.base + ["--val_fraction", "0.6", "--test_fraction", "0.5"])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
