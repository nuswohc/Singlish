"""Tests for scripts/extract_common_voice_subset.py on a synthetic archive."""

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
import extract_common_voice_subset as extract  # noqa: E402
import scan_common_voice_metadata as scan  # noqa: E402
import synthetic_cv  # noqa: E402

AUDIO_EXT = (".mp3", ".wav", ".flac", ".ogg", ".opus")
DEFAULT_GROUPS = {"Malaysian", "Hong Kong", "Mainland Chinese", "Korean",
                  "South Asian", "Filipino", "American"}
SMALL = ["--clips_per_accent", "6", "--max_clips_per_speaker", "2",
         "--val_fraction", "0.2", "--test_fraction", "0.2"]


def _quiet(fn, argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = fn(argv)
    return code, out.getvalue()


class ExtractTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.data = cls.dir / "data"
        cls.archive = synthetic_cv.make_synthetic_archive(cls.dir, "tar.gz")
        code, out = _quiet(scan.main, ["--archive", str(cls.archive), "--data_dir", str(cls.data)])
        assert code == 0, out
        cls.base = ["--archive", str(cls.archive), "--data_dir", str(cls.data)]
        code, cls.out = _quiet(extract.main, cls.base + SMALL)
        assert code == 0, cls.out
        cls.stats = dict(extract.LAST_STATS)
        cls.df = cv_utils.load_cv_subset(data_dir=cls.data, decode_audio=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_default_groups_exclude_special(self):
        self.assertEqual(set(self.stats["groups"]), DEFAULT_GROUPS)
        self.assertEqual(set(self.df["accent_group"]), DEFAULT_GROUPS)
        self.assertFalse(self.df["is_singaporean"].any())

    def test_caps(self):
        self.assertTrue((self.df.groupby("accent_group").size() <= 6).all())
        self.assertTrue((self.df.groupby("speaker_id").size() <= 2).all())
        # The heavy Malaysian speaker (12 clips) was capped and others still included.
        mal = self.df[self.df["accent_group"] == "Malaysian"]
        self.assertEqual(len(mal), 6)
        self.assertEqual(mal["speaker_id"].nunique(), 5)

    def test_schema_and_constants(self):
        shards = sorted((self.data / "cv_subset").glob("part-*.parquet"))
        self.assertTrue(shards)
        for s in shards:
            self.assertTrue(pq.read_schema(s).equals(cv_utils.SUBSET_SCHEMA, check_metadata=False))
        self.assertEqual(list(self.df.columns), cv_utils.SUBSET_SCHEMA.names)
        self.assertTrue((self.df["source"] == "common_voice").all())
        self.assertTrue((self.df["sampling_rate"] == 16000).all())

    def test_durations_and_flac(self):
        self.assertTrue(self.df["duration_sec"].between(3.0, 16.0).all())
        for key in ("too_short_prefilter", "too_short_trim", "too_long"):
            self.assertNotIn(synthetic_cv.SPECIAL_CLIPS[key], set(self.df["clip_id"]))
        for b, dur in zip(self.df["audio_flac"], self.df["duration_sec"]):
            info = sf.info(io.BytesIO(b))
            self.assertEqual(info.format, "FLAC")
            self.assertEqual(info.samplerate, 16000)
            self.assertEqual(info.channels, 1)
            self.assertAlmostEqual(info.frames / 16000, dur, delta=0.01)
            self.assertTrue(3.0 <= info.frames / 16000 <= 16.0)

    def test_speaker_disjoint_splits(self):
        for group, g in self.df.groupby("accent_group"):
            sets = {s: set(x["speaker_id"]) for s, x in g.groupby("split")}
            self.assertEqual(set(sets), {"train", "val", "test"}, group)
            self.assertFalse(sets["train"] & sets["val"], group)
            self.assertFalse(sets["train"] & sets["test"], group)
            self.assertFalse(sets["val"] & sets["test"], group)

    def test_stream_and_outputs(self):
        self.assertTrue(self.stats["stopped_early"])
        self.assertFalse((self.data / "cv_subset" / "_staging").exists())
        self.assertTrue((self.data / "cv_subset_summary.csv").is_file())
        info = json.loads((self.data / "cv_subset_split_info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["archive"], self.archive.name)
        self.assertNotIn(str(self.dir), json.dumps(info))
        self.assertEqual(info["audio_processing"]["target_sr"], 16000)
        summary = pd.read_csv(self.data / "cv_subset_summary.csv")
        all_rows = summary[summary["split"] == "ALL"]
        self.assertEqual(int(all_rows["n_clips"].sum()), len(self.df))

    def test_no_audio_files_on_disk(self):
        found = [p for p in self.dir.rglob("*")
                 if p.is_file() and p.suffix.lower() in AUDIO_EXT]
        self.assertEqual(found, [])

    def test_loader(self):
        full = cv_utils.load_cv_subset(data_dir=self.data)
        self.assertNotIn("audio_flac", full.columns)
        for a, dur in zip(full["audio"], full["duration_sec"]):
            self.assertEqual(a.dtype.name, "float32")
            self.assertEqual(a.ndim, 1)
            self.assertAlmostEqual(len(a) / 16000, dur, delta=0.01)
        train = cv_utils.load_cv_subset(split="train", data_dir=self.data)
        self.assertTrue(len(train) > 0)
        self.assertTrue((train["split"] == "train").all())
        kor = cv_utils.load_cv_subset(accent_groups=["Korean"], data_dir=self.data)
        self.assertEqual(set(kor["accent_group"]), {"Korean"})

    def test_rerun_deterministic_and_singaporean(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Copy pass-1 metadata into a fresh data dir so the class outputs stay intact.
            data2 = Path(tmp) / "data"
            data2.mkdir()
            (data2 / "cv_validated_with_groups.parquet").write_bytes(
                (self.data / "cv_validated_with_groups.parquet").read_bytes())
            argv = ["--archive", str(self.archive), "--data_dir", str(data2)]
            code, out = _quiet(extract.main, argv + SMALL)
            self.assertEqual(code, 0, out)
            again = cv_utils.load_cv_subset(data_dir=data2, decode_audio=False)
            self.assertEqual(sorted(again["clip_id"]), sorted(self.df["clip_id"]))
            self.assertEqual(
                dict(zip(again["clip_id"], again["split"])),
                dict(zip(self.df["clip_id"], self.df["split"])))

            code, out = _quiet(extract.main, argv + ["--accent_groups", "Singaporean"])
            self.assertEqual(code, 0, out)
            sg = cv_utils.load_cv_subset(data_dir=data2, decode_audio=False)
            self.assertEqual(set(sg["accent_group"]), {"Singaporean"})
            self.assertTrue(sg["is_singaporean"].all())
            self.assertIn("all clips go to train", out)   # only 2 speakers

    def test_bad_args(self):
        code, out = _quiet(extract.main, self.base + ["--accent_groups", "Martian"])
        self.assertEqual(code, 1)
        self.assertIn("Available groups", out)
        code, out = _quiet(extract.main, self.base + ["--val_fraction", "0.6",
                                                      "--test_fraction", "0.5"])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
