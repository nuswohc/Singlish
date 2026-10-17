"""Tests for cv_utils.load_accent_data (multi-source loader)."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import cv_utils  # noqa: E402
import prepare_speech_accent_archive as saa  # noqa: E402
import prepare_speechocean762 as so  # noqa: E402
import synthetic_sources  # noqa: E402


def _quiet_call(fn, *args, **kwargs):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = fn(*args, **kwargs)
    return result, out.getvalue()


def _rows(source, prefix, group, n_speakers, splits):
    rng = np.random.default_rng(0)
    rows = []
    for s in range(n_speakers):
        for c in range(2):
            t = np.arange(int(3.5 * 16000)) / 16000
            y = (0.1 * np.sin(2 * np.pi * rng.uniform(120, 300) * t)).astype(np.float32)
            rows.append({
                "clip_id": f"{prefix}{s}_{c}", "source": source,
                "speaker_id": f"{prefix}{s}", "accent_raw": group,
                "accent_group": group, "is_singaporean": False,
                "duration_sec": 3.5, "sentence": "x", "sampling_rate": 16000,
                "audio_flac": cv_utils.encode_flac(y), "split": splits[s % len(splits)],
            })
    return rows


class LoaderUnitTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = Path(cls.tmp.name) / "data"
        for name, rows in (
            ("cv_subset", _rows("common_voice", "cv", "American", 3, ["train", "val", "test"])),
            ("speechocean_subset", _rows("speechocean762", "so762_", "Mainland Chinese", 2,
                                         ["train", "test"])),
        ):
            d = cls.data / name
            d.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist(rows, schema=cv_utils.SUBSET_SCHEMA),
                           str(d / "part-0000.parquet"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_default_loads_available_sources(self):
        df, out = _quiet_call(cv_utils.load_accent_data, data_dir=self.data)
        self.assertEqual(set(df["source"]), {"common_voice", "speechocean762"})
        self.assertEqual(len(df), 6 + 4)
        self.assertIn("no shards for speech_accent_archive", out)
        cv = cv_utils.load_cv_subset(data_dir=self.data)
        self.assertEqual(list(df.columns), list(cv.columns))
        self.assertNotIn("audio_flac", df.columns)
        self.assertTrue(all(a.dtype == np.float32 and a.ndim == 1 for a in df["audio"]))

    def test_columns_without_decode(self):
        df, _ = _quiet_call(cv_utils.load_accent_data, data_dir=self.data, decode_audio=False)
        self.assertEqual(list(df.columns), cv_utils.SUBSET_SCHEMA.names)
        self.assertEqual(list(df.columns),
                         list(cv_utils.load_cv_subset(data_dir=self.data, decode_audio=False).columns))

    def test_filters(self):
        df = cv_utils.load_accent_data(sources=["speechocean762"], data_dir=self.data,
                                       decode_audio=False)
        self.assertEqual(set(df["source"]), {"speechocean762"})
        df, _ = _quiet_call(cv_utils.load_accent_data, split="train", data_dir=self.data,
                            decode_audio=False)
        self.assertTrue((df["split"] == "train").all())
        self.assertEqual(set(df["source"]), {"common_voice", "speechocean762"})
        df, _ = _quiet_call(cv_utils.load_accent_data, accent_groups=["Mainland Chinese"],
                            data_dir=self.data, decode_audio=False)
        self.assertEqual(set(df["accent_group"]), {"Mainland Chinese"})

    def test_empty_after_filter(self):
        df, _ = _quiet_call(cv_utils.load_accent_data, accent_groups=["Martian"],
                            data_dir=self.data)
        self.assertEqual(len(df), 0)
        self.assertEqual(list(df.columns),
                         [c for c in cv_utils.SUBSET_SCHEMA.names if c != "audio_flac"] + ["audio"])

    def test_errors(self):
        with self.assertRaises(ValueError):
            cv_utils.load_accent_data(sources=["timit"], data_dir=self.data)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                _quiet_call(cv_utils.load_accent_data, data_dir=Path(tmp))

    def test_load_cv_subset_unchanged(self):
        cv = cv_utils.load_cv_subset(data_dir=self.data, decode_audio=False)
        self.assertEqual(set(cv["source"]), {"common_voice"})
        self.assertEqual(len(cv), 6)


class LoaderIntegrationTests(unittest.TestCase):

    def test_both_prepare_scripts_into_one_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            synthetic_sources.make_speechocean_parquets(tmp / "so")
            zip_path = synthetic_sources.make_saa_kaggle_zip(tmp / "saa.zip")
            split = ["--val_fraction", "0.2", "--test_fraction", "0.2"]
            code, out = _quiet_call(so.main, ["--source_dir", str(tmp / "so"), "--data_dir",
                                              str(data), "--no_download"] + split)
            self.assertEqual(code, 0, out)
            code, out = _quiet_call(saa.main, ["--source_dir", str(zip_path),
                                               "--data_dir", str(data)] + split)
            self.assertEqual(code, 0, out)

            df, out = _quiet_call(cv_utils.load_accent_data, data_dir=data, decode_audio=False)
            self.assertIn("no shards for common_voice", out)
            self.assertEqual(set(df["source"]), {"speechocean762", "speech_accent_archive"})
            self.assertEqual(set(df["accent_group"]), {"Mainland Chinese", "Korean"})
            spk_by_source = df.groupby("source")["speaker_id"].apply(set)
            self.assertFalse(spk_by_source["speechocean762"] & spk_by_source["speech_accent_archive"])
            self.assertTrue(df["clip_id"].is_unique)
            self.assertTrue((df.groupby("speaker_id")["split"].nunique() == 1).all())


if __name__ == "__main__":
    unittest.main()
