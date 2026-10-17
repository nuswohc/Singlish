"""Tests for scripts/prepare_mnsc_singaporean.py on synthetic MNSC-style parquet."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import cv_utils  # noqa: E402
import prepare_mnsc_singaporean as mnsc  # noqa: E402
import synthetic_sources  # noqa: E402

AUDIO_EXT = (".mp3", ".wav", ".flac", ".ogg", ".opus")
BASE = ["--clips_total", "40", "--max_clips_per_speaker", "6",
        "--val_fraction", "0.2", "--test_fraction", "0.2"]


def _quiet(fn, argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = fn(argv)
    return code, out.getvalue()


def _load(data):
    return cv_utils.load_accent_data(sources=["mnsc"], data_dir=data, decode_audio=False)


class HelperTests(unittest.TestCase):

    def test_normalize_transcript(self):
        n = mnsc.normalize_transcript
        self.assertEqual(n("<Speaker1>: [ah] Go {MRT} (uh) #Sentosa#  lah"),
                         "ah go mrt uh sentosa lah")
        self.assertEqual(n("[[nested] thing] ok"), "nested thing ok")
        self.assertEqual(n("<speaker2>:   Hello   World "), "hello world")
        self.assertEqual(n(None), "")
        self.assertEqual(n(""), "")

    def test_extract_speaker_id(self):
        e = mnsc.extract_speaker_id
        self.assertEqual(e({"speaker1": {"speaker_id": "A1"}, "speaker2": {"speaker_id": "B"}}), "A1")
        self.assertEqual(e({"speaker1": {"speaker_id": ""}, "speaker2": {"speaker_id": "B2"}}), "B2")
        self.assertEqual(e(json.dumps({"speaker1": {"speaker_id": "J1"}})), "J1")
        self.assertEqual(e({"speaker1": json.dumps({"speaker_id": "null"}),
                            "speaker2": json.dumps({"speaker_id": "J2"})}), "J2")
        self.assertEqual(e({"speaker": {"speaker_id": "01820000"}}), "01820000")
        self.assertEqual(e({"speaker1": {"speaker_id": None}, "speaker": {"speaker_id": " 7 "}}), "7")
        for bad in (None, "", "not json", "[1, 2]", 5, {"speaker1": "x"}, {"speaker": {}}):
            self.assertEqual(e(bad), "", bad)

    def test_duration_from_attrs(self):
        self.assertAlmostEqual(mnsc.duration_from_attrs({"start": "1.5", "end": "6"}), 4.5)
        self.assertAlmostEqual(mnsc.duration_from_attrs('{"start": 0, "end": 3}'), 3.0)
        for bad in ({}, {"start": "x", "end": "2"}, {"start": 5, "end": 2}, None):
            self.assertNotEqual(mnsc.duration_from_attrs(bad), mnsc.duration_from_attrs(bad))  # NaN

    def test_metadata_columns(self):
        self.assertEqual(mnsc.metadata_columns(synthetic_sources.MNSC_PART1_SCHEMA),
                         ["answer", "other_attributes.speaker.speaker_id"])
        self.assertEqual(mnsc.metadata_columns(synthetic_sources.MNSC_PART3_SCHEMA),
                         ["answer", "other_attributes"])
        spk = pa.struct([("speaker_id", pa.string())])
        part3 = pa.schema([("answer", pa.string()), ("other_attributes", pa.struct([
            ("start", pa.string()), ("end", pa.string()), ("speaker1", spk),
            ("speaker2", pa.string())]))])
        self.assertEqual(mnsc.metadata_columns(part3),
                         ["answer", "other_attributes.speaker1.speaker_id",
                          "other_attributes.speaker2", "other_attributes.start",
                          "other_attributes.end"])

    def test_scan_should_stop(self):
        s = mnsc.scan_should_stop
        self.assertEqual(s(30, 50, 10, 10, 0.01, 1, 8), (False, ""))
        self.assertEqual(s(60, 50, 0, 10, 0.01, 2, 8), (True, "target+saturated"))
        self.assertEqual(s(60, 50, 5, 100, 0.01, 2, 8), (False, ""))
        self.assertEqual(s(10, 50, 0, 10, 0.01, 8, 8), (True, "max_shards"))
        self.assertEqual(mnsc.pool_capacity({"a": 10, "b": 2}, 6), 8)

    def test_names(self):
        self.assertEqual(mnsc.accent_raw_for("ASR-PART1-Train"), "Singaporean (MNSC PART1)")
        self.assertEqual(mnsc.accent_raw_for("ASR-PART3-Train"), "Singaporean (MNSC PART3)")
        self.assertTrue(mnsc.is_test_config("ASR-PART1-Test"))
        self.assertFalse(mnsc.is_test_config("ASR-PART1-Train"))
        self.assertEqual(mnsc.clip_id_for("ASR-PART1-Train", 2, 1234),
                         "mnsc_part1-train_00002_01234")
        self.assertEqual(mnsc.shard_index_from_name("x/train-00771-of-00772.parquet", 0), 771)
        self.assertEqual(mnsc.shard_index_from_name("odd.parquet", 4), 4)


class MnscPipelineTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.src = cls.dir / "mnsc"
        cls.expect = synthetic_sources.make_mnsc_source(cls.src)
        cls.data = cls.dir / "data"
        code, cls.out = _quiet(mnsc.main, ["--source_dir", str(cls.src), "--data_dir",
                                           str(cls.data), "--exclude_speakers_json", ""] + BASE)
        assert code == 0, cls.out
        cls.stats = dict(mnsc.LAST_STATS)
        cls.df = _load(cls.data)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run(self, extra, data_name):
        data = self.dir / data_name
        code, out = _quiet(mnsc.main, ["--source_dir", str(self.src), "--data_dir", str(data)]
                           + extra)
        return code, out, data

    def test_schema_and_constants(self):
        shards = sorted((self.data / "mnsc_subset").glob("part-*.parquet"))
        self.assertTrue(shards)
        for s in shards:
            self.assertTrue(pq.read_schema(s).equals(cv_utils.SUBSET_SCHEMA, check_metadata=False))
        self.assertEqual(list(self.df.columns), cv_utils.SUBSET_SCHEMA.names)
        self.assertTrue((self.df["source"] == "mnsc").all())
        self.assertTrue((self.df["accent_group"] == "Singaporean").all())
        self.assertTrue(self.df["is_singaporean"].all())
        self.assertTrue((self.df["accent_raw"] == "Singaporean (MNSC PART1)").all())

    def test_ids_and_dropped_rows(self):
        self.assertTrue(self.df["speaker_id"].str.startswith("mnsc_").all())
        self.assertTrue(self.df["clip_id"].str.startswith("mnsc_part1-train_").all())
        self.assertTrue(self.df["clip_id"].is_unique)
        self.assertLessEqual(set(self.df["speaker_id"]),
                             {f"mnsc_{s}" for s in self.expect["speakers"]})
        sp = self.expect["special"]
        for key in ("no_id", "no_answer", "duplicate", "too_short", "too_long"):
            self.assertNotIn(sp[key], set(self.df["clip_id"]), key)
        scan = self.stats["scan"]
        self.assertEqual(scan["no_speaker_id"], 1)
        self.assertEqual(scan["empty_transcript"], 1)
        self.assertEqual(scan["duplicate_utterance"], 1)
        self.assertTrue((self.df["sentence"] != "").all())

    def test_per_speaker_cap_and_total(self):
        self.assertLessEqual(len(self.df), 40)
        self.assertEqual(len(self.df), 40)
        self.assertLessEqual(self.df.groupby("speaker_id").size().max(), 6)
        self.assertEqual(self.df["speaker_id"].nunique(), 10)

    def test_incremental_scan(self):
        self.assertEqual(self.stats["shards_scanned"], 2)
        self.assertEqual(self.stats["stop_reason"], "target+saturated")
        self.assertEqual(self.stats["n_selected"], 50)
        self.assertEqual(set(c.split("_")[2] for c in self.df["clip_id"]), {"00000", "00001"})

    def test_durations_and_flac(self):
        self.assertTrue(self.df["duration_sec"].between(3.0, 16.0).all())
        for b, dur in zip(self.df["audio_flac"], self.df["duration_sec"]):
            info = sf.info(io.BytesIO(b))
            self.assertEqual(info.samplerate, 16000)
            self.assertEqual(info.channels, 1)
            self.assertAlmostEqual(info.frames / 16000, dur, delta=0.01)

    def test_speaker_disjoint_splits(self):
        self.assertTrue((self.df.groupby("speaker_id")["split"].nunique() == 1).all())
        self.assertEqual(set(self.df["split"]), {"train", "val", "test"})

    def test_outputs(self):
        info = json.loads((self.data / "mnsc_subset_split_info.json").read_text(encoding="utf-8"))
        self.assertIn("license", info)
        self.assertIn("IMDA", info["license"]["note"])
        self.assertNotIn(str(self.dir), json.dumps(info))
        self.assertNotIn(str(self.dir).replace("\\", "\\\\"), json.dumps(info))
        self.assertTrue((self.data / "mnsc_subset_summary.csv").is_file())
        self.assertIn("IMDA", self.out)

    def test_no_audio_files_written(self):
        found = [p for p in self.dir.rglob("*")
                 if p.is_file() and p.suffix.lower() in AUDIO_EXT]
        self.assertEqual(found, [])

    def test_all_rows_skips_and_normalised_sentence(self):
        code, out, data = self._run(["--clips_total", "500", "--max_clips_per_speaker", "50",
                                     "--exclude_speakers_json", ""], "data_all")
        self.assertEqual(code, 0, out)
        self.assertEqual(mnsc.LAST_STATS["shards_scanned"], 3)
        self.assertEqual(mnsc.LAST_STATS["stop_reason"], "all_shards_scanned")
        self.assertEqual(mnsc.LAST_STATS["skips"]["too_short"], 1)
        self.assertEqual(mnsc.LAST_STATS["skips"]["too_long"], 1)
        self.assertIn("WARNING: only", out)                         # < 300 speakers
        df = _load(data)
        sp = self.expect["special"]
        row = df[df["clip_id"] == sp["normalised"]]
        self.assertEqual(row["sentence"].tolist(), [self.expect["normalised_sentence"]])
        # 44.1 kHz shard resampled to 16 kHz
        self.assertTrue(any("_00002_" in c for c in df["clip_id"]))
        self.assertTrue((df["sampling_rate"] == 16000).all())

    def test_exclude_speakers(self):
        js = self.dir / "split_info.json"
        js.write_text(json.dumps({"split": {"train_speakers": ["S01", "S02", "nobody"]}}),
                      encoding="utf-8")
        code, out, data = self._run(BASE + ["--exclude_speakers_json", str(js)], "data_excl")
        self.assertEqual(code, 0, out)
        self.assertEqual(mnsc.LAST_STATS["n_overlap"], 2)
        self.assertIn("forced into train", out)
        df = _load(data)
        for spk in ("mnsc_S01", "mnsc_S02"):
            self.assertEqual(set(df.loc[df["speaker_id"] == spk, "split"]), {"train"})
        self.assertTrue((df.groupby("speaker_id")["split"].nunique() == 1).all())
        info = json.loads((data / "mnsc_subset_split_info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["exclude_speakers"]["n_overlap"], 2)
        self.assertEqual(info["exclude_speakers"]["file"], "split_info.json")

        bad = self.dir / "bad.json"
        bad.write_text(json.dumps({"split": {"val_speakers": []}}), encoding="utf-8")
        code, out, _ = self._run(BASE + ["--exclude_speakers_json", str(bad)], "data_bad")
        self.assertEqual(code, 1)
        self.assertIn("split.train_speakers", out)

    def test_test_configs_guard(self):
        code, out, _ = self._run(BASE + ["--configs", "ASR-PART1-Test",
                                         "--exclude_speakers_json", ""], "data_t1")
        self.assertEqual(code, 1)
        self.assertIn("evaluation set", out)
        code, out, _ = self._run(BASE + ["--configs", "ASR-PART1-Test", "--allow_test_configs",
                                         "--exclude_speakers_json", ""], "data_t2")
        self.assertEqual(code, 0, out)

    def test_part3_speaker2_fallback(self):
        code, out, data = self._run(BASE + ["--configs", "ASR-PART1-Train", "ASR-PART3-Train",
                                            "--exclude_speakers_json", ""], "data_p3")
        self.assertEqual(code, 0, out)
        df = _load(data)
        p3 = df[df["speaker_id"] == "mnsc_P3-2"]
        self.assertGreater(len(p3), 0)
        self.assertTrue((p3["accent_raw"] == "Singaporean (MNSC PART3)").all())
        self.assertTrue(p3["clip_id"].str.startswith("mnsc_part3-train_").all())
        self.assertTrue((p3["sentence"].str.startswith("part three")).all())

    def test_bad_configs_and_args(self):
        code, out, _ = self._run(BASE + ["--configs", "ASR-PART9-Train",
                                         "--exclude_speakers_json", ""], "data_u")
        self.assertEqual(code, 1)
        self.assertIn("ERROR", out)
        code, _, _ = self._run(BASE + ["--oversample", "0.5"], "data_o")
        self.assertEqual(code, 1)
        code, _, _ = self._run(BASE + ["--saturation", "1.5"], "data_s")
        self.assertEqual(code, 1)

    def test_loader_includes_mnsc_by_default(self):
        self.assertIn("mnsc", cv_utils.SOURCE_SUBSETS)
        df = cv_utils.load_accent_data(data_dir=self.data, decode_audio=False)
        self.assertEqual(set(df["source"]), {"mnsc"})
        train = cv_utils.load_accent_data(sources=["mnsc"], split="train",
                                          data_dir=self.data, decode_audio=False)
        self.assertEqual(set(train["split"]), {"train"})
        self.assertLess(len(train), len(df))


if __name__ == "__main__":
    unittest.main()
