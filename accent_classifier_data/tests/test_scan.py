"""Tests for scripts/scan_common_voice_metadata.py on synthetic archives."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import scan_common_voice_metadata as scan  # noqa: E402
import synthetic_cv  # noqa: E402

AUDIO_EXT = (".mp3", ".wav", ".flac", ".ogg", ".opus")

EXPECTED_MAPPINGS = {
    "Singaporean English": "Singaporean",
    "Hong Kong English": "Hong Kong",
    "Hong Kong Chinese English": "Hong Kong",
    "Taiwanese English": "Taiwanese",
    "Chinese English": "Mainland Chinese",
    "Mandarin": "Mainland Chinese",
    "Korean": "Korean",
    "Japanese English": "Japanese",
    "Filipino": "Filipino",
    "Philippines English": "Filipino",
    "Indonesian English": "Indonesian",
    "Vietnamese": "Vietnamese",
    "Thai English": "Thai",
    "Myanmar English": "Myanmar",
    "Burmese": "Myanmar",
    "Malaysian English": "Malaysian",
    "MALAYSIAN": "Malaysian",
    "India and South Asia (India, Pakistan, Sri Lanka)": "South Asian",
    "West Indies and Bermuda (Bahamas, Barbados, Jamaica, Trinidad)": "Unmapped",
    "Latin American": "Unmapped",
    "United States English": "American",
    "United States English|Canadian English": "American",
    "United States English|slight drawl": "American",
    "England English": "British",
    "Scottish English": "British",
    "Australian English": "Australian",
    "Malaysian English|Singaporean English": "Ambiguous",
    "Irish English": "Unmapped",
    "something unusual": "Unmapped",
    "": "Unlabelled",
    "   ": "Unlabelled",
}


def run_scan(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = scan.main(argv)
    return code, out.getvalue()


def audio_files_under(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT]


class MappingTests(unittest.TestCase):

    def test_expected_mappings(self):
        for raw, group in EXPECTED_MAPPINGS.items():
            self.assertEqual(scan.map_accent(raw), group, raw)


class ScanTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.data = self.dir / "data"

    def tearDown(self):
        self.tmp.cleanup()

    def _check_outputs(self):
        meta_dir = self.data / "cv_metadata"
        written = sorted(p.name for p in meta_dir.iterdir())
        self.assertEqual(written, sorted(["validated.tsv", "train.tsv", "dev.tsv",
                                          "test.tsv", "clip_durations.tsv"]))
        self.assertTrue(all(n.endswith(".tsv") for n in written))
        counts = pd.read_csv(self.data / "cv_accent_counts.csv", keep_default_na=False)
        got = {r.accent_group: (r.n_clips, r.n_speakers) for r in counts.itertuples()}
        exp = {g: (e["n_clips"], e["n_speakers"]) for g, e in synthetic_cv.EXPECTED.items()}
        self.assertEqual(got, exp)
        self.assertTrue((pd.to_numeric(counts["total_hours"]) > 0).all())
        meta = pq.read_table(self.data / "cv_validated_with_groups.parquet").to_pandas()
        self.assertEqual(len(meta), sum(e["n_clips"] for e in synthetic_cv.EXPECTED.values()))
        self.assertTrue(meta["duration_sec"].notna().all())
        self.assertTrue(meta["sentence"].str.contains('"hello"').all())
        # Only the archive itself may contain audio.
        self.assertEqual(audio_files_under(self.dir), [])

    def test_tar_gz_tsvs_after_clips_reads_through(self):
        arc = synthetic_cv.make_synthetic_archive(self.dir, "tar.gz", tsv_first=False)
        code, out = run_scan(["--archive", str(arc), "--data_dir", str(self.data)])
        self.assertEqual(code, 0, out)
        self.assertFalse(scan.LAST_STATS["stopped_early"])
        self._check_outputs()
        self.assertIn("something unusual", out)

    def test_tar_gz_tsvs_first_stops_early(self):
        arc = synthetic_cv.make_synthetic_archive(self.dir, "tar.gz", tsv_first=True)
        code, out = run_scan(["--archive", str(arc), "--data_dir", str(self.data)])
        self.assertEqual(code, 0, out)
        self.assertTrue(scan.LAST_STATS["stopped_early"])
        self.assertEqual(scan.LAST_STATS["members_seen"], 6)   # 5 TSVs + first clip
        self._check_outputs()

    def test_plain_tar_and_zip(self):
        for kind in ("tar", "zip"):
            arc = synthetic_cv.make_synthetic_archive(self.dir / kind, kind)
            data = self.dir / kind / "data"
            code, out = run_scan(["--archive", str(arc), "--data_dir", str(data)])
            self.assertEqual(code, 0, out)
            self.assertEqual(scan.LAST_STATS["archive_type"], kind if kind == "zip" else "tar")
            self.data = data
            self._check_outputs()

    def test_legacy_accent_column(self):
        arc = synthetic_cv.make_synthetic_archive(self.dir, "tar.gz", accent_column="accent")
        code, out = run_scan(["--archive", str(arc), "--data_dir", str(self.data)])
        self.assertEqual(code, 0, out)
        self._check_outputs()

    def test_missing_validated(self):
        import tarfile
        arc = self.dir / "empty.tar.gz"
        with tarfile.open(arc, "w:gz") as tf:
            info = tarfile.TarInfo("x/en/other.tsv")
            info.size = 2
            tf.addfile(info, io.BytesIO(b"a\n"))
        code, out = run_scan(["--archive", str(arc), "--data_dir", str(self.data)])
        self.assertEqual(code, 1)
        self.assertIn("validated.tsv was not found", out)


if __name__ == "__main__":
    unittest.main()
