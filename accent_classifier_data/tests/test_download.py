"""Tests for scripts/download_common_voice.py (no network, no real download)."""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import download_common_voice as dl  # noqa: E402


def _run(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = dl.main(argv)
    return code, out.getvalue()


class DownloadTests(unittest.TestCase):

    def test_list_segments(self):
        code, out = _run(["--list_segments"])
        self.assertEqual(code, 0)
        for slug in dl.KNOWN_SEGMENTS:
            self.assertIn(slug, out)

    def test_missing_key(self):
        saved = os.environ.pop("MDC_API_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                env_file = Path(tmp) / "does_not_exist.env"
                raw_dir = Path(tmp) / "raw"
                with mock.patch("datacollective.download_dataset",
                                side_effect=AssertionError("must not be called")) as m:
                    code, out = _run(["--env_file", str(env_file), "--raw_dir", str(raw_dir)])
                    m.assert_not_called()
                self.assertEqual(code, 1)
                self.assertIn("MDC_API_KEY", out)
                self.assertIn("terms", out)
                self.assertFalse(raw_dir.exists())
        finally:
            if saved is not None:
                os.environ["MDC_API_KEY"] = saved

    def test_progress_bar_on_cp1252_stdout(self):
        """SDK progress output with non-ASCII chars must not crash on cp1252."""
        saved = os.environ.pop("MDC_API_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["MDC_API_KEY"] = "dummy-key-for-test"
                raw_dir = Path(tmp) / "raw"
                fake = raw_dir / "fake.tar.gz"

                def fake_download(slug, download_directory=None, **kw):
                    sys.stdout.write("\u2588\u2591 \U0001f98a 50%\n")
                    fake.write_bytes(b"\x1f\x8b")
                    return fake

                buf = io.BytesIO()
                cp1252 = io.TextIOWrapper(buf, encoding="cp1252")
                with mock.patch("datacollective.download_dataset", side_effect=fake_download), \
                        contextlib.redirect_stdout(cp1252):
                    code = dl.main(["--env_file", str(Path(tmp) / "none.env"),
                                    "--raw_dir", str(raw_dir)])
                cp1252.flush()
                self.assertEqual(code, 0, buf.getvalue().decode("cp1252", "replace"))
                cp1252.detach()
        finally:
            os.environ.pop("MDC_API_KEY", None)
            if saved is not None:
                os.environ["MDC_API_KEY"] = saved

    def test_key_from_env_file_not_printed(self):
        saved = os.environ.pop("MDC_API_KEY", None)
        secret = "sk-test-SECRET-1234567890"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                env_file = Path(tmp) / ".env"
                env_file.write_text(f"MDC_API_KEY={secret}\n", encoding="utf-8")
                raw_dir = Path(tmp) / "raw"
                fake = Path(tmp) / "raw" / "fake.tar.gz"

                def fake_download(slug, download_directory=None, **kw):
                    self.assertEqual(Path(download_directory), raw_dir)
                    fake.write_bytes(b"\x1f\x8b" + b"\x00" * 10)
                    return fake

                with mock.patch("datacollective.download_dataset", side_effect=fake_download):
                    code, out = _run(["--env_file", str(env_file), "--raw_dir", str(raw_dir)])
                self.assertEqual(code, 0)
                self.assertNotIn(secret, out)
                self.assertIn("fake.tar.gz", out)
        finally:
            os.environ.pop("MDC_API_KEY", None)
            if saved is not None:
                os.environ["MDC_API_KEY"] = saved


if __name__ == "__main__":
    unittest.main()
