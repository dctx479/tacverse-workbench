import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import download_dataset as dd


class PullDatasetRetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.day_dir = Path(self.tmp.name)
        self.repo_id = "TacVerse/example-dataset"
        self.local_dir = self.day_dir / "example-dataset"
        self.cache_error = FileNotFoundError(
            2, "No such file or directory",
            str(self.local_dir / ".cache" / "huggingface" / "download" /
                "meta" / "episodes" / "blob.12345678.incomplete"),
        )

    @patch.object(dd, "build_summary", return_value={"ok": True})
    @patch.object(dd.time, "sleep")
    def test_retries_cache_temp_error_serially(self, _sleep, summary):
        with patch.object(
                dd, "_snapshot_to_local",
                side_effect=[self.cache_error, str(self.local_dir)]) as download:
            logs = []
            result = dd.pull_dataset(
                self.repo_id, self.day_dir, None, "token", log=logs.append)

        self.assertEqual({"ok": True}, result)
        self.assertEqual(
            [call(self.repo_id, None, self.local_dir, "token",
                  max_workers=8),
             call(self.repo_id, None, self.local_dir, "token",
                  max_workers=1)],
            download.call_args_list,
        )
        self.assertTrue(any("单线程续传" in line for line in logs))
        summary.assert_called_once_with(self.repo_id, str(self.local_dir))

    @patch.object(dd, "build_summary")
    def test_does_not_retry_unrelated_download_error(self, summary):
        error = PermissionError("denied")
        with patch.object(dd, "_snapshot_to_local", side_effect=error) as download:
            with self.assertRaises(PermissionError):
                dd.pull_dataset(self.repo_id, self.day_dir, None, None, log=lambda _: None)

        self.assertEqual(1, download.call_count)
        summary.assert_not_called()

    def test_only_matches_huggingface_incomplete_cache_paths(self):
        self.assertTrue(dd._is_local_cache_temp_error(self.cache_error))
        self.assertFalse(dd._is_local_cache_temp_error(FileNotFoundError("info.json")))
        self.assertFalse(dd._is_local_cache_temp_error(RuntimeError("blob.incomplete")))

    @patch.object(dd.os, "name", "nt")
    def test_windows_hub_local_dir_uses_extended_path_prefix(self):
        path = dd._hub_local_dir(self.local_dir)
        self.assertTrue(path.startswith("\\\\?\\"))
        self.assertTrue(path.endswith("example-dataset"))


if __name__ == "__main__":
    unittest.main()
