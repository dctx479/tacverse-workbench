import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import dataset_quality as quality


class DatasetQualityTests(unittest.TestCase):
    def setUp(self):
        self.cfg = quality._cfg({
            "start_end_window": 1,
            "boundary_abs_threshold": 0.5,
            "boundary_mad_factor": 3.0,
            "jump_abs_threshold": 0.5,
            "jump_mad_factor": 3.0,
        })

    @staticmethod
    def _episode(values):
        return {
            "frames": list(range(len(values))),
            "columns": {"observation.state": [
                (index, [float(value)]) for index, value in enumerate(values)
            ]},
        }

    def test_boundary_outlier_is_localized(self):
        episodes = {
            0: self._episode([0, 0, 0]),
            1: self._episode([0, 0, 0]),
            2: self._episode([4, 0, 0]),
        }
        issues = quality._check_boundaries(episodes, self.cfg, fps=30)
        starts = [item for item in issues if item.rule == "boundary_start"]
        self.assertEqual([item.episode_index for item in starts], [2])

    def test_jumps_do_not_compare_across_episode_boundaries(self):
        episodes = {
            0: self._episode([0, 0, 0, 0, 0, 0]),
            1: self._episode([10, 10, 10, 10, 10, 10]),
        }
        self.assertEqual(quality._check_jumps(episodes, self.cfg, fps=30), [])

    def test_dataset_dir_resolves_flat_upstream_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info = root / "sample" / "meta" / "info.json"
            info.parent.mkdir(parents=True)
            info.write_text("{}", encoding="utf-8")
            result = quality.dataset_dir(
                {"dataset_name": "TacVerse/sample"}, out_dir=root)
            self.assertEqual(result, root / "sample")

    def test_review_status_round_trip_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp)
            rows = [{
                "issue_id": "issue-1", "episode": 3, "rule": "trajectory_jump",
                "field": "action", "review_status": "未确认",
            }]
            (report / "issues.json").write_text(
                json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            rows[0]["review_status"] = "误报"
            quality.save_review_status(report, rows)
            loaded = quality.load_report_records(report)
            self.assertEqual(loaded[0]["review_status"], "误报")
            self.assertEqual(quality.summarize_records(loaded)["unconfirmed"], 0)

    def test_cancel_stops_scan_before_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(quality.QualityCancelled):
                quality._scan_path_uncached(
                    Path(tmp), self.cfg, cancel=lambda: True)

    def test_tactile_stream_excludes_blur_and_freeze(self):
        fake_video = Path("ep000001_left_tactile_left.mp4")
        fake_cv2 = mock.Mock()
        fake_cv2.CAP_PROP_FPS = 5
        fake_cv2.COLOR_BGR2GRAY = 6
        fake_cv2.CV_64F = 7
        capture = mock.Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 30.0
        capture.read.side_effect = [(True, object())] * 8 + [(False, None)]
        fake_cv2.VideoCapture.return_value = capture
        gray = mock.Mock()
        gray.mean.return_value = 100.0
        fake_cv2.cvtColor.return_value = gray
        fake_cv2.absdiff.return_value.mean.return_value = 0.0

        with mock.patch.dict("sys.modules", {"cv2": fake_cv2}), \
                mock.patch.object(quality, "_video_paths", return_value=[
                    ("left_tactile_left", fake_video)
                ]):
            issues = quality._check_flicker(Path("."), quality._cfg({
                "flicker_luma_threshold": 1000,
                "blur_min_sec": 0.1,
                "freeze_min_sec": 0.1,
                "exposure_min_sec": 0.1,
                "max_video_frames": 20,
            }))

        rules = {item.rule for item in issues}
        self.assertNotIn("motion_blur", rules)
        self.assertNotIn("camera_freeze", rules)
        fake_cv2.Laplacian.assert_not_called()
        fake_cv2.absdiff.assert_not_called()


if __name__ == "__main__":
    unittest.main()
