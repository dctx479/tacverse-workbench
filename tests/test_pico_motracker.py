import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

import pico_motracker as pico


class PicoMotrackerTests(unittest.TestCase):
    def _dataset(self, rows):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        (root / "meta").mkdir()
        (root / "data" / "chunk-000").mkdir(parents=True)
        names = [
            "left_tcp.x", "left_tcp.y", "left_tcp.z",
            "right_tcp.x", "right_tcp.y", "right_tcp.z",
        ]
        info = {"features": {"observation.state": {"names": names}}}
        (root / "meta" / "info.json").write_text(json.dumps(info))
        table = pa.table({
            "episode_index": [row[0] for row in rows],
            "frame_index": [row[1] for row in rows],
            "timestamp": [row[2] for row in rows],
            "observation.state": [row[3] for row in rows],
        })
        pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
        return temp, root

    def test_axis_and_xyz_hits_share_one_event(self):
        temp, root = self._dataset([
            (0, 0, 0.0, [0, 0, 0, 0, 0, 0]),
            (0, 1, 1 / 30, [0, 0, 0, 0.21, 0.21, 0]),
        ])
        self.addCleanup(temp.cleanup)
        result = pico.detect(root, {
            "hands": ["right"],
            "axis_step_threshold": {"x": 0.2, "y": 0.3, "z": 0.3},
            "xyz_step_threshold": 0.25,
        })
        self.assertEqual(result.total_events, 1)
        event = result.events[0]
        self.assertEqual(event.axis_hits, ("x",))
        self.assertTrue(event.xyz_hit)
        self.assertAlmostEqual(event.xyz_step, 0.2969848, places=6)

    def test_xyz_rule_can_trigger_without_single_axis_hit(self):
        temp, root = self._dataset([
            (0, 0, 0.0, [0, 0, 0, 0, 0, 0]),
            (0, 1, 1 / 30, [0, 0, 0, 0.15, 0.15, 0.15]),
        ])
        self.addCleanup(temp.cleanup)
        result = pico.detect(root, {
            "hands": ["right"],
            "axis_step_threshold": {"x": 0.2, "y": 0.2, "z": 0.2},
            "xyz_step_threshold": 0.25,
        })
        self.assertEqual(result.events[0].axis_hits, ())
        self.assertTrue(result.events[0].xyz_hit)

    def test_episode_boundaries_and_frame_gaps_are_not_compared(self):
        temp, root = self._dataset([
            (0, 0, 0.0, [0, 0, 0, 0, 0, 0]),
            (0, 2, 2 / 30, [0, 0, 0, 10, 10, 10]),
            (1, 0, 0.0, [0, 0, 0, -10, -10, -10]),
        ])
        self.addCleanup(temp.cleanup)
        result = pico.detect(root, {"hands": ["right"]})
        self.assertEqual(result.total_events, 0)
        self.assertEqual(result.scanned_transitions, 0)

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            pico.resolve_config({"xyz_step_threshold": 0})

    def test_missing_scan_dependency_is_reported_without_import_crash(self):
        real_import = __import__

        def blocked_import(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("missing numpy")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked_import):
            with self.assertRaisesRegex(ValueError, "numpy/pyarrow"):
                pico.detect(Path("/does/not/matter"))


if __name__ == "__main__":
    unittest.main()
