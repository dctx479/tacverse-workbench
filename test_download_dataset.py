import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

import download_dataset as dd


class DatasetLogTests(unittest.TestCase):
    def test_present_arrays_are_compact_and_round_trip(self):
        log = {
            "dataset_index": [f"org/dataset-{i}" for i in range(40)],
            "daily_totals": [{
                "pulled_at": "2026-07-24T12:00:00",
                "date": "260724",
                "org": "TacVerse",
                "total_datasets": 40,
                "total_episodes": 100,
                "total_frames": 1000,
                "total_hours": 1.0,
                "present": list(range(40)),
            }],
            "datasets": {},
        }

        text = dd._dataset_log_text(log)

        self.assertEqual(json.loads(text), log)
        present_lines = []
        inside = False
        for line in text.splitlines():
            if line.strip() == '"present": [':
                inside = True
                continue
            if inside and line.strip() == "]":
                break
            if inside:
                present_lines.append(line)
        self.assertEqual([16, 16, 8], [
            len(line.strip().rstrip(",").split(", "))
            for line in present_lines
        ])

    def test_dataset_changes_and_metadata_use_compact_lines(self):
        change = {
            "date": "260724",
            "pulled_at": "2026-07-24T14:04:19",
            "total_episodes": 10,
            "total_frames": 6017,
            "duration_hours": 0.056,
            "d_episodes": 10,
            "d_frames": 6017,
            "d_hours": 0.056,
        }
        log = {
            "dataset_index": ["TacVerse/taccap-g1-cucumber-0702"],
            "daily_totals": [],
            "datasets": {
                "TacVerse/taccap-g1-cucumber-0702": {
                    "changes": [change],
                    "fps": 30,
                    "robot_type": "bi_taccap_gripper",
                    "total_tasks": 1,
                    "uploader": "WBH333",
                    "last_modified": "2026-07-02T09:16:55+00:00",
                },
            },
        }

        text = dd._dataset_log_text(log)

        self.assertEqual(log, json.loads(text))
        self.assertIn(
            '        {"date": "260724", "pulled_at": '
            '"2026-07-24T14:04:19", "total_episodes": 10,', text)
        self.assertIn(
            '      "fps": 30, "robot_type": "bi_taccap_gripper", '
            '"total_tasks": 1, "uploader": "WBH333", '
            '"last_modified": "2026-07-02T09:16:55+00:00"}', text)

    def test_manual_snapshot_upserts_and_real_pull_replaces_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset_log.json"
            today = dt.date(2026, 7, 24)
            dd.upsert_manual_totals(
                "260723", "TacVerse", 156, 7898, 18289577, 169.311,
                path=path, today=today,
            )
            dd.upsert_manual_totals(
                "260723", "TacVerse", 157, 8000, 19000000, 170.0,
                path=path, today=today,
            )
            dd.upsert_manual_totals(
                "260723", "Xense", 20, 500, 1000000, 9.0,
                path=path, today=today,
            )
            log = dd.load_dataset_log(path)
            manual = [r for r in log["daily_totals"]
                      if r.get("source") == "manual"]
            self.assertEqual(2, len(manual))
            tacverse = next(r for r in manual if r["org"] == "TacVerse")
            self.assertEqual(8000, tacverse["total_episodes"])
            self.assertEqual([], tacverse["present"])

            report = {
                "pulled_at": "2026-07-23T20:00:00",
                "date": "260723",
                "org": "TacVerse",
                "total_datasets": 1,
                "total_episodes": 10,
                "total_frames": 10800,
                "total_hours": 0.1,
                "datasets": [{
                    "dataset_name": "TacVerse/example",
                    "total_episodes": 10,
                    "total_frames": 10800,
                    "duration_hours": 0.1,
                }],
            }
            dd.append_pull(report, path=path)
            log = dd.load_dataset_log(path)
            remaining_manual = [r for r in log["daily_totals"]
                                if r.get("source") == "manual"]
            self.assertEqual(["Xense"], [r["org"] for r in remaining_manual])
            real = next(r for r in log["daily_totals"]
                        if r.get("source") != "manual")
            self.assertEqual([0], real["present"])

    def test_manual_history_drives_daily_totals_without_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "dataset_log.json"
            today = dt.date(2026, 7, 24)
            dd.upsert_manual_totals(
                "260722", "TacVerse", 150, 7030, 17000000, 149.911,
                path=path, today=today,
            )
            dd.upsert_manual_totals(
                "260723", "TacVerse", 156, 7898, 18289577, 169.311,
                path=path, today=today,
            )

            history = dd.load_history(root / "pulls", log_file=path)
            self.assertEqual(["260722", "260723"],
                             [row["date"] for row in history])
            self.assertEqual([], history[-1]["datasets"])
            self.assertEqual("manual", history[-1]["source"])
            series = dd.daily_series(history)
            self.assertEqual(19.4, series[-1]["new_hours"])
            self.assertEqual(7898, series[-1]["total_episodes"])
            self.assertIs(history[-2], dd.find_baseline(history[-1], history))
            self.assertEqual((19.4, 868),
                             dd.aggregate_deltas(history[-1], history))

    def test_future_manual_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                dd.upsert_manual_totals(
                    "260725", "TacVerse", 1, 1, 1, 1.0,
                    path=Path(tmp) / "dataset_log.json",
                    today=dt.date(2026, 7, 24),
                )

    def test_only_latest_pull_is_kept_for_each_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset_log.json"

            def report(pulled_at, date, episodes):
                return {
                    "pulled_at": pulled_at,
                    "date": date,
                    "org": "TacVerse",
                    "total_datasets": 1,
                    "total_episodes": episodes,
                    "total_frames": episodes * 100,
                    "total_hours": episodes / 10,
                    "datasets": [{
                        "dataset_name": "TacVerse/example",
                        "total_episodes": episodes,
                        "total_frames": episodes * 100,
                        "duration_hours": episodes / 10,
                    }],
                }

            dd.append_pull(report("2026-07-23T18:00:00", "260723", 5), path)
            early = report("2026-07-24T14:04:19", "260724", 10)
            latest = report("2026-07-24T15:12:00", "260724", 15)
            dd.append_pull(early, path)
            dd.append_pull(latest, path)

            log = dd.load_dataset_log(path)
            today_rows = [r for r in log["daily_totals"]
                          if r["date"] == "260724" and r["org"] == "TacVerse"]
            self.assertEqual(1, len(today_rows))
            self.assertEqual("2026-07-24T15:12:00", today_rows[0]["pulled_at"])
            changes = log["datasets"]["TacVerse/example"]["changes"]
            today_changes = [c for c in changes if c["date"] == "260724"]
            self.assertEqual(1, len(today_changes))
            self.assertEqual(15, today_changes[0]["total_episodes"])
            self.assertEqual(10, today_changes[0]["d_episodes"])

            # Replaying an older result must not roll the day back.
            dd.append_pull(early, path)
            log = dd.load_dataset_log(path)
            today_row = next(r for r in log["daily_totals"]
                             if r["date"] == "260724")
            self.assertEqual("2026-07-24T15:12:00", today_row["pulled_at"])
            today_change = next(c for c in
                                log["datasets"]["TacVerse/example"]["changes"]
                                if c["date"] == "260724")
            self.assertEqual(15, today_change["total_episodes"])

    def test_compaction_keeps_metadata_and_recomputes_daily_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset_log.json"
            log = {
                "dataset_index": ["TacVerse/example", "TacVerse/orphan"],
                "daily_totals": [
                    {"pulled_at": "2026-07-23T18:00:00", "date": "260723",
                     "org": "TacVerse", "total_datasets": 1,
                     "total_episodes": 5, "total_frames": 500,
                     "total_hours": 0.5, "present": [0]},
                    {"pulled_at": "2026-07-24T14:04:19", "date": "260724",
                     "org": "TacVerse", "total_datasets": 1,
                     "total_episodes": 10, "total_frames": 1000,
                     "total_hours": 1.0, "present": [0]},
                    {"pulled_at": "2026-07-24T15:12:00", "date": "260724",
                     "org": "TacVerse", "total_datasets": 1,
                     "total_episodes": 15, "total_frames": 1500,
                     "total_hours": 1.5, "present": [0]},
                ],
                "datasets": {
                    "TacVerse/example": {
                        "robot_type": "test_robot",
                        "changes": [
                            {"pulled_at": "2026-07-23T18:00:00", "date": "260723",
                             "total_episodes": 5, "total_frames": 500,
                             "duration_hours": 0.5, "d_episodes": 5,
                             "d_frames": 500, "d_hours": 0.5},
                            {"pulled_at": "2026-07-24T14:04:19", "date": "260724",
                             "total_episodes": 10, "total_frames": 1000,
                             "duration_hours": 1.0, "d_episodes": 5,
                             "d_frames": 500, "d_hours": 0.5},
                            {"pulled_at": "2026-07-24T15:12:00", "date": "260724",
                             "total_episodes": 15, "total_frames": 1500,
                             "duration_hours": 1.5, "d_episodes": 5,
                             "d_frames": 500, "d_hours": 0.5},
                        ],
                    },
                    "TacVerse/orphan": {
                        "robot_type": "keep_me",
                        "changes": [],
                    },
                },
            }
            dd.write_dataset_log(log, path)

            self.assertEqual((3, 2), dd.compact_dataset_log(path))
            compacted = dd.load_dataset_log(path)
            self.assertIn("TacVerse/orphan", compacted["datasets"])
            self.assertEqual("keep_me",
                             compacted["datasets"]["TacVerse/orphan"]["robot_type"])
            changes = compacted["datasets"]["TacVerse/example"]["changes"]
            self.assertEqual(2, len(changes))
            self.assertEqual("2026-07-24T15:12:00", changes[-1]["pulled_at"])
            self.assertEqual(10, changes[-1]["d_episodes"])
            self.assertEqual(1000, changes[-1]["d_frames"])
            self.assertEqual(1.0, changes[-1]["d_hours"])


if __name__ == "__main__":
    unittest.main()
