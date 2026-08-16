import os
import sys
import unittest
import datetime as dt
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

import main_app


class MainWindowUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        main_app.configure_application_ui(cls.app)

    def test_rollup_tab_is_present(self):
        report = {
            "date": "260101",
            "org": "TacVerse",
            "datasets": [{
                "dataset_name": "TacVerse/taccap-g1-test-task-0101",
                "uploader": "alice",
                "total_episodes": 2,
                "total_frames": 120,
                "duration_hours": 0.1,
                "last_modified": "2026-01-01T00:00:00+00:00",
            }],
        }
        with patch.object(main_app.dd, "migrate_pull_history_to_log"), \
                patch.object(main_app.dd, "load_history", return_value=[report]), \
                patch.object(main_app.dd, "load_hf_change_history", return_value={}), \
                patch.object(main_app.MainWindow, "_refresh_identity"):
            win = main_app.MainWindow()
        self.addCleanup(win.close)

        tab_names = [win.tabs.tabText(i) for i in range(win.tabs.count())]
        self.assertIn("分组统计", tab_names)
        self.assertNotIn("趋势", tab_names)
        self.assertEqual(["上传者", "任务", "robot_type"], [
            win.dim_combo.itemText(i) for i in range(win.dim_combo.count())
        ])
        self.assertTrue(hasattr(win, "rollup_start_date"))
        self.assertTrue(hasattr(win, "rollup_end_date"))
        self.assertEqual(Qt.Horizontal, win.rollup_splitter.orientation())
        self.assertEqual(2, win.rollup_splitter.count())
        self.assertEqual(Qt.Vertical, win.rollup_daily_splitter.orientation())
        self.assertEqual(2, win.rollup_daily_splitter.count())
        self.assertTrue(hasattr(win, "trend_plot"))
        self.assertIn("2026-01-01", win.trend_plot.plotItem.titleLabel.text)
        self.assertTrue(hasattr(win, "detail_scroll"))
        self.assertEqual(win.detail_scroll.widget(), win.detail_stack)
        self.assertEqual(win.detail_scroll.sizePolicy().verticalPolicy(), main_app.QSizePolicy.Ignored)
        self.assertGreaterEqual(win.daily_group_table.columnCount(), 5)
        self.assertGreaterEqual(win.rollup_table.columnCount(), 5)
        self.assertIs(win.rollup_table, win.range_group_table)
        self.assertEqual(1, win.rollup_table.rowCount())
        margins = win.layout().contentsMargins()
        self.assertLessEqual(margins.top(), 7)
        self.assertLessEqual(margins.bottom(), 6)
        self.assertLessEqual(
            win.kpi_labels["total_datasets"].parentWidget().minimumHeight(), 66)
        win._set_episode_length_note("需要先下载本地数据集，才能查看 episode 时长。")
        self.assertEqual(
            "需要先下载本地数据集，才能查看 episode 时长。",
            win.episode_length_tree.topLevelItem(0).text(0),
        )

    def test_pull_done_does_not_reference_missing_viewer_note_or_unlock(self):
        report = {
            "date": "260101",
            "org": "TacVerse",
            "count": 1,
            "requested": 1,
            "total_hours": 0.1,
            "total_episodes": 2,
            "total_frames": 120,
            "datasets": [{
                "dataset_name": "TacVerse/taccap-g1-test-task-0101",
                "uploader": "alice",
                "total_episodes": 2,
                "total_frames": 120,
                "duration_hours": 0.1,
                "last_modified": "2026-01-01T00:00:00+00:00",
            }],
        }
        with patch.object(main_app.dd, "migrate_pull_history_to_log"), \
                patch.object(main_app.dd, "load_history", return_value=[report]), \
                patch.object(main_app.dd, "load_hf_change_history", return_value={}), \
                patch.object(main_app.MainWindow, "_refresh_identity"):
            win = main_app.MainWindow()
        self.addCleanup(win.close)

        win._set_busy(True)
        win._on_pull_done(report, "pull_result_260101.json")

        self.assertIn("拉取完成: 1/1 个数据集", win.status.text())
        self.assertFalse(win.btn_pull.isEnabled())
        win._set_busy(False)

    def test_trend_axis_labels_are_adaptive(self):
        report = {
            "date": "260130",
            "org": "TacVerse",
            "datasets": [],
        }
        history = []
        start = dt.date(2026, 1, 1)
        for offset in range(30):
            day = start + dt.timedelta(days=offset)
            history.append({
                "date": day.strftime("%y%m%d"),
                "org": "TacVerse",
                "total_hours": offset + 1,
                "total_episodes": offset + 1,
                "total_frames": (offset + 1) * 100,
                "datasets": [],
            })
        with patch.object(main_app.dd, "migrate_pull_history_to_log"), \
                patch.object(main_app.dd, "load_history", return_value=history), \
                patch.object(main_app.dd, "load_hf_change_history", return_value={}), \
                patch.object(main_app.MainWindow, "_refresh_identity"):
            win = main_app.MainWindow()
        self.addCleanup(win.close)

        win.trend_plot.resize(380, 180)
        ticks = win._trend_x_ticks("260101", "260130")

        self.assertLess(len(ticks), 30)
        self.assertGreaterEqual(len(ticks), 7)
        self.assertEqual(0, ticks[0][0])
        self.assertEqual(29, ticks[-1][0])
        self.assertGreaterEqual(ticks[-1][0] - ticks[-2][0], 3)
        self.assertEqual("01-01", ticks[0][1])
        self.assertEqual("01-30", ticks[-1][1])
        self.assertNotIn("\n", ticks[0][1])

        sparse = [
            {"date": "260812", "new_hours": 20, "total_hours": 700},
            {"date": "260816", "new_hours": 10, "total_hours": 850},
        ]
        x, new_hours, total_hours = win._trend_plot_points(sparse, "260801")
        self.assertEqual([11, 15], x)
        self.assertEqual([20, 10], new_hours)
        self.assertEqual([700, 850], total_hours)

        win.trend_plot.resize(980, 180)
        short_range_ticks = win._trend_x_ticks("260801", "260816")
        gaps = [
            short_range_ticks[i + 1][0] - short_range_ticks[i][0]
            for i in range(len(short_range_ticks) - 1)
        ]
        self.assertGreaterEqual(len(short_range_ticks), 7)
        self.assertTrue(all(gap >= 2 for gap in gaps))

        daily_rows = [
            {"date": "260812", "group": "alice", "hours": 20},
            {"date": "260812", "group": "bob", "hours": 5},
            {"date": "260816", "group": "alice", "hours": 10},
        ]
        trend_series = win._trend_series_from_daily_rows(
            daily_rows, "260801", "260816")
        trend_by_date = {row["date"]: row for row in trend_series}
        self.assertEqual(16, len(trend_series))
        self.assertEqual(0, trend_by_date["260801"]["new_hours"])
        self.assertEqual(25, trend_by_date["260812"]["new_hours"])
        self.assertEqual(25, trend_by_date["260812"]["total_hours"])
        self.assertEqual(10, trend_by_date["260816"]["new_hours"])
        self.assertEqual(35, trend_by_date["260816"]["total_hours"])

        self.assertFalse(win._trend_uses_dual_axis([30, 10], [30, 40]))
        self.assertTrue(win._trend_uses_dual_axis([1, 1], [1, 10]))
        self.assertFalse(win._trend_uses_dual_axis([0], [10]))

        dual_axis_rows = [
            {"date": (dt.date(2026, 8, 1) + dt.timedelta(days=offset)).strftime("%y%m%d"),
             "group": "alice", "hours": 1}
            for offset in range(10)
        ]
        win._refresh_trends(dual_axis_rows, "260801", "260810", "2026-08-01 ~ 2026-08-10")
        self.assertTrue(win.trend_plot.getAxis("right").isVisible())


if __name__ == "__main__":
    unittest.main()
