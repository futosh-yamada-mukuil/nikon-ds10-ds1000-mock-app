"""Night dates and local reports; no account usage or AI calls."""
from datetime import datetime
from pathlib import Path
import sys
import tempfile
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nightly import night_window, write_report


class NightlyScheduleTests(unittest.TestCase):
    def test_midnight_keeps_the_same_operational_date(self):
        zone = ZoneInfo("Asia/Tokyo")
        before = night_window(datetime(2026, 10, 7, 23, 30, tzinfo=zone), ["23:00", "04:00"])
        after = night_window(datetime(2026, 10, 8, 1, 0, tzinfo=zone), ["23:00", "04:00"])
        self.assertEqual(before, after)
        self.assertEqual(before[0], "2026-10-07")
        self.assertEqual(before[1], datetime(2026, 10, 8, 4, tzinfo=zone).timestamp())

    def test_window_boundaries(self):
        zone = ZoneInfo("Asia/Tokyo")
        for day, hour, minute in [(7, 22, 59), (8, 4, 0), (8, 12, 0)]:
            self.assertIsNone(night_window(datetime(2026, 10, day, hour, minute, tzinfo=zone), ["23:00", "04:00"]))
        self.assertIsNotNone(night_window(datetime(2026, 10, 7, 23, tzinfo=zone), ["23:00", "04:00"]))

    def test_same_day_window_and_empty_window(self):
        now = datetime(2026, 10, 8, 2, tzinfo=ZoneInfo("Asia/Tokyo"))
        self.assertEqual(night_window(now, ["00:00", "04:00"])[0], "2026-10-08")
        with self.assertRaises(ValueError):
            night_window(now, ["04:00", "04:00"])

    def test_report_uses_existing_state_without_ai(self):
        with tempfile.TemporaryDirectory() as temp:
            local = Path(temp)
            state = {"stopped": True, "reason": "remaining_at_or_below_stop_threshold",
                     "last_run_date": "2026-10-07", "quota_windows": [
                         {"limit_id": "codex", "window": "primary", "remaining_percent": 80}],
                     "task_summaries": ["切り出しを修正しました。"], "unfinished_tasks": ["未検証作業"],
                     "executed_tests": [{"status": "interrupted", "exit_code": -15, "log": "local.log"}],
                     "changed_files_status": [" M app/inference.py"]}
            write_report(local, state, local, {"model": "gpt-6-luna", "effort": "low", "time_window": ["23:00", "04:00"]})
            text = (local / "report.md").read_text()
            for expected in ["80%", "切り出しを修正", "未検証作業", "interrupted", "コミットおよびPushは実行していません"]:
                self.assertIn(expected, text)
