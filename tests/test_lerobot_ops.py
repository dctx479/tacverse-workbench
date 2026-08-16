import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import lerobot_ops as lops


class LerobotOpsProcessTests(unittest.TestCase):
    def _runner(self, source):
        temp = tempfile.TemporaryDirectory()
        path = Path(temp.name) / "runner.py"
        path.write_text(textwrap.dedent(source), encoding="utf-8")
        self.addCleanup(temp.cleanup)
        return path

    def test_run_op_parses_result_with_interleaved_logs(self):
        runner = self._runner("""
            import json
            import sys

            json.loads(sys.stdin.read())
            print("loading", flush=True)
            print("progress", file=sys.stderr, flush=True)
            print('RESULT_JSON:' + json.dumps({
                "ok": True, "op": "split", "outputs": [{"repo_id": "x/y"}],
                "error": None,
            }), flush=True)
        """)
        logs = []

        with patch.object(lops, "RUNNER", runner):
            result = lops.run_op({"op": "split"}, log=logs.append)

        self.assertTrue(result["ok"])
        self.assertEqual("split", result["op"])
        self.assertIn("loading", logs)
        self.assertIn("progress", logs)

    def test_run_op_cancel_terminates_child(self):
        runner = self._runner("""
            import sys
            import time

            sys.stdin.read()
            time.sleep(30)
        """)
        with patch.object(lops, "RUNNER", runner):
            with self.assertRaisesRegex(RuntimeError, "操作已取消"):
                lops.run_op({"op": "slow"}, cancel=lambda: True)


if __name__ == "__main__":
    unittest.main()
