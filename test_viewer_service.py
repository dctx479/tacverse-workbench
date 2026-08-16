import json
import unittest
from unittest.mock import patch

import viewer_service as vsvc


class _Response:
    def __init__(self, lines):
        self.lines = [line.encode("utf-8") for line in lines]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.lines)


class ViewerDoctorTests(unittest.TestCase):
    def test_doctor_parses_progress_and_result_stream(self):
        events = [
            {"type": "progress", "progress": {"overall_percent": 42,
                                                  "message": "Running"}},
            {"type": "result", "result": {"ok": True, "report": {}}},
        ]
        progress = []
        with patch("viewer_service.urllib.request.urlopen",
                   return_value=_Response(
                       [json.dumps(event) + "\n" for event in events])) as open_url:
            result, error = vsvc.ViewerService(port=3001).doctor(
                "TacVerse/example", max_episodes=25,
                on_progress=progress.append)
        self.assertIsNone(error)
        self.assertEqual(result["ok"], True)
        self.assertEqual(progress[0]["overall_percent"], 42)
        request = open_url.call_args.args[0]
        self.assertEqual(request.method, "POST")
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["maxEpisodes"], 25)

    def test_doctor_reports_stream_error(self):
        with patch("viewer_service.urllib.request.urlopen",
                   return_value=_Response([
                       json.dumps({"type": "error", "error": "failed"}) + "\n"
                   ])):
            result, error = vsvc.ViewerService(port=3001).doctor("TacVerse/example")
        self.assertIsNone(result)
        self.assertEqual(error, "failed")


if __name__ == "__main__":
    unittest.main()
