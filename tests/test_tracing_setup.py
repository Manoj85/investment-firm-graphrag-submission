"""Verify opt-in tracing does not contact LangSmith when disabled/incomplete."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from langsmith.utils import LangSmithNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.config.project_config import PROJECT_ROOT, load_settings
from scripts.services.service_clients import ServiceError
from scripts.services.tracing_setup import TracingSession


class TracingTests(unittest.TestCase):
    def test_readback_retries_until_uploaded_run_visible(self):
        tracing = object.__new__(TracingSession)
        tracing.tracer = Mock()
        tracing.run_id = "invented-run"
        tracing.report = {}
        tracing.client = Mock()
        tracing.client.read_run.side_effect = [
            LangSmithNotFoundError("not visible yet"),
            Mock(end_time="finished", error=None),
        ]
        tracing.client.get_run_url.return_value = "https://example.test/run"
        with patch("scripts.services.tracing_setup.wait_for_all_tracers"), patch("scripts.services.tracing_setup.sleep") as pause:
            self.assertEqual(tracing.finish()["status"], "verified")
            self.assertEqual(tracing.client.read_run.call_count, 2)
            pause.assert_called_once_with(2)

    def test_disabled_tracing_makes_no_client(self):
        settings = load_settings(PROJECT_ROOT / "work" / "absent-test.env", {})
        with patch("scripts.services.tracing_setup.Client") as client:
            tracing = TracingSession(settings)
            self.assertEqual(tracing.finish()["status"], "disabled")
            self.assertEqual(tracing.invoke_config()["callbacks"], [])
            tracing.close()
            client.assert_not_called()

    def test_enabled_tracing_without_key_fails_before_client(self):
        settings = replace(load_settings(PROJECT_ROOT / "work" / "absent-test.env", {}),
                           langsmith_tracing=True)
        with patch("scripts.services.tracing_setup.Client") as client:
            with self.assertRaisesRegex(ServiceError, "missing"):
                TracingSession(settings)
            client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
