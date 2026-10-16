"""Verify opt-in tracing does not contact LangSmith when disabled/incomplete."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from project_config import PROJECT_ROOT, load_settings
from service_clients import ServiceError
from tracing_setup import TracingSession


class TracingTests(unittest.TestCase):
    def test_disabled_tracing_makes_no_client(self):
        settings = load_settings(PROJECT_ROOT / "work" / "absent-test.env", {})
        with patch("tracing_setup.Client") as client:
            tracing = TracingSession(settings)
            self.assertEqual(tracing.finish()["status"], "disabled")
            self.assertEqual(tracing.invoke_config()["callbacks"], [])
            tracing.close()
            client.assert_not_called()

    def test_enabled_tracing_without_key_fails_before_client(self):
        settings = replace(load_settings(PROJECT_ROOT / "work" / "absent-test.env", {}),
                           langsmith_tracing=True)
        with patch("tracing_setup.Client") as client:
            with self.assertRaisesRegex(ServiceError, "missing"):
                TracingSession(settings)
            client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
