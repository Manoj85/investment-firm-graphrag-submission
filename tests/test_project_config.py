"""Check precedence, credential redaction, and invalid configuration offline."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from project_config import load_settings


class ConfigurationTests(unittest.TestCase):
    def test_shell_overrides_file_and_secrets_are_not_in_repr(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("NEBIUS_API_KEY=file-test-secret\nLANGSMITH_TRACING=false\n", encoding="utf-8")
            settings = load_settings(path, {"NEBIUS_API_KEY": "shell-test-secret"})
            self.assertEqual(settings.nebius_api_key, "shell-test-secret")
            self.assertNotIn("shell-test-secret", repr(settings))
            self.assertNotIn("file-test-secret", repr(settings))
            self.assertFalse(settings.service_readiness()["Nebius"])

    def test_missing_file_allows_environment_only_and_defaults_to_no_tracing(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = load_settings(Path(directory) / "absent.env", {})
            self.assertFalse(settings.langsmith_tracing)
            self.assertFalse(any(settings.service_readiness().values()))

    def test_invalid_tracing_value_fails_without_echoing_value(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "^LANGSMITH_TRACING must be true or false$"):
                load_settings(Path(directory) / "absent.env", {"LANGSMITH_TRACING": "invalid-sensitive-value"})


if __name__ == "__main__":
    unittest.main()
