"""Offline tests for diagnostics: no real credentials and no provider charges."""

from pathlib import Path
import sys
import unittest

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.config.project_config import load_settings
from scripts.services.verify_nebius import run_probe, summarize_chat, summarize_embedding, validate_settings


class DiagnosticTests(unittest.TestCase):
    def test_rejects_incomplete_or_untrusted_endpoint_before_requests(self):
        path = Path(__file__).resolve().parents[1] / "tests" / "nonexistent.env"
        with self.assertRaises(ValueError):
            validate_settings(load_settings(path, {}), "all")
        with self.assertRaises(ValueError):
            validate_settings(load_settings(path, {"NEBIUS_API_KEY": "invented-test-secret",
                "NEBIUS_BASE_URL": "https://example.com/v1/", "NEBIUS_CHAT_MODEL": "test"}), "chat")

    def test_rejects_truncated_or_empty_chat(self):
        for content, reason in [("", "stop"), ("Partial answer", "length")]:
            with self.assertRaises(ValueError):
                summarize_chat({"choices": [{"message": {"content": content}, "finish_reason": reason}]})

    def test_rejects_invalid_vectors(self):
        for vector in [[], [float("nan")], [float("inf")], [True], ["0.2"]]:
            with self.assertRaises(ValueError):
                summarize_embedding({"data": [{"embedding": vector}]})

    def test_measures_returned_vector_not_advertised_size(self):
        result = summarize_embedding({"data": [{"embedding": [0.1, -0.2, 0.3]}]})
        self.assertEqual(result["dimension"], 3)
        self.assertEqual(result["vector_count"], 1)

    def test_http_failure_is_not_retried_or_echoed(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(401, json={"detail": "invented-secret-not-to-log"})
        with httpx.Client(base_url="https://api.tokenfactory.nebius.com/v1/",
                          transport=httpx.MockTransport(handler)) as client:
            result = run_probe(client, "chat/completions", {"model": "test"}, summarize_chat)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["http_status"], 401)
        self.assertNotIn("invented-secret-not-to-log", str(result))


if __name__ == "__main__":
    unittest.main()
