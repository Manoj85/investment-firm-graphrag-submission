"""Frozen identity, exactly-once collection and error preservation without APIs."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluation.comparison import collect_comparison, validate_benchmark
from scripts.services.service_clients import ServiceError


class ComparisonTests(unittest.TestCase):
    def benchmark(self):
        questions = [{"id": f"T{i:02}", "question": f"Example question {i}?", "expected_status": "answered",
                      "expected": "Evaluation-only text", "required_sources": []} for i in range(10)]
        return {"status": "frozen_for_comparison", "questions": questions, "corpus_version": "v1",
                "question_sha256": hashlib.sha256(json.dumps(questions, sort_keys=True).encode()).hexdigest()}

    def test_frozen_benchmark_rejects_tampering_duplicates_or_stale_corpus(self):
        benchmark = self.benchmark()
        validate_benchmark(benchmark, "v1")
        for field, value in (("corpus_version", "old"), ("question_sha256", "changed")):
            modified = deepcopy(benchmark); modified[field] = value
            with self.assertRaises(ValueError):
                validate_benchmark(modified, "v1")
        benchmark["questions"][1]["id"] = benchmark["questions"][0]["id"]
        with self.assertRaises(ValueError):
            validate_benchmark(benchmark, "v1")

    def test_twenty_once_balanced_order_and_failure_kept_without_answer_key_leak(self):
        def answer(question, *, mode, settings, prompt_version):
            if question == "Example question 1?" and mode == "graph":
                raise ServiceError("Controlled failure")
            return {"question": question, "mode": mode, "answer": {"status": "insufficient_evidence",
                    "answer": "Not established.", "citations": []}, "sources": []}
        invoke = Mock(side_effect=answer)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "initial"
            summary = collect_comparison(self.benchmark(), output, None, invoke, {}, progress=lambda text: None)
            self.assertEqual(invoke.call_count, 20)
            self.assertEqual(summary["completed_attempts"], 20)
            self.assertEqual(summary["errors"], 1)
            self.assertEqual([row["mode"] for row in summary["attempts"][:4]], ["vector", "graph", "graph", "vector"])
            failure = json.loads((output / "T01-graph.json").read_text())
            self.assertEqual(failure["outcome"], "error")
            self.assertEqual(failure["error"], "Controlled failure")
            first = json.loads((output / "T00-vector.json").read_text())
            self.assertIsNone(first["diagnostics"]["semantic_grade"])
            with self.assertRaises(FileExistsError):
                collect_comparison(self.benchmark(), output, None, invoke, {})
            self.assertEqual(invoke.call_count, 20)

    def test_guard_failure_prevents_any_answer_call(self):
        invoke = Mock()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                collect_comparison(self.benchmark(), Path(folder) / "initial", None, invoke, {},
                                   guard=Mock(side_effect=ValueError("Changed runtime")))
        invoke.assert_not_called()
