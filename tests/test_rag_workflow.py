"""Check evidence/citation boundaries and actual graph execution offline."""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qwen_tokenizer import TOKENIZER_PATH, count_tokens, load_tokenizer
from rag_workflow import build_workflow, select_evidence, validate_answer
from service_clients import ServiceError


SOURCE = {"id": "source-id", "text": "Maya leads Atlas.",
          "metadata": {"document_id": "DOC-002", "section": "Ownership"}}


class CitationTests(unittest.TestCase):
    def test_unknown_and_missing_citations_rejected(self):
        sources = [{**SOURCE, "citation": "S1"}]
        for answer in [
            {"status": "answered", "answer": "Maya leads Atlas.", "citations": []},
            {"status": "answered", "answer": "Maya leads Atlas [S2].", "citations": ["S2"]},
            {"status": "answered", "answer": "Maya leads Atlas [S1].", "citations": []},
            {"status": [], "answer": "Invalid", "citations": []},
        ]:
            with self.assertRaises(ServiceError):
                validate_answer(json.dumps(answer), sources)

    def test_valid_refusal_does_not_require_invented_citation(self):
        answer = {"status": "insufficient_evidence", "answer": "The sources do not establish that.", "citations": []}
        self.assertEqual(validate_answer(json.dumps(answer), []), answer)


@unittest.skipUnless(TOKENIZER_PATH.is_file(), "Prepare the pinned tokenizer first")
class WorkflowTests(unittest.TestCase):
    def test_complete_passages_obey_evidence_budget(self):
        tokenizer = load_tokenizer()
        sources, context = select_evidence([SOURCE], tokenizer, budget=1)
        self.assertEqual(sources, [])
        self.assertEqual(context, "")
        sources, context = select_evidence([SOURCE], tokenizer, budget=100)
        self.assertLessEqual(count_tokens(tokenizer, context), 100)
        self.assertIn(SOURCE["text"], context)
        self.assertEqual(sources[0]["citation"], "S1")

    def test_graph_passes_actual_evidence_and_question_to_generation(self):
        calls = []
        class FakeChat:
            def generate(self, question, context, sources):
                calls.append((question, context, sources))
                return {"answer": {"status": "answered", "answer": "Maya [S1].", "citations": ["S1"]},
                        "usage": {}, "response_model": "mock-model"}
        graph = build_workflow(lambda question: {"question": question, "evidence": [SOURCE]}, FakeChat())
        result = graph.invoke({"question": "Who leads Atlas?"})
        self.assertEqual(calls[0][0], "Who leads Atlas?")
        self.assertIn("Maya leads Atlas.", calls[0][1])
        self.assertEqual(result["answer"]["citations"], ["S1"])

    def test_empty_retrieval_skips_model_and_returns_insufficient_evidence(self):
        class MustNotCall:
            def generate(self, *args):
                raise AssertionError("No evidence should skip the model")
        graph = build_workflow(lambda question: {"evidence": []}, MustNotCall())
        result = graph.invoke({"question": "Unknown fact?"})
        self.assertEqual(result["answer"]["status"], "insufficient_evidence")
        self.assertIsNone(result["response_model"])


if __name__ == "__main__":
    unittest.main()
