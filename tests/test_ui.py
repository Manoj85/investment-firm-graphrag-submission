"""Exercise explicit UI actions without contacting any real provider."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
from scripts.services.service_clients import ServiceError


ROOT = Path(__file__).resolve().parents[1]


def fake_report(question, *, mode):
    source = {"id": "chunk", "text": "Taylor approved Decision X.", "citation": "S1",
              "metadata": {"document_id": "TEST", "section": "Approval", "source": "fiction.md"},
              "retrieval_origins": ["graph_path"] if mode == "graph" else ["vector"]}
    step = {"edge_id": "edge", "source": "decision:x", "source_label": "Decision X",
            "target": "person:taylor", "target_label": "Taylor", "relation": "APPROVED_BY",
            "traversal": "forward", "evidence": [{"chunk_id": "chunk", "document_id": "TEST",
                                                      "section": "Approval", "excerpt": source["text"]}]}
    return {"question": question, "mode": mode, "answer": {"status": "answered",
            "answer": "Taylor approved Decision X [S1].", "citations": ["S1"]},
            "sources": [source], "elapsed_seconds": 1.2, "usage": {"total_tokens": 100},
            "packing": {"budget": 2000, "context_tokens": 20, "selected_ids": ["chunk"], "dropped_ids": []},
            "namespace": "test", "corpus_version": "v1", "prompt_version": "test-prompt",
            "tracing": {"status": "disabled"}, "graph_selection": {
                "status": "candidates_found", "paths": [{"steps": [deepcopy(step)]}], "truncated": False}}


class WorkspaceTests(unittest.TestCase):
    @patch("scripts.ui.app.run_question", side_effect=fake_report)
    def test_load_selection_compare_history_and_clear_make_only_explicit_calls(self, backend):
        def with_claims(question, *, mode):
            report = fake_report(question, mode=mode)
            report["structured_claims"] = {"status": "answered", "message": "", "claims": [
                {"text": "Taylor approved Decision X.", "evidence": [
                    {"source": "S1", "quote": "Taylor approved Decision X."}]}]}
            return report
        backend.side_effect = with_claims
        app = AppTest.from_file(ROOT / "streamlit_app.py", default_timeout=20).run()
        self.assertFalse(app.exception)
        backend.assert_not_called()
        app.radio(key="retrieval_choice").set_value("Compare both").run()
        backend.assert_not_called()
        app.radio(key="retrieval_choice").set_value("Compare both")
        app.text_area(key="question_input").set_value("Who approved Decision X?")
        next(button for button in app.button if button.label == "Run question").click().run()
        self.assertFalse(app.exception)
        self.assertEqual([call.kwargs["mode"] for call in backend.call_args_list], ["vector", "graph"])
        self.assertEqual([tab.label for tab in app.tabs], ["Answer", "Sources", "Graph paths", "Run details"])
        self.assertEqual(len(app.session_state.run_history), 1)
        self.assertEqual(len(app.session_state.run_history[0]["reports"]), 2)
        self.assertEqual(sum(item.label == "Claims and their exact source excerpts" for item in app.expander), 2)
        self.assertEqual(sum(item.value == "Taylor approved Decision X." for item in app.text), 2)
        app.run()  # All tab renderers execute; none may call the backend again.
        self.assertEqual(backend.call_count, 2)
        next(button for button in app.button if button.label == "Clear session view").click().run()
        self.assertEqual(backend.call_count, 2)
        self.assertEqual(app.session_state.run_history, [])

    @patch("scripts.ui.app.run_question")
    def test_structured_format_is_explicit_and_shared_between_modes(self, backend):
        backend.side_effect = lambda question, mode, prompt_version: fake_report(question, mode=mode)
        for label, version in [("Structured claims (experimental)", "grounded-answer-v5-claims"),
                               ("Schema-constrained claims (experimental)", "grounded-answer-v6-claims-schema")]:
            with self.subTest(format=label):
                backend.reset_mock()
                app = AppTest.from_file(ROOT / "streamlit_app.py", default_timeout=20).run()
                app.selectbox(key="answer_format").set_value(label).run()
                backend.assert_not_called()
                # Form edits are submitted together; a bare rerun does not commit them.
                app.selectbox(key="answer_format").set_value(label)
                app.radio(key="retrieval_choice").set_value("Compare both")
                next(button for button in app.button if button.label == "Run question").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(backend.call_count, 2)
                self.assertTrue(all(call.kwargs["prompt_version"] == version for call in backend.call_args_list))

    @patch("scripts.ui.app.run_question")
    def test_partial_comparison_failure_keeps_the_successful_side(self, backend):
        backend.side_effect = lambda question, mode: fake_report(question, mode=mode) if mode == "vector" else (_ for _ in ()).throw(ServiceError("Test service failure"))
        app = AppTest.from_file(ROOT / "streamlit_app.py", default_timeout=20).run()
        app.radio(key="retrieval_choice").set_value("Compare both")
        next(button for button in app.button if button.label == "Run question").click().run()
        self.assertFalse(app.exception)
        entry = app.session_state.run_history[0]
        self.assertIn("vector", entry["reports"])
        self.assertIn("graph", entry["errors"])
        self.assertTrue(app.error)
        self.assertEqual(backend.call_count, 2)

    @patch("scripts.ui.app.run_question", side_effect=fake_report)
    def test_blank_input_does_not_call_backend(self, backend):
        app = AppTest.from_file(ROOT / "streamlit_app.py", default_timeout=20).run()
        app.text_area(key="question_input").set_value("   ")
        next(button for button in app.button if button.label == "Run question").click().run()
        backend.assert_not_called()
        self.assertEqual(app.session_state.run_history, [])
        self.assertTrue(app.warning)

    @patch("scripts.ui.app.run_question")
    def test_clarification_renders_without_a_fake_generated_answer(self, backend):
        report = fake_report("What does Alex do?", mode="graph")
        report.update(answer={"status": "clarification", "answer": "Alex Morgan or Alex Rivera?", "citations": []},
                      sources=[], usage=None, graph_selection={"status": "clarification_required", "mentions": []})
        backend.return_value = report
        app = AppTest.from_file(ROOT / "streamlit_app.py", default_timeout=20).run()
        next(button for button in app.button if button.label == "Run question").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Alex Morgan" in text.value for text in app.markdown))
        self.assertTrue(any(metric.value == "No model call" for metric in app.metric))
