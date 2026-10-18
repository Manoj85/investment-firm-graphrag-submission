import json
from types import SimpleNamespace
import unittest

import httpx

from scripts.config.project_config import PROJECT_ROOT
from scripts.rag.excerpt_selection import (EXCERPT_VERSION, excerpt_catalog,
    render_excerpt_selection)
from scripts.rag.rag_workflow import GroundedChat, build_workflow
from scripts.services.service_clients import ServiceError


SOURCES = [{"citation": "S1", "text": "## Review\n\nLina prepared the review assessment.",
            "metadata": {"document_id": "TEST-1", "section": "Review"}}]


class ExcerptSelectionTests(unittest.TestCase):
    def test_source_wording_prevents_the_observed_unsupported_clause(self):
        original = json.loads((PROJECT_ROOT / "outputs/evaluation/runs/revised-20261018T013123324572Z/Q07-graph.json").read_text(encoding="utf-8"))
        sources = original["report"]["sources"]
        catalog = excerpt_catalog(sources)
        chosen = next(e for e in catalog if e["source"] == "S4")
        answer, claims, payload = render_excerpt_selection(json.dumps({"status": "answered", "excerpt_ids": [chosen["id"]]}), sources, catalog)
        self.assertIn("Priya Shah prepared the completeness-review assessment.", answer["answer"])
        self.assertNotIn("supporting DEC-004 recommendation", answer["answer"])
        self.assertEqual(claims["claims"][0]["text"], chosen["quote"])
        self.assertEqual(answer["citations"], ["S4"])

    def test_factual_text_injection_is_rejected_not_silently_dropped(self):
        catalog = excerpt_catalog(SOURCES)
        for field in ["text", "claims", "message"]:
            value = {"status": "answered", "excerpt_ids": ["E1"], field: "Lina approved a recommendation."}
            with self.assertRaises(ServiceError):
                render_excerpt_selection(json.dumps(value), SOURCES, catalog)

    def test_invalid_duplicate_empty_or_ambiguous_selection_fails_closed(self):
        catalog = excerpt_catalog(SOURCES)
        for status, ids in [("answered", []), ("answered", ["E9"]),
                            ("answered", ["E1", "E1"]), ("answered", [1]),
                            ("clarification", ["E1"])]:
            with self.assertRaises(ServiceError):
                render_excerpt_selection(json.dumps({"status": status, "excerpt_ids": ids}), SOURCES, catalog)

    def test_changed_source_cannot_reuse_a_previous_excerpt(self):
        catalog = excerpt_catalog(SOURCES)
        with self.assertRaises(ServiceError):
            render_excerpt_selection('{"status":"answered","excerpt_ids":["E1"]}',
                [{**SOURCES[0], "text": "Lina did not prepare an assessment."}], catalog)

    def test_long_sentence_is_not_truncated_into_a_new_fact(self):
        text = "Lina " + "carefully " * 50 + "prepared the assessment."
        source = {**SOURCES[0], "text": text + " Omar reviewed the record."}
        catalog = excerpt_catalog([source])
        self.assertEqual([e["quote"] for e in catalog], ["Omar reviewed the record."])
        for item in catalog:
            self.assertIn(item["quote"], source["text"])

    def test_nonanswers_have_no_free_generated_background(self):
        for status in ["clarification", "insufficient_evidence"]:
            answer, claims, raw = render_excerpt_selection(json.dumps({"status": status, "excerpt_ids": []}), SOURCES, excerpt_catalog(SOURCES))
            self.assertEqual(answer["status"], status)
            self.assertEqual(claims["claims"], [])
            self.assertEqual(answer["citations"], [])
            self.assertNotIn("Lina", answer["answer"])

    def test_schema_and_transport_accept_only_ids_and_preserve_provenance(self):
        calls = []
        def response(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
                "content": '{"status":"answered","excerpt_ids":["E1"]}'}}]})
        settings = SimpleNamespace(nebius_base_url="https://api.tokenfactory.nebius.com/v1/", nebius_api_key="fake", nebius_chat_model="test")
        chat = GroundedChat(settings, transport=httpx.MockTransport(response), prompt_version=EXCERPT_VERSION)
        try:
            result = chat.generate("Who prepared the review?", "Original context", SOURCES)
        finally:
            chat.close()
        self.assertEqual(len(calls), 1)
        schema = calls[0]["response_format"]["json_schema"]["schema"]
        self.assertEqual(set(schema["properties"]), {"status", "excerpt_ids"})
        self.assertEqual(schema["properties"]["excerpt_ids"]["items"]["enum"], ["E1"])
        self.assertEqual(result["generation_payload"]["excerpt_ids"], ["E1"])
        self.assertEqual(result["excerpt_selection"]["catalog"][0]["source"], "S1")
        self.assertEqual(result["structured_claims"]["claims"][0]["text"], "Lina prepared the review assessment.")

    def test_reviewed_bridge_connects_selected_endpoint_excerpts(self):
        original = json.loads((PROJECT_ROOT / "outputs/evaluation/runs/revised-20261018T013123324572Z/Q07-graph.json").read_text(encoding="utf-8"))["report"]
        class SelectionChat:
            prompt_version = EXCERPT_VERSION
            def generate(self, question, context, sources):
                catalog = excerpt_catalog(sources)
                ids = [next(e["id"] for e in catalog if e["source"] == label and (label != "S6" or "LedgerCheck checks" in e["quote"])) for label in ["S4", "S6"]]
                answer, claims, raw = render_excerpt_selection(json.dumps({"status": "answered", "excerpt_ids": ids}), sources, catalog)
                return {"answer": answer, "structured_claims": claims, "generation_payload": raw}
        workflow = build_workflow(lambda question: {"evidence": original["sources"],
            "graph_selection": original["graph_selection"]}, SelectionChat())
        result = workflow.invoke({"question": original["question"]})
        self.assertNotIn("S3", result["selected_excerpt_answer"]["citations"])
        self.assertIn("S3", result["answer"]["citations"])
        bridge = next(b for b in result["answer_checks"]["source_backed_bridges"] if b["relation"] == "REQUIRES_TOOL")
        self.assertIn(bridge["quote"], next(s["text"] for s in result["sources"] if s["citation"] == bridge["citation"]))
        self.assertNotIn("supporting DEC-004 recommendation", result["answer"]["answer"])
        self.assertNotIn("model_answer", result)
