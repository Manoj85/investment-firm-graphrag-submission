"""Citation rendering, quote provenance, failure boundaries and shared workflow."""

from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from scripts.rag.structured_claims import CLAIMS_PROMPT_VERSION, validate_and_render_claims
from scripts.rag.rag_workflow import GroundedChat, build_workflow, validate_answer
from scripts.services.service_clients import ServiceError


SOURCES = [
    {"id": "a", "citation": "S1", "text": "Lina works on Juniper. Juniper uses Lantern.",
     "metadata": {"document_id": "TEST-1", "section": "Work"}},
    {"id": "b", "citation": "S2", "text": "Lina has the skill Record Review.",
     "metadata": {"document_id": "TEST-2", "section": "Skill"}},
]
PAYLOAD = {"status": "answered", "claims": [
    {"text": "Lina works on Juniper.", "evidence": [{"source": "S1", "quote": "Lina works on Juniper."}]},
    {"text": "Lina has Record Review expertise.", "evidence": [{"source": "S2", "quote": "Lina has the skill Record Review."}]},
], "message": ""}


class StructuredClaimsTests(unittest.TestCase):
    def test_separate_relationship_claims_render_consistent_citations(self):
        answer, payload = validate_and_render_claims(json.dumps(PAYLOAD), SOURCES)
        self.assertEqual(answer["citations"], ["S1", "S2"])
        self.assertIn("Juniper. [S1]", answer["answer"])
        self.assertIn("expertise. [S2]", answer["answer"])
        self.assertEqual(validate_answer(json.dumps(answer), SOURCES), answer)
        self.assertEqual(payload, PAYLOAD)

    def test_quotes_cannot_be_invented_or_borrowed_from_another_source(self):
        for source, quote in [("S9", "Lina works on Juniper."),
                              ("S2", "Lina works on Juniper."),
                              ("S1", "Lina approved Juniper."),
                              ("S1", "Lina"), ("S1", [])]:
            value = deepcopy(PAYLOAD)
            value["claims"][0]["evidence"] = [{"source": source, "quote": quote}]
            with self.assertRaises(ServiceError):
                validate_and_render_claims(json.dumps(value), SOURCES)

    def test_missing_claim_support_and_inconsistent_status_fail_closed(self):
        cases = []
        for field, value in [("claims", []), ("status", []), ("message", "Missing requested fact.")]:
            item = deepcopy(PAYLOAD); item[field] = value; cases.append(item)
        item = deepcopy(PAYLOAD); item["claims"][0]["evidence"] = []; cases.append(item)
        item = deepcopy(PAYLOAD); item["claims"][0]["text"] += " [S1]"; cases.append(item)
        item = deepcopy(PAYLOAD); item["extra"] = "unexpected"; cases.append(item)
        for item in cases:
            with self.assertRaises(ServiceError):
                validate_and_render_claims(json.dumps(item), SOURCES)

    def test_partial_refusal_and_clarification_do_not_need_fake_claims(self):
        for status, claims, message in [
            ("insufficient_evidence", [], "No return guarantee appears in the retrieved passages."),
            ("insufficient_evidence", PAYLOAD["claims"][:1], "The requested skill is missing."),
            ("clarification", [], "Do you mean Robin Bell or Robin Wood?"),
        ]:
            answer, _ = validate_and_render_claims(json.dumps(dict(status=status, claims=claims, message=message)), SOURCES)
            self.assertEqual(answer["status"], status)
            self.assertEqual(len(answer["citations"]), len(claims))
        with self.assertRaises(ServiceError):
            validate_and_render_claims(json.dumps(dict(status="clarification", claims=PAYLOAD["claims"], message="Which Lina?")), SOURCES)

    def test_quote_presence_is_not_a_semantic_entailment_checker(self):
        value = deepcopy(PAYLOAD)
        value["claims"][0]["text"] = "Lina approved Juniper."
        answer, _ = validate_and_render_claims(json.dumps(value), SOURCES)
        # Deliberate limitation: exact source quotes do not validate a paraphrase.
        # Review/evaluation must catch this; do not claim semantic verification.
        self.assertIn("approved", answer["answer"])

    def test_workflow_retains_claims_and_uses_single_bounded_model_call(self):
        requests = []
        def response(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(PAYLOAD)}}],
                                             "usage": {"total_tokens": 100}, "model": "test-model"})
        settings = SimpleNamespace(nebius_base_url="https://api.tokenfactory.nebius.com/v1/",
                                   nebius_api_key="fake", nebius_chat_model="test-model")
        chat = GroundedChat(settings, prompt_version=CLAIMS_PROMPT_VERSION, transport=httpx.MockTransport(response))
        try:
            result = build_workflow(lambda question: {"evidence": SOURCES}, chat).invoke({"question": "Who works on Juniper and has review expertise?"})
            self.assertEqual(result["structured_claims"], PAYLOAD)
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0]["max_tokens"], 1024)
            self.assertEqual(result["answer"]["citations"], ["S1", "S2"])
        finally:
            chat.close()

    def test_invalid_quote_saves_actual_rejection_without_retry(self):
        value = deepcopy(PAYLOAD); value["claims"][0]["evidence"][0]["quote"] = "An invented supporting quotation."
        settings = SimpleNamespace(nebius_base_url="https://api.tokenfactory.nebius.com/v1/",
                                   nebius_api_key="fake", nebius_chat_model="test-model")
        calls = []
        def respond(request):
            calls.append(request)
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}], "usage": {"total_tokens": 50}})
        chat = GroundedChat(settings, prompt_version=CLAIMS_PROMPT_VERSION, transport=httpx.MockTransport(respond))
        try:
            with patch("scripts.rag.rag_workflow.atomic_json") as save:
                with self.assertRaises(ServiceError):
                    chat.generate("Question?", "Context", SOURCES)
            self.assertEqual(len(calls), 1)
            self.assertEqual(json.loads(save.call_args.args[1]["content"]), value)
            self.assertEqual(save.call_args.args[1]["usage"]["total_tokens"], 50)
        finally:
            chat.close()
