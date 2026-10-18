"""Derive honest statuses from inspectable requirement/claim links."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
import httpx

from scripts.rag.requirement_claims import REQUIREMENTS_VERSION, validate_requirement_claims
from scripts.rag.rag_workflow import GroundedChat, build_workflow
from scripts.services.service_clients import ServiceError

SOURCE = {"id": "one", "citation": "S1", "text": "Juniper depends on Cedar.",
          "metadata": {"document_id": "TEST", "section": "Dependency"}}
VALUE = {"claims": [{"text": "Juniper depends on Cedar.", "evidence": [{"source": "S1", "quote": SOURCE["text"]}]}],
         "requirements": [{"part": "Identify upstream project", "claim_ids": [1]}], "clarification": ""}

class RequirementClaimsTests(unittest.TestCase):
    def test_missing_parts_derive_insufficient_without_discarding_supported_claim(self):
        value = deepcopy(VALUE)
        value["requirements"] += [{"part": "Identify upstream lead", "claim_ids": []},
                                   {"part": "Identify that lead's skill", "claim_ids": []}]
        answer, canonical, original = validate_requirement_claims(json.dumps(value), [SOURCE])
        self.assertEqual(answer["status"], "insufficient_evidence")
        self.assertIn("upstream lead", answer["answer"])
        self.assertEqual(answer["citations"], ["S1"])
        self.assertEqual(original, value)
        self.assertEqual(canonical["status"], "insufficient_evidence")

    def test_links_reject_unknown_boolean_duplicate_or_unreferenced_claims(self):
        for ids in [[2], [True], [1, 1], []]:
            value = deepcopy(VALUE); value["requirements"][0]["claim_ids"] = ids
            with self.assertRaises(ServiceError):
                validate_requirement_claims(json.dumps(value), [SOURCE])

    def test_clarification_and_no_evidence_do_not_fabricate_answer_claims(self):
        value = {"claims": [], "requirements": [], "clarification": "Robin Bell or Robin Wood?"}
        self.assertEqual(validate_requirement_claims(json.dumps(value), [SOURCE])[0]["status"], "clarification")
        value = {"claims": [], "requirements": [{"part": "Identify a return guarantee", "claim_ids": []}], "clarification": ""}
        self.assertEqual(validate_requirement_claims(json.dumps(value), [SOURCE])[0]["status"], "insufficient_evidence")
        value = deepcopy(VALUE); value["clarification"] = "Which project?"
        with self.assertRaises(ServiceError):
            validate_requirement_claims(json.dumps(value), [SOURCE])

    def test_workflow_retains_requirement_payload_and_uses_one_schema_request(self):
        calls = []
        def respond(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(VALUE)}}], "usage": {"total_tokens": 50}})
        settings = SimpleNamespace(nebius_base_url="https://api.tokenfactory.nebius.com/v1/", nebius_api_key="fake", nebius_chat_model="test")
        chat = GroundedChat(settings, prompt_version=REQUIREMENTS_VERSION, transport=httpx.MockTransport(respond))
        try:
            result = build_workflow(lambda question: {"evidence": [SOURCE]}, chat).invoke({"question": "Which project is upstream?"})
            self.assertEqual(result["generation_payload"], VALUE)
            self.assertEqual(result["answer"]["status"], "answered")
            self.assertEqual(len(calls), 1)
            self.assertNotIn("status", calls[0]["response_format"]["json_schema"]["schema"]["properties"])
        finally:
            chat.close()

    def test_structural_links_cannot_prove_semantic_coverage(self):
        value = deepcopy(VALUE); value["requirements"][0]["part"] = "Identify a person's skill"
        # Deliberately document the semantic boundary: a valid index is not proof
        # that a dependency claim establishes a skill. Source review remains.
        self.assertEqual(validate_requirement_claims(json.dumps(value), [SOURCE])[0]["status"], "answered")
