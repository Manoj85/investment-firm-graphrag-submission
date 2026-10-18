"""Citation rendering, quote provenance, failure boundaries and shared workflow."""

from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from scripts.rag.structured_claims import (CLAIMS_PROMPT_VERSION, SCHEMA_CLAIMS_PROMPT_VERSION,
    claims_schema, validate_and_render_claims, normalize_quote_punctuation)
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
    def test_only_unique_exact_semicolon_clause_normalizes_with_audit(self):
        value = deepcopy(PAYLOAD); value["claims"] = value["claims"][:1]
        value["claims"][0]["evidence"][0]["quote"] = "Lina works on Juniper."
        sources = [{**SOURCES[0], "text": "Lina works on Juniper; Omar reviews records."}]
        content, audit = normalize_quote_punctuation(json.dumps(value), sources)
        answer, canonical = validate_and_render_claims(content, sources)
        self.assertEqual(canonical["claims"][0]["evidence"][0]["quote"], "Lina works on Juniper")
        self.assertEqual(audit[0]["model_quote"], "Lina works on Juniper.")
        self.assertEqual(answer["citations"], ["S1"])

    def test_punctuation_normalizer_does_not_fix_words_or_ambiguous_spans(self):
        for text in ["Lina does not work on Juniper; Omar reviews records.",
                     "Lina works on Juniper; Lina works on Juniper;", "Lina works on Juniper today."]:
            value = deepcopy(PAYLOAD); value["claims"] = value["claims"][:1]
            original = json.dumps(value)
            content, audit = normalize_quote_punctuation(original, [{**SOURCES[0], "text": text}])
            self.assertEqual(content, original)
            self.assertEqual(audit, [])
    def test_default_promotes_tested_schema_policy_and_retains_legacy(self):
        from inspect import signature
        from scripts.rag.answer_service import run_question
        from scripts.rag.rag_workflow import PROMPT_VERSION, BASELINE_PROMPT_VERSION, PROMPTS
        self.assertEqual(PROMPT_VERSION, SCHEMA_CLAIMS_PROMPT_VERSION)
        self.assertEqual(BASELINE_PROMPT_VERSION, "grounded-answer-v2")
        self.assertIn(BASELINE_PROMPT_VERSION, PROMPTS)
        self.assertEqual(PROMPTS[CLAIMS_PROMPT_VERSION], PROMPTS[SCHEMA_CLAIMS_PROMPT_VERSION])
        self.assertEqual(signature(run_question).parameters["prompt_version"].default, SCHEMA_CLAIMS_PROMPT_VERSION)

    def test_schema_is_closed_bounded_and_uses_only_available_labels(self):
        schema = claims_schema(SOURCES)
        claim = schema["properties"]["claims"]["items"]
        evidence = claim["properties"]["evidence"]["items"]
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(claim["additionalProperties"])
        self.assertFalse(evidence["additionalProperties"])
        self.assertEqual(schema["properties"]["claims"]["maxItems"], 8)
        self.assertEqual(evidence["properties"]["source"]["enum"], ["S1", "S2"])
        self.assertNotIn(SOURCES[0]["text"], json.dumps(schema))
        self.assertEqual(claims_schema(SOURCES[:1])["properties"]["claims"]["items"]["properties"]["evidence"]["items"]["properties"]["source"]["enum"], ["S1"])
        for sources in [[], [SOURCES[0], SOURCES[0]]]:
            with self.assertRaises(ValueError):
                claims_schema(sources)

    def test_schema_generation_preserves_v5_prompt_limits_and_single_call(self):
        requests = []
        def response(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(PAYLOAD)}}]})
        settings = SimpleNamespace(nebius_base_url="https://api.tokenfactory.nebius.com/v1/",
                                   nebius_api_key="fake", nebius_chat_model="test-model")
        for version in [CLAIMS_PROMPT_VERSION, SCHEMA_CLAIMS_PROMPT_VERSION]:
            chat = GroundedChat(settings, prompt_version=version, transport=httpx.MockTransport(response))
            try:
                result = chat.generate("Who works on Juniper?", "Same evidence", SOURCES)
                self.assertEqual(result["structured_claims"], PAYLOAD)
            finally:
                chat.close()
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["messages"], requests[1]["messages"])
        self.assertEqual(requests[0]["max_tokens"], requests[1]["max_tokens"])
        self.assertEqual(requests[0]["response_format"]["type"], "json_object")
        self.assertEqual(requests[1]["response_format"]["type"], "json_schema")
        self.assertTrue(requests[1]["response_format"]["json_schema"]["strict"])

    def test_schema_mode_still_rejects_fabricated_quote_without_retry(self):
        value = deepcopy(PAYLOAD)
        value["claims"][0]["evidence"][0]["quote"] = "Lina approved Project Juniper."
        calls = []
        def response(request):
            calls.append(request)
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}],
                                             "usage": {"total_tokens": 60}})
        settings = SimpleNamespace(nebius_base_url="https://api.tokenfactory.nebius.com/v1/",
                                   nebius_api_key="fake", nebius_chat_model="test-model")
        chat = GroundedChat(settings, prompt_version=SCHEMA_CLAIMS_PROMPT_VERSION, transport=httpx.MockTransport(response))
        try:
            with patch("scripts.rag.rag_workflow.atomic_json") as save:
                with self.assertRaises(ServiceError):
                    chat.generate("Who works on Juniper?", "Same evidence", SOURCES)
            self.assertEqual(len(calls), 1)
            self.assertEqual(save.call_args.args[1]["usage"]["total_tokens"], 60)
            self.assertEqual(save.call_args.args[1]["prompt_version"], SCHEMA_CLAIMS_PROMPT_VERSION)
        finally:
            chat.close()

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
