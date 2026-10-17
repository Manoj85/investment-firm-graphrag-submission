"""Bounded review contracts, provenance boundary and visible failure retention."""

from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

import httpx

from scripts.rag.claim_verifier import ClaimVerifier, prepare_review, validate_review, review_schema
from scripts.services.service_clients import ServiceError


SOURCE = {"citation": "S1", "text": "Lina leads Project Juniper."}
PAYLOAD = {"status": "answered", "message": "", "claims": [
    {"text": "Lina leads Juniper.", "evidence": [{"source": "S1", "quote": SOURCE["text"]}]}]}
REVIEW = {"claims": [{"claim_id": 1, "verdict": "supported", "reason": "Source explicitly identifies Lina as lead."}],
          "coverage": "complete", "missing": [], "message_verdict": "acceptable", "message_reason": "Empty message is appropriate."}


class ClaimVerifierTests(unittest.TestCase):
    def test_schema_mode_sends_closed_shape_and_count_including_zero_claims(self):
        calls = []
        def respond(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(REVIEW)}}]})
        verifier = ClaimVerifier(self.settings(), schema=True, transport=httpx.MockTransport(respond))
        try:
            record = verifier.review("Who leads Juniper?", PAYLOAD, [SOURCE])
            schema = calls[0]["response_format"]["json_schema"]["schema"]
            self.assertFalse(schema["additionalProperties"])
            self.assertFalse(schema["properties"]["claims"]["items"]["additionalProperties"])
            self.assertEqual(schema["properties"]["claims"]["maxItems"], 1)
            self.assertEqual(review_schema(0)["properties"]["claims"]["maxItems"], 0)
            self.assertEqual(record["verifier_version"], "claim-review-v2-schema")
        finally:
            verifier.close()

    def settings(self):
        return SimpleNamespace(nebius_base_url="https://api.tokenfactory.nebius.com/v1/",
                               nebius_api_key="fake", nebius_chat_model="test-model")

    def test_code_derives_decision_from_all_claims_coverage_and_message(self):
        self.assertEqual(validate_review(json.dumps(REVIEW), 1)["decision"], "judged_pass")
        values = []
        item = deepcopy(REVIEW); item["claims"][0]["verdict"] = "unsupported"; values.append(item)
        item = deepcopy(REVIEW); item["coverage"] = "incomplete"; item["missing"] = ["Required participation link"]; values.append(item)
        item = deepcopy(REVIEW); item["coverage"] = "uncertain"; values.append(item)
        item = deepcopy(REVIEW); item["message_verdict"] = "needs_review"; values.append(item)
        for item in values:
            self.assertEqual(validate_review(json.dumps(item), 1)["decision"], "needs_review")

    def test_missing_duplicate_invented_and_boolean_claim_ids_reject(self):
        for ids in [[], [1, 1], [1, 3], [True, 2]]:
            item = deepcopy(REVIEW)
            item["claims"] = [{**REVIEW["claims"][0], "claim_id": cid} for cid in ids]
            with self.assertRaises(ServiceError):
                validate_review(json.dumps(item), 2)

    def test_inconsistent_coverage_extra_fields_and_bad_json_reject(self):
        for coverage, missing in [("complete", ["A missing skill"]), ("incomplete", [])]:
            item = deepcopy(REVIEW); item.update(coverage=coverage, missing=missing)
            with self.assertRaises(ServiceError):
                validate_review(json.dumps(item), 1)
        item = deepcopy(REVIEW); item["pass"] = True
        with self.assertRaises(ServiceError):
            validate_review(json.dumps(item), 1)
        with self.assertRaises(ServiceError):
            validate_review("not JSON", 1)

    def test_input_provenance_and_bounds_are_checked_before_api(self):
        calls = []
        verifier = ClaimVerifier(self.settings(), transport=httpx.MockTransport(lambda request: calls.append(request)))
        value = deepcopy(PAYLOAD); value["claims"][0]["evidence"][0]["quote"] = "An invented quotation here."
        try:
            with self.assertRaises(ServiceError):
                verifier.review("Who leads Juniper?", value, [SOURCE])
            with self.assertRaises(ValueError):
                prepare_review("Who leads Juniper?", PAYLOAD, [SOURCE, SOURCE])
            with self.assertRaises(ValueError):
                prepare_review("Who leads Juniper?", PAYLOAD, [{**SOURCE, "text": "x" * 30001}])
            self.assertEqual(calls, [])
        finally:
            verifier.close()

    def test_one_review_call_gets_actual_claims_sources_and_no_answer_repair(self):
        calls = []
        def respond(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json={"model": "test-model", "usage": {"total_tokens": 80},
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(REVIEW)}}]})
        verifier = ClaimVerifier(self.settings(), transport=httpx.MockTransport(respond))
        before = deepcopy(PAYLOAD)
        try:
            record = verifier.review("Who leads Juniper?", PAYLOAD, [SOURCE])
            self.assertEqual(record["review"]["decision"], "judged_pass")
            self.assertEqual(PAYLOAD, before)
            self.assertEqual(len(calls), 1)
            body = json.loads(calls[0]["messages"][1]["content"])
            self.assertEqual(set(body), {"question", "claims", "sources", "status", "message"})
            self.assertEqual(body["sources"][0]["text"], SOURCE["text"])
            self.assertNotIn("answer", record)
        finally:
            verifier.close()

    def test_bad_model_review_is_recorded_as_error_with_usage_no_retry(self):
        for finish, content in [("length", "partial JSON"), ("stop", "not JSON")]:
            calls = []
            def respond(request):
                calls.append(request)
                return httpx.Response(200, json={"usage": {"total_tokens": 90}, "choices": [
                    {"finish_reason": finish, "message": {"content": content}}]})
            verifier = ClaimVerifier(self.settings(), transport=httpx.MockTransport(respond))
            try:
                record = verifier.review("Who leads Juniper?", PAYLOAD, [SOURCE])
                self.assertEqual(record["outcome"], "error")
                self.assertEqual(record["usage"]["total_tokens"], 90)
                self.assertEqual(record["content"], content)
                self.assertEqual(len(calls), 1)
                self.assertNotIn("review", record)
            finally:
                verifier.close()
