import json
from types import SimpleNamespace
import unittest

import httpx

from scripts.config.project_config import PROJECT_ROOT
from scripts.rag.coverage_excerpts import (COVERAGE_VERSION, question_parts,
    render_coverage, bare_name_clarification)
from scripts.rag.excerpt_selection import excerpt_catalog
from scripts.rag.rag_workflow import GroundedChat
from scripts.services.service_clients import ServiceError


def saved_sources(question, mode="graph"):
    return json.loads((PROJECT_ROOT / f"outputs/evaluation/runs/revised-20261018T151922774598Z/{question}-{mode}.json").read_text())["report"]


class CoverageExcerptsTests(unittest.TestCase):
    def test_missing_part_downgrades_even_when_model_says_answered(self):
        sources = [{"citation": "S1", "text": "Lina has Delivery Planning expertise."}]
        answer, claims, raw, audit = render_coverage(json.dumps({"status": "answered",
            "coverage": {"skill": ["E1"], "team": []}}), sources, excerpt_catalog(sources), ["skill", "team"])
        self.assertEqual(answer["status"], "insufficient_evidence")
        self.assertEqual(raw["status"], "answered")
        self.assertEqual(audit["unfilled_parts"], ["team"])
        self.assertTrue(audit["status_downgraded"])
        self.assertNotIn("Lina belongs", answer["answer"])

    def test_historical_participation_and_tool_omissions_have_explicit_slots(self):
        for q, expected in [("Q05", {"participation", "skill", "team"}),
                            ("Q08", {"project", "tool"})]:
            record = saved_sources(q)
            self.assertEqual(set(question_parts(record["question"])), expected)

    def test_slots_do_not_encode_known_entities_or_expected_answers(self):
        self.assertEqual(question_parts("Which Vega participant has Scheduling expertise, and which team are they in?"),
                         ["participation", "skill", "team"])
        self.assertEqual(question_parts("Who approved a tool selection, and on what date?"),
                         ["approval", "date", "tool_relationship"])

    def test_missing_extra_unknown_duplicate_and_free_text_fields_reject(self):
        sources = [{"citation": "S1", "text": "Lina has Delivery Planning expertise."}]
        for value in [{"status": "answered", "coverage": {}},
                      {"status": "answered", "coverage": {"skill": [], "invented": []}},
                      {"status": "answered", "coverage": {"skill": ["E9"]}},
                      {"status": "answered", "coverage": {"skill": ["E1", "E1"]}},
                      {"status": "answered", "coverage": {"skill": ["E1"]}, "text": "Lina approved."}]:
            with self.subTest(value=value), self.assertRaises(ServiceError):
                render_coverage(json.dumps(value), sources, excerpt_catalog(sources), ["skill"])

    def test_reused_evidence_renders_once_and_keeps_source_wording(self):
        sources = [{"citation": "S1", "text": "Lina has Delivery Planning expertise and belongs to Operations."}]
        answer, claims, _, audit = render_coverage(json.dumps({"status": "answered",
            "coverage": {"skill": ["E1"], "team": ["E1"]}}), sources, excerpt_catalog(sources), ["skill", "team"])
        self.assertEqual(len(claims["claims"]), 1)
        self.assertEqual(answer["citations"], ["S1"])
        self.assertEqual(claims["claims"][0]["text"], sources[0]["text"])
        self.assertIn("not semantic verification", audit["limitation"])

    def test_bare_first_name_uses_retrieved_names_without_guessing(self):
        report = saved_sources("Q09", "vector")
        result = bare_name_clarification(report["question"], report["sources"])
        self.assertEqual(result["answer"]["status"], "clarification")
        self.assertIn("Alex Morgan or Alex Rivera", result["answer"]["answer"])
        self.assertEqual(result["answer"]["citations"], [])
        for question in ["What is Alex Morgan responsible for?", "Which Alex works in Data Platform?",
                         "What is Alex responsible for in Risk Analytics?"]:
            self.assertIsNone(bare_name_clarification(question, report["sources"]))
        self.assertIsNone(bare_name_clarification("What does Alex do?", report["sources"][:1]))

    def test_other_names_work_and_person_name_must_occur_in_source(self):
        sources = [{"citation": f"S{i}", "metadata": {"section": name}, "text": f"{name} is an analyst."}
                   for i, name in enumerate(["Robin Vale", "Robin Lake"], 1)]
        self.assertIn("Robin Lake or Robin Vale", bare_name_clarification("What does Robin do?", sources)["answer"]["answer"])
        sources[1]["text"] = "Unrelated operational record."
        self.assertIsNone(bare_name_clarification("What does Robin do?", sources))

    def test_transport_uses_question_specific_schema_and_one_call(self):
        calls = []
        record = saved_sources("Q08")
        catalog = excerpt_catalog(record["sources"])
        project = next(e["id"] for e in catalog if e["source"] == "S1")
        tool = next(e["id"] for e in catalog if e["source"] == "S2")
        def respond(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                "status": "answered", "coverage": {"project": [project], "tool": [tool]}})}}]})
        settings = SimpleNamespace(nebius_base_url="https://api.tokenfactory.nebius.com/v1/", nebius_api_key="fake", nebius_chat_model="fake")
        chat = GroundedChat(settings, prompt_version=COVERAGE_VERSION, transport=httpx.MockTransport(respond))
        try:
            result = chat.generate(record["question"], "ignored old context", record["sources"])
            schema = calls[0]["response_format"]["json_schema"]["schema"]
            self.assertEqual(schema["properties"]["coverage"]["required"], ["project", "tool"])
            self.assertEqual(result["coverage_selection"]["unfilled_parts"], [])
            self.assertIn("Signal Lineage Board", result["answer"]["answer"])
            ambiguous = saved_sources("Q09", "vector")
            result = chat.generate(ambiguous["question"], "", ambiguous["sources"])
            self.assertEqual(result["answer"]["status"], "clarification")
            self.assertIsNone(result["usage"])
            self.assertEqual(len(calls), 1)
        finally:
            chat.close()
