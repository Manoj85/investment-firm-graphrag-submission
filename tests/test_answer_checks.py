"""Check disclosed omissions and real graph excerpts without invented support."""
from copy import deepcopy
import unittest
from scripts.rag.answer_checks import apply_answer_checks

def result(text, claims, status="answered", citations=None):
    return {"answer": {"status": status, "answer": text, "citations": citations or ["S1"]},
            "structured_claims": {"claims": [{"text": item} for item in claims]}}

class AnswerChecksTests(unittest.TestCase):
    def test_missing_lead_and_skill_downgrades_but_preserves_original(self):
        value = result("Cedar supplies records [S1].", ["Cedar supplies records."])
        output = apply_answer_checks(value, {}, [], "Who leads the upstream project and what skill does the lead have?")
        self.assertEqual(output["answer"]["status"], "insufficient_evidence")
        self.assertEqual(value["answer"]["status"], "answered")
        self.assertEqual(output["model_answer"], value["answer"])

    def test_clarification_request_keeps_only_request_sentences(self):
        value = result("Two Robins have different roles. Please clarify which Robin is intended.", [], "insufficient_evidence", [])
        output = apply_answer_checks(value, {}, [], "What does Robin do?")
        self.assertEqual(output["answer"]["status"], "clarification")
        self.assertNotIn("different roles", output["answer"]["answer"])

    def test_graph_bridge_adds_actual_separate_excerpt_not_label_to_prose(self):
        quote = "Decision D-23 requires Lantern for Juniper review."
        sources = [{"id": "bridge", "citation": "S2", "text": quote}]
        edge = {"relation": "REQUIRES_TOOL", "source_label": "D-23", "target_label": "Lantern", "edge_id": "edge",
                "evidence": [{"chunk_id": "bridge", "excerpt": quote}]}
        retrieval = {"graph_selection": {"paths": [{"steps": [edge]}]}}
        value = result("Lina maintains Lantern; Omar approved D-23 [S1].", ["Lina maintains Lantern.", "Omar approved D-23."])
        output = apply_answer_checks(value, retrieval, sources, "Who maintains the checker and who approved the decision?")
        self.assertIn("Source-backed relationship excerpt: " + quote + " [S2]", output["answer"]["answer"])
        self.assertEqual(output["answer"]["citations"], ["S1", "S2"])
        self.assertEqual(output["model_answer"]["citations"], ["S1"])
        for bad in ["An invented relationship excerpt here.", "Lantern is useful."]:
            altered = deepcopy(retrieval); altered["graph_selection"]["paths"][0]["steps"][0]["evidence"][0]["excerpt"] = bad
            self.assertEqual(apply_answer_checks(value, altered, sources, "Who approved D-23?")["answer_checks"]["source_backed_bridges"], [])

    def test_no_graph_fact_added_for_unnamed_endpoint_or_unselected_source(self):
        edge = {"relation": "REQUIRES_TOOL", "source_label": "D-23", "target_label": "Lantern", "edge_id": "edge",
                "evidence": [{"chunk_id": "absent", "excerpt": "D-23 requires Lantern for review."}]}
        retrieval = {"graph_selection": {"paths": [{"steps": [edge]}]}}
        output = apply_answer_checks(result("Omar approved D-23 [S1].", ["Omar approved D-23."]), retrieval, [], "Who approved D-23?")
        self.assertEqual(output["answer_checks"]["source_backed_bridges"], [])

    def test_project_team_does_not_satisfy_organizational_team_warning(self):
        output = apply_answer_checks(result("Lina is on the Project Juniper team [S1].", ["Lina is on the Project Juniper team."]), {}, [], "Which team is Lina in?")
        self.assertEqual(output["answer"]["status"], "insufficient_evidence")
