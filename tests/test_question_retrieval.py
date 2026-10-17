"""Question selection on invented, non-benchmark names and supported roles."""

from pathlib import Path
import sys
import unittest

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.graph.question_retrieval import question_mentions, select_question_paths


class QuestionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.network = nx.MultiDiGraph(graph_version="g1", corpus_version="c1")
        for node_id, label, kind, aliases in [
            ("decision:d1", "D-101", "Decision", []), ("tool:lantern", "Lantern", "Tool", []),
            ("person:lee", "Lee Jones", "Person", ["Lee"]),
            ("person:lee2", "Lee Davis", "Person", ["Lee"]),
            ("team:review", "Review Team", "Team", [])]:
            self.network.add_node(node_id, label=label, type=kind, aliases=aliases)
        for source, target, relation in [
            ("decision:d1", "tool:lantern", "SELECTS_TOOL"),
            ("decision:d1", "person:lee", "APPROVED_BY"),
            ("decision:d1", "person:lee2", "PREPARED_BY"),
            ("tool:lantern", "team:review", "OWNED_BY"),
            ("person:lee", "team:review", "LEADS_TEAM")]:
            self.network.add_edge(source, target, key=relation, relation=relation,
                                  evidence=[{"chunk_id": relation, "excerpt": "Source"}])

    def test_approval_question_finds_unmentioned_person_without_leadership_shortcut(self):
        result = select_question_paths(self.network, "Who approved Lantern?")
        self.assertEqual(result["seed_entities"], ["tool:lantern"])
        self.assertEqual(len(result["paths"]), 1)
        path = result["paths"][0]
        self.assertEqual(path["nodes"][-1], "person:lee")
        self.assertEqual([step["relation"] for step in path["steps"]], ["SELECTS_TOOL", "APPROVED_BY"])
        self.assertNotIn("LEADS_TEAM", result["evidence_chunk_ids"])
        self.assertNotIn("PREPARED_BY", result["evidence_chunk_ids"])

    def test_longest_name_avoids_false_first_name_ambiguity(self):
        self.assertEqual(question_mentions(self.network, "Who is Lee Jones?")[0]["status"], "resolved")
        result = select_question_paths(self.network, "What does Lee approve?")
        self.assertEqual(result["status"], "clarification_required")
        self.assertEqual(result["examined_edges"], 0)

    def test_no_guessing_for_unknown_entities_unsupported_intent_or_negation(self):
        for question, reason in [("Who approved Unknown?", "no_named_entities"),
                                 ("How much does Lantern cost?", "no_relationship_intent"),
                                 ("Who did not approve D-101?", "unsupported_negation")]:
            result = select_question_paths(self.network, question)
            self.assertEqual(result["stop_reason"], reason)
            self.assertEqual(result["paths"], [])

    def test_limits_determinism_and_evidence_copy(self):
        first = select_question_paths(self.network, "Who approved D-101?")
        self.assertEqual(first, select_question_paths(self.network, "Who approved D-101?"))
        first["paths"][0]["steps"][0]["evidence"][0]["excerpt"] = "Changed"
        self.assertEqual(self.network.edges["decision:d1", "person:lee", "APPROVED_BY"]["evidence"][0]["excerpt"], "Source")
        for i in range(50):
            self.network.add_node(f"person:{i}", label=f"Person {i}", type="Person", aliases=[])
            self.network.add_edge("decision:d1", f"person:{i}", key=f"x{i}", relation="PREPARED_BY", evidence=[])
        result = select_question_paths(self.network, "Who approved D-101?")
        self.assertLessEqual(result["examined_edges"], 40)
        self.assertTrue(result["truncated"])
        self.assertTrue(all(len(path["steps"]) <= 3 for path in result["paths"]))

    def test_upstream_lead_skill_uses_ordered_roles_despite_distractors(self):
        for node, kind in [("project:juniper", "Project"), ("project:cedar", "Project"),
                           ("Actual Lead", "Person"), ("Sponsor", "Person"),
                           ("Downstream Lead", "Person"), ("Access Review", "Skill")]:
            self.network.add_node(node, label={"project:juniper": "Juniper", "project:cedar": "Cedar"}.get(node, node), type=kind, aliases=[])
        edges = [("project:juniper", "project:cedar", "DEPENDS_ON"),
                 ("Actual Lead", "project:cedar", "LEADS_PROJECT"),
                 ("Sponsor", "project:cedar", "SPONSORS_PROJECT"),
                 ("Downstream Lead", "project:juniper", "LEADS_PROJECT"),
                 ("Actual Lead", "Access Review", "HAS_SKILL")]
        for source, target, relation in edges:
            self.network.add_edge(source, target, key=source + relation, relation=relation,
                                  evidence=[{"chunk_id": source + relation}])
        for index in range(60):
            self.network.add_edge("project:juniper", "Sponsor", key=str(index),
                                  relation="ABOUT_PROJECT", evidence=[])
        result = select_question_paths(self.network, "Which upstream project does Juniper depend on, who leads it, and what skill does that lead have?")
        self.assertFalse(result["truncated"])
        self.assertEqual(result["examined_edges"], 3)
        self.assertEqual(result["paths"][-1]["nodes"],
                         ["project:juniper", "project:cedar", "Actual Lead", "Access Review"])
        self.assertNotIn("Sponsor", result["paths"][-1]["nodes"])

    def test_upstream_pattern_does_not_invent_missing_lead_or_skill(self):
        self.network.add_node("project:a", label="Juniper", type="Project", aliases=[])
        self.network.add_node("project:b", label="Cedar", type="Project", aliases=[])
        self.network.add_edge("project:a", "project:b", key="dependency", relation="DEPENDS_ON", evidence=[])
        result = select_question_paths(self.network, "Which upstream project does Juniper depend on, who leads it and what skill do they have?")
        self.assertEqual(len(result["paths"]), 1)
        self.assertEqual(result["paths"][0]["nodes"], ["project:a", "project:b"])
