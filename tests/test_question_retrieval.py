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
