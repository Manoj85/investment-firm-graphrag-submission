from copy import deepcopy
import json
import unittest

import networkx as nx

from scripts.config.project_config import PROJECT_ROOT
from scripts.graph.graph_retrieval import descriptive_project_selection
from scripts.graph.question_retrieval import select_question_paths
from scripts.rag.answer_checks import apply_answer_checks


class ReviewedBridgeTests(unittest.TestCase):
    def test_historical_participation_gap_uses_question_anchor_and_exact_source(self):
        root = PROJECT_ROOT / "outputs/evaluation/runs/revised-20261018T151922774598Z"
        r = json.loads((root / "Q05-graph.json").read_text())["report"]
        value = {"answer": r["selected_excerpt_answer"], "structured_claims": r["structured_claims"]}
        selection = {"graph_selection": r["graph_selection"]}
        old = apply_answer_checks(value, selection, r["sources"], r["question"])
        self.assertEqual(old["answer_checks"]["source_backed_bridges"], [])
        fixed = apply_answer_checks(value, selection, r["sources"], r["question"], extend_reviewed_bridges=True)
        self.assertIn("Maya Chen and Priya Shah work on Project Atlas.", fixed["answer"]["answer"])
        for b in fixed["answer_checks"]["source_backed_bridges"]:
            self.assertIn(b["quote"], next(s["text"] for s in r["sources"] if s["id"] == b["chunk_id"]))

    def test_requested_tool_is_supplied_only_by_a_fetched_reviewed_edge(self):
        quote = "Project Vega uses Lantern Board to record lineage links."
        value = {"answer": {"status": "answered", "answer": "Project Vega traces reporting inputs [S1].", "citations": ["S1"]},
                 "structured_claims": {"claims": [{"text": "Project Vega traces reporting inputs."}]}}
        step = {"relation": "USES_TOOL", "source_label": "Project Vega", "target_label": "Lantern Board",
                "edge_id": "vega-tool", "evidence": [{"chunk_id": "tool", "excerpt": quote}]}
        selection = {"graph_selection": {"paths": [{"steps": [step]}]}}
        sources = [{"id": "tool", "citation": "S2", "text": quote}]
        question = "Which project traces inputs and which tool records those links?"
        fixed = apply_answer_checks(value, selection, sources, question, extend_reviewed_bridges=True)
        self.assertIn(quote + " [S2]", fixed["answer"]["answer"])
        for evidence, query in [([], question), (sources, "Which project traces inputs?")]:
            self.assertEqual(apply_answer_checks(value, selection, evidence, query,
                extend_reviewed_bridges=True)["answer_checks"]["source_backed_bridges"], [])
        altered = deepcopy(selection)
        altered["graph_selection"]["paths"][0]["steps"][0]["evidence"][0]["excerpt"] = "An invented tool relationship."
        self.assertEqual(apply_answer_checks(value, altered, sources, question,
            extend_reviewed_bridges=True)["answer_checks"]["source_backed_bridges"], [])

    def test_project_team_wording_does_not_establish_named_organizational_team(self):
        text = "Lina works on Project Vega. The team uses Lantern."
        value = {"answer": {"status": "answered", "answer": text + " [S1]", "citations": ["S1"]},
                 "structured_claims": {"claims": [{"text": text}]}}
        fixed = apply_answer_checks(value, {}, [], "Which team is Lina in?", extend_reviewed_bridges=True)
        self.assertEqual(fixed["answer"]["status"], "insufficient_evidence")

    def test_descriptive_seed_is_bounded_to_one_explicit_project_in_top_passage(self):
        graph = nx.MultiDiGraph(graph_version="g", corpus_version="c")
        for key, label, kind in [("v", "Project Vega", "Project"), ("w", "Project Willow", "Project"),
                                  ("t", "Lantern Board", "Tool")]:
            graph.add_node(key, label=label, type=kind, aliases=[])
        graph.add_edge("v", "t", key="uses", relation="USES_TOOL", evidence=[{"chunk_id": "tool", "excerpt": "Vega uses Lantern."}])
        question = "Which project traces reporting inputs and which tool records the links?"
        original = select_question_paths(graph, question)
        vector = {"evidence": [{"id": "purpose", "text": "Project Vega traces reporting inputs."}]}
        fixed = descriptive_project_selection(graph, question, original, vector)
        self.assertEqual(fixed["descriptive_seed"]["project_id"], "v")
        self.assertIn("tool", fixed["evidence_chunk_ids"])
        self.assertEqual(fixed["question"], question)
        for text in ["Project Vega and Project Willow exchange inputs.", "An unrelated paragraph."]:
            self.assertEqual(descriptive_project_selection(graph, question, original,
                {"evidence": [{"id": "ambiguous", "text": text}]}), original)
        self.assertEqual(descriptive_project_selection(graph, question, original, {"evidence": []}), original)
        self.assertEqual(descriptive_project_selection(graph, question + " except Vega", original, vector), original)
