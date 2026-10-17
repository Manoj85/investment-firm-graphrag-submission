"""Traversal semantics and resource limits, using a tiny offline graph."""

import unittest
from pathlib import Path
import sys

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.graph.graph_paths import bounded_paths


class BoundedPathTests(unittest.TestCase):
    def setUp(self):
        self.network = nx.MultiDiGraph()
        for entity in ("tool", "decision", "approver", "isolated"):
            self.network.add_node(entity, label=entity)
        self.network.add_edge("decision", "tool", key="selects", relation="SELECTS_TOOL",
                              evidence=[{"chunk_id": "selection", "excerpt": "selects tool"}])
        self.network.add_edge("decision", "approver", key="approves", relation="APPROVED_BY",
                              evidence=[{"chunk_id": "approval", "excerpt": "approved decision"}])

    def test_reverse_navigation_preserves_fact_and_evidence(self):
        result = bounded_paths(self.network, "tool", "approver", max_hops=2)
        steps = result["paths"][0]["steps"]
        self.assertEqual([s["traversal"] for s in steps], ["reverse", "forward"])
        self.assertEqual((steps[0]["source"], steps[0]["target"]), ("decision", "tool"))
        self.assertEqual(result["evidence_chunk_ids"], ["selection", "approval"])
        steps[0]["evidence"][0]["excerpt"] = "changed"
        self.assertEqual(self.network.edges["decision", "tool", "selects"]["evidence"][0]["excerpt"], "selects tool")

    def test_hop_edge_and_path_limits_report_incomplete_search(self):
        self.assertEqual(bounded_paths(self.network, "tool", "approver", max_hops=1)["paths"], [])
        limited = bounded_paths(self.network, "tool", "approver", max_examined_edges=1)
        self.assertEqual(limited["examined_edges"], 1)
        self.assertEqual(limited["stop_reason"], "edge_budget")
        self.assertTrue(limited["truncated"])
        self.assertEqual(bounded_paths(self.network, "tool", "approver", max_paths=1)["stop_reason"], "path_limit")

    def test_parallel_facts_cycles_and_deterministic_order(self):
        self.network.add_edge("decision", "tool", key="other", relation="REQUIRES_TOOL",
                              evidence=[{"chunk_id": "selection"}])
        first = bounded_paths(self.network, "tool", "approver")
        self.assertEqual(len(first["paths"]), 2)
        self.assertEqual(first, bounded_paths(self.network, "tool", "approver"))
        for path in first["paths"]:
            self.assertEqual(len(path["nodes"]), len(set(path["nodes"])))
        self.assertEqual(len(first["evidence_chunk_ids"]), 2)

    def test_missing_disconnected_and_invalid_inputs(self):
        self.assertEqual(bounded_paths(self.network, "tool", "isolated")["paths"], [])
        for start, target in (("missing", "tool"), ("tool", "tool")):
            with self.assertRaises(ValueError):
                bounded_paths(self.network, start, target)
        for kwargs in ({"max_hops": 4}, {"max_examined_edges": 0}, {"max_paths": True}):
            with self.assertRaises(ValueError):
                bounded_paths(self.network, "tool", "approver", **kwargs)
