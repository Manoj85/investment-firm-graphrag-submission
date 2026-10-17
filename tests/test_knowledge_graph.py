"""Reject graph facts whose structure or provenance no longer matches sources."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.graph.knowledge_graph import load_graph, validate_graph, to_networkx, outgoing_relationships
from scripts.config.project_config import PROJECT_ROOT, load_settings
from scripts.corpus.corpus_pipeline import prepare_corpus
from scripts.corpus.qwen_tokenizer import TOKENIZER_PATH


@unittest.skipUnless(TOKENIZER_PATH.is_file(), "Prepare the pinned tokenizer first")
class GraphEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = load_graph()
        # Prepare committed sources with controlled settings: no credentials,
        # ignored active manifest, live database, or personal .env is required.
        settings = load_settings(PROJECT_ROOT / "work" / "absent-graph-test.env", {
            "NEBIUS_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-8B",
            "NEBIUS_BASE_URL": "https://api.tokenfactory.nebius.com/v1/",
            "PINECONE_INDEX": "graph-unit-test", "PINECONE_NAMESPACE": "unit"})
        cls.manifest = prepare_corpus(settings)
        # Unit settings have their own version identity. The runtime inspection
        # separately checks the committed graph against the actual active version.
        cls.graph["corpus_version"] = cls.manifest["corpus_version"]

    def test_authored_graph_matches_current_source_records(self):
        summary = validate_graph(self.graph, self.manifest)
        self.assertEqual(summary["supporting_documents"], [f"DOC-{i:03}" for i in range(1, 11)])
        self.assertGreaterEqual(summary["nodes"], 20)

    def test_source_version_change_requires_explicit_graph_review(self):
        graph = deepcopy(self.graph)
        graph["corpus_version"] = "different-version"
        with self.assertRaisesRegex(ValueError, "corpus_version"):
            validate_graph(graph, self.manifest)

    def test_missing_node_and_wrong_relationship_direction_rejected(self):
        graph = deepcopy(self.graph)
        graph["relationships"][0]["target"] = "project:unknown"
        with self.assertRaisesRegex(ValueError, "endpoint"):
            validate_graph(graph, self.manifest)
        graph = deepcopy(self.graph)
        edge = graph["relationships"][0]
        edge["source"], edge["target"] = edge["target"], edge["source"]
        with self.assertRaisesRegex(ValueError, "endpoint types"):
            validate_graph(graph, self.manifest)

    def test_invented_excerpt_or_mislabelled_document_rejected(self):
        for field, value in [("excerpt", "Priya approved every decision."),
                             ("document_id", "DOC-008"), ("chunk_id", "unknown-chunk")]:
            with self.subTest(field=field):
                graph = deepcopy(self.graph)
                graph["relationships"][0]["evidence"][0][field] = value
                with self.assertRaises(ValueError):
                    validate_graph(graph, self.manifest)

    def test_duplicate_identity_or_missing_evidence_rejected(self):
        graph = deepcopy(self.graph)
        graph["nodes"].append(deepcopy(graph["nodes"][0]))
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_graph(graph, self.manifest)
        graph = deepcopy(self.graph)
        graph["relationships"][0]["evidence"] = []
        with self.assertRaisesRegex(ValueError, "requires source evidence"):
            validate_graph(graph, self.manifest)

    def test_networkx_preserves_edge_direction_ids_and_evidence(self):
        network = to_networkx(self.graph, self.manifest)
        rows = outgoing_relationships(network, "person:priya-shah")
        self.assertEqual({row["edge_id"] for row in rows},
                         {edge["id"] for edge in self.graph["relationships"]
                          if edge["source"] == "person:priya-shah"})
        edge = self.graph["relationships"][0]
        self.assertTrue(network.has_edge(edge["source"], edge["target"], edge["id"]))
        self.assertFalse(network.has_edge(edge["target"], edge["source"], edge["id"]))
        self.assertEqual(rows[0]["evidence"], self.graph["relationships"][0]["evidence"])
        with self.assertRaises(ValueError):
            outgoing_relationships(network, "person:unknown")

    def test_parallel_edges_do_not_overwrite_each_other(self):
        graph = deepcopy(self.graph)
        parallel = deepcopy(graph["relationships"][0])
        parallel["id"] = "rel:second-reviewed-reference"
        graph["relationships"].append(parallel)
        network = to_networkx(graph, self.manifest)
        pair = network.get_edge_data(parallel["source"], parallel["target"])
        self.assertEqual(set(pair), {graph["relationships"][0]["id"], parallel["id"]})

    def test_in_memory_mutation_does_not_change_json_or_query_results(self):
        graph = deepcopy(self.graph)
        network = to_networkx(graph, self.manifest)
        edge = graph["relationships"][0]
        rows = outgoing_relationships(network, edge["source"])
        rows[0]["evidence"][0]["excerpt"] = "Query output changed"
        self.assertEqual(network.edges[edge["source"], edge["target"], edge["id"]]["evidence"], edge["evidence"])
        network.edges[edge["source"], edge["target"], edge["id"]]["evidence"][0]["excerpt"] = "Graph changed"
        self.assertNotEqual(graph["relationships"][0]["evidence"][0]["excerpt"], "Graph changed")

    def test_approval_and_preparation_are_distinct_and_alex_is_not_merged(self):
        network = to_networkx(self.graph, self.manifest)
        approvals = [row for row in outgoing_relationships(network, "decision:dec-001")
                     if row["relation"] == "APPROVED_BY"]
        self.assertEqual([row["target"] for row in approvals], ["person:elena-brooks"])
        preparers = [row for row in outgoing_relationships(network, "decision:dec-001")
                     if row["relation"] == "PREPARED_BY"]
        self.assertEqual([row["target"] for row in preparers], ["person:priya-shah"])
        alex = [node_id for node_id, attrs in network.nodes(data=True)
                if attrs["type"] == "Person" and "Alex" in attrs["aliases"]]
        self.assertEqual(set(alex), {"person:alex-morgan", "person:alex-rivera"})

    def test_every_expanded_edge_has_literal_source_support(self):
        chunks = {record["id"]: record["metadata"] for record in self.manifest["records"]}
        for edge in self.graph["relationships"]:
            for citation in edge["evidence"]:
                self.assertIn(citation["excerpt"], chunks[citation["chunk_id"]]["chunk_text"])
        network = to_networkx(self.graph, self.manifest)
        for tool in ("tool:beacon-evidence-hub", "tool:harbor-access-register", "tool:signal-lineage-board"):
            self.assertTrue(any(row["relation"] == "OWNED_BY" for row in outgoing_relationships(network, tool)))
