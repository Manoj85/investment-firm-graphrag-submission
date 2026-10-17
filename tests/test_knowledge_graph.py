"""Reject graph facts whose structure or provenance no longer matches sources."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from knowledge_graph import load_graph, validate_graph
from project_config import PROJECT_ROOT, load_settings
from corpus_pipeline import prepare_corpus
from qwen_tokenizer import TOKENIZER_PATH


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
        self.assertEqual(summary["supporting_documents"], ["DOC-001", "DOC-002", "DOC-009"])

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
