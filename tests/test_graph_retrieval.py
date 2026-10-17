"""Graph/vector merge boundaries, fallback and no-model clarification."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.graph.graph_retrieval import merge_evidence, retrieve_with_graph
from scripts.services.service_clients import ServiceError


class GraphRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.source = {"id": "one", "text": "A source fact.", "score": None,
                       "metadata": {"document_id": "TEST", "section": "Fact"}}

    def test_deduplication_preserves_origins_real_score_and_graph_priority(self):
        vector = {**deepcopy(self.source), "score": 0.7}
        result = merge_evidence([self.source], [vector, {**vector, "id": "two"}])
        self.assertEqual([row["id"] for row in result], ["one", "two"])
        self.assertEqual(result[0]["retrieval_origins"], ["graph_path", "vector"])
        self.assertEqual(result[0]["vector_score"], 0.7)
        self.assertIsNone(result[0]["score"])
        result[0]["metadata"]["section"] = "Changed"
        self.assertEqual(self.source["metadata"]["section"], "Fact")
        with self.assertRaises(ServiceError):
            merge_evidence([self.source], [{**vector, "text": "Substituted"}])

    @patch("scripts.graph.graph_retrieval.select_question_paths")
    @patch("scripts.graph.graph_retrieval.to_networkx")
    @patch("scripts.graph.graph_retrieval.load_graph")
    @patch("scripts.graph.graph_retrieval.active_manifest")
    @patch("scripts.graph.graph_retrieval.retrieve")
    def test_ambiguity_returns_names_without_vector_or_remote_calls(self, retrieve, manifest, load, convert, select):
        manifest.return_value = {"namespace": "unit", "corpus_version": "v1"}
        select.return_value = {"status": "clarification_required", "mentions": [
            {"status": "ambiguous", "candidates": [{"label": "Sam One"}, {"label": "Sam Two"}]}]}
        store, embeddings = Mock(), Mock()
        result = retrieve_with_graph(None, store, embeddings, "What does Sam do?")
        self.assertEqual(result["clarification"]["status"], "clarification")
        self.assertIn("Sam Two", result["clarification"]["answer"])
        retrieve.assert_not_called()
        self.assertEqual(store.mock_calls, [])
        self.assertEqual(embeddings.mock_calls, [])

    @patch("scripts.graph.graph_retrieval.fetch_path_evidence")
    @patch("scripts.graph.graph_retrieval.select_question_paths")
    @patch("scripts.graph.graph_retrieval.to_networkx")
    @patch("scripts.graph.graph_retrieval.load_graph")
    @patch("scripts.graph.graph_retrieval.active_manifest")
    @patch("scripts.graph.graph_retrieval.retrieve")
    def test_unsupported_selector_preserves_vector_fallback_and_checks_version(self, retrieve, manifest, load, convert, select, fetch):
        manifest.return_value = {"namespace": "unit", "corpus_version": "v1"}
        select.return_value = {"status": "unsupported", "stop_reason": "no_named_entities"}
        retrieve.return_value = {"namespace": "unit", "corpus_version": "v1", "evidence": [self.source]}
        fetch.return_value = {"status": "no_path_evidence", "evidence": []}
        result = retrieve_with_graph(None, Mock(), Mock(), "A descriptive question?")
        self.assertEqual(result["evidence"][0]["id"], "one")
        self.assertEqual(result["graph_retrieval"]["fallback_reason"], "no_named_entities")
        retrieve.return_value["corpus_version"] = "stale"
        with self.assertRaises(ServiceError):
            retrieve_with_graph(None, Mock(), Mock(), "A descriptive question?")
