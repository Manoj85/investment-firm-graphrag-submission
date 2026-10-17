"""Identity ambiguity and graph-selected remote evidence integrity."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.graph.graph_evidence import fetch_path_evidence, resolve_entity
from scripts.services.service_clients import ServiceError


class GraphEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.network = nx.MultiDiGraph()
        for key, label in (("person:morgan", "Alex Morgan"), ("person:rivera", "Alex Rivera")):
            self.network.add_node(key, label=label, type="Person", aliases=["Alex"])
        self.meta = {"document_id": "DOC-001", "section": "Approval", "source": "example.md",
                     "version": "1", "source_sha256": "hash", "chunk_id": "chunk",
                     "chunk_text": "Elena approved the decision.", "corpus_version": "v1"}
        self.citation = {k: v for k, v in self.meta.items() if k not in ("chunk_text", "corpus_version")}
        self.citation["excerpt"] = self.meta["chunk_text"]
        self.manifest = {"corpus_version": "v1", "namespace": "unit-v1",
                         "records": [{"id": "chunk", "metadata": self.meta}]}
        self.report = {"corpus_version": "v1", "evidence_chunk_ids": ["chunk"],
                       "paths": [{"steps": [{"edge_id": "edge1", "evidence": [self.citation]},
                                             {"edge_id": "edge2", "evidence": [self.citation]}]}]}
        self.store = Mock()
        self.store.fetch.return_value = {"chunk": {"id": "chunk", "metadata": deepcopy(self.meta)}}

    def test_exact_id_normalized_label_alias_and_ambiguity(self):
        self.assertEqual(resolve_entity(self.network, "person:morgan")["status"], "resolved")
        self.assertEqual(resolve_entity(self.network, "  aLex   Morgan ")["candidates"][0]["id"], "person:morgan")
        result = resolve_entity(self.network, "Alex")
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(resolve_entity(self.network, "Who is Alex?")["status"], "not_found")
        with self.assertRaises(ValueError):
            resolve_entity(self.network, " ")

    def test_fetch_uses_namespace_deduplicates_and_returns_remote_text_without_vectors(self):
        self.store.fetch.return_value["chunk"]["values"] = [1, 2]
        result = fetch_path_evidence(self.report, self.manifest, self.store)
        self.store.fetch.assert_called_once_with("unit-v1", ["chunk"])
        row = result["evidence"][0]
        self.assertEqual(row["text"], self.meta["chunk_text"])
        self.assertEqual(row["supporting_edge_ids"], ["edge1", "edge2"])
        self.assertIsNone(row["score"])
        self.assertNotIn("values", row)

    def test_stale_or_forged_path_rejected_before_remote_call(self):
        for field, value in (("corpus_version", "old"), ("evidence_chunk_ids", ["other"])):
            report = deepcopy(self.report)
            report[field] = value
            with self.assertRaises(ValueError):
                fetch_path_evidence(report, self.manifest, self.store)
        report = deepcopy(self.report)
        report["paths"][0]["steps"][0]["evidence"][0]["excerpt"] = "Invented approval"
        with self.assertRaises(ValueError):
            fetch_path_evidence(report, self.manifest, self.store)
        self.store.fetch.assert_not_called()

    def test_missing_extra_substituted_identity_or_text_rejected(self):
        bad_text = deepcopy(self.meta)
        bad_text["chunk_text"] = "Wrong text"
        for response in ({}, {"chunk": {"id": "other", "metadata": self.meta}},
                         {"chunk": {"id": "chunk", "metadata": bad_text}},
                         {"chunk": {"id": "chunk", "metadata": self.meta}, "extra": {}}):
            self.store.fetch.return_value = response
            with self.assertRaises(ServiceError):
                fetch_path_evidence(self.report, self.manifest, self.store)

    def test_empty_paths_make_no_fetch(self):
        report = {"corpus_version": "v1", "paths": [], "evidence_chunk_ids": []}
        self.assertEqual(fetch_path_evidence(report, self.manifest, self.store)["status"], "no_path_evidence")
        self.store.fetch.assert_not_called()
