"""Bound expansion, source isolation, deduplication, and remote evidence checks."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.retrieval.neighbor_retrieval import expand_neighbors
from scripts.services.service_clients import ServiceError


class NeighborTests(unittest.TestCase):
    def setUp(self):
        records = []
        for doc in ("A", "B"):
            for index in range(5):
                records.append({"id": f"{doc}{index}", "metadata": {
                    "source": f"{doc}.md", "document_id": doc, "version": "1",
                    "char_start": str(index * 10), "section": f"Part {index}",
                    "chunk_text": f"{doc} passage {index}"}})
        self.manifest = {"records": records}
        self.stored = {r["id"]: r for r in records}
        self.store = Mock()
        self.store.fetch.side_effect = lambda namespace, ids: {
            key: deepcopy(self.stored[key]) for key in ids}

    def report(self, ids):
        return {"namespace": "test", "evidence": [{"id": key, "score": 0.7,
            "text": self.stored[key]["metadata"]["chunk_text"],
            "metadata": {k: v for k, v in self.stored[key]["metadata"].items() if k != "chunk_text"}}
            for key in ids]}

    def test_no_cross_document_neighbors_or_duplicate_seed_records(self):
        result = expand_neighbors(self.report(["A4", "A3"]), self.manifest, self.store)
        ids = [item["id"] for item in result["evidence"]]
        self.assertEqual(ids, ["A4", "A3", "A2"])
        self.assertTrue(all(item["metadata"]["document_id"] == "A" for item in result["evidence"]))
        self.assertEqual(len(ids), len(set(ids)))
        neighbor = result["evidence"][-1]
        self.assertEqual(neighbor["neighbor_of"], "A3")
        self.assertIsNone(neighbor["score"])

    def test_added_candidates_are_capped_without_losing_seeds(self):
        result = expand_neighbors(self.report(["A1", "B1", "A3", "B3"]), self.manifest, self.store)
        self.assertEqual(result["neighbor_expansion"]["added_passages"], 4)
        self.assertEqual(len(result["evidence"]), 8)
        self.assertEqual(len([item for item in result["evidence"] if item["retrieval_origin"] == "vector"]), 4)

    def test_missing_or_substituted_remote_neighbor_is_rejected(self):
        self.store.fetch.side_effect = None
        self.store.fetch.return_value = {}
        with self.assertRaises(ServiceError):
            expand_neighbors(self.report(["A1"]), self.manifest, self.store)
        corrupt = deepcopy(self.stored["A0"])
        corrupt["metadata"]["chunk_text"] = "Wrong evidence"
        self.store.fetch.return_value = {"A0": corrupt}
        with self.assertRaises(ServiceError):
            expand_neighbors(self.report(["A1"]), self.manifest, self.store, max_neighbors=1)

    def test_empty_search_makes_no_neighbor_request(self):
        result = expand_neighbors(self.report([]), self.manifest, self.store)
        self.assertEqual(result["evidence"], [])
        self.store.fetch.assert_not_called()
