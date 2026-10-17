"""Check that diagnostic conclusions distinguish coverage, corruption and scores."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.retrieval.diagnose_retrieval import analyze
from scripts.services.service_clients import DIMENSION, ServiceError


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.vector = [1.0] + [0.0] * (DIMENSION - 1)
        self.meta = {"document_id": "DOC-TEST", "section": "Approval", "chunk_text": "A approved B."}
        self.manifest = {"records": [{"id": "target", "metadata": self.meta}]}
        self.fetched = {"target": {"metadata": self.meta, "values": self.vector}}
        self.matches = [{"id": "target", "metadata": self.meta, "score": 0.9999}]

    def analyze(self, fetched=None, matches=None):
        return analyze(self.manifest, self.fetched if fetched is None else fetched,
                       self.matches if matches is None else matches,
                       self.vector, {"target": self.vector}, "target")

    def test_small_score_difference_is_reported_not_hidden_or_storage_failure(self):
        result = self.analyze()
        self.assertTrue(all(result["checks"].values()))
        self.assertFalse(result["score_comparison"]["all_within_1e_5"])
        self.assertAlmostEqual(result["score_comparison"]["max_absolute_difference"], 0.0001)
        self.assertEqual(result["target"]["rank"], 1)

    def test_missing_fetch_or_duplicate_query_prevents_exhaustive_claim(self):
        with self.assertRaises(ServiceError):
            self.analyze(fetched={})
        with self.assertRaises(ServiceError):
            self.analyze(matches=self.matches * 2)

    def test_corrupted_stored_vector_and_metadata_are_flagged(self):
        fetched = deepcopy(self.fetched)
        fetched["target"]["values"][1] = 0.1
        fetched["target"]["metadata"]["chunk_text"] = "Substituted text."
        result = self.analyze(fetched=fetched)
        self.assertFalse(result["checks"]["all_metadata_matches"])
        self.assertFalse(result["checks"]["all_stored_vectors_match_cache_within_1e_6"])


if __name__ == "__main__":
    unittest.main()
