"""Retry transient Windows replacement locks without losing the prior receipt."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.corpus.corpus_pipeline import atomic_json


class AtomicJsonTests(unittest.TestCase):
    def denial(self, code):
        error = PermissionError("simulated replacement lock")
        error.winerror = code
        return error

    def test_transient_windows_lock_retries_only_replace_then_persists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text('{"old": true}', encoding="utf-8")
            original = os.replace
            calls = []
            def replace(source, target):
                calls.append((source, target))
                if len(calls) < 3:
                    self.assertEqual(json.loads(path.read_text()), {"old": True})
                    raise self.denial(32)
                return original(source, target)
            with patch("scripts.corpus.corpus_pipeline.os.replace", side_effect=replace), \
                 patch("scripts.corpus.corpus_pipeline.time.sleep") as sleep:
                atomic_json(path, {"new": True})
            self.assertEqual(len(calls), 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(json.loads(path.read_text()), {"new": True})

    def test_persistent_lock_raises_and_preserves_prior_and_staged_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text('{"old": true}', encoding="utf-8")
            with patch("scripts.corpus.corpus_pipeline.os.replace", side_effect=self.denial(5)) as replace, \
                 patch("scripts.corpus.corpus_pipeline.time.sleep"):
                with self.assertRaises(PermissionError):
                    atomic_json(path, {"new": True})
            self.assertEqual(replace.call_count, 3)
            self.assertEqual(json.loads(path.read_text()), {"old": True})
            self.assertEqual(json.loads(path.with_suffix('.tmp').read_text()), {"new": True})

    def test_other_permission_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("scripts.corpus.corpus_pipeline.os.replace", side_effect=PermissionError("denied")) as replace, \
                 patch("scripts.corpus.corpus_pipeline.time.sleep") as sleep:
                with self.assertRaises(PermissionError):
                    atomic_json(Path(directory) / "receipt.json", {})
            self.assertEqual(replace.call_count, 1)
            sleep.assert_not_called()
