"""Offline checks with the real pinned tokenizer; prepare its assets first."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from inspect_atlas import ATLAS_PATH, SourceDocument, load_markdown
from qwen_tokenizer import TOKENIZER_PATH, count_tokens, load_tokenizer
from token_chunks import split_token_chunks, validate_chunks


@unittest.skipUnless(TOKENIZER_PATH.is_file(), "Run scripts/qwen_tokenizer.py to enable tokenizer checks")
class TokenChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = load_tokenizer()

    def test_atlas_coverage_limits_overlap_and_metadata_independence(self):
        document = load_markdown(ATLAS_PATH)
        chunks = split_token_chunks(document, self.tokenizer, 32, 8)
        validate_chunks(document, chunks, self.tokenizer, 32)
        self.assertTrue(any(int(chunk.metadata["actual_overlap_tokens"]) > 0 for chunk in chunks))
        for chunk in chunks:
            for name, value in document.metadata.items():
                self.assertEqual(chunk.metadata[name], value)
            self.assertLessEqual(int(chunk.metadata["actual_overlap_tokens"]), 8)
        chunks[0].metadata["title"] = "changed locally"
        self.assertNotEqual(chunks[1].metadata["title"], "changed locally")
        self.assertNotEqual(document.metadata["title"], "changed locally")

    def test_ids_repeat_and_change_with_configuration_or_source(self):
        document = load_markdown(ATLAS_PATH)
        def ids(doc, budget=32):
            return [chunk.metadata["chunk_id"] for chunk in split_token_chunks(doc, self.tokenizer, budget, 8)]
        self.assertEqual(ids(document), ids(document))
        self.assertNotEqual(ids(document), ids(document, 33))
        edited = SourceDocument(document.page_content + " New fact.", dict(document.metadata))
        self.assertNotEqual(ids(document), ids(edited))

    def test_unicode_and_no_overlap_preserve_every_character(self):
        text = "## Evidence\n" + "Maya 中文 🙂 café e\u0301\n" * 30
        document = SourceDocument(text, {"source": "synthetic.md", "document_id": "TEST"})
        chunks = split_token_chunks(document, self.tokenizer, 17, 0)
        validate_chunks(document, chunks, self.tokenizer, 17)
        self.assertEqual("".join(chunk.page_content for chunk in chunks), text)
        overlapped = split_token_chunks(document, self.tokenizer, 17, 5)
        validate_chunks(document, overlapped, self.tokenizer, 17)
        self.assertTrue(all(count_tokens(self.tokenizer, chunk.page_content) <= 17 for chunk in overlapped))

    def test_invalid_budgets_rejected(self):
        document = load_markdown(ATLAS_PATH)
        for size, overlap in [(0, 0), (8, 8), (8, -1), (True, 0)]:
            with self.assertRaises(ValueError):
                split_token_chunks(document, self.tokenizer, size, overlap)

    def test_default_preserves_short_sections_and_validator_detects_gaps(self):
        document = load_markdown(ATLAS_PATH)
        chunks = split_token_chunks(document, self.tokenizer)
        self.assertEqual(len(chunks), 6)
        self.assertEqual("".join(chunk.page_content for chunk in chunks), document.page_content)
        with self.assertRaises(ValueError):
            validate_chunks(document, chunks[1:], self.tokenizer, 256)


if __name__ == "__main__":
    unittest.main()
