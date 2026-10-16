"""Split our controlled Markdown sections into source-preserving token windows."""

import hashlib
import json

from tokenizers import Tokenizer

from inspect_atlas import SourceDocument
from inspect_atlas_chunks import split_by_section
from qwen_tokenizer import MODEL_ID, REVISION, TOKENIZER_SHA256, count_tokens


CHUNKING_VERSION = "section-token-window-v1"


def section_windows(text: str, tokenizer: Tokenizer, max_tokens: int, overlap: int):
    """Yield character spans bounded by actual re-encoded token counts.

    Token offsets locate original text, so Unicode is never reconstructed by
    decoding an arbitrary slice of byte-based token IDs. Overlap is a target
    upper bound: Unicode boundaries or progress requirements can reduce it.
    """
    start, previous_end = 0, 0
    while start < len(text):
        encoding = tokenizer.encode(text[start:], add_special_tokens=False)
        if len(encoding.ids) <= max_tokens:
            end = len(text)
        else:
            # Offsets are relative to the encoded suffix. Python slices exclude end.
            end = start + max(stop for _, stop in encoding.offsets[:max_tokens])
            # Cutting/restarting can change BPE boundaries. Recount the actual
            # substring, and move back by whole characters until the limit holds.
            while end > start and count_tokens(tokenizer, text[start:end]) > max_tokens:
                end -= 1
        if end <= start:
            raise ValueError("Token budget cannot fit the next complete Unicode character")
        if end <= previous_end:
            # An overlap must not produce a window with no new source characters.
            start = previous_end
            continue
        actual_overlap = count_tokens(tokenizer, text[start:previous_end])
        yield start, end, actual_overlap
        if end == len(text):
            break

        next_start = end
        if overlap:
            piece = tokenizer.encode(text[start:end], add_special_tokens=False)
            relative_start = piece.offsets[max(0, len(piece.ids) - overlap)][0]
            next_start = start + relative_start
            # The overlap substring is also re-encoded to enforce its own budget.
            while next_start < end and count_tokens(tokenizer, text[next_start:end]) > overlap:
                next_start += 1
        previous_end = end
        # Ensure termination even with an unusually small budget or token offsets.
        start = next_start if next_start > start else end


def split_token_chunks(document: SourceDocument, tokenizer: Tokenizer,
                       max_tokens: int = 256, overlap: int = 32) -> list[SourceDocument]:
    if type(max_tokens) is not int or type(overlap) is not int or not 0 <= overlap < max_tokens:
        raise ValueError("Require integer settings: 0 <= overlap < max_tokens")
    config = {"version": CHUNKING_VERSION, "tokenizer_model": MODEL_ID,
              "tokenizer_revision": REVISION, "tokenizer_sha256": TOKENIZER_SHA256,
              "max_tokens": max_tokens, "overlap": overlap}
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    source_hash = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
    chunks, section_start = [], 0
    for section in split_by_section(document):
        for number, (start, end, repeated) in enumerate(
                section_windows(section.page_content, tokenizer, max_tokens, overlap), start=1):
            text = section.page_content[start:end]
            # Add section-relative offsets to the section's position in the document.
            absolute_start, absolute_end = section_start + start, section_start + end
            identity = [document.metadata["source"], document.metadata["document_id"],
                        document.metadata.get("version", "unversioned"), source_hash,
                        config_id, absolute_start, absolute_end]
            chunk_id = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
            # All metadata stays string-valued to match SourceDocument's contract.
            metadata = dict(section.metadata)
            metadata.update({"parent_section_id": section.metadata["chunk_id"],
                "chunk_id": chunk_id, "chunk_in_section": str(number),
                "char_start": str(absolute_start), "char_end": str(absolute_end),
                "token_count": str(count_tokens(tokenizer, text)),
                "actual_overlap_tokens": str(repeated),
                "max_tokens": str(max_tokens), "target_overlap_tokens": str(overlap),
                "chunking_version": CHUNKING_VERSION, "chunking_config_id": config_id,
                "source_sha256": source_hash, "tokenizer_model": MODEL_ID,
                "tokenizer_revision": REVISION, "tokenizer_sha256": TOKENIZER_SHA256})
            chunks.append(SourceDocument(text, metadata))
        section_start += len(section.page_content)
    return chunks


def validate_chunks(document: SourceDocument, chunks: list[SourceDocument],
                    tokenizer: Tokenizer, max_tokens: int) -> None:
    """Check text coverage by source spans, not by concatenating overlapping text."""
    covered_until = 0
    if not chunks:
        raise ValueError("No chunks produced")
    for chunk in chunks:
        start, end = int(chunk.metadata["char_start"]), int(chunk.metadata["char_end"])
        if not 0 <= start <= covered_until < end <= len(document.page_content):
            raise ValueError("Chunk spans have a gap, invalid boundary, or no new text")
        if chunk.page_content != document.page_content[start:end]:
            raise ValueError("Chunk differs from original source slice")
        if count_tokens(tokenizer, chunk.page_content) > max_tokens:
            raise ValueError("Chunk exceeds token budget")
        covered_until = end
    if covered_until != len(document.page_content):
        raise ValueError("Source ending was lost")
