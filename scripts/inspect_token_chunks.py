"""Inspect token-bounded Atlas chunks offline after preparing tokenizer assets."""

import argparse
from dataclasses import asdict
import json

from inspect_atlas import ATLAS_PATH, PROJECT_ROOT, load_markdown
from qwen_tokenizer import load_tokenizer
from token_chunks import split_token_chunks, validate_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    args = parser.parse_args()
    try:
        tokenizer = load_tokenizer()
        document = load_markdown(ATLAS_PATH)
        chunks = split_token_chunks(document, tokenizer, args.max_tokens, args.overlap)
        validate_chunks(document, chunks, tokenizer, args.max_tokens)
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Could not prepare token chunks: {error}") from error

    folder = PROJECT_ROOT / "work" / "chunks"
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / f"atlas-{args.max_tokens}-{args.overlap}.json"
    receipt.write_text(json.dumps([asdict(chunk) for chunk in chunks], indent=2,
                                 ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Token budget: {args.max_tokens}; overlap target: {args.overlap}")
    print(f"Chunks: {len(chunks)}; exact source slices, full coverage, and size limits verified")
    for number, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        print(f"{number:02d} | {meta['section']} | {meta['token_count']} tokens | "
              f"chars [{meta['char_start']}:{meta['char_end']}] | "
              f"overlap {meta['actual_overlap_tokens']}")
    print(f"Full text and metadata: {receipt.relative_to(PROJECT_ROOT).as_posix()}")
    print("No embedding calls or Pinecone writes made.")


if __name__ == "__main__":
    main()
