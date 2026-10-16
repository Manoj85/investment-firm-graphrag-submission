"""Show text, token IDs, and source spans locally. No model/API calls."""

import argparse
import json

from inspect_atlas import ATLAS_PATH, load_markdown
from inspect_atlas_chunks import split_by_section
from qwen_tokenizer import MODEL_ID, REVISION, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--text", help="A short sentence to inspect instead of the default")
    inputs.add_argument("--atlas-ownership", action="store_true",
                        help="Inspect the actual passage used by our embedding diagnostic")
    parser.add_argument("--limit", type=int, default=30, help="Maximum token rows to print")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    text = args.text if args.text is not None else "Maya leads Project Atlas."
    if args.atlas_ownership:
        sections = split_by_section(load_markdown(ATLAS_PATH))
        text = next(section.page_content for section in sections
                    if section.metadata["section"] == "Ownership")
    if not text.strip():
        parser.error("Choose nonempty text")
    try:
        tokenizer = load_tokenizer()
    except (OSError, ValueError) as error:
        raise SystemExit(f"Could not inspect tokens: {error}") from error
    # Encoding maps text to vocabulary IDs; it does not produce embedding vectors.
    encoding = tokenizer.encode(text, add_special_tokens=False)
    print(f"Tokenizer: {MODEL_ID}; revision {REVISION}")
    print("Input text (JSON-escaped so newlines/Unicode remain visible):")
    print(json.dumps(text, ensure_ascii=True))
    print(f"Characters: {len(text)}; whitespace-separated words: {len(text.split())}")
    print(f"Local content tokens: {len(encoding.ids)}")
    print("Row | token ID | source span | original text covered by this token")
    for number, (token_id, (start, end)) in enumerate(
            zip(encoding.ids, encoding.offsets), start=1):
        if number > args.limit:
            break
        # Slice the original text instead of decoding an arbitrary token alone.
        # Multiple byte-based tokens can cover the same Unicode character span.
        original = json.dumps(text[start:end], ensure_ascii=True)
        print(f"{number:02d} | {token_id} | [{start}:{end}] | {original}")
    if len(encoding.ids) > args.limit:
        print(f"Showing first {args.limit} tokens; increase --limit to see more.")
    print("Token IDs are vocabulary labels, not meaning vectors or provider usage counts.")
    print("Offline inspection only. No API request, embedding, or Pinecone write.")


if __name__ == "__main__":
    main()
