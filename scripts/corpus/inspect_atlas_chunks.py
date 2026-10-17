"""Lesson 3: split Atlas at section headings and inspect every chunk.

Only the Python standard library is needed. This script does not call a model,
create embeddings, write an index, or change the source document.
"""

import json

# Reuse lesson 2's loader and data structure rather than reading the file again
# with a second, potentially different implementation.
# Running `python -m scripts.corpus.inspect_atlas_chunks` uses the repository package imports.
from scripts.corpus.inspect_atlas import ATLAS_PATH, SourceDocument, load_markdown


def split_by_section(document: SourceDocument) -> list[SourceDocument]:
    """Return chunks containing original text plus copied source metadata.

    This deliberately handles our controlled Markdown format: a document header
    followed by sections that begin with '## '. It is not a general Markdown
    parser; fenced code blocks and nested heading hierarchies are out of scope.
    """
    if not document.page_content.strip():
        raise ValueError("Cannot split an empty document.")

    # Keep newline characters. Joining the resulting chunks should reproduce
    # exactly the text that lesson 2 loaded, including blank lines.
    lines = document.page_content.splitlines(keepends=True)

    # Line zero starts the header chunk. Each later '## ' line starts a section.
    # If a document starts directly with '## ', zero is already in this list.
    starts = [0]
    for line_number, line in enumerate(lines):
        if line.startswith("## ") and line_number > 0:
            starts.append(line_number)

    # The final boundary is just after the last line. It lets us use the same
    # slicing rule for the last section instead of treating it as a special case.
    starts.append(len(lines))

    chunks = []
    for chunk_number, (start, end) in enumerate(
        zip(starts, starts[1:]), start=1
    ):
        # Python slices include start and exclude end. The heading itself stays
        # in the text, giving a future model context about the passage's topic.
        chunk_text = "".join(lines[start:end])
        first_line = lines[start]
        section = (
            first_line[3:].strip()
            if first_line.startswith("## ")
            else "Document header"
        )

        # Copy the dictionary so chunks do not share mutable metadata with each
        # other or with the original document. All seven source fields survive.
        metadata = dict(document.metadata)

        # Add passage-level identity without replacing document-level identity.
        metadata["section"] = section
        document_id = metadata["document_id"]
        version = metadata.get("version", "unversioned")
        metadata["chunk_id"] = (
            f"{document_id}:{version}:section-{chunk_number:03d}"
        )

        # Reuse the same text-plus-metadata shape for a smaller unit of content.
        # Neither object is a LangChain Document yet; that comes later.
        chunks.append(SourceDocument(page_content=chunk_text, metadata=metadata))

    return chunks


def main() -> None:
    try:
        document = load_markdown(ATLAS_PATH)
        chunks = split_by_section(document)
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Could not split Atlas: {error}") from error

    print(f"Loaded document: {document.metadata['document_id']}")
    print(f"Section-based chunks: {len(chunks)}")

    # This inspection checks that splitting lost no text and duplicated no text.
    reconstructed = "".join(chunk.page_content for chunk in chunks)
    print(f"Original text preserved: {reconstructed == document.page_content}")

    # Display the whole object for each chunk, not just a shortened preview.
    # That makes it possible to inspect the metadata alongside its evidence.
    for number, chunk in enumerate(chunks, start=1):
        print(f"\n--- CHUNK {number}: {chunk.metadata['section']} ---")
        print("METADATA")
        print(json.dumps(chunk.metadata, indent=2, ensure_ascii=False))
        print("TEXT")
        print(chunk.page_content, end="")


if __name__ == "__main__":
    main()
