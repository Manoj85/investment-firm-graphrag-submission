"""Load our first source document and inspect it. Requires only Python 3.10+."""

from dataclasses import dataclass
import json
from pathlib import Path


# Resolve paths from this script, so running it from another folder still works.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ATLAS_PATH = PROJECT_ROOT / "outputs" / "corpus" / "02-project-atlas-charter.md"

# These labels belong to our simple source-document format, not all Markdown.
HEADER_FIELDS = {
    "Document ID": "document_id",
    "Company": "company",
    "Version": "version",
    "Updated": "updated",
    "Source owner": "source_owner",
}


@dataclass
class SourceDocument:
    """Source text plus identifying information; not a LangChain object yet."""

    page_content: str
    metadata: dict[str, str]


def load_markdown(source: Path) -> SourceDocument:
    """Read one project Markdown document without changing its content."""
    source = source.resolve()
    text = source.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"Source document is empty: {source}")

    metadata = {"source": source.relative_to(PROJECT_ROOT).as_posix()}

    # Read identifying fields before the first level-two section heading.
    # partition(':') preserves any additional colons in a field's value.
    for line in text.splitlines():
        if line.startswith("## "):
            break
        if line.startswith("# "):
            metadata["title"] = line[2:].strip()
        label, separator, value = line.partition(":")
        if separator and label in HEADER_FIELDS:
            metadata[HEADER_FIELDS[label]] = value.strip()

    for required in ("title", "document_id"):
        if not metadata.get(required):
            raise ValueError(f"Missing required metadata '{required}': {source}")

    return SourceDocument(page_content=text, metadata=metadata)


def main() -> None:
    try:
        document = load_markdown(ATLAS_PATH)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Could not load Atlas: {error}") from error

    print("METADATA")
    print(json.dumps(document.metadata, indent=2, ensure_ascii=False))

    sections = [
        line[3:].strip()
        for line in document.page_content.splitlines()
        if line.startswith("## ")
    ]
    print("\nSECTION HEADINGS")
    for section in sections:
        print(f"- {section}")

    print("\nTEXT CHECK")
    print(f"Characters: {len(document.page_content)}")
    print(f"Whitespace-separated words: {len(document.page_content.split())}")

    print("\nFULL SOURCE TEXT")
    print(document.page_content)


if __name__ == "__main__":
    main()
