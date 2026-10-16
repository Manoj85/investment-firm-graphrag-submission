"""Inspect local setup only: no network requests and no credential values."""

import sys

from project_config import CORPUS_DIR, PROJECT_ROOT, load_settings


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Interpreter: {sys.executable}")
    print(f"Virtual environment: {sys.prefix != sys.base_prefix}")
    print(f"Local .env present: {(PROJECT_ROOT / '.env').is_file()}")
    # Restrict inspection to our source directory, never docs or evaluation keys.
    sources = sorted(CORPUS_DIR.glob("*.md"))
    print(f"Corpus: outputs/corpus ({len(sources)} Markdown document(s))")
    if not CORPUS_DIR.is_dir() or not sources:
        print("Local setup incomplete: source corpus is missing or empty.")
        return 1

    try:
        settings = load_settings()
    except ValueError as error:
        print(f"Configuration error: {error}")
        return 1

    for service, ready in settings.service_readiness().items():
        status = "present; access NOT verified" if ready else "incomplete (expected before setup)"
        print(f"{service} settings: {status}")
    print(f"LangSmith tracing requested: {settings.langsmith_tracing}")
    print("Local configuration check passed. No service calls made.")
    # Service credentials are deliberately optional for this offline checkpoint.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
