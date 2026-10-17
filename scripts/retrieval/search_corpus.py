"""Embed a question, search the persisted active corpus, and show evidence."""

import argparse
from datetime import datetime, timezone
import json

from scripts.corpus.corpus_pipeline import atomic_json, retrieve
from scripts.config.project_config import PROJECT_ROOT, load_settings
from scripts.services.service_clients import NebiusEmbeddings, PineconeStore, ServiceError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    store = embeddings = None
    try:
        settings = load_settings()
        store, embeddings = PineconeStore(settings), NebiusEmbeddings(settings)
        report = retrieve(settings, store, embeddings, args.question, top_k=args.top_k)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = PROJECT_ROOT / "work" / "pinecone" / f"search-{stamp}.json"
        atomic_json(path, report)
        print(json.dumps(report, indent=2, ensure_ascii=True))
        print(f"Receipt: {path.relative_to(PROJECT_ROOT).as_posix()}")
        return 0
    except (ValueError, ServiceError, OSError) as error:
        print(f"Search failed: {error}")
        return 1
    finally:
        if embeddings is not None:
            embeddings.close()
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
