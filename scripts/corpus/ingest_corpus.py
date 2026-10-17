"""Prepare (--dry-run), or embed and persist the configured source corpus."""

import argparse
import json

from scripts.corpus.corpus_pipeline import atomic_json, ingest, prepare_corpus
from scripts.config.project_config import PROJECT_ROOT, load_settings
from scripts.services.service_clients import NebiusEmbeddings, PineconeStore, ServiceError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Local preparation only; no credentials or requests needed")
    args = parser.parse_args()
    store = embeddings = None
    try:
        settings = load_settings()
        if args.dry_run:
            manifest = prepare_corpus(settings)
            atomic_json(PROJECT_ROOT / "work" / "pinecone" / "prepared-corpus.json", manifest)
            result = {"status": "prepared locally; NOT ingested", "documents": len(manifest["documents"]),
                      "records": len(manifest["records"]), "namespace": manifest["namespace"],
                      "corpus_version": manifest["corpus_version"], "dimension": manifest["dimension"]}
        else:
            store = PineconeStore(settings)
            embeddings = NebiusEmbeddings(settings)
            result = ingest(settings, store, embeddings)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, ServiceError, OSError) as error:
        print(f"Ingestion failed: {error}")
        return 1
    finally:
        if embeddings is not None:
            embeddings.close()
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
