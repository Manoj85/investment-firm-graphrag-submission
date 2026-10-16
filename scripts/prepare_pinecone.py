"""Inspect the project index, or create it explicitly with --create."""

import argparse
import json

from project_config import PROJECT_ROOT, load_settings
from corpus_pipeline import atomic_json
from service_clients import DIMENSION, PineconeStore, ServiceError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create", action="store_true", help="Create the configured index if absent")
    args = parser.parse_args()
    store = None
    try:
        store = PineconeStore(load_settings())
        description = store.ensure_index(create=args.create)
        # Check at the point of use as well as inside the shared client.
        # After this guard, description[...] always operates on a dictionary.
        if not isinstance(description, dict):
            raise ServiceError("Expected a Pinecone index description dictionary")
        report = {"status": "ready", "index": description["name"],
            "dimension": DIMENSION, "metric": description["metric"],
            "host": description["host"], "spec": description["spec"], "stats": store.stats()}
        atomic_json(PROJECT_ROOT / "work" / "pinecone" / "index-check.json", report)
        print(json.dumps(report, indent=2))
        return 0
    except (ValueError, ServiceError, OSError) as error:
        print(f"Pinecone setup failed: {error}. No index was deleted or replaced.")
        return 1
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
