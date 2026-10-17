"""Inspect question-selected graph evidence without supplying an answer endpoint."""

import argparse
import json

from scripts.config.project_config import PROJECT_ROOT, load_settings
from scripts.corpus.corpus_pipeline import active_manifest, atomic_json
from scripts.graph.graph_evidence import fetch_path_evidence
from scripts.graph.knowledge_graph import load_graph, to_networkx
from scripts.graph.question_retrieval import select_question_paths
from scripts.services.service_clients import PineconeStore, ServiceError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="Who approved DEC-002?")
    parser.add_argument("--fetch-pinecone", action="store_true")
    args = parser.parse_args()
    store = None
    try:
        settings = load_settings()
        manifest = active_manifest(settings)
        network = to_networkx(load_graph(), manifest)
        selection = select_question_paths(network, args.question)
        report = {"selection": selection, "retrieval": {"status": "not_requested"}}
        if args.fetch_pinecone and selection["evidence_chunk_ids"]:
            store = PineconeStore(settings)
            store.ensure_index()
            report["retrieval"] = fetch_path_evidence(selection, manifest, store)
        path = PROJECT_ROOT / "work" / "graph" / ("question-evidence-live.json" if args.fetch_pinecone else "question-selection.json")
        atomic_json(path, report)
        print(json.dumps(report, indent=2))
        print(f"\nReceipt: {path.relative_to(PROJECT_ROOT).as_posix()}")
        print("Candidate evidence only: no generated answer, embedding or index write.")
        return 0
    except (ValueError, OSError, KeyError, TypeError, ServiceError) as error:
        print(f"Question graph inspection failed: {error}")
        return 1
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
