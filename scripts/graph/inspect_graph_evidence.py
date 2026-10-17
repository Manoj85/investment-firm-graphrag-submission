"""Resolve named entities; optionally fetch their path evidence from Pinecone."""

import argparse
import json

from scripts.corpus.corpus_pipeline import active_manifest, atomic_json
from scripts.graph.graph_evidence import fetch_path_evidence, resolve_entity
from scripts.graph.graph_paths import bounded_paths
from scripts.graph.knowledge_graph import load_graph, to_networkx
from scripts.config.project_config import PROJECT_ROOT, load_settings
from scripts.services.service_clients import PineconeStore, ServiceError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="Beacon Evidence Hub")
    parser.add_argument("--target", default="Elena")
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--fetch-pinecone", action="store_true", help="Read-only live passage fetch")
    args = parser.parse_args()
    store = None
    try:
        settings = load_settings()
        manifest = active_manifest(settings)
        network = to_networkx(load_graph(), manifest)
        report = {"selection": {"start": resolve_entity(network, args.start),
                                 "target": resolve_entity(network, args.target)}}
        resolved = all(item["status"] == "resolved" for item in report["selection"].values())
        if resolved:
            start = report["selection"]["start"]["candidates"][0]["id"]
            target = report["selection"]["target"]["candidates"][0]["id"]
            paths = bounded_paths(network, start, target, max_hops=args.max_hops)
            paths.update(graph_version=network.graph["graph_version"], corpus_version=manifest["corpus_version"])
            report["path_search"] = paths
            if args.fetch_pinecone and paths["evidence_chunk_ids"]:
                store = PineconeStore(settings)
                store.ensure_index()  # Check existing index only; never create/write.
                report["retrieval"] = fetch_path_evidence(paths, manifest, store)
            else:
                report["retrieval"] = {"status": "not_requested" if not args.fetch_pinecone else "no_path_evidence"}
        else:
            report["retrieval"] = {"status": "selection_required"}
        # Separate files preserve the live receipt when a later offline check runs.
        name = "graph-evidence-live.json" if args.fetch_pinecone and resolved else "graph-entity-selection.json"
        path = PROJECT_ROOT / "work" / "graph" / name
        atomic_json(path, report)
        print(json.dumps(report, indent=2))
        print(f"\nReceipt: {path.relative_to(PROJECT_ROOT).as_posix()}")
        print("No embeddings or generated answer. Paths are candidates, not an approval conclusion.")
        return 0 if resolved else 2
    except (ValueError, OSError, KeyError, TypeError, ServiceError) as error:
        print(f"Graph evidence inspection failed: {error}")
        return 1
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
