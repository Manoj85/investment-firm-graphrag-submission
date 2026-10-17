"""Inspect bounded graph paths and their evidence offline, using stable entity IDs."""

import argparse
import json

from scripts.corpus.corpus_pipeline import active_manifest, atomic_json
from scripts.graph.graph_paths import bounded_paths
from scripts.graph.knowledge_graph import load_graph, to_networkx
from scripts.config.project_config import PROJECT_ROOT, load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="tool:beacon-evidence-hub")
    parser.add_argument("--target", default="person:elena-brooks")
    parser.add_argument("--max-hops", type=int, default=2)
    args = parser.parse_args()
    try:
        network = to_networkx(load_graph(), active_manifest(load_settings()))
        report = bounded_paths(network, args.start, args.target, max_hops=args.max_hops)
        report["graph_version"] = network.graph["graph_version"]
        report["corpus_version"] = network.graph["corpus_version"]
        summary = {key: value for key, value in report.items() if key != "paths"}
        summary["path_count"] = len(report["paths"])
        print(json.dumps(summary, indent=2))
        for number, path in enumerate(report["paths"], 1):
            print(f"\nPath {number}: {len(path['steps'])} hops")
            for step in path["steps"]:
                print(f"  Fact: {step['source_label']} --{step['relation']}--> {step['target_label']}")
                print(f"    Walk: {step['traversal']} | edge: {step['edge_id']}")
                for citation in step["evidence"]:
                    print(f"    {citation['document_id']} | {citation['section']} | {citation['excerpt']}")
        if not report["paths"]:
            print("No path found within these limits; this does not establish absence of a fact.")
        path = PROJECT_ROOT / "work" / "graph" / "path-inspection.json"
        atomic_json(path, report)
        print(f"\nReceipt: {path.relative_to(PROJECT_ROOT).as_posix()}")
        print("Local inspection only; no Pinecone fetch or generated answer.")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Path inspection failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
