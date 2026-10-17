"""Inspect the first reviewed graph and its supporting passages; no API calls."""

import json

from scripts.corpus.corpus_pipeline import active_manifest, atomic_json
from scripts.graph.knowledge_graph import load_graph, validate_graph
from scripts.config.project_config import PROJECT_ROOT, load_settings


def main():
    try:
        # Recompute current corpus identity so stale source references cannot
        # silently be used with a different ingested version.
        manifest = active_manifest(load_settings())
        graph = load_graph()
        summary = validate_graph(graph, manifest)
        labels = {node["id"]: node["label"] for node in graph["nodes"]}
        print(json.dumps(summary, indent=2))
        for edge in graph["relationships"]:
            print(f"\n{labels[edge['source']]} --{edge['relation']}--> {labels[edge['target']]}")
            for evidence in edge["evidence"]:
                print(f"  {evidence['document_id']} | {evidence['section']} | {evidence['source']}")
                print(f"  Evidence: {evidence['excerpt']}")
        path = PROJECT_ROOT / "work" / "graph" / "first-graph-inspection.json"
        atomic_json(path, {"summary": summary, "graph": graph})
        print(f"\nReceipt: {path.relative_to(PROJECT_ROOT).as_posix()}")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Graph inspection failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
