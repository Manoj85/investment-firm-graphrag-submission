"""Load reviewed JSON into NetworkX and inspect one entity's direct edges."""

import argparse
import json

import networkx as nx

from scripts.corpus.corpus_pipeline import active_manifest, atomic_json
from scripts.graph.knowledge_graph import load_graph, outgoing_relationships, to_networkx
from scripts.config.project_config import PROJECT_ROOT, load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity-id", default="person:priya-shah", help="Stored node ID, not a free-text question")
    args = parser.parse_args()
    try:
        manifest = active_manifest(load_settings())
        network = to_networkx(load_graph(), manifest)
        rows = outgoing_relationships(network, args.entity_id)
        summary = {"status": "validated graph loaded into NetworkX locally",
            "networkx_version": nx.__version__, "graph_class": type(network).__name__,
            "directed": network.is_directed(), "allows_parallel_edges": network.is_multigraph(),
            "nodes": network.number_of_nodes(), "relationships": network.number_of_edges(),
            "graph_version": network.graph["graph_version"],
            "corpus_version": network.graph["corpus_version"],
            "selected_entity": args.entity_id, "selected_label": network.nodes[args.entity_id]["label"],
            "outgoing_edges": network.out_degree(args.entity_id),
            "incoming_edges": network.in_degree(args.entity_id)}
        report = {"summary": summary, "outgoing_relationships": rows}
        print(json.dumps(summary, indent=2))
        for row in rows:
            print(f"\n{row['source_label']} --{row['relation']}--> {row['target_label']}")
            print(f"  Edge ID: {row['edge_id']}")
            for citation in row["evidence"]:
                print(f"  {citation['document_id']} | {citation['section']} | {citation['excerpt']}")
        if not rows:
            print("\nNo outgoing relationships for this entity. Incoming links may still exist.")
        path = PROJECT_ROOT / "work" / "graph" / "networkx-inspection.json"
        atomic_json(path, report)
        print(f"\nReceipt: {path.relative_to(PROJECT_ROOT).as_posix()}")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"NetworkX inspection failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
