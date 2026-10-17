"""Validate reviewed graph facts against the active source corpus, offline."""

import json
from copy import deepcopy
from pathlib import Path

import networkx as nx

from project_config import PROJECT_ROOT


GRAPH_PATH = PROJECT_ROOT / "outputs" / "graph" / "knowledge-graph.json"
NODE_TYPES = {"Person", "Project", "Skill", "Team", "Tool", "Decision", "Document"}
# A deliberately small vocabulary for our first learning checkpoint.
RELATION_ENDPOINTS = {
    "WORKED_ON": ("Person", "Project"),
    "HAS_SKILL": ("Person", "Skill"),
    "MEMBER_OF": ("Person", "Team"),
}
EVIDENCE_FIELDS = {"document_id", "section", "source", "version", "source_sha256", "chunk_id", "excerpt"}


def nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def validate_graph(graph, manifest):
    """Check structure and exact provenance, not natural-language entailment."""
    if not isinstance(graph, dict) or graph.get("schema_version") != 1:
        raise ValueError("Unsupported graph schema_version; expected 1")
    if not nonempty_string(graph.get("graph_version")):
        raise ValueError("Graph version must be a nonempty string")
    if graph.get("corpus_version") != manifest["corpus_version"]:
        raise ValueError("Graph corpus_version differs from the active source version; review provenance")
    nodes, edges = graph.get("nodes"), graph.get("relationships")
    if not isinstance(nodes, list) or not nodes or not isinstance(edges, list) or not edges:
        raise ValueError("Graph requires nonempty nodes and relationships lists")
    by_id = {}
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("Each node must be an object")
        node_id, node_type = node.get("id"), node.get("type")
        if not nonempty_string(node_id) or node_id in by_id:
            raise ValueError("Node IDs must be nonempty and unique")
        if not isinstance(node_type, str) or node_type not in NODE_TYPES:
            raise ValueError("Unknown graph node type")
        if not node_id.startswith(node_type.lower() + ":") or not nonempty_string(node.get("label")):
            raise ValueError("Node ID prefix must match type; label must be nonempty")
        aliases = node.get("aliases")
        if (not isinstance(aliases, list) or any(not nonempty_string(alias) for alias in aliases)
                or len(aliases) != len(set(aliases))):
            raise ValueError("Node aliases must be a list of unique nonempty strings")
        # Aliases may be shared across people: future identity resolution must
        # detect ambiguity rather than merging people with the same first name.
        by_id[node_id] = node

    chunks = {record["id"]: record["metadata"] for record in manifest["records"]}
    edge_ids, used_nodes, documents = set(), set(), set()
    evidence_count = 0
    for edge in edges:
        if not isinstance(edge, dict) or not nonempty_string(edge.get("id")):
            raise ValueError("Each relationship needs a nonempty ID")
        if edge["id"] in edge_ids:
            raise ValueError("Relationship IDs must be unique")
        edge_ids.add(edge["id"])
        source, target, relation = edge.get("source"), edge.get("target"), edge.get("relation")
        if not isinstance(source, str) or not isinstance(target, str) or source not in by_id or target not in by_id:
            raise ValueError("Relationship endpoint is not an existing node")
        if not isinstance(relation, str) or relation not in RELATION_ENDPOINTS:
            raise ValueError("Unknown relationship type")
        if (by_id[source]["type"], by_id[target]["type"]) != RELATION_ENDPOINTS[relation]:
            raise ValueError("Relationship endpoint types do not match its direction/type")
        evidence = edge.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("Every relationship requires source evidence")
        for citation in evidence:
            if not isinstance(citation, dict) or set(citation) != EVIDENCE_FIELDS:
                raise ValueError("Evidence must contain the complete provenance fields")
            if any(not nonempty_string(value) for value in citation.values()):
                raise ValueError("Evidence fields must be nonempty strings")
            meta = chunks.get(citation["chunk_id"])
            if meta is None:
                raise ValueError("Evidence chunk is absent from the active manifest")
            for key in EVIDENCE_FIELDS - {"excerpt"}:
                if citation[key] != meta[key]:
                    raise ValueError(f"Evidence {key} differs from the source chunk")
            if citation["excerpt"] not in meta["chunk_text"]:
                raise ValueError("Evidence excerpt is not an exact substring of the source chunk")
            documents.add(citation["document_id"])
            evidence_count += 1
        used_nodes.update((source, target))
    if used_nodes != set(by_id):
        raise ValueError("Every node must participate in a supported relationship")
    return {"status": "graph structure and source references verified locally",
            "graph_version": graph["graph_version"], "corpus_version": graph["corpus_version"],
            "nodes": len(nodes), "relationships": len(edges), "evidence_references": evidence_count,
            "supporting_documents": sorted(documents),
            "semantic_support": "manually reviewed; not established by substring validation"}


def load_graph(path: Path = GRAPH_PATH):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def to_networkx(graph, manifest) -> nx.MultiDiGraph:
    """Validate before construction; preserve direction, edge IDs and evidence."""
    validate_graph(graph, manifest)
    network = nx.MultiDiGraph(
        schema_version=graph["schema_version"], graph_version=graph["graph_version"],
        corpus_version=graph["corpus_version"])
    for node in graph["nodes"]:
        # The stable ID is the NetworkX key. Copy nested aliases/evidence so
        # modifying an in-memory graph cannot mutate our loaded JSON object.
        attributes = deepcopy(node)
        node_id = attributes.pop("id")
        network.add_node(node_id, **attributes)
    for edge in graph["relationships"]:
        network.add_edge(edge["source"], edge["target"], key=edge["id"],
                         relation=edge["relation"], evidence=deepcopy(edge["evidence"]))
    return network


def outgoing_relationships(network: nx.MultiDiGraph, node_id: str) -> list[dict]:
    """Return direct outgoing edges with their evidence; no inferred paths."""
    if node_id not in network:
        raise ValueError("Unknown graph entity ID; use a stored stable ID")
    rows = []
    for source, target, edge_id, attributes in network.out_edges(node_id, keys=True, data=True):
        rows.append({"edge_id": edge_id, "source": source,
            "source_label": network.nodes[source]["label"], "relation": attributes["relation"],
            "target": target, "target_label": network.nodes[target]["label"],
            "evidence": deepcopy(attributes["evidence"])})
    return rows
