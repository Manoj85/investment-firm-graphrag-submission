"""Resolve explicit entity names and fetch graph-selected source passages."""

from copy import deepcopy

import networkx as nx

from knowledge_graph import EVIDENCE_FIELDS
from service_clients import ServiceError


def resolve_entity(network: nx.MultiDiGraph, text: str) -> dict:
    """Exact stable ID first; otherwise normalized whole label/alias matching.

    This accepts an entity name, not an arbitrary question. Shared aliases stay
    ambiguous; no first-match, substring or model-based identity guessing.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Entity selection must be nonempty text")
    query = text.strip()
    if query in network:
        ids = [query]
    else:
        normalize = lambda value: " ".join(value.split()).casefold()
        ids = sorted(node_id for node_id, attrs in network.nodes(data=True)
                     if any(normalize(query) == normalize(name)
                            for name in [attrs["label"], *attrs["aliases"]]))
    return {"input": text, "status": "resolved" if len(ids) == 1 else "ambiguous" if ids else "not_found",
            "candidates": [{"id": node_id, "label": network.nodes[node_id]["label"],
                            "type": network.nodes[node_id]["type"]} for node_id in ids]}


def fetch_path_evidence(report: dict, manifest: dict, store) -> dict:
    """Fetch actual Pinecone text and reject stale/missing/substituted evidence.

    The graph selects IDs; fetched records supply runtime passage text. No
    vector scores are invented for ID fetches, and no vectors enter the receipt.
    """
    if report["corpus_version"] != manifest["corpus_version"]:
        raise ValueError("Path corpus version differs from active manifest")
    expected = {record["id"]: record["metadata"] for record in manifest["records"]}
    refs = {}
    for path in report["paths"]:
        for step in path["steps"]:
            for citation in step["evidence"]:
                chunk_id = citation["chunk_id"]
                meta = expected.get(chunk_id)
                if meta is None or set(citation) != EVIDENCE_FIELDS:
                    raise ValueError("Path evidence is absent or malformed")
                if any(citation[key] != meta[key] for key in EVIDENCE_FIELDS - {"excerpt"}):
                    raise ValueError("Path provenance differs from active manifest")
                if not citation["excerpt"] or citation["excerpt"] not in meta["chunk_text"]:
                    raise ValueError("Path excerpt differs from active source")
                refs.setdefault(chunk_id, []).append(step["edge_id"])
    ids = list(refs)
    if ids != report["evidence_chunk_ids"] or len(ids) > 40:
        raise ValueError("Path evidence IDs differ from references or exceed fetch limit")
    fetched = store.fetch(manifest["namespace"], ids) if ids else {}
    if set(fetched) != set(ids):
        raise ServiceError("Graph fetch did not return exactly the requested passages")
    evidence = []
    for chunk_id in ids:
        record = fetched[chunk_id]
        if not isinstance(record, dict) or record.get("id") != chunk_id or record.get("metadata") != expected[chunk_id]:
            raise ServiceError("Graph passage identity/metadata/text differs from active manifest")
        meta = deepcopy(record["metadata"])
        evidence.append({"id": chunk_id, "score": None, "text": meta.pop("chunk_text"),
                         "metadata": meta, "retrieval_origin": "graph_path",
                         "supporting_edge_ids": list(dict.fromkeys(refs[chunk_id]))})
    return {"namespace": manifest["namespace"], "corpus_version": manifest["corpus_version"],
            "status": "verified" if ids else "no_path_evidence", "evidence": evidence}
