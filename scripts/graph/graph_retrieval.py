"""Graph-selected source passages plus the unchanged vector candidates."""

from copy import deepcopy

from scripts.corpus.corpus_pipeline import active_manifest, retrieve
from scripts.graph.graph_evidence import fetch_path_evidence
from scripts.graph.knowledge_graph import load_graph, to_networkx
from scripts.graph.question_retrieval import select_question_paths
from scripts.services.service_clients import ServiceError


def merge_evidence(graph_evidence, vector_evidence):
    """Graph-first order, one copy per passage, with both origins inspectable."""
    merged = {}
    for origin, rows in (("graph_path", graph_evidence), ("vector", vector_evidence)):
        for item in rows:
            row = deepcopy(item)
            chunk_id = row["id"]
            if chunk_id in merged:
                existing = merged[chunk_id]
                if existing["text"] != row["text"] or existing["metadata"] != row["metadata"]:
                    raise ServiceError("Graph/vector copies of a passage disagree")
                if origin not in existing["retrieval_origins"]:
                    existing["retrieval_origins"].append(origin)
                if origin == "vector":
                    existing["vector_score"] = row.get("score")
                continue
            row["retrieval_origins"] = [origin]
            row["retrieval_origin"] = origin
            merged[chunk_id] = row
    return list(merged.values())


def retrieve_with_graph(settings, store, embeddings, question):
    """Resolve graph ambiguity before spending a query embedding/model call."""
    if not question.strip() or len(question) > 2000:
        raise ValueError("Question must contain 1-2000 characters")
    manifest = active_manifest(settings)
    selection = select_question_paths(to_networkx(load_graph(), manifest), question)
    base = {"question": question, "namespace": manifest["namespace"],
            "corpus_version": manifest["corpus_version"], "graph_selection": selection}
    if selection["status"] == "clarification_required":
        names = sorted({candidate["label"] for mention in selection["mentions"]
                        if mention["status"] == "ambiguous" for candidate in mention["candidates"]})
        return {**base, "evidence": [], "clarification": {
            "status": "clarification", "answer": "Which entity do you mean: " + " or ".join(names) + "?",
            "citations": []}, "graph_retrieval": {"status": "clarification_required"}}

    # The same top-four vector query remains available for descriptive questions
    # and evidence the reviewed graph does not represent. No answer-key access.
    vector = retrieve(settings, store, embeddings, question)
    if vector["corpus_version"] != manifest["corpus_version"] or vector["namespace"] != manifest["namespace"]:
        raise ServiceError("Graph/vector corpus identities differ")
    fetched = fetch_path_evidence(selection, manifest, store)
    evidence = merge_evidence(fetched["evidence"], vector["evidence"])
    return {**base, "evidence": evidence, "seed_evidence": vector["evidence"],
            "graph_retrieval": {"status": fetched["status"], "ordering": "graph then vector; deduplicated",
                "graph_passages": len(fetched["evidence"]), "vector_passages": len(vector["evidence"]),
                "merged_passages": len(evidence),
                "fallback_reason": selection["stop_reason"] if not fetched["evidence"] else None}}
