"""Read-only small-corpus fetch, cached-vector comparison, and complete ranking."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math

from corpus_pipeline import active_manifest, atomic_json
from project_config import PROJECT_ROOT, load_settings
from service_clients import (DIMENSION, QUERY_INSTRUCTION,
                             PineconeStore, ServiceError, validate_vector,
                             document_embedding_input, embedding_identity)


DEFAULT_QUESTION = "Who approved the selection of Beacon Evidence Hub for Project Atlas?"


def cached_vector(settings, text, *, query=False):
    """Read the exact existing cache entry; this diagnostic never calls Nebius."""
    # Share the client's identity function so diagnostics cannot drift from it.
    identity = embedding_identity(settings, text, query=query)
    digest = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    path = PROJECT_ROOT / "data" / "cache" / "embeddings" / f"{digest}.json"
    if not path.is_file():
        raise ServiceError("Required exact-input embedding cache is missing; no model call made")
    cached = json.loads(path.read_text(encoding="utf-8"))
    if cached.get("identity") != identity:
        raise ServiceError("Cached embedding input identity does not match")
    return validate_vector(cached["vector"])


def cosine(left, right):
    """Compare vector directions independently of Pinecone's returned score."""
    dot = math.fsum(a * b for a, b in zip(left, right))
    norms = math.sqrt(math.fsum(a * a for a in left) * math.fsum(b * b for b in right))
    return dot / norms


def analyze(manifest, fetched, matches, query_vector, cached_vectors, target_id):
    """Validate complete coverage, then compare remote and local evidence."""
    expected = {record["id"]: record["metadata"] for record in manifest["records"]}
    if set(fetched) != set(expected):
        raise ServiceError("Fetch did not return every expected corpus record")
    match_ids = [item["id"] for item in matches]
    if len(match_ids) != len(expected) or set(match_ids) != set(expected):
        raise ServiceError("Query did not return each expected record exactly once")
    validate_vector(query_vector)
    local_scores, differences = {}, {}
    for chunk_id in expected:
        vector = validate_vector(fetched[chunk_id]["values"])
        cached = validate_vector(cached_vectors[chunk_id])
        differences[chunk_id] = max(abs(a - b) for a, b in zip(vector, cached))
        local_scores[chunk_id] = cosine(query_vector, vector)
    local_order = sorted(local_scores, key=lambda chunk_id: local_scores[chunk_id], reverse=True)
    local_ranks = {chunk_id: rank for rank, chunk_id in enumerate(local_order, 1)}
    rows = []
    for rank, match in enumerate(matches, 1):
        chunk_id = match["id"]
        score = match.get("score")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ServiceError("Query returned a nonfinite or missing similarity score")
        meta = expected[chunk_id]
        rows.append({"rank": rank, "id": chunk_id, "document_id": meta["document_id"],
            "section": meta["section"], "pinecone_score": score,
            "local_cosine": local_scores[chunk_id], "local_rank": local_ranks[chunk_id],
            "score_absolute_difference": abs(score - local_scores[chunk_id]),
            "fetched_metadata_matches": fetched[chunk_id].get("metadata") == meta,
            "query_metadata_matches": match.get("metadata") == meta,
            "stored_vs_cached_max_absolute_difference": differences[chunk_id]})
    target = next(row for row in rows if row["id"] == target_id)
    return {"checks": {
        "all_expected_ids_fetched_and_queried": True,
        "all_metadata_matches": all(row["fetched_metadata_matches"] and row["query_metadata_matches"] for row in rows),
        "all_vectors_finite_nonzero_dimension_4096": True,
        # Pinecone stores floating-point values with lower precision than JSON doubles.
        "all_stored_vectors_match_cache_within_1e_6": max(differences.values()) <= 1e-6,
        "remote_order_matches_local_order": match_ids == local_order},
        # Exact service scores can differ from locally calculated cosine values.
        # Report the observed error rather than assuming an API accuracy guarantee.
        "score_comparison": {
            "max_absolute_difference": max(row["score_absolute_difference"] for row in rows),
            "all_within_1e_5": all(row["score_absolute_difference"] <= 1e-5 for row in rows)},
        "target": {**target, "text": fetched[target_id]["metadata"]["chunk_text"],
                   "dimension": DIMENSION},
        "ranking": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--document-id", default="DOC-003")
    parser.add_argument("--section", default="Approval and accountability")
    args = parser.parse_args()
    store = None
    try:
        settings = load_settings()
        manifest = active_manifest(settings)
        records = manifest["records"]
        # Exhaustive ranking is a teaching diagnostic, not a scalable retriever.
        if not 1 <= len(records) <= 50:
            raise ServiceError("Complete-ranking diagnostic is limited to 1-50 chunks")
        targets = [record for record in records if record["metadata"]["document_id"] == args.document_id
                   and record["metadata"]["section"] == args.section]
        if len(targets) != 1:
            raise ServiceError("Choose a document/section with exactly one chunk for this diagnostic")
        target_id = targets[0]["id"]
        query_vector = cached_vector(settings, args.question, query=True)
        cached_vectors = {record["id"]: cached_vector(settings, document_embedding_input(
                              record["metadata"], context=settings.embedding_document_context))
                          for record in records}
        store = PineconeStore(settings)
        store.ensure_index()
        namespace = manifest["namespace"]
        print("1. Fetching the exact approval record by ID (no similarity search)...", flush=True)
        target_fetch = store.fetch(namespace, [target_id])
        if target_id not in target_fetch:
            raise ServiceError("Approval record missing from direct fetch in the active namespace")
        print(f"   Found {target_id}", flush=True)
        print("2. Fetching all corpus vectors and querying all candidates...", flush=True)
        fetched = store.fetch(namespace, [record["id"] for record in records])
        result = store.search(namespace, query_vector, manifest["corpus_version"], top_k=len(records))
        analysis = analyze(manifest, fetched, result.get("matches", []), query_vector, cached_vectors, target_id)
        analysis["target"].update({
            "document_embedding_input": document_embedding_input(
                targets[0]["metadata"], context=settings.embedding_document_context),
            "embedding_input_includes_source_title": settings.embedding_document_context})
        # Include the separate targeted fetch in the checks, not just the batch read.
        analysis["checks"]["targeted_fetch_matches_batch"] = target_fetch[target_id] == fetched[target_id]
        report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "question": args.question, "namespace": namespace, "corpus_version": manifest["corpus_version"],
            "documents": len(manifest["documents"]), "records": len(records),
            "query_input": f"Instruct: {QUERY_INSTRUCTION}\nQuery:{args.question}",
            "embedding_protocol": manifest["embedding_protocol"], "model_calls": 0,
            "cache_entries_read": len(records) + 1, **analysis}
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = PROJECT_ROOT / "work" / "pinecone" / f"diagnosis-{stamp}.json"
        atomic_json(path, report)
        print("3. Check results:")
        print(json.dumps(report["checks"], indent=2))
        for row in report["ranking"]:
            print(f"{row['rank']:02} | {row['document_id']} | {row['section']} | "
                  f"Pinecone {row['pinecone_score']:.6f} | local {row['local_cosine']:.6f}")
        print(f"Approval rank: {report['target']['rank']} of {len(records)}")
        print(f"Receipt: {path.relative_to(PROJECT_ROOT).as_posix()}")
        return 0 if all(report["checks"].values()) else 1
    except (ValueError, ServiceError, OSError, KeyError, TypeError) as error:
        print(f"Diagnosis failed: {error}")
        return 1
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
