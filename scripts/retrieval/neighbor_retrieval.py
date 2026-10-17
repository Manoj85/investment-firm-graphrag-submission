"""Opt-in, bounded source-neighbor expansion after ordinary vector retrieval."""

from scripts.corpus.corpus_pipeline import active_manifest, retrieve
from scripts.services.service_clients import ServiceError


def expand_neighbors(report, manifest, store, *, max_neighbors=4):
    """Fetch adjacent passages from the same source; never infer semantic edges."""
    if not 0 <= max_neighbors <= 4:
        raise ValueError("Neighbor experiment supports at most four added passages")
    records = manifest["records"]
    expected = {record["id"]: record["metadata"] for record in records}
    groups = {}
    for record in records:
        meta = record["metadata"]
        # Source identity prevents crossing documents or versions by accident.
        key = (meta["source"], meta["document_id"], meta["version"])
        groups.setdefault(key, []).append(record)
    positions = {}
    for group in groups.values():
        group.sort(key=lambda record: int(record["metadata"]["char_start"]))
        for index, record in enumerate(group):
            positions[record["id"]] = (group, index)

    seeds = report["evidence"]
    seen = {seed["id"] for seed in seeds}
    additions = []
    for seed in seeds:
        group, index = positions[seed["id"]]
        for offset in (-1, 1):
            adjacent = index + offset
            if not 0 <= adjacent < len(group):
                continue
            neighbor_id = group[adjacent]["id"]
            if neighbor_id in seen or len(additions) >= max_neighbors:
                continue
            seen.add(neighbor_id)
            additions.append({"id": neighbor_id, "seed_id": seed["id"],
                              "direction": "previous" if offset < 0 else "next"})

    # The manifest selects IDs, not runtime answer text. Fetch actual source
    # records from Pinecone and reject missing/substituted passages.
    fetched = store.fetch(report["namespace"], [item["id"] for item in additions]) if additions else {}
    if set(fetched) != {item["id"] for item in additions}:
        raise ServiceError("Neighbor fetch did not return every requested passage")
    neighbors_by_seed = {}
    for item in additions:
        meta = fetched[item["id"]].get("metadata")
        if meta != expected[item["id"]]:
            raise ServiceError("Neighbor passage metadata/text differs from the active manifest")
        neighbors_by_seed.setdefault(item["seed_id"], []).append({
            "id": item["id"], "score": None, "text": meta["chunk_text"],
            "metadata": {key: value for key, value in meta.items() if key != "chunk_text"},
            "retrieval_origin": "neighbor", "neighbor_of": item["seed_id"],
            "direction": item["direction"]})

    # Keep a seed and its local context together. The shared evidence packer
    # still enforces the same token budget and labels only retained passages.
    expanded = []
    for seed in seeds:
        expanded.append({**seed, "retrieval_origin": "vector"})
        expanded.extend(neighbors_by_seed.get(seed["id"], []))
    return {**report, "seed_evidence": seeds, "evidence": expanded,
            "neighbor_expansion": {"policy": "adjacent-source-passages-v1",
                "max_added_passages": max_neighbors, "added_passages": len(additions),
                "seed_count": len(seeds), "candidate_count": len(expanded),
                "ordering": "seed then previous/next neighbors"}}


def retrieve_with_neighbors(settings, store, embeddings, question):
    report = retrieve(settings, store, embeddings, question)  # Same top-four seeds.
    manifest = active_manifest(settings)
    return expand_neighbors(report, manifest, store)
