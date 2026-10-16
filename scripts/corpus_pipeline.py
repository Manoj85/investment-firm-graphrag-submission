"""Prepare a versioned corpus, persist passages, and retrieve real evidence."""

import hashlib
import json
import os
import time

from inspect_atlas import load_markdown
from project_config import CORPUS_DIR, PROJECT_ROOT
from qwen_tokenizer import MODEL_ID, load_tokenizer
from service_clients import DIMENSION, EMBEDDING_PROTOCOL, QUERY_INSTRUCTION, ServiceError
from token_chunks import split_token_chunks, validate_chunks


MANIFEST_PATH = PROJECT_ROOT / "data" / "indexes" / "active-manifest.json"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_corpus(settings):
    if settings.nebius_embedding_model != MODEL_ID:
        raise ValueError("Selected embedding model does not match the pinned Qwen tokenizer")
    root = (PROJECT_ROOT / settings.corpus_dir).resolve()
    # Never allow a setting to ingest learning docs or evaluation answer keys.
    if not root.is_relative_to(CORPUS_DIR.resolve()):
        raise ValueError("CORPUS_DIR must be outputs/corpus or a directory beneath it")
    paths = sorted(root.rglob("*.md"))
    if not paths:
        raise ValueError("No source Markdown documents found")
    tokenizer = load_tokenizer()
    documents, chunks, seen = [], [], set()
    for path in paths:
        if not path.resolve().is_relative_to(CORPUS_DIR.resolve()):
            raise ValueError("Source symlink escapes outputs/corpus")
        document = load_markdown(path)
        doc_id = document.metadata["document_id"]
        if doc_id in seen:
            raise ValueError(f"Duplicate source document ID: {doc_id}")
        seen.add(doc_id)
        prepared = split_token_chunks(document, tokenizer, settings.chunk_max_tokens, settings.chunk_overlap)
        validate_chunks(document, prepared, tokenizer, settings.chunk_max_tokens)
        documents.append({"source": document.metadata["source"], "document_id": doc_id,
                          "source_sha256": prepared[0].metadata["source_sha256"]})
        chunks.extend(prepared)
    manifest = {"manifest_version": 1, "documents": documents,
        "embedding_model": MODEL_ID, "dimension": DIMENSION, "metric": "cosine",
        "embedding_endpoint": settings.nebius_base_url.rstrip("/"),
        "embedding_protocol": EMBEDDING_PROTOCOL, "query_instruction": QUERY_INSTRUCTION,
        "chunk_config_id": chunks[0].metadata["chunking_config_id"],
        "index": settings.pinecone_index,
        "records": [{"id": chunk.metadata["chunk_id"],
                     "metadata": {**chunk.metadata, "chunk_text": chunk.page_content}}
                    for chunk in chunks]}
    version = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    namespace = f"{settings.pinecone_namespace}-{version[:16]}"
    if len(namespace) > 200:
        raise ValueError("Namespace prefix too long")
    manifest.update({"corpus_version": version, "namespace": namespace})
    for record in manifest["records"]:
        record["metadata"].update({"corpus_version": version, "embedding_model": MODEL_ID})
    return manifest


def verify_persisted(store, manifest, *, timeout=30):
    expected = {record["id"]: record["metadata"] for record in manifest["records"]}
    deadline = time.monotonic() + timeout
    while True:
        found = store.fetch(manifest["namespace"], list(expected))
        valid = all(item in found and found[item].get("metadata") == metadata
                    for item, metadata in expected.items())
        stats = store.stats()
        namespace_stats = stats.get("namespaces", {}).get(manifest["namespace"], {})
        count = namespace_stats.get("vectorCount", namespace_stats.get("vector_count", 0))
        if valid and count == len(expected):
            return count
        if count > len(expected):
            raise ServiceError("Namespace contains unexpected extra records; refusing activation")
        if time.monotonic() >= deadline:
            raise ServiceError("Records not yet fully visible; manifest not activated. Rerun ingest/check later.")
        time.sleep(2)


def ingest(settings, store, embeddings, *, manifest_path=MANIFEST_PATH):
    manifest = prepare_corpus(settings)
    store.ensure_index()  # Creation is a separate explicit command.
    records = [{**record, "values": embeddings.embed(record["metadata"]["chunk_text"])}
               for record in manifest["records"]]
    store.upsert(manifest["namespace"], records)
    count = verify_persisted(store, manifest)
    # Publish only after all remote IDs, text, metadata, and count are confirmed.
    atomic_json(manifest_path, manifest)
    return {"status": "persisted", "documents": len(manifest["documents"]),
            "records": count, "namespace": manifest["namespace"],
            "corpus_version": manifest["corpus_version"],
            "new_embeddings": embeddings.embedded_texts, "cache_hits": embeddings.cache_hits}


def active_manifest(settings, *, manifest_path=MANIFEST_PATH):
    if not manifest_path.is_file():
        raise ValueError("No active ingestion manifest. Ingest the corpus first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = prepare_corpus(settings)
    if (manifest.get("corpus_version") != current["corpus_version"]
            or manifest.get("namespace") != current["namespace"]
            or manifest.get("records") != current["records"]):
        raise ValueError("Corpus/configuration changed since ingestion; ingest the new version before querying")
    return manifest


def retrieve(settings, store, embeddings, question, *, top_k=4, manifest_path=MANIFEST_PATH):
    if not question.strip() or len(question) > 2000:
        raise ValueError("Question must contain 1-2000 characters")
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10")
    manifest = active_manifest(settings, manifest_path=manifest_path)
    store.ensure_index()
    vector = embeddings.embed(question, query=True)
    response = store.search(manifest["namespace"], vector, manifest["corpus_version"], top_k)
    expected = {record["id"]: record["metadata"] for record in manifest["records"]}
    evidence, seen = [], set()
    for match in response.get("matches", []):
        chunk_id = match["id"]
        if chunk_id not in expected or match.get("metadata") != expected[chunk_id]:
            raise ServiceError("Retrieved record does not match the active source manifest")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        evidence.append({"id": chunk_id, "score": match.get("score"),
                         "text": match["metadata"]["chunk_text"],
                         "metadata": {key: value for key, value in match["metadata"].items()
                                      if key != "chunk_text"}})
    return {"question": question, "namespace": manifest["namespace"],
            "corpus_version": manifest["corpus_version"], "evidence": evidence}
