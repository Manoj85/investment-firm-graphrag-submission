"""Small explicit REST adapters for the first persisted retrieval milestone."""

import hashlib
import json
import math
from pathlib import Path
import re
import ssl
import time
from typing import Any

import httpx

from project_config import PROJECT_ROOT, Settings
from qwen_tokenizer import MODEL_ID
from verify_nebius import validate_settings


DIMENSION = 4096  # Measured in the successful Qwen diagnostic, not guessed.
API_VERSION = "2025-10"
QUERY_INSTRUCTION = "Given a question about organizational knowledge, retrieve passages that answer it."
EMBEDDING_PROTOCOL = "qwen-document-raw-query-instruct-v1"
CONTEXTUAL_EMBEDDING_PROTOCOL = "qwen-document-title-section-query-instruct-v2"
# The query input did not change. Retain its prior cache identity so this
# experiment compares passage representations using the exact same query vector.
QUERY_CACHE_PROTOCOL = "qwen-document-raw-query-instruct-v1"
DOCUMENT_INPUT_TEMPLATE = "Document: {title}\nSection: {section}\n\n{chunk_text}"


def document_embedding_input(metadata, *, context=True):
    """Enrich embedding text without modifying the source slice used in citations."""
    return DOCUMENT_INPUT_TEMPLATE.format(**metadata) if context else metadata["chunk_text"]


def embedding_protocol(settings):
    return CONTEXTUAL_EMBEDDING_PROTOCOL if settings.embedding_document_context else EMBEDDING_PROTOCOL


def embedding_identity(settings, text, *, query=False):
    prepared = f"Instruct: {QUERY_INSTRUCTION}\nQuery:{text}" if query else text
    protocol = QUERY_CACHE_PROTOCOL if query else embedding_protocol(settings)
    return [settings.nebius_base_url.rstrip("/"), MODEL_ID, protocol, prepared]


class ServiceError(RuntimeError):
    """Sanitized errors suitable for console output; never raw request headers."""


def tls_context():
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_default_certs()
    return context


def request_json(client, method, url, *, allow_missing=False, **kwargs):
    try:
        response = client.request(method, url, **kwargs)
        if response.status_code == 404 and allow_missing:
            return None
        if not response.is_success:
            hints = {401: "authentication rejected", 403: "access denied",
                     402: "billing access required", 404: "resource/model not found",
                     409: "resource conflict", 429: "quota/rate limit reached"}
            raise ServiceError(f"HTTP {response.status_code}: "
                               f"{hints.get(response.status_code, 'provider request rejected')}")
        return response.json()
    except httpx.RequestError:
        raise ServiceError("Network/TLS/timeout failure; no automatic retry") from None
    except ValueError:
        raise ServiceError("Provider returned invalid JSON") from None


def validate_vector(vector):
    if (not isinstance(vector, list) or len(vector) != DIMENSION
            or any(type(x) not in (float, int) or not math.isfinite(x) for x in vector)
            or not any(vector)):
        raise ServiceError(f"Expected a finite, nonzero {DIMENSION}-dimensional vector")
    return vector


class NebiusEmbeddings:
    def __init__(self, settings: Settings, *, transport=None, cache_dir: Path | None = None):
        base_url = validate_settings(settings, "embedding")
        if settings.nebius_embedding_model != MODEL_ID:
            raise ValueError("This preparation protocol requires Qwen/Qwen3-Embedding-8B")
        self.settings = settings
        self.cache_dir = cache_dir or PROJECT_ROOT / "data" / "cache" / "embeddings"
        self.client = httpx.Client(base_url=base_url, transport=transport,
            headers={"Authorization": f"Bearer {settings.nebius_api_key}"},
            verify=tls_context(), timeout=60, follow_redirects=False)
        self.cache_hits = 0
        self.embedded_texts = 0

    def close(self):
        self.client.close()

    def embed(self, text: str, *, query=False):
        if not text.strip():
            raise ValueError("Embedding input is empty")
        # Qwen's upstream retrieval examples instruct queries, not document text.
        # We send this string explicitly; no SDK silently inserts another prefix.
        identity = embedding_identity(self.settings, text, query=query)
        prepared = identity[-1]
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                vector = validate_vector(cached["vector"])
                if cached["identity"] != identity:
                    raise ServiceError("Embedding cache identity mismatch")
                self.cache_hits += 1
                return vector
            except (ValueError, KeyError, TypeError):
                raise ServiceError("Invalid embedding cache; rebuild that cache entry") from None
        result = request_json(self.client, "POST", "embeddings", json={
            "model": MODEL_ID, "input": prepared, "encoding_format": "float"})
        try:
            if len(result["data"]) != 1:
                raise ServiceError("Expected one embedding per passage")
            vector = validate_vector(result["data"][0]["embedding"])
        except (KeyError, TypeError, IndexError):
            raise ServiceError("Unexpected embedding response") from None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"identity": identity, "vector": vector}) + "\n",
                              encoding="utf-8")
        self.embedded_texts += 1
        return vector


class PineconeStore:
    def __init__(self, settings: Settings, *, transport=None):
        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is missing; save it locally in .env")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,43}[a-z0-9]|[a-z0-9]", settings.pinecone_index):
            raise ValueError("PINECONE_INDEX must be 1-45 lowercase letters, digits, or internal hyphens")
        if not settings.pinecone_namespace.strip():
            raise ValueError("PINECONE_NAMESPACE must be a nonempty project prefix")
        self.settings, self.host = settings, None
        self.client = httpx.Client(base_url="https://api.pinecone.io/", transport=transport,
            headers={"Api-Key": settings.pinecone_api_key, "X-Pinecone-Api-Version": API_VERSION},
            verify=tls_context(), timeout=60, follow_redirects=False)

    def close(self):
        self.client.close()

    def describe(self):
        return request_json(self.client, "GET", f"indexes/{self.settings.pinecone_index}", allow_missing=True)

    def ensure_index(self, *, create=False) -> dict[str, Any]:
        """Return a checked index description, or raise; never return None."""
        description = self.describe()
        if description is None:
            if not create:
                raise ServiceError("Index missing. Run prepare_pinecone.py --create explicitly.")
            description = request_json(self.client, "POST", "indexes", json={
                "name": self.settings.pinecone_index, "vector_type": "dense",
                "dimension": DIMENSION, "metric": "cosine",
                "spec": {"serverless": {"cloud": self.settings.pinecone_cloud,
                                         "region": self.settings.pinecone_region}},
                "deletion_protection": "enabled",
                "tags": {"project": "investment-firm-graphrag"}})
        # The HTTP helper can return None for a missing resource. Even creation
        # must produce an object before callers can safely use description[...].
        if not isinstance(description, dict):
            raise ServiceError("Pinecone returned an invalid index description")
        self.validate_index(description)
        # Readiness polls have a short bound and do not create more resources.
        deadline = time.monotonic() + 30
        while not description.get("status", {}).get("ready"):
            if time.monotonic() >= deadline:
                raise ServiceError("Index still initializing; rerun the read-only setup check later")
            time.sleep(2)
            description = self.describe()
            if description is None:
                raise ServiceError("Index disappeared during readiness check")
            if not isinstance(description, dict):
                raise ServiceError("Pinecone returned an invalid index description")
            self.validate_index(description)
        host = description.get("host", "")
        if not re.fullmatch(r"[A-Za-z0-9.-]+\.pinecone\.io", host):
            raise ServiceError("Unexpected index host; refusing to forward the API key")
        self.host = host
        return description

    @staticmethod
    def validate_index(description):
        if (description.get("dimension") != DIMENSION or description.get("metric") != "cosine"
                or description.get("vector_type", "dense") != "dense"
                or "serverless" not in description.get("spec", {}) or description.get("embed")):
            raise ServiceError("Existing index incompatible: require serverless dense, cosine, dimension 4096, external vectors. Nothing changed.")

    def data_request(self, method, path, **kwargs):
        if self.host is None:
            raise ServiceError("Index has not been checked")
        return request_json(self.client, method, f"https://{self.host}/{path}", **kwargs)

    def upsert(self, namespace, records):
        for start in range(0, len(records), 50):
            batch = records[start:start + 50]
            for record in batch:
                validate_vector(record["values"])
            result = self.data_request("POST", "vectors/upsert", json={"namespace": namespace, "vectors": batch})
            if result.get("upsertedCount") != len(batch):
                raise ServiceError("Upsert did not confirm the complete batch")

    def fetch(self, namespace, ids):
        found = {}
        for start in range(0, len(ids), 50):
            params = [("namespace", namespace)] + [("ids", item) for item in ids[start:start + 50]]
            result = self.data_request("GET", "vectors/fetch", params=params)
            found.update(result.get("vectors", {}))
        return found

    def stats(self):
        return self.data_request("POST", "describe_index_stats", json={})

    def search(self, namespace, vector, corpus_version, top_k=4) -> dict[str, Any]:
        """Return a query response object, or raise before callers use .get()."""
        validate_vector(vector)
        result = self.data_request("POST", "query", json={"namespace": namespace,
            "vector": vector, "topK": top_k, "includeMetadata": True, "includeValues": False,
            "filter": {"corpus_version": {"$eq": corpus_version}}})
        if not isinstance(result, dict):
            raise ServiceError("Pinecone returned an invalid query response object")
        return result
