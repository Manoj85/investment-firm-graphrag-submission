"""Exercise persistence/version boundaries with HTTP mocks, not real accounts."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from corpus_pipeline import active_manifest, ingest, prepare_corpus, retrieve, verify_persisted
from project_config import PROJECT_ROOT, load_settings
from qwen_tokenizer import TOKENIZER_PATH
from service_clients import (DIMENSION, NebiusEmbeddings, PineconeStore,
                             QUERY_INSTRUCTION, ServiceError)


def settings():
    return load_settings(PROJECT_ROOT / "work" / "absent-test.env", {
        "NEBIUS_API_KEY": "invented-nebius-test-key",
        "NEBIUS_BASE_URL": "https://api.tokenfactory.nebius.com/v1/",
        "NEBIUS_CHAT_MODEL": "test-chat", "NEBIUS_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-8B",
        "PINECONE_API_KEY": "invented-pinecone-test-key", "PINECONE_INDEX": "test-index",
        "PINECONE_NAMESPACE": "test"})


class FakePinecone:
    def __init__(self):
        self.records, self.requests = {}, []
        self.description = {"name": "test-index", "dimension": DIMENSION,
            "metric": "cosine", "vector_type": "dense", "host": "test.svc.pinecone.io",
            "status": {"ready": True}, "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}}}
        self.exists = True

    def handle(self, request):
        payload = json.loads(request.content) if request.content else {}
        self.requests.append((request.method, request.url.path, payload))
        if request.url.path == "/indexes/test-index":
            return httpx.Response(200 if self.exists else 404, json=self.description if self.exists else {})
        if request.url.path == "/indexes":
            self.exists = True
            return httpx.Response(201, json=self.description)
        if request.url.path == "/vectors/upsert":
            records = self.records.setdefault(payload["namespace"], {})
            records.update({record["id"]: record for record in payload["vectors"]})
            return httpx.Response(200, json={"upsertedCount": len(payload["vectors"])})
        if request.url.path == "/vectors/fetch":
            records = self.records.get(request.url.params["namespace"], {})
            return httpx.Response(200, json={"vectors": {key: records[key]
                for key in request.url.params.get_list("ids") if key in records}})
        if request.url.path == "/describe_index_stats":
            return httpx.Response(200, json={"namespaces": {ns: {"vectorCount": len(records)}
                for ns, records in self.records.items()}})
        if request.url.path == "/query":
            records = self.records.get(payload["namespace"], {})
            version = payload["filter"]["corpus_version"]["$eq"]
            matches = [{"id": key, "score": 0.8, "metadata": record["metadata"]}
                for key, record in records.items() if record["metadata"]["corpus_version"] == version]
            return httpx.Response(200, json={"matches": matches[:payload["topK"]]})
        raise AssertionError(f"Unexpected path {request.url.path}")


@unittest.skipUnless(TOKENIZER_PATH.is_file(), "Prepare the pinned tokenizer first")
class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name)
        self.settings = settings()
        self.remote = FakePinecone()
        self.store = PineconeStore(self.settings, transport=httpx.MockTransport(self.remote.handle))
        self.inputs = []
        def embedding_response(request):
            payload = json.loads(request.content)
            self.inputs.append(payload["input"])
            return httpx.Response(200, json={"data": [{"embedding": [0.1] * DIMENSION}]})
        self.embeddings = NebiusEmbeddings(self.settings,
            transport=httpx.MockTransport(embedding_response), cache_dir=self.path / "cache")

    def tearDown(self):
        self.store.close()
        self.embeddings.close()
        self.folder.cleanup()

    def test_repeated_ingestion_keeps_count_and_reuses_vectors(self):
        manifest_path = self.path / "active.json"
        first = ingest(self.settings, self.store, self.embeddings, manifest_path=manifest_path)
        count_before = len(self.inputs)
        second = ingest(self.settings, self.store, self.embeddings, manifest_path=manifest_path)
        self.assertEqual(first["records"], second["records"])
        self.assertEqual(first["namespace"], second["namespace"])
        self.assertEqual(len(self.inputs), count_before)
        self.assertEqual(len(self.remote.records[first["namespace"]]), first["records"])

    def test_configuration_changes_isolate_namespace_and_block_stale_manifest(self):
        path = self.path / "active.json"
        ingest(self.settings, self.store, self.embeddings, manifest_path=path)
        changed = replace(self.settings, chunk_max_tokens=128)
        self.assertNotEqual(prepare_corpus(self.settings)["namespace"], prepare_corpus(changed)["namespace"])
        with self.assertRaisesRegex(ValueError, "changed since ingestion"):
            active_manifest(changed, manifest_path=path)

    def test_document_boundary_rejects_docs_and_answer_keys(self):
        for directory in ("docs", "outputs/evaluation", "outputs"):
            with self.assertRaisesRegex(ValueError, "must be outputs/corpus"):
                prepare_corpus(replace(self.settings, corpus_dir=directory))

    def test_query_uses_instructions_and_remote_namespace_with_provenance(self):
        path = self.path / "active.json"
        ingest(self.settings, self.store, self.embeddings, manifest_path=path)
        result = retrieve(self.settings, self.store, self.embeddings, "Who leads Atlas?", manifest_path=path)
        self.assertTrue(result["evidence"])
        self.assertEqual(self.inputs[-1], f"Instruct: {QUERY_INSTRUCTION}\nQuery:Who leads Atlas?")
        self.assertTrue(all(not item.startswith("Instruct:") for item in self.inputs[:-1]))
        query = [payload for _, endpoint, payload in self.remote.requests if endpoint == "/query"][-1]
        self.assertEqual(query["namespace"], result["namespace"])
        self.assertFalse(query["includeValues"])

    def test_wrong_dimension_does_not_modify_existing_index(self):
        self.remote.description["dimension"] = 1536
        with self.assertRaisesRegex(ServiceError, "incompatible"):
            self.store.ensure_index(create=True)
        self.assertFalse(any(method != "GET" for method, _, _ in self.remote.requests))

    def test_missing_index_created_only_explicitly(self):
        self.remote.exists = False
        with self.assertRaisesRegex(ServiceError, "Index missing"):
            self.store.ensure_index()
        self.store.ensure_index(create=True)
        creates = [payload for method, endpoint, payload in self.remote.requests if method == "POST" and endpoint == "/indexes"]
        self.assertEqual(len(creates), 1)
        self.assertEqual(creates[0]["dimension"], DIMENSION)
        self.assertEqual(creates[0]["deletion_protection"], "enabled")

    def test_untrusted_host_is_not_used(self):
        self.remote.description["host"] = "example.com"
        with self.assertRaisesRegex(ServiceError, "Unexpected index host"):
            self.store.ensure_index()

    def test_incomplete_remote_records_cannot_be_published(self):
        manifest = prepare_corpus(self.settings)
        self.store.ensure_index()
        with self.assertRaisesRegex(ServiceError, "not yet fully visible"):
            verify_persisted(self.store, manifest, timeout=0)

    def test_tampered_remote_metadata_is_rejected(self):
        path = self.path / "active.json"
        result = ingest(self.settings, self.store, self.embeddings, manifest_path=path)
        first = next(iter(self.remote.records[result["namespace"]].values()))
        first["metadata"]["chunk_text"] = "unexpected substituted text"
        with self.assertRaisesRegex(ServiceError, "does not match"):
            retrieve(self.settings, self.store, self.embeddings, "Who leads Atlas?", manifest_path=path)


if __name__ == "__main__":
    unittest.main()
