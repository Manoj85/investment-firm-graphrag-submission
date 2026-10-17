"""Shared answering lifecycle and receipt persistence, with fake providers."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.rag.answer_service import run_question
from scripts.services.service_clients import ServiceError


class AnswerServiceTests(unittest.TestCase):
    def providers(self):
        settings = Mock(nebius_chat_model="chat", nebius_embedding_model="embedding")
        chat = Mock(prompt_version="grounded-answer-v2")
        trace = Mock()
        trace.invoke_config.return_value = {}
        trace.finish.return_value = {"status": "disabled"}
        return settings, Mock(), Mock(), chat, trace

    def test_shared_result_is_saved_and_all_clients_close(self):
        settings, store, embeddings, chat, trace = self.providers()
        workflow = Mock()
        workflow.invoke.return_value = {"answer": {"status": "answered", "answer": "Fact [S1]", "citations": ["S1"]},
            "sources": [], "retrieval": {"namespace": "test", "corpus_version": "v1", "packing": {}},
            "usage": {}, "response_model": "chat", "structured_claims": {"claims": [{"text": "Fact", "evidence": []}]}}
        with patch("scripts.rag.answer_service.PineconeStore", return_value=store), \
             patch("scripts.rag.answer_service.NebiusEmbeddings", return_value=embeddings), \
             patch("scripts.rag.answer_service.GroundedChat", return_value=chat), \
             patch("scripts.rag.answer_service.TracingSession", return_value=trace), \
             patch("scripts.rag.answer_service.application", return_value=workflow), \
             patch("scripts.rag.answer_service.atomic_json") as save:
            report = run_question("A question?", settings=settings)
        self.assertEqual(report["mode"], "vector")
        self.assertTrue(report["receipt_path"].startswith("work/answers/"))
        self.assertEqual(save.call_args.args[1], report)
        self.assertEqual(report["structured_claims"], workflow.invoke.return_value["structured_claims"])
        for client in (store, embeddings, chat, trace):
            client.close.assert_called_once()

    def test_failure_closes_clients_without_saving_a_success_receipt(self):
        settings, store, embeddings, chat, trace = self.providers()
        workflow = Mock()
        workflow.invoke.side_effect = ServiceError("Test failure")
        with patch("scripts.rag.answer_service.PineconeStore", return_value=store), \
             patch("scripts.rag.answer_service.NebiusEmbeddings", return_value=embeddings), \
             patch("scripts.rag.answer_service.GroundedChat", return_value=chat), \
             patch("scripts.rag.answer_service.TracingSession", return_value=trace), \
             patch("scripts.rag.answer_service.application", return_value=workflow), \
             patch("scripts.rag.answer_service.atomic_json") as save:
            with self.assertRaises(ServiceError):
                run_question("A question?", settings=settings)
        save.assert_not_called()
        for client in (store, embeddings, chat, trace):
            client.close.assert_called_once()
