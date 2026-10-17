"""Run a new question through persisted retrieval and cited generation."""

import argparse
from datetime import datetime, timezone
import json
from time import perf_counter

from scripts.corpus.corpus_pipeline import atomic_json
from scripts.config.project_config import PROJECT_ROOT, load_settings
from scripts.rag.rag_workflow import EVIDENCE_BUDGET, PROMPT_VERSION, PROMPTS, GroundedChat, application
from scripts.services.service_clients import NebiusEmbeddings, PineconeStore, ServiceError
from scripts.services.tracing_setup import TracingSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--expand-neighbors", action="store_true",
                        help="Experiment: fetch at most four adjacent source passages after vector search")
    parser.add_argument("--prompt-version", choices=tuple(PROMPTS), default=PROMPT_VERSION,
                        help="Versioned answer instructions; v2 remains the default")
    args = parser.parse_args()
    store = embeddings = chat = tracing = None
    try:
        settings = load_settings()
        store = PineconeStore(settings)
        embeddings = NebiusEmbeddings(settings)
        chat = GroundedChat(settings, prompt_version=args.prompt_version)
        mode = "vector_neighbors" if args.expand_neighbors else "vector"
        graph = application(settings, store, embeddings, chat,
                            expand_source_neighbors=args.expand_neighbors)
        tracing = TracingSession(settings)
        started = perf_counter()
        print("Retrieving persisted evidence, then generating a cited answer...", flush=True)
        config = tracing.invoke_config()
        config["metadata"] = {"prompt_version": chat.prompt_version,
            "retrieval_mode": mode,
            "chat_model": settings.nebius_chat_model, "embedding_model": settings.nebius_embedding_model}
        result = graph.invoke({"question": args.question}, config=config)
        elapsed = round(perf_counter() - started, 3)
        report = {"mode": mode, "question": args.question,
            "answer": result["answer"], "sources": result["sources"],
            "namespace": result["retrieval"]["namespace"],
            "corpus_version": result["retrieval"]["corpus_version"],
            "prompt_version": chat.prompt_version, "evidence_budget": EVIDENCE_BUDGET,
            "evidence_budget_method": "pinned Qwen content tokens",
            "requested_chat_model": settings.nebius_chat_model,
            "response_model": result["response_model"], "usage": result["usage"],
            "elapsed_seconds": elapsed, "tracing": tracing.finish()}
        if args.expand_neighbors:
            report["seed_evidence"] = result["retrieval"]["seed_evidence"]
            report["neighbor_expansion"] = result["retrieval"]["neighbor_expansion"]
            report["neighbor_expansion"]["selected_passages"] = len(result["sources"])
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = PROJECT_ROOT / "work" / "answers" / f"answer-{stamp}.json"
        atomic_json(path, report)
        print(f"Status: {report['answer']['status']}")
        print(report["answer"]["answer"])
        for source in report["sources"]:
            if source["citation"] in report["answer"]["citations"]:
                meta = source["metadata"]
                print(f"[{source['citation']}] {meta['document_id']} | {meta['section']} | {meta['source']}")
        print(f"Elapsed: {report['elapsed_seconds']} seconds")
        print(f"Full evidence/usage receipt: {path.relative_to(PROJECT_ROOT).as_posix()}")
        print(f"LangSmith tracing: {report['tracing']['status']}")
        return 0
    except (ValueError, ServiceError, OSError) as error:
        print(f"Answering failed: {error}")
        return 1
    finally:
        for client in (tracing, chat, embeddings, store):
            if client is not None:
                client.close()


if __name__ == "__main__":
    raise SystemExit(main())
