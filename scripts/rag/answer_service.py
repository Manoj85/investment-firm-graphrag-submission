"""One explicit answering action, shared by CLI, UI and later evaluation."""

from datetime import datetime, timezone
from time import perf_counter

from scripts.config.project_config import PROJECT_ROOT, load_settings
from scripts.corpus.corpus_pipeline import atomic_json
from scripts.rag.rag_workflow import EVIDENCE_BUDGET, PROMPT_VERSION, GroundedChat, application, SCHEMA_CLAIMS_PROMPT_VERSION
from scripts.services.service_clients import NebiusEmbeddings, PineconeStore
from scripts.services.tracing_setup import TracingSession
from scripts.rag.answer_checks import ANSWER_POLICY
from scripts.rag.excerpt_selection import EXCERPT_VERSION
from scripts.rag.coverage_excerpts import COVERAGE_VERSION, COVERAGE_POLICY, COVERAGE_VERSIONS


def run_question(question, *, mode="vector", settings=None,
                 expand_source_neighbors=False, prompt_version=PROMPT_VERSION):
    """Make a single requested run; do not cache answers or retry their content."""
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise ValueError("Question must contain 1-2000 characters")
    if mode not in {"vector", "graph"} or (mode == "graph" and expand_source_neighbors):
        raise ValueError("Choose vector or graph mode; keep neighbor expansion separate")
    store = embeddings = chat = tracing = None
    try:
        settings = settings or load_settings()
        store = PineconeStore(settings)
        embeddings = NebiusEmbeddings(settings)
        chat = GroundedChat(settings, prompt_version=prompt_version)
        selected_mode = "vector_neighbors" if expand_source_neighbors else mode
        workflow = application(settings, store, embeddings, chat, mode=mode,
                               expand_source_neighbors=expand_source_neighbors)
        tracing = TracingSession(settings)
        config = tracing.invoke_config()
        config["run_name"] = "organizational-graph-rag" if mode == "graph" else "organizational-vector-rag"
        config["tags"] = [selected_mode, "graph-integration" if mode == "graph" else "friday-baseline"]
        config["metadata"] = {"prompt_version": chat.prompt_version, "retrieval_mode": selected_mode,
            "answer_policy": (COVERAGE_POLICY if chat.prompt_version in COVERAGE_VERSIONS else ANSWER_POLICY if chat.prompt_version == SCHEMA_CLAIMS_PROMPT_VERSION
                              else "source-owned-excerpts-with-bridges-v1" if chat.prompt_version == EXCERPT_VERSION else "none"),
            "chat_model": settings.nebius_chat_model, "embedding_model": settings.nebius_embedding_model}
        started = perf_counter()
        result = workflow.invoke({"question": question}, config=config)
        elapsed = round(perf_counter() - started, 3)
        report = {"mode": selected_mode, "question": question,
            "answer": result["answer"], "sources": result["sources"],
            "namespace": result["retrieval"]["namespace"], "corpus_version": result["retrieval"]["corpus_version"],
            "prompt_version": chat.prompt_version, "evidence_budget": EVIDENCE_BUDGET,
            "evidence_budget_method": "pinned Qwen content tokens",
            "requested_chat_model": settings.nebius_chat_model, "response_model": result["response_model"],
            "usage": result["usage"], "elapsed_seconds": elapsed, "tracing": tracing.finish(),
            "packing": result["retrieval"]["packing"]}
        if mode == "graph":
            for key in ("graph_selection", "graph_retrieval"):
                report[key] = result["retrieval"][key]
            report["seed_evidence"] = result["retrieval"].get("seed_evidence", [])
        if "structured_claims" in result:
            report["structured_claims"] = result["structured_claims"]
        if "generation_payload" in result:
            report["generation_payload"] = result["generation_payload"]
        for key in ("answer_checks", "model_answer", "quote_normalizations", "excerpt_selection", "selected_excerpt_answer", "coverage_selection"):
            if key in result:
                report[key] = result[key]
        if expand_source_neighbors:
            report["seed_evidence"] = result["retrieval"]["seed_evidence"]
            report["neighbor_expansion"] = {**result["retrieval"]["neighbor_expansion"],
                                             "selected_passages": len(result["sources"])}
        now = datetime.now(timezone.utc)
        report["timestamp_utc"] = now.isoformat()
        path = PROJECT_ROOT / "work" / "answers" / f"answer-{now.strftime('%Y%m%dT%H%M%S%fZ')}.json"
        report["receipt_path"] = path.relative_to(PROJECT_ROOT).as_posix()
        atomic_json(path, report)
        return report
    finally:
        for client in (tracing, chat, embeddings, store):
            if client is not None:
                client.close()
