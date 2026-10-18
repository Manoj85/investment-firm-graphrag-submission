"""Explicitly run benchmark v1 once per mode; no retries or automated fact grades."""

from datetime import datetime, timezone
import argparse
import hashlib
import json
import subprocess

from scripts.config.project_config import PROJECT_ROOT, load_settings
from scripts.corpus.corpus_pipeline import active_manifest
from scripts.evaluation.comparison import BENCHMARK_PATH, collect_comparison, runtime_fingerprint, validate_benchmark
from scripts.graph.knowledge_graph import load_graph, validate_graph
from scripts.rag.answer_service import run_question
from scripts.rag.rag_workflow import (EVIDENCE_BUDGET, PROMPT_VERSION, PROMPTS, INSTRUCTIONS,
    COMPLETION_LIMITS, generation_response_format, SCHEMA_CLAIMS_PROMPT_VERSION)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Make the 20 live answering attempts; otherwise local preflight only")
    parser.add_argument("--prompt-version", choices=tuple(PROMPTS), default=PROMPT_VERSION)
    args = parser.parse_args()
    settings = load_settings()
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8-sig"))
    manifest = active_manifest(settings)
    validate_benchmark(benchmark, manifest["corpus_version"])
    graph = load_graph()
    validate_graph(graph, manifest)
    fingerprint = runtime_fingerprint()
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    now = datetime.now(timezone.utc)
    output_dir = PROJECT_ROOT / "outputs/evaluation/runs" / ("revised-" + now.strftime("%Y%m%dT%H%M%S%fZ"))
    metadata = {"started_utc": now.isoformat(), "git_revision": revision,
        "runtime_fingerprint": fingerprint, "benchmark_version": benchmark["benchmark_version"],
        "benchmark_file_sha256": hashlib.sha256(BENCHMARK_PATH.read_bytes()).hexdigest(),
        "question_sha256": benchmark["question_sha256"], "corpus_version": manifest["corpus_version"],
        "namespace": manifest["namespace"], "graph_version": graph["graph_version"],
        "chat_model": settings.nebius_chat_model, "embedding_model": settings.nebius_embedding_model,
        "prompt_version": args.prompt_version, "system_instructions_sha256": hashlib.sha256(INSTRUCTIONS[args.prompt_version].encode()).hexdigest(),
        "temperature": 0, "max_completion_tokens": COMPLETION_LIMITS[args.prompt_version], "evidence_budget": EVIDENCE_BUDGET,
        "response_format_type": generation_response_format(args.prompt_version, [{"citation": "S1"}])["type"],
        "schema_labels": "bound to actual packed source labels per request; no answer keys" if args.prompt_version == SCHEMA_CLAIMS_PROMPT_VERSION else None,
        "vector_top_k": 4, "order": "odd question: vector then graph; even: graph then vector",
        "answer_content_retries": 0, "cache_policy": "existing shared query embedding cache; no cold-cache performance claim",
        "disclosure": benchmark["disclosure"]}

    if not args.run:
        print(json.dumps({"status": "local preflight passed; no API calls", "planned_attempts": 20,
                          "settings": metadata}, indent=2))
        return

    def guard():
        if runtime_fingerprint() != fingerprint or hashlib.sha256(BENCHMARK_PATH.read_bytes()).hexdigest() != metadata["benchmark_file_sha256"]:
            raise ValueError("Runtime/corpus/graph/benchmark changed during collection; stopping without replay")

    print("Collecting 20 revised attempts; every outcome is preserved.", flush=True)
    print(f"Results: {output_dir.relative_to(PROJECT_ROOT).as_posix()}", flush=True)
    summary = collect_comparison(benchmark, output_dir, settings, run_question, metadata,
                                 guard=guard, progress=lambda message: print(message, flush=True), prompt_version=args.prompt_version)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
