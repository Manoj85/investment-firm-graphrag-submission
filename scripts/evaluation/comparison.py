"""Collect frozen comparisons without answer-key access in the answering service."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter, time

from scripts.config.project_config import PROJECT_ROOT
from scripts.corpus.corpus_pipeline import atomic_json
from scripts.rag.rag_workflow import EVIDENCE_BUDGET, PROMPT_VERSION, validate_answer


BENCHMARK_PATH = PROJECT_ROOT / "outputs/evaluation/benchmark-v1.json"


def validate_benchmark(benchmark, corpus_version):
    questions = benchmark.get("questions", [])
    if benchmark.get("status") != "frozen_for_comparison" or len(questions) != 10:
        raise ValueError("Expected a frozen ten-question benchmark")
    if benchmark.get("corpus_version") != corpus_version:
        raise ValueError("Benchmark and active corpus versions differ")
    if len({question["id"] for question in questions}) != 10:
        raise ValueError("Benchmark question IDs must be unique")
    digest = hashlib.sha256(json.dumps(questions, sort_keys=True).encode()).hexdigest()
    if digest != benchmark.get("question_sha256"):
        raise ValueError("Frozen question/expectation hash differs")
    for question in questions:
        if not isinstance(question["question"], str) or not 1 <= len(question["question"].strip()) <= 2000:
            raise ValueError("Invalid benchmark question text")


def runtime_fingerprint():
    """No credentials: hash runtime code, dependency pins, graph and source files."""
    paths = [*sorted((PROJECT_ROOT / "scripts").rglob("*.py")),
             *sorted((PROJECT_ROOT / "outputs/corpus").glob("*.md")),
             PROJECT_ROOT / "outputs/graph/knowledge-graph.json", PROJECT_ROOT / "requirements.txt"]
    inventory = {path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in paths}
    return hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()


def diagnostics(question, report):
    """Structural/reference coverage only; never substitute for semantic grading."""
    sources = report.get("sources", [])
    supplied = {(item["metadata"]["document_id"], item["metadata"]["section"]) for item in sources}
    expected = {(item["document_id"], item["section"]) for item in question["required_sources"]}
    answer = report["answer"]
    try:
        validate_answer(json.dumps(answer), sources)
        valid = True
    except Exception:
        valid = False
    return {"status_matches_expectation": answer["status"] == question["expected_status"],
            "citation_structure_valid": valid,
            "reference_sections_supplied": [list(item) for item in sorted(supplied & expected)],
            "reference_sections_not_supplied": [list(item) for item in sorted(expected - supplied)],
            "semantic_grade": None,
            "note": "Reference sections are diagnostic targets, not the only possible valid evidence. Manual source review required."}


def collect_comparison(benchmark, output_dir: Path, settings, invoke, metadata, *, guard=None, progress=print):
    """One attempt per question/mode; checkpoint successes and errors durably.

    Alternating order balances which mode benefits from warmed query caches.
    Interrupted/running records are never silently replayed by this function.
    """
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(output_dir / "run.json", metadata)
    summary = {"status": "running", "run_id": output_dir.name, "attempts": [],
               "planned_attempts": 20, "completed_attempts": 0, "successful_calls": 0, "errors": 0}
    atomic_json(output_dir / "summary.json", summary)
    rejected_path = PROJECT_ROOT / "work/answers/rejected-answer.json"
    for number, question in enumerate(benchmark["questions"]):
        modes = ("vector", "graph") if number % 2 == 0 else ("graph", "vector")
        for mode in modes:
            if guard:
                guard()
            name = f"{question['id']}-{mode}.json"
            attempt = {"question_id": question["id"], "question": question["question"], "mode": mode,
                       "started_utc": datetime.now(timezone.utc).isoformat(), "outcome": "running"}
            atomic_json(output_dir / name, attempt)
            began, wall_start = time(), perf_counter()
            try:
                # Only the question text/settings reach production answering.
                report = invoke(question["question"], mode=mode, settings=settings, prompt_version=PROMPT_VERSION)
                attempt.update(outcome="success", report=report, diagnostics=diagnostics(question, report))
                summary["successful_calls"] += 1
            except Exception as error:
                attempt.update(outcome="error", error_type=type(error).__name__,
                               error=str(error) if type(error).__name__ in {"ServiceError", "ValueError", "OSError"}
                               else "Unexpected execution failure; inspect the recorded exception type.")
                # Save a fresh matching rejection before the next call overwrites
                # the legacy diagnostic path. Never attach a prior question's text.
                if rejected_path.exists() and rejected_path.stat().st_mtime >= began:
                    rejected = json.loads(rejected_path.read_text(encoding="utf-8-sig"))
                    if rejected.get("question") == question["question"]:
                        attempt["rejected_response"] = rejected
                summary["errors"] += 1
            attempt.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                           wall_seconds=round(perf_counter() - wall_start, 3))
            atomic_json(output_dir / name, attempt)
            summary["attempts"].append({"question_id": question["id"], "mode": mode,
                                        "outcome": attempt["outcome"], "file": name})
            summary["completed_attempts"] += 1
            atomic_json(output_dir / "summary.json", summary)
            status = attempt.get("report", {}).get("answer", {}).get("status", attempt["outcome"])
            progress(f"{summary['completed_attempts']:02}/20 {question['id']} {mode}: {status}")
    summary.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(output_dir / "summary.json", summary)
    return summary
