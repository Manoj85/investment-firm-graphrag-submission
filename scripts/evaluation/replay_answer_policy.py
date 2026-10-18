"""Replay deterministic answer checks on saved model outputs without API calls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts.config.project_config import PROJECT_ROOT
from scripts.corpus.corpus_pipeline import atomic_json
from scripts.rag.answer_checks import ANSWER_POLICY, apply_answer_checks
from scripts.rag.structured_claims import normalize_quote_punctuation, validate_and_render_claims
from scripts.rag.rag_workflow import validate_answer
from scripts.services.service_clients import ServiceError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", help="Existing run directory under outputs/evaluation/runs")
    args = parser.parse_args()
    source = (PROJECT_ROOT / args.source_run).resolve()
    if not source.is_relative_to((PROJECT_ROOT / "outputs/evaluation/runs").resolve()):
        raise ValueError("Replay only accepts saved evaluation runs")
    inputs = sorted(source.glob("Q*.json"))
    if len(inputs) != 20:
        raise ValueError("Expected twenty saved attempts")
    output = source.parent / ("policy-replay-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    atomic_json(output / "run.json", {"kind": "offline deterministic policy replay, not fresh model evaluation",
        "source_run": source.relative_to(PROJECT_ROOT).as_posix(), "answer_policy": ANSWER_POLICY,
        "additional_provider_calls": 0, "additional_tokens": 0,
        "limitations": "Fixed previously generated outputs, including failures; no sampling or new trace. Does not estimate future model reliability."})
    outcomes = []
    for path in inputs:
        original = json.loads(path.read_text(encoding="utf-8"))
        record = {"question_id": original["question_id"], "question": original["question"], "mode": original["mode"],
            "source_file": path.relative_to(PROJECT_ROOT).as_posix(), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "kind": "offline_policy_replay", "additional_provider_calls": 0}
        saved = original.get("report", original.get("rejected_response", {}))
        try:
            if not saved:
                raise ServiceError("No saved model evidence available to replay")
            if saved.get("structured_claims") is None and saved.get("answer", {}).get("status") == "clarification":
                result = {"answer": saved["answer"]}
            else:
                raw = saved.get("content") or json.dumps(saved["structured_claims"])
                normalized, changes = normalize_quote_punctuation(raw, saved["sources"])
                answer, payload = validate_and_render_claims(normalized, saved["sources"])
                result = apply_answer_checks({"answer": answer, "structured_claims": payload}, saved, saved["sources"], original["question"])
                if changes:
                    result["quote_normalizations"] = changes
            validate_answer(json.dumps(result["answer"]), saved.get("sources", []))
            record.update(outcome="success", report={**result, "sources": saved.get("sources", []),
                "mode": original["mode"], "question": original["question"],
                "tracing": {"status": "not_requested", "note": "Offline replay; original trace is in source receipt."}})
        except (ServiceError, ValueError, KeyError, TypeError) as error:
            record.update(outcome="error", error=str(error))
        atomic_json(output / path.name, record)
        outcomes.append({"file": path.name, "outcome": record["outcome"]})
    summary = {"kind": "offline_policy_replay", "additional_provider_calls": 0,
        "attempts": outcomes, "accepted": sum(item["outcome"] == "success" for item in outcomes)}
    atomic_json(output / "summary.json", summary)
    print(output.relative_to(PROJECT_ROOT).as_posix())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
