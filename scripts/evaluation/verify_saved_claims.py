"""Six saved-evidence verifier probes; default preflight, --run makes six calls."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json

from scripts.config.project_config import PROJECT_ROOT, load_settings
from scripts.corpus.corpus_pipeline import atomic_json
from scripts.evaluation.comparison import runtime_fingerprint
from scripts.rag.claim_verifier import ClaimVerifier, VERIFIER_VERSION, SCHEMA_VERIFIER_VERSION, VERIFIER_INSTRUCTIONS, prepare_review
from scripts.services.service_clients import ServiceError


SUITE_PATH = PROJECT_ROOT / "outputs/evaluation/claim-review-cases-v1.json"


def load_cases():
    suite = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    cases = suite["cases"]
    if len(cases) != 6 or len({case["id"] for case in cases}) != 6:
        raise ValueError("Expected six distinct bounded review cases")
    prepared = []
    for case in cases:
        path = (PROJECT_ROOT / case["receipt"]).resolve()
        if not path.is_relative_to((PROJECT_ROOT / "outputs/evaluation/runs").resolve()):
            raise ValueError("Review input must be a saved evaluation receipt")
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["receipt_sha256"]:
            raise ValueError("Saved review input changed")
        report = json.loads(path.read_text(encoding="utf-8"))["report"]
        payload = deepcopy(report["structured_claims"])
        for change in case.get("claim_changes", []):
            payload["claims"][change["claim_id"] - 1]["text"] = change["text"]
        prepare_review(report["question"], payload, report["sources"])
        prepared.append((case, {"question": report["question"], "payload": payload, "sources": report["sources"]}))
    return suite, prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--schema", action="store_true", help="Use versioned API JSON-schema experiment instead of v1 JSON-object mode")
    args = parser.parse_args()
    suite, prepared = load_cases()
    if not args.run:
        print(json.dumps({"status": "local review preflight passed; no API calls", "cases": len(prepared),
                          "verifier_version": SCHEMA_VERIFIER_VERSION if args.schema else VERIFIER_VERSION, "extra_calls_per_case": 1}, indent=2))
        return
    settings = load_settings()
    fingerprint = runtime_fingerprint()
    suite_hash = hashlib.sha256(SUITE_PATH.read_bytes()).hexdigest()
    now = datetime.now(timezone.utc)
    output = PROJECT_ROOT / "outputs/evaluation/runs" / ("verifier-" + now.strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    atomic_json(output / "run.json", {"started_utc": now.isoformat(), "verifier_version": SCHEMA_VERIFIER_VERSION if args.schema else VERIFIER_VERSION,
        "runtime_fingerprint": fingerprint, "suite_sha256": suite_hash, "model": settings.nebius_chat_model,
        "instructions_sha256": hashlib.sha256(VERIFIER_INSTRUCTIONS.encode()).hexdigest(),
        "temperature": 0, "max_completion_tokens": 1024, "content_retries": 0,
        "scope": "Six saved inputs; no Pinecone, embedding, regeneration or workflow integration. Local receipts; no LangSmith trace for this standalone probe.",
        "disclosure": suite["disclosure"]})
    summary = {"status": "running", "cases": []}
    verifier = ClaimVerifier(settings, schema=args.schema)
    print(f"Results: {output.relative_to(PROJECT_ROOT).as_posix()}", flush=True)
    try:
        for case, inputs in prepared:
            if runtime_fingerprint() != fingerprint or hashlib.sha256(SUITE_PATH.read_bytes()).hexdigest() != suite_hash:
                raise ValueError("Review code/input suite changed during collection")
            path = output / f"{case['id']}.json"
            snapshot = {"case": case, "input": inputs, "outcome": "running"}
            atomic_json(path, snapshot)
            try:
                # Expected decisions and mutations' explanations are not sent.
                record = verifier.review(inputs["question"], inputs["payload"], inputs["sources"])
            except (ServiceError, ValueError, OSError) as error:
                record = {"outcome": "error", "error": str(error)}
            snapshot.update(outcome=record["outcome"], result=record)
            decision = record.get("review", {}).get("decision")
            snapshot["matches_expected_decision"] = decision == case["expected_decision"] if decision else None
            atomic_json(path, snapshot)
            summary["cases"].append({"id": case["id"], "outcome": record["outcome"],
                "decision": decision, "expected": case["expected_decision"],
                "matches_expected_decision": snapshot["matches_expected_decision"]})
            atomic_json(output / "summary.json", summary)
            print(f"{case['id']}: {decision or record['outcome']}", flush=True)
            if record.get("error", "").startswith("HTTP "):
                # Unsupported schema/model/access is not improved by sending
                # the same rejected request shape five more times.
                summary["stop_reason"] = record["error"]
                break
    finally:
        verifier.close()
    summary["status"] = "complete" if len(summary["cases"]) == len(prepared) else "stopped"
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
