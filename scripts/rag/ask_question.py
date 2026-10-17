"""Run a new question through persisted retrieval and cited generation."""

import argparse

from scripts.rag.answer_service import run_question
from scripts.rag.rag_workflow import PROMPT_VERSION, PROMPTS
from scripts.services.service_clients import ServiceError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--mode", choices=("vector", "graph"), default="vector",
                        help="Graph mode adds question-selected source passages to vector evidence")
    parser.add_argument("--expand-neighbors", action="store_true",
                        help="Experiment: fetch at most four adjacent source passages after vector search")
    parser.add_argument("--prompt-version", choices=tuple(PROMPTS), default=PROMPT_VERSION,
                        help="Versioned answer instructions; v2 remains the default")
    args = parser.parse_args()
    if args.mode == "graph" and args.expand_neighbors:
        parser.error("Use graph mode or the neighbor experiment separately")
    try:
        print("Retrieving persisted evidence, then generating a cited answer...", flush=True)
        report = run_question(args.question, mode=args.mode,
                              expand_source_neighbors=args.expand_neighbors,
                              prompt_version=args.prompt_version)
        print(f"Status: {report['answer']['status']}")
        print(report["answer"]["answer"])
        for source in report["sources"]:
            if source["citation"] in report["answer"]["citations"]:
                meta = source["metadata"]
                print(f"[{source['citation']}] {meta['document_id']} | {meta['section']} | {meta['source']}")
        print(f"Elapsed: {report['elapsed_seconds']} seconds")
        print(f"Full evidence/usage receipt: {report['receipt_path']}")
        print(f"LangSmith tracing: {report['tracing']['status']}")
        return 0
    except (ValueError, ServiceError, OSError) as error:
        print(f"Answering failed: {error}")
        return 1



if __name__ == "__main__":
    raise SystemExit(main())
