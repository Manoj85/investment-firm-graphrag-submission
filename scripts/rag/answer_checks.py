"""Narrow, disclosed checks and verbatim graph evidence, never a truth oracle."""
from copy import deepcopy
import re

ANSWER_POLICY = "coverage-source-bridges-and-quote-spans-v2"
ROLE_PATTERNS = {
    "project lead": (r"\b(?:leads?|leader)\b", r"\b(?:leads?|leader)\b"),
    "executive sponsor": (r"\bsponsors?\b", r"\bsponsors?\b"),
    "skill or expertise": (r"\b(?:skills?|expertise)\b", r"\b(?:skills?|expertise)\b"),
    "rule maintainer": (r"\bmaintain(?:s|er|ers)?\b", r"\bmaintain(?:s|er|ers)?\b"),
    "decision approver": (r"\bapprov(?:ed|er|al)\b", r"\bapprov(?:ed|er|al)\b"),
    "organizational team": (r"\bteam\b", r"\b(?:team|member|belongs?)\b"),
}


def apply_answer_checks(result, retrieval, sources, question):
    output = deepcopy(result)
    original = deepcopy(result["answer"])
    answer = output["answer"]
    claims = result.get("structured_claims", {}).get("claims", [])
    texts = [claim["text"] for claim in claims]
    events, bridges = [], []
    # This extracts an already-generated clarification request; it neither picks
    # an identity nor treats unsourced role assertions as a clarification answer.
    if not claims and answer["status"] != "answered":
        sentences = re.split(r"(?<=[.?])\s+", answer["answer"])
        questions = [text for text in sentences if re.search(r"please clarify|which .+ (?:intended|mean)|do you mean", text, re.I)]
        if questions:
            answer.update(status="clarification", answer=" ".join(questions), citations=[])
            events.append("Aligned status with the explicit clarification request; removed uncited background assertions.")

    selected = {source["id"]: source for source in sources}
    # Supplement a missing bridge with its actual reviewed source excerpt. Do
    # not append a label to model prose or infer a new factual relationship.
    for path in retrieval.get("graph_selection", {}).get("paths", []):
        for step in path["steps"]:
            if step["relation"] not in {"REQUIRES_TOOL", "SELECTS_TOOL", "WORKED_ON", "DEPENDS_ON"}:
                continue
            if not all(re.search(r"(?<!\w)" + re.escape(step[key]) + r"(?!\w)", original["answer"], re.I)
                       for key in ["source_label", "target_label"]):
                continue
            for citation in step["evidence"]:
                source = selected.get(citation["chunk_id"])
                excerpt = citation["excerpt"]
                if (not source or source["citation"] in answer["citations"] or len(bridges) >= 3
                        or not 12 <= len(excerpt) <= 600 or excerpt not in source["text"]):
                    continue
                bridges.append({"edge_id": step["edge_id"], "relation": step["relation"],
                    "source_label": step["source_label"], "target_label": step["target_label"],
                    "citation": source["citation"], "quote": excerpt, "chunk_id": source["id"]})
                answer["answer"] += f"\n\nSource-backed relationship excerpt: {excerpt} [{source['citation']}]"
                answer["citations"].append(source["citation"])
                texts.append(excerpt)
    if bridges:
        events.append("Added separately labeled verbatim source excerpts for omitted graph relationships between named answer entities.")

    missing = []
    for role, (asked, stated) in ROLE_PATTERNS.items():
        candidates = texts
        if role == "organizational team":
            candidates = [text for text in texts if not re.search(r"\bProject\s+\w+\s+team\b", text)]
        if re.search(asked, question, re.I) and not any(re.search(stated, text, re.I) for text in candidates):
            missing.append(role)
    if missing and answer["status"] == "answered":
        answer["status"] = "insufficient_evidence"
        answer["answer"] += "\n\nThis answer does not establish the requested " + ", ".join(missing) + "."
        events.append("Downgraded answered status because explicit requested role wording is absent; no missing facts were fabricated.")
    output["answer_checks"] = {"policy": ANSWER_POLICY, "events": events,
        "missing_explicit_roles": missing, "source_backed_bridges": bridges,
        "limitation": "Lexical omission warnings and reviewed graph provenance, not semantic entailment or complete question understanding."}
    output["model_answer"] = original
    return output
