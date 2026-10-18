"""Complete one supported typed proof using fetched, packed source excerpts."""
from copy import deepcopy
import re

from scripts.rag.coverage_excerpts import question_parts

APPROVAL_VERSION = "grounded-answer-v12-approval-proof"
APPROVAL_POLICY = "unique-reviewed-approval-proof-v1"
PARTIAL_MESSAGE = "The selected evidence does not establish every requested part of this question."


def complete_approval_proof(result, retrieval, sources, question):
    """Bind maintainer -> tool <- decision -> approver, for one named project.

    This is intentionally not a general entailment checker. The reviewed graph
    supplies the semantics; every displayed word still comes from packed sources.
    No model call, benchmark lookup, entity-name rule or broad status override.
    """
    output = deepcopy(result)
    audit = {"policy": APPROVAL_POLICY, "status": "not_applicable",
        "limitation": "A narrow reviewed-graph proof, not general question understanding or source-truth verification."}
    output["approval_completion"] = audit
    # Recognize only this two-part question family. An unrecognized additional
    # request must not inherit an answered status from the narrower proof.
    family = re.fullmatch(r"\s*Who maintains (?P<subject>[^?]+?) required by (?P<scope>[^?]+?),?\s+and who approved (?:that|the) decision\??\s*", question, re.I)
    if (not family or re.search(r"\b(?:who|which|what|when|where|why|how)\b",
            family["subject"] + " " + family["scope"], re.I)
            or set(question_parts(question)) != {"maintainer", "approval", "tool_relationship"}
            or not re.search(r"\brequir(?:ed|es?)\b", question, re.I)
            or result["answer"]["status"] == "clarification"):
        return output
    selection = retrieval.get("graph_selection", {})
    projects = {c["id"] for mention in selection.get("mentions", [])
                if mention["status"] == "resolved" for c in mention["candidates"] if c["type"] == "Project"}
    if len(projects) != 1:
        audit.update(status="ambiguous_project" if projects else "no_project_anchor")
        return output
    steps = {step["edge_id"]: step for path in selection.get("paths", []) for step in path["steps"]}
    packed = {s["id"]: s for s in sources}

    def evidence(step):
        for ref in step["evidence"]:
            source = packed.get(ref["chunk_id"])
            quote = ref["excerpt"]
            if source and 12 <= len(quote) <= 600 and quote in source["text"]:
                return {"edge_id": step["edge_id"], "relation": step["relation"],
                    "chunk_id": ref["chunk_id"], "citation": source["citation"], "quote": quote}
        return None

    def named(label):
        return bool(re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", result["answer"]["answer"], re.I))

    # Anchor the proof in a maintainer relationship already quoted, rather than
    # merely finding two names somewhere in the answer or graph neighborhood.
    candidates = {}
    for maintainer in steps.values():
        if maintainer["relation"] != "MAINTAINS_RULES":
            continue
        support = evidence(maintainer)
        if (not support or support["quote"] not in result["answer"]["answer"]
                or not named(maintainer["source_label"]) or not named(maintainer["target_label"])):
            continue
        for required in steps.values():
            if required["relation"] != "REQUIRES_TOOL" or required["target"] != maintainer["target"]:
                continue
            about = [s for s in steps.values() if s["relation"] == "ABOUT_PROJECT"
                     and s["source"] == required["source"] and s["target"] in projects]
            approvals = [s for s in steps.values() if s["relation"] == "APPROVED_BY"
                         and s["source"] == required["source"]]
            for project in about:
                for approval in approvals:
                    key = (required["source"], maintainer["source"], maintainer["target"],
                           project["target"], approval["target"])
                    candidates[key] = [maintainer, project, required, approval]
    # Do not resolve multiple decisions/approvers by whichever happened to fit
    # in the source budget. Ambiguity is checked before evidence availability.
    if len(candidates) != 1:
        audit.update(status="ambiguous" if candidates else "no_bound_proof", candidate_count=len(candidates))
        return output
    key, proof = next(iter(candidates.items()))
    refs = [evidence(step) for step in proof]
    if any(ref is None for ref in refs):
        audit.update(status="missing_packed_evidence", candidate_count=1)
        return output

    before = deepcopy(output["answer"])
    answer = output["answer"]
    # Remove only application-owned warnings made obsolete by this full proof.
    # Original selection/status and the pre-completion answer stay in the receipt.
    answer["answer"] = answer["answer"].replace(PARTIAL_MESSAGE, "").strip()
    missing = output.get("answer_checks", {}).get("missing_explicit_roles", [])
    warning = "This answer does not establish the requested " + ", ".join(missing) + "."
    if missing:
        answer["answer"] = answer["answer"].replace(warning, "").strip()
    added = []
    for ref in refs:
        # The graph excerpt may include a heading absent from a selected quote.
        # If the body is already quoted, retain its provenance without repetition.
        body = re.sub(r"^#+[^\n]*\n\s*", "", ref["quote"]).strip()
        if (ref["citation"] not in answer["citations"]
                or (ref["quote"] not in answer["answer"] and body not in answer["answer"])):
            answer["answer"] += f"\n\nSource-backed approval evidence: {ref['quote']} [{ref['citation']}]"
            added.append(ref)
        if ref["citation"] not in answer["citations"]:
            answer["citations"].append(ref["citation"])
    answer["status"] = "answered"
    if "answer_checks" in output:
        output["answer_checks"]["missing_explicit_roles"] = []
        output["answer_checks"]["events"].append("Completed the requested maintainer/required-decision/approval proof from one reviewed decision and its packed sources.")
    audit.update(status="completed", candidate_count=1, decision_id=key[0], maintainer_id=key[1],
        tool_id=key[2], project_id=key[3], approver_id=key[4], evidence=refs,
        added_excerpts=added, pre_completion_answer=before,
        status_rule="All parts of the supported question pattern have a unique typed proof with exact packed evidence.")
    return output
