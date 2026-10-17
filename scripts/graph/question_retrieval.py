"""Small, inspectable question-to-graph candidate selector; no answer keys."""

from collections import deque
from copy import deepcopy
import re


# This is a disclosed vocabulary, not a general language parser. Rules name
# schema relations, never answer people, document IDs or benchmark questions.
INTENTS = {
    "APPROVED_BY": r"\bapprov(?:e|ed|es|al|er|ers)\b",
    "PREPARED_BY": r"\bprepar(?:e|ed|es|er|ation)\b",
    "LEADS_PROJECT": r"\b(?:leads?|leader)\b",
    "LEADS_TEAM": r"\b(?:leads?|leader)\b",
    "SPONSORS_PROJECT": r"\bsponsors?\b",
    "WORKED_ON": r"\b(?:participants?|works?|worked)\b",
    "HAS_SKILL": r"\b(?:skills?|expertise)\b",
    "MEMBER_OF": r"\b(?:team|members?|belongs?)\b",
    "MAINTAINS_RULES": r"\bmaintain(?:s|er|ers)?\b",
    "OWNED_BY": r"\b(?:owns?|owners?|ownership)\b",
    "USES_TOOL": r"\b(?:uses?|tools?)\b",
    "DEPENDS_ON": r"\b(?:upstream|depend(?:s|ency|encies)?)\b",
}
CONNECTORS = {
    "APPROVED_BY": {"ABOUT_PROJECT", "SELECTS_TOOL", "REQUIRES_TOOL", "REGISTERS_SOURCE"},
    "PREPARED_BY": {"ABOUT_PROJECT", "SELECTS_TOOL", "REQUIRES_TOOL", "REGISTERS_SOURCE"},
    "LEADS_PROJECT": {"USES_TOOL", "USES_SOURCE", "DEPENDS_ON", "ABOUT_PROJECT"},
    "LEADS_TEAM": {"OWNED_BY", "MEMBER_OF"},
    "SPONSORS_PROJECT": {"USES_TOOL", "DEPENDS_ON", "ABOUT_PROJECT"},
    "WORKED_ON": {"HAS_SKILL", "USES_TOOL"},
    "HAS_SKILL": {"WORKED_ON", "LEADS_PROJECT", "DEPENDS_ON"},
    "MEMBER_OF": {"WORKED_ON", "HAS_SKILL"},
    "MAINTAINS_RULES": {"ABOUT_PROJECT", "REQUIRES_TOOL", "USES_TOOL"},
    "OWNED_BY": {"USES_TOOL", "USES_SOURCE"},
    "USES_TOOL": {"DEPENDS_ON", "ABOUT_PROJECT"},
    "DEPENDS_ON": {"USES_SOURCE"},
}


def question_mentions(network, question: str) -> list[dict]:
    """Match whole stored names/aliases; longest overlapping name wins.

    Offsets refer to whitespace-normalized question text. An exact shared alias
    keeps every identity candidate. Alex Morgan must not also match bare Alex.
    """
    normalized = " ".join(question.split())
    spans = {}
    for node_id, attrs in network.nodes(data=True):
        for name in [node_id, attrs["label"], *attrs["aliases"]]:
            pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in name.split()) + r"(?!\w)"
            for match in re.finditer(pattern, normalized, re.IGNORECASE):
                spans.setdefault(match.span(), set()).add(node_id)
    chosen = []
    for (start, end), ids in sorted(spans.items(), key=lambda item: (-(item[0][1]-item[0][0]), item[0])):
        if any(start < row["end"] and end > row["start"] for row in chosen):
            continue
        chosen.append({"text": normalized[start:end], "start": start, "end": end,
                       "status": "resolved" if len(ids) == 1 else "ambiguous",
                       "candidates": [{"id": node_id, "label": network.nodes[node_id]["label"],
                                       "type": network.nodes[node_id]["type"]} for node_id in sorted(ids)]})
    return sorted(chosen, key=lambda row: row["start"])


def select_question_paths(network, question: str) -> dict:
    """Find bounded candidate paths ending in a requested relation.

    Candidate relevance is deliberately limited: there is no entailment check,
    free-form semantic planning, automatic clarification answer or generation.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be nonempty text")
    mentions = question_mentions(network, question)
    intents = [relation for relation, pattern in INTENTS.items() if re.search(pattern, question, re.IGNORECASE)]
    seeds = list(dict.fromkeys(row["candidates"][0]["id"] for row in mentions if row["status"] == "resolved"))
    report = {"question": question, "selector_version": "schema-lexical-v1",
              "graph_version": network.graph["graph_version"], "corpus_version": network.graph["corpus_version"],
              "mentions": mentions, "requested_relations": intents, "seed_entities": seeds,
              "max_hops": 3, "max_examined_edges": 40, "max_paths": 10,
              "examined_edges": 0, "truncated": False, "paths": [], "evidence_chunk_ids": []}
    if any(row["status"] == "ambiguous" for row in mentions):
        report.update(status="clarification_required", stop_reason="ambiguous_entity")
        return report
    if not seeds or not intents or len(seeds) > 4 or re.search(r"\b(?:not|never|except)\b|n't\b", question, re.IGNORECASE):
        reason = ("no_named_entities" if not seeds else "no_relationship_intent" if not intents
                  else "too_many_entities" if len(seeds) > 4 else "unsupported_negation")
        report.update(status="unsupported", stop_reason=reason)
        return report

    # A dependent project's own lead is not its upstream project's lead.
    # Bind this supported compound pattern in order, rather than mixing all
    # requested relation families into one undirected neighborhood search.
    ordered = (len(seeds) == 1 and network.nodes[seeds[0]]["type"] == "Project"
               and {"DEPENDS_ON", "LEADS_PROJECT", "HAS_SKILL"} <= set(intents))
    stages = [("DEPENDS_ON", "forward", "Project"),
              ("LEADS_PROJECT", "reverse", "Person"),
              ("HAS_SKILL", "forward", "Skill")]
    if ordered:
        report.update(selector_version="schema-lexical-v2", path_pattern=[
            {"relation": relation, "traversal": direction, "target_type": kind}
            for relation, direction, kind in stages])
    connectors = set().union(*(CONNECTORS[relation] for relation in intents))
    allowed = connectors | set(intents)
    queue = deque((seed, [seed], []) for seed in seeds)
    seen_paths, paths, stopped = set(), [], None
    while queue and stopped is None:
        current, nodes, steps = queue.popleft()
        adjacent = [(u, v, key, attrs, "forward") for u, v, key, attrs in network.out_edges(current, keys=True, data=True)]
        adjacent += [(u, v, key, attrs, "reverse") for u, v, key, attrs in network.in_edges(current, keys=True, data=True)]
        if ordered:
            relation, direction, kind = stages[len(steps)]
            # Count only typed candidates for this constrained plan. Unrelated
            # edges cannot consume its budget; the same 40-candidate cap holds.
            adjacent = [row for row in adjacent if row[3]["relation"] == relation
                        and row[4] == direction
                        and network.nodes[row[1] if direction == "forward" else row[0]]["type"] == kind]
        # Inspect desired facts before connectors, with a stable tie breaker.
        adjacent.sort(key=lambda row: (row[3]["relation"] not in intents, row[2], row[4]))
        for source, target, edge_id, attrs, direction in adjacent:
            if report["examined_edges"] >= 40:
                stopped = "edge_budget"
                break
            report["examined_edges"] += 1
            neighbor = target if direction == "forward" else source
            relation = attrs["relation"]
            if relation not in allowed or neighbor in nodes:
                continue
            step = {"edge_id": edge_id, "source": source, "target": target,
                    "source_label": network.nodes[source]["label"], "target_label": network.nodes[target]["label"],
                    "relation": relation, "traversal": direction, "from_entity": current,
                    "to_entity": neighbor, "evidence": deepcopy(attrs["evidence"])}
            extended, visited = steps + [step], nodes + [neighbor]
            key = tuple((item["edge_id"], item["traversal"]) for item in extended)
            if relation in intents and key not in seen_paths:
                seen_paths.add(key)
                paths.append({"nodes": visited, "steps": extended,
                              "matched_seed_count": len(set(visited) & set(seeds)),
                              "terminal_relation": relation})
                if len(paths) == 10:
                    stopped = "path_limit"
                    break
            if (ordered or relation in connectors) and len(extended) < 3:
                queue.append((neighbor, visited, extended))
    paths.sort(key=lambda path: (-path["matched_seed_count"], len(path["steps"]),
                                tuple(step["edge_id"] for step in path["steps"])))
    report.update(status="candidates_found" if paths else "no_candidate_paths", paths=paths,
                  truncated=stopped is not None, stop_reason=stopped or "search_complete_within_hop_limit")
    report["evidence_chunk_ids"] = list(dict.fromkeys(citation["chunk_id"] for path in paths
                                        for step in path["steps"] for citation in step["evidence"]))
    return report
