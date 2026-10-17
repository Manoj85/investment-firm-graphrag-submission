"""Bounded local path inspection; paths retain original directed source facts."""

from collections import deque
from copy import deepcopy

import networkx as nx


def bounded_paths(network: nx.MultiDiGraph, start: str, target: str, *,
                  max_hops: int = 3, max_examined_edges: int = 40,
                  max_paths: int = 10) -> dict:
    """Explore simple paths breadth first, including explicitly marked reverse steps.

    These are connectivity candidates, not inferred facts or ranked answers.
    The shared edge-examination budget bounds work across the whole search.
    """
    if start not in network or target not in network:
        raise ValueError("Unknown entity ID; use exact stored IDs")
    if start == target:
        raise ValueError("Choose distinct start and target entities")
    for name, value, ceiling in (("max_hops", max_hops, 3),
                                 ("max_examined_edges", max_examined_edges, 40),
                                 ("max_paths", max_paths, 10)):
        if type(value) is not int or not 1 <= value <= ceiling:
            raise ValueError(f"{name} must be an integer from 1 to {ceiling}")

    queue = deque([(start, [start], [])])
    paths, examined, stopped = [], 0, None
    while queue and stopped is None:
        current, visited, steps = queue.popleft()
        # Incoming navigation does not invert APPROVED_BY into a new fact.
        # Keep source/target intact and record the direction we walked separately.
        adjacent = [(u, v, key, attrs, "forward")
                    for u, v, key, attrs in network.out_edges(current, keys=True, data=True)]
        adjacent += [(u, v, key, attrs, "reverse")
                     for u, v, key, attrs in network.in_edges(current, keys=True, data=True)]
        adjacent.sort(key=lambda row: (row[2], row[4]))
        for source, destination, edge_id, attrs, direction in adjacent:
            if examined >= max_examined_edges:
                stopped = "edge_budget"
                break
            examined += 1
            neighbor = destination if direction == "forward" else source
            if neighbor in visited:
                continue
            step = {"edge_id": edge_id, "source": source, "target": destination,
                    "source_label": network.nodes[source]["label"],
                    "target_label": network.nodes[destination]["label"],
                    "relation": attrs["relation"], "traversal": direction,
                    "from_entity": current, "to_entity": neighbor,
                    "evidence": deepcopy(attrs["evidence"])}
            extended = steps + [step]
            if neighbor == target:
                paths.append({"nodes": visited + [neighbor], "steps": extended})
                if len(paths) >= max_paths:
                    stopped = "path_limit"
                    break
            elif len(extended) < max_hops:
                queue.append((neighbor, visited + [neighbor], extended))

    # Multiple edges can cite one passage. Fetch that passage only once later.
    chunk_ids = list(dict.fromkeys(citation["chunk_id"]
                     for path in paths for step in path["steps"]
                     for citation in step["evidence"]))
    return {"start": start, "target": target, "max_hops": max_hops,
            "max_examined_edges": max_examined_edges, "max_paths": max_paths,
            "examined_edges": examined, "stop_reason": stopped or "search_complete_within_hop_limit",
            "truncated": stopped is not None, "paths": paths,
            "evidence_chunk_ids": chunk_ids}
