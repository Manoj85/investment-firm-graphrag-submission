"""Pure presentation helpers: no API calls or mutation of saved receipts."""

import json


def path_diagram(paths):
    """Graphviz uses original fact arrows, even when traversal was reversed."""
    nodes, edges = {}, {}
    for path in paths:
        for step in path["steps"]:
            nodes[step["source"]] = step["source_label"]
            nodes[step["target"]] = step["target_label"]
            edges[step["edge_id"]] = step
    colors = {"person": "#e2edff", "decision": "#d4f2ed", "tool": "#fff0cc",
              "project": "#e8e4ff", "team": "#e8edf3", "skill": "#fce6ed"}
    lines = ['digraph evidence {', 'rankdir=LR; bgcolor="transparent"; pad=0.25;',
             'node [shape=box style="rounded,filled" fontname="Arial" fontsize=12 color="#cbd5e1" margin="0.18,0.12"];',
             'edge [fontname="Arial" fontsize=10 color="#64748b" fontcolor="#475569"];']
    for node_id, label in nodes.items():
        color = colors.get(node_id.split(":")[0], "#edf2f6")
        lines.append(f'{json.dumps(node_id)} [label={json.dumps(label)} fillcolor="{color}"];')
    for step in edges.values():
        lines.append(f'{json.dumps(step["source"])} -> {json.dumps(step["target"])} '
                     f'[label={json.dumps(step["relation"].replace("_", " "))}];')
    return "\n".join(lines + ['}'])


def citation_usage(report, chunk_id):
    for source in report.get("sources", []):
        if source["id"] == chunk_id:
            label = source["citation"]
            return label, label in report["answer"]["citations"]
    return None, False
