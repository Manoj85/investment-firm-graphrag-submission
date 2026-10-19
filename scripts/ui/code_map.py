"""Display only the packaged Graphify snapshot; never scan or call a model."""

import json

import streamlit as st

from scripts.config.project_config import PROJECT_ROOT


ASSET_DIR = PROJECT_ROOT / "assets" / "code-map"


def load_code_map():
    """Read application-owned assets, never uploaded or generated answer HTML."""
    metadata = json.loads((ASSET_DIR / "snapshot.json").read_text(encoding="utf-8"))
    html = (ASSET_DIR / "graph.html").read_text(encoding="utf-8")
    return metadata, html


def render_code_map():
    st.subheader("Explore the application code")
    st.write("Search for run_question, select it, then click a connected function or component.")
    st.caption("Developer code map · Separate from the organizational graph used for answers. "
               "Opening this tab makes no model or Pinecone request.")
    try:
        metadata, html = load_code_map()
    except (OSError, ValueError):
        st.warning("The packaged code-map snapshot is unavailable. Answering remains available in Ask & compare.")
        return
    st.caption(f"Graphify {metadata['graphify_version']} · {metadata['files']} Python files · "
               f"{metadata['nodes']} nodes · {metadata['edges']} edges · "
               f"Snapshot {metadata['built_date']}. Regenerate after code changes.")
    # Only a reviewed, repository-owned HTML asset is embedded. The viewer and
    # its pinned JavaScript are packaged locally; arbitrary HTML is not accepted.
    st.iframe(html, height=680, tab_index=0, alt="Interactive Graphify code map")
    st.caption("Links come from static code analysis. Inferred links need source verification. "
               "Clustering was skipped, so zero communities is expected.")
    st.download_button("Download code map", html, file_name="investment-firm-code-map.html",
                       mime="text/html", key="download-code-map", on_click="ignore")
