"""A local learning workspace; submitting a form is the only answering trigger."""

from datetime import datetime, timezone
import json
from uuid import uuid4

import streamlit as st

from scripts.config.project_config import PROJECT_ROOT, load_settings
from scripts.rag.answer_service import run_question
from scripts.rag.structured_claims import CLAIMS_PROMPT_VERSION
from scripts.rag.rag_workflow import BASELINE_PROMPT_VERSION
from scripts.services.service_clients import ServiceError
from scripts.ui.presentation import citation_usage, path_diagram


SAMPLES = {
    "Decision approval": "Who approved DEC-002?",
    "Connect a tool to its approver": "Who approved the selection of Beacon Evidence Hub for Project Atlas?",
    "Project leadership": "Who leads Project Atlas, and who is its executive sponsor?",
    "An ambiguous name": "What is Alex responsible for?",
    "A question outside the sources": "What annual investment return does Project Atlas guarantee?",
}
MODES = {"Vector": ["vector"], "Graph-assisted": ["graph"], "Compare both": ["vector", "graph"]}
MODE_NAMES = {"vector": "Vector", "graph": "Graph-assisted"}
ANSWER_FORMATS = {"Current default": None,
                  "Structured claims (experimental)": CLAIMS_PROMPT_VERSION,
                  "Original cited answer": BASELINE_PROMPT_VERSION}
CSS = """
<style>
.block-container {max-width: 1280px; padding-top: 2.2rem; padding-bottom: 3rem;}
h1 {font-size: 2.5rem !important; letter-spacing: -0.055em; color:#142b40;}
h2,h3 {letter-spacing: -0.025em;}
.eyebrow {font-size:.72rem; letter-spacing:.17em; font-weight:700; color:#0f766e; margin-bottom:.45rem;}
.intro {font-size:1.05rem; color:#526578; max-width:760px; margin-bottom:1.5rem;}
[data-testid="stSidebar"] {border-right:1px solid #e0e7ed;}
[data-testid="stMetricValue"] {font-size:1.5rem;}
[data-testid="stForm"] {background:white; border:1px solid #dce5ec; border-radius:16px; padding:1.25rem;}
[data-testid="stTabs"] {margin-top:.8rem;}
.status {display:inline-block; font-size:.75rem; font-weight:650; padding:.25rem .65rem; border-radius:30px; margin-bottom:1rem;}
.answered {background:#dff3ed;color:#0c6759;} .clarification {background:#fff0cf;color:#805800;}
.insufficient_evidence {background:#edf1f6;color:#526578;}
</style>
"""


def local_snapshot():
    """Read only non-secret local inventories, not service health or answer keys."""
    result = {"documents": 0, "passages": 0, "entities": 0, "relationships": 0, "version": None}
    try:
        manifest = json.loads((PROJECT_ROOT / "data/indexes/active-manifest.json").read_text(encoding="utf-8-sig"))
        result.update(documents=len(manifest["documents"]), passages=len(manifest["records"]),
                      version=manifest["corpus_version"], namespace=manifest["namespace"])
        graph = json.loads((PROJECT_ROOT / "outputs/graph/knowledge-graph.json").read_text(encoding="utf-8-sig"))
        result.update(entities=len(graph["nodes"]), relationships=len(graph["relationships"]))
    except (OSError, ValueError, KeyError, TypeError):
        result["incomplete"] = True
    return result


def execute_request(question, choice, *, prompt_version=None):
    """Keep a successful comparison side even when the other side fails."""
    entry = {"id": uuid4().hex, "question": question, "choice": choice,
             "created_utc": datetime.now(timezone.utc).isoformat(), "reports": {}, "errors": {}}
    for mode in MODES[choice]:
        try:
            options = {"prompt_version": prompt_version} if prompt_version else {}
            entry["reports"][mode] = run_question(question, mode=mode, **options)
        except (ValueError, ServiceError, OSError) as error:
            entry["errors"][mode] = str(error)
        except Exception as error:
            # Never dump credentials, request headers or arbitrary client objects.
            entry["errors"][mode] = f"Unexpected {type(error).__name__}; the run did not complete."
    return entry


def choose_sample():
    selected = st.session_state.sample_question
    if selected in SAMPLES:
        st.session_state.question_input = SAMPLES[selected]


def render_answer(report):
    answer = report["answer"]
    status = answer["status"]
    style = status if status in {"answered", "clarification", "insufficient_evidence"} else "insufficient_evidence"
    label = {"answered": "Answered", "clarification": "Clarification needed",
             "insufficient_evidence": "Insufficient evidence"}.get(status, "Review result")
    st.markdown(f'<span class="status {style}">{label}</span>', unsafe_allow_html=True)
    st.markdown(answer["answer"])
    if answer["citations"]:
        st.caption("Cited sources: " + ", ".join(answer["citations"]) + " · Open Sources to read the supporting text.")
    metrics = st.columns(3)
    metrics[0].metric("Answer time", f"{report['elapsed_seconds']:.2f}s")
    metrics[1].metric("Selected passages", len(report["sources"]))
    tokens = (report.get("usage") or {}).get("total_tokens")
    metrics[2].metric("Chat tokens", tokens if tokens is not None else "No model call")
    st.caption("An answered status is not an automatic factual-quality grade. Check the cited passages.")


def render_sources(report):
    if report.get("generation_payload"):
        with st.expander("Question parts and supporting claims"):
            st.caption("Model-declared coverage; inspect whether the linked claims actually establish each requested relationship.")
            for part in report["generation_payload"]["requirements"]:
                st.write(part["part"])
                st.caption("Claims: " + ", ".join(map(str, part["claim_ids"])) if part["claim_ids"] else "Not established in retrieved evidence")
    if report.get("structured_claims"):
        with st.expander("Claims and their exact source excerpts"):
            st.caption("Excerpts match the cited passages. Check whether they support each claim and cover the question.")
            for number, claim in enumerate(report["structured_claims"]["claims"], 1):
                st.write(f"{number}. {claim['text']}")
                for evidence in claim["evidence"]:
                    st.caption("[" + evidence["source"] + "]")
                    st.text(evidence["quote"])
    if not report["sources"]:
        st.info("No passages were sent to the model for this result.")
    for source in report["sources"]:
        meta = source["metadata"]
        cited = source["citation"] in report["answer"]["citations"]
        title = f"[{source['citation']}] {meta['document_id']} · {meta['section']} · " + ("Cited" if cited else "Context only")
        with st.expander(title, expanded=cited):
            st.markdown(source["text"])
            st.caption("Retrieved by: " + " + ".join(source.get("retrieval_origins", ["vector"])))
            st.caption(meta.get("source", ""))
            with st.expander("Source identity"):
                st.json({"chunk_id": source["id"], "document_id": meta["document_id"],
                         "section": meta["section"], "version": meta.get("version"),
                         "source_sha256": meta.get("source_sha256")})
    packing = report.get("packing", {})
    if packing.get("dropped_ids"):
        st.warning(f"{len(packing['dropped_ids'])} candidate passages did not fit the evidence budget. See Run details.")


def render_paths(report):
    if report["mode"] != "graph":
        st.info("Vector mode retrieves passages by semantic similarity; it does not traverse graph paths.")
        return
    selection = report.get("graph_selection", {})
    paths = selection.get("paths", [])
    retrieval = report.get("graph_retrieval", {})
    if selection.get("status") == "clarification_required":
        st.warning("A shared name matched more than one entity. No answer person was silently chosen.")
        st.json(selection.get("mentions", []))
        return
    if not paths:
        st.info("No graph paths selected. Vector evidence was used when available. Reason: " +
                str(retrieval.get("fallback_reason") or selection.get("stop_reason", "not available")))
        return
    st.caption("Candidate paths, not a proof of every claim. Arrows show the original fact direction.")
    st.graphviz_chart(path_diagram(paths), width="stretch")
    if selection.get("truncated"):
        st.warning("Search stopped at a limit: " + str(selection.get("stop_reason")))
    for index, path in enumerate(paths, 1):
        with st.expander(f"Path {index} · {len(path['steps'])} relationship(s)", expanded=index == 1):
            for step in path["steps"]:
                st.markdown(f"**{step['source_label']} → {step['relation']} → {step['target_label']}**")
                st.caption("Walked " + step["traversal"] + " · " + step["edge_id"])
                for citation in step["evidence"]:
                    label, cited = citation_usage(report, citation["chunk_id"])
                    state = (f"{label} · cited" if cited else f"{label} · context only") if label else "Not in selected answer context"
                    st.caption(f"{citation['document_id']} · {citation['section']} · {state}")
                    st.markdown(citation["excerpt"])


def render_details(report, key):
    trace = report.get("tracing", {})
    st.write("Trace status: " + trace.get("status", "unknown"))
    url = trace.get("url", "")
    if trace.get("status") == "verified" and url.startswith("https://"):
        st.link_button("Open LangSmith trace", url)
    st.caption("Answer time excludes trace setup/readback. Chat token usage does not include query embedding cost.")
    st.json({"model": report.get("response_model"), "prompt": report["prompt_version"],
             "namespace": report["namespace"], "corpus_version": report["corpus_version"],
             "packing": report.get("packing"), "graph_retrieval": report.get("graph_retrieval")})
    if report.get("receipt_path"):
        st.caption("Saved locally: " + report["receipt_path"])
    payload = json.dumps(report, indent=2)
    st.download_button("Download full receipt", payload, file_name=f"{key}-{report['mode']}.json",
                       mime="application/json", key=f"download-{key}-{report['mode']}", on_click="ignore")
    with st.expander("Full JSON receipt"):
        st.json(report)


def render_app():
    st.set_page_config(page_title="Aster Vale · Knowledge Lab", page_icon="◈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("run_history", [])
    snapshot = local_snapshot()
    with st.sidebar:
        st.markdown("### ◈ Aster Vale")
        st.caption("KNOWLEDGE LAB")
        st.divider()
        st.markdown("**One question. Two ways to find evidence.**")
        st.caption("Vector finds similar passages. Graph-assisted adds source-backed relationships.")
        with st.expander("Models & corpus"):
            try:
                settings = load_settings()
                st.caption("Chat: " + settings.nebius_chat_model)
                st.caption("Embeddings: " + settings.nebius_embedding_model)
                st.caption("Settings presence: " + ", ".join(name for name, ready in settings.service_readiness().items() if ready))
                st.caption("Presence does not verify API access. Services are checked when you run a question.")
            except ValueError as error:
                st.warning(str(error))
            st.caption("Corpus: " + str(snapshot["version"] or "No active manifest"))
        st.caption("Fictional organizational data · Local testing workspace")

    st.markdown('<div class="eyebrow">SOURCE-BACKED ORGANIZATIONAL KNOWLEDGE</div>', unsafe_allow_html=True)
    st.title("Explore the answer. Understand the evidence.")
    st.markdown('<div class="intro">Ask a question, compare retrieval methods, and follow the sources and relationships behind each answer.</div>', unsafe_allow_html=True)
    for column, name, value in zip(st.columns(4), ["Documents", "Indexed passages", "Graph entities", "Relationships"],
                                   [snapshot["documents"], snapshot["passages"], snapshot["entities"], snapshot["relationships"]]):
        column.metric(name, value)
    st.caption("Local inventory snapshot · source/version integrity is checked on each submitted run.")
    if snapshot.get("incomplete"):
        st.warning("Local inventory is incomplete. Use the documented setup/ingestion steps before answering.")

    st.selectbox("Try a question", ["Write your own", *SAMPLES], key="sample_question", on_change=choose_sample)
    st.session_state.setdefault("question_input", "Who approved DEC-002?")
    with st.form("question_form"):
        question = st.text_area("Your question", key="question_input", height=90, max_chars=2000)
        choice = st.radio("Retrieval method", list(MODES), index=1, horizontal=True, key="retrieval_choice")
        answer_format = st.selectbox("Answer format", list(ANSWER_FORMATS), key="answer_format")
        if answer_format != "Original cited answer":
            st.caption("Shows individual claims and exact source excerpts. Stricter checks may reject an answer; matching excerpts do not guarantee correct claims.")
        if answer_format == "Current default":
            st.caption("Current default: schema-constrained claims.")
        st.caption("Compare both deliberately runs both methods. Changing views or reopening history makes no model request.")
        submitted = st.form_submit_button("Run question", type="primary", width="stretch")
    if submitted:
        if not question.strip():
            st.warning("Enter a question before running.")
        else:
            with st.spinner("Retrieving source evidence and preparing your answer…"):
                entry = execute_request(question.strip(), choice,
                    prompt_version=ANSWER_FORMATS[answer_format])
            st.session_state.run_history = (st.session_state.run_history + [entry])[-20:]
            st.session_state.history_choice = entry["id"]
            st.rerun()

    history = st.session_state.run_history
    with st.sidebar:
        st.divider()
        st.markdown("**Session history**")
        if history:
            by_id = {entry["id"]: entry for entry in history}
            selected = st.selectbox("Open a saved run", list(reversed(by_id)), key="history_choice",
                format_func=lambda key: by_id[key]["question"][:58] + " · " + by_id[key]["choice"])
            if st.button("Clear session view", width="stretch"):
                st.session_state.run_history = []
                st.session_state.pop("history_choice", None)
                st.rerun()
            st.caption("Up to 20 actions in this browser session. Clearing this view keeps saved receipt files.")
        else:
            st.caption("Your submitted runs appear here. They stay available while this browser session is open.")

    if not history:
        st.divider()
        st.subheader("Start with a question")
        st.write("Read the answer first. Then open Sources to verify it and Graph paths to understand how the facts connect.")
        st.caption("No answer request has been made by loading this page.")
        return
    entry = next(item for item in history if item["id"] == selected)
    st.divider()
    st.subheader(entry["question"])
    st.caption("Saved action: " + entry["choice"] + " · " + entry["created_utc"] + " · This is the result of that submitted question.")
    tabs = st.tabs(["Answer", "Sources", "Graph paths", "Run details"])
    for tab, renderer in zip(tabs, [render_answer, render_sources, render_paths, render_details]):
        with tab:
            columns = st.columns(len(MODES[entry["choice"]]), gap="large")
            for column, mode in zip(columns, MODES[entry["choice"]]):
                with column:
                    st.markdown("#### " + MODE_NAMES[mode])
                    if mode in entry["errors"]:
                        st.error(entry["errors"][mode])
                        st.caption("This method did not complete. Any successful result from the other method is preserved.")
                    elif renderer is render_details:
                        renderer(entry["reports"][mode], entry["id"])
                    else:
                        renderer(entry["reports"][mode])
