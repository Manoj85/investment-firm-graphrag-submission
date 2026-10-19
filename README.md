# Knowledge Lab: organizational Graph RAG

Track 2 / Project 3. A local Streamlit application comparing Vector and Graph-assisted retrieval over ten fictional investment-firm documents. Pinecone persists source passages and embeddings; a reviewed NetworkX graph links organizational entities to their evidence. LangGraph coordinates retrieval and cited generation using Nebius models. LangSmith provides tracing.

## Setup (Python 3.12)

Run from this repository in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set your own NEBIUS_API_KEY, PINECONE_API_KEY and LANGSMITH_API_KEY in the local .env. The template includes the tested model IDs. Set LANGSMITH_TRACING=true to enable tracing. Keep .env private; it is ignored. Credentials are never committed.

Prepare the pinned tokenizer and inspect the corpus without provider calls (the tokenizer preparation downloads a public asset):

```powershell
.\.venv\Scripts\python.exe -m scripts.corpus.qwen_tokenizer
.\.venv\Scripts\python.exe -m scripts.corpus.ingest_corpus --dry-run
```

For your first run, explicitly create or inspect the configured 4096-dimensional cosine index and ingest the corpus. These steps use your service account and ingestion may incur embedding charges:

```powershell
.\.venv\Scripts\python.exe -m scripts.retrieval.prepare_pinecone --create
.\.venv\Scripts\python.exe -m scripts.corpus.ingest_corpus
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open http://127.0.0.1:8501/ and leave the terminal running. After a restart, run only the last command. Re-ingest deliberately when the corpus or embedding/chunk configuration changes. The active manifest and caches are generated locally and are not part of Git.

## Inspect the implementation

- scripts/config: environment settings and validation.
- scripts/corpus: pinned tokenizer, token-bounded chunking, provenance and explicit ingestion.
- scripts/services and scripts/retrieval: service clients, persistent retrieval and tracing.
- scripts/graph: reviewed graph loading, entity resolution and bounded relationship traversal.
- scripts/rag: retrieve/generate workflow, structured claims and evidence checks.
- scripts/ui: the shared backend's testing interface.
- outputs/corpus: the only retrieval input, ten fictional Markdown sources.
- outputs/graph/knowledge-graph.json: 31 reviewed entities and 56 source-backed relationships.
- tests and the small outputs/evaluation fixture subset: offline regression tests and reproducible comparison tooling. Answer keys/fixtures are not ingested.

Vector selects four similar passages. Graph-assisted also follows reviewed relationships and fetches their source passages from Pinecone. Evidence is deduplicated and packed within a 2000-content-token budget. The same generator then returns claims with source labels and quotations. NetworkX represents organizational facts; LangGraph represents execution steps. Both modes use real service calls, with no prewritten question-to-answer mapping.

Choose Vector, Graph-assisted or Compare both. Answer, Sources, Graph paths and Run details show the result and its provenance. Only Run question makes an answer request; changing tabs/history does not. Current default is V6 schema-constrained claims. Reviewed bridges (candidate) exposes optional V12 approval proofs; it was not promoted to the shared default. Exact quotes, schema validity and trace verification do not prove semantic correctness. Manual source review and broader unseen-question evaluation remain necessary.

## Verify

After preparing the tokenizer:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -m scripts.evaluation.run_comparison
```

The comparison command without --run is a local preflight. Adding --run deliberately makes live attempts using the exposed ten-question authored benchmark; it is not a blind accuracy estimate. Diagnostic and replay commands that accept a saved-run path require the corresponding receipt files.

## Design choices and limitations

Pinecone provides persistent passage retrieval across restarts. NetworkX with reviewed JSON keeps this small organizational graph inspectable without an additional graph database. Graph-assisted retrieval was chosen to recover explicit relationships that similarity search can miss. Bounded traversal and a shared evidence budget limit context growth, but can still omit useful evidence. Streamlit provides a compact Python interface for inspecting both modes.

The ten authored benchmark questions were exposed during development, so their scores do not estimate performance on unseen questions. Structured output and quote checks improve inspectability but cannot establish semantic truth. The reviewed graph is manually authored and would require further work to scale. All people, organizations and events in the source data are fictional.

Codex assisted implementation, debugging, testing and documentation. The Google Doc report explains the dataset, prompts, iterations, comparison results and learnings; the demo shows the running local application.

## Optional code exploration with Graphify

Graphify is a developer aid for exploring this repository's functions and dependencies. It is separate from the organizational knowledge graph used to answer questions. A local Graphify 0.9.92 code-only run over 52 Python files produced 372 nodes and 911 edges without model calls. It traced `render_app → execute_request → run_question`; the first call edge was extracted and the cross-file edge inferred. A broad query exceeded its context budget, so focused symbol/path queries and source verification remain necessary. No answer-quality improvement is claimed.

Install it separately from the application environment if you want to reproduce the inspection:

```powershell
uv tool install graphifyy==0.9.92
graphify extract scripts --code-only --no-cluster --out work/graphify-review/result --max-workers 1
graphify path render_app run_question --graph work/graphify-review/result/graphify-out/graph.json
graphify export html --graph work/graphify-review/result/graphify-out/graph.json
```

The official package name is `graphifyy`. This optional workflow does not install assistant hooks, send documents to a semantic backend, or change the app's dependencies, prompts or benchmark results.

[Graphify project](https://github.com/Graphify-Labs/graphify) · [Code-only workflow](https://docs.graphify.com/guides/local-development)

[View the code-graph screenshot](https://github.com/Manoj85/investment-firm-graphrag/blob/main/docs/assets/graphify/run-question-code-graph.png)
