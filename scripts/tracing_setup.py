"""Opt-in LangSmith callbacks with explicit settings and verified root-run readback."""

from uuid import uuid4

from langchain_core.tracers.langchain import LangChainTracer, wait_for_all_tracers
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from service_clients import ServiceError, tls_context


class SystemTrustAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = tls_context()
        return super().init_poolmanager(*args, **kwargs)


class TracingSession:
    def __init__(self, settings):
        self.client = self.session = self.tracer = None
        self.run_id = uuid4()
        self.report = {"status": "disabled", "project": settings.langsmith_project}
        if not settings.langsmith_tracing:
            return
        if not settings.langsmith_api_key or not settings.langsmith_project:
            raise ServiceError("Tracing enabled but LANGSMITH_API_KEY/project missing")
        endpoint = settings.langsmith_endpoint.rstrip("/")
        if endpoint not in {"https://api.smith.langchain.com", "https://eu.api.smith.langchain.com"}:
            raise ServiceError("Use a documented US/EU LangSmith endpoint for this milestone")
        self.session = requests.Session()
        self.session.mount("https://", SystemTrustAdapter())
        try:
            self.client = Client(api_url=endpoint, api_key=settings.langsmith_api_key,
                session=self.session, auto_batch_tracing=False, timeout_ms=(5000, 10000),
                retry_config=Retry(total=0))
            try:
                self.client.read_project(project_name=settings.langsmith_project)
            except LangSmithNotFoundError:
                self.client.create_project(project_name=settings.langsmith_project)
            self.tracer = LangChainTracer(client=self.client, project_name=settings.langsmith_project)
            self.report = {"status": "configured; run upload not yet verified",
                           "project": settings.langsmith_project, "run_id": str(self.run_id)}
        except Exception:
            self.close()
            raise ServiceError("LangSmith setup failed; check key, endpoint region, and workspace access") from None

    def invoke_config(self):
        return {"run_id": self.run_id, "callbacks": [self.tracer] if self.tracer else [],
                "run_name": "organizational-vector-rag", "tags": ["vector", "friday-baseline"]}

    def finish(self):
        if self.tracer is None:
            return self.report
        try:
            wait_for_all_tracers()
            run = self.client.read_run(self.run_id)
            if run.end_time is None or run.error:
                raise ServiceError("Trace has not completed successfully")
            self.report.update({"status": "verified", "url": self.client.get_run_url(run=run)})
        except Exception:
            self.report["status"] = "upload/readback not verified; inspect LangSmith access"
        return self.report

    def close(self):
        if self.client is not None:
            self.client.close()
        if self.session is not None:
            self.session.close()
