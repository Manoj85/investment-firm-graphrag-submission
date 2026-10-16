"""Opt-in LangSmith callbacks with explicit settings and verified root-run readback."""

from uuid import uuid4
from time import sleep

from langchain_core.tracers.langchain import LangChainTracer, wait_for_all_tracers
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError
import requests
from requests.adapters import BaseAdapter
import httpx
from urllib3.util.retry import Retry

from service_clients import ServiceError, tls_context


class SystemTrustAdapter(BaseAdapter):
    """Use our verified HTTPX transport behind LangSmith's Requests interface.

    This adapter avoids the managed Windows environment's failing default
    Requests transport and shares the working, verified
    transport used by Nebius/Pinecone; it does not disable certificate checks.
    Only ordinary JSON requests are needed for this trace/readback milestone.
    """
    def __init__(self):
        self.http = httpx.Client(verify=tls_context(), timeout=30, follow_redirects=False)

    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
        if verify is False or cert is not None or stream:
            raise requests.RequestException("Unsupported trace transport option")
        if isinstance(timeout, tuple):
            timeout = httpx.Timeout(timeout[1], connect=timeout[0])
        try:
            result = self.http.request(request.method, request.url,
                headers=dict(request.headers), content=request.body, timeout=timeout or 30)
        except httpx.RequestError:
            raise requests.ConnectionError("Verified trace transport connection failed") from None
        response = requests.Response()
        response.status_code = result.status_code
        response.headers.update(result.headers)
        response._content = result.content
        response.encoding = result.encoding
        response.url = str(result.url)
        response.request = request
        return response

    def close(self):
        self.http.close()


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
        try:
            self.client = Client(api_url=endpoint, api_key=settings.langsmith_api_key,
                session=self.session, auto_batch_tracing=False, timeout_ms=(5000, 10000),
                retry_config=Retry(total=0))
            # Client installs its own Requests adapters during construction.
            # Mount afterward so our verified Windows transport is retained.
            self.session.mount("https://", SystemTrustAdapter())
            try:
                self.client.read_project(project_name=settings.langsmith_project)
            except LangSmithNotFoundError:
                self.client.create_project(project_name=settings.langsmith_project)
            self.tracer = LangChainTracer(client=self.client, project_name=settings.langsmith_project)
            self.report = {"status": "configured; run upload not yet verified",
                           "project": settings.langsmith_project, "run_id": str(self.run_id)}
        except Exception as error:
            self.close()
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)
            detail = f"{type(error).__name__}" + (f", HTTP {status}" if status else "")
            raise ServiceError(f"LangSmith setup failed ({detail}); check key, endpoint region, and workspace access") from None

    def invoke_config(self):
        return {"run_id": self.run_id, "callbacks": [self.tracer] if self.tracer else [],
                "run_name": "organizational-vector-rag", "tags": ["vector", "friday-baseline"]}

    def finish(self):
        if self.tracer is None:
            return self.report
        try:
            wait_for_all_tracers()
            # A successful upload may not be immediately visible to reads.
            for attempt in range(4):
                try:
                    run = self.client.read_run(self.run_id)
                    if run.end_time is not None:
                        break
                except LangSmithNotFoundError:
                    if attempt == 3:
                        raise
                if attempt < 3:
                    sleep(2)
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
