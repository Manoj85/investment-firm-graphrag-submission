"""Make small live provider checks. No retrieval, indexing, or automatic retries."""

import argparse
from datetime import datetime, timezone
import json
import math
import ssl
from time import perf_counter
from urllib.parse import urlsplit

import httpx

from inspect_atlas import ATLAS_PATH, load_markdown
from inspect_atlas_chunks import split_by_section
from project_config import PROJECT_ROOT, Settings, load_settings


def validate_settings(settings: Settings, operation: str) -> str:
    """Fail locally before sending credentials; this milestone uses the global API."""
    required = {"NEBIUS_API_KEY": settings.nebius_api_key,
                "NEBIUS_BASE_URL": settings.nebius_base_url}
    if operation in {"all", "chat"}:
        required["NEBIUS_CHAT_MODEL"] = settings.nebius_chat_model
    if operation in {"all", "embedding"}:
        required["NEBIUS_EMBEDDING_MODEL"] = settings.nebius_embedding_model
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError("Missing settings: " + ", ".join(missing))
    # Do not forward the key to a mistyped host, HTTP URL, or redirect destination.
    url = urlsplit(settings.nebius_base_url)
    if (url.scheme != "https" or url.netloc != "api.tokenfactory.nebius.com"
            or url.path.rstrip("/") != "/v1" or url.query or url.fragment):
        raise ValueError("This diagnostic requires https://api.tokenfactory.nebius.com/v1/")
    return settings.nebius_base_url.rstrip("/") + "/"


def summarize_chat(data: dict) -> dict:
    """A successful HTTP status is insufficient: require completed visible text."""
    choice = data["choices"][0]
    text = choice["message"].get("content")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Chat returned no visible answer")
    if choice.get("finish_reason") != "stop":
        raise ValueError("Chat did not finish normally; inspect output limits/reasoning settings")
    return {"response_model": data.get("model"), "text": text.strip(),
            "finish_reason": choice["finish_reason"], "usage": data.get("usage")}


def summarize_embedding(data: dict) -> dict:
    """Require one nonempty finite numeric vector before reporting its dimension."""
    records = data["data"]
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("Expected exactly one embedding record")
    vector = records[0]["embedding"]
    if (not isinstance(vector, list) or not vector
            or any(type(number) not in (int, float) or not math.isfinite(number)
                   for number in vector)):
        raise ValueError("Embedding must be a nonempty vector of finite numbers")
    return {"response_model": data.get("model"), "vector_count": len(records),
            "dimension": len(vector), "first_five_values": vector[:5],
            "usage": data.get("usage")}


def run_probe(client: httpx.Client, endpoint: str, payload: dict, summarize) -> dict:
    started = perf_counter()
    try:
        response = client.post(endpoint, json=payload)
        response.raise_for_status()
        result = {"status": "passed", **summarize(response.json())}
    except httpx.HTTPStatusError as error:
        code = error.response.status_code
        # Do not print raw exceptions, request headers, or arbitrary error bodies.
        hints = {400: "Check request parameters and model ID",
                 401: "Authentication rejected; check the key locally",
                 403: "Access denied; check project/model permissions",
                 404: "Endpoint or model not found",
                 402: "Check account billing/credit access",
                 429: "Rate/quota limit; check account details before retrying"}
        result = {"status": "failed", "http_status": code,
                  "error": hints.get(code, "Provider HTTP error; inspect account status")}
    except httpx.RequestError:
        result = {"status": "failed", "error": "Network/TLS/timeout failure; no automatic retry"}
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        result = {"status": "failed", "error": "Unexpected or incomplete response; no valid result confirmed"}
    result["elapsed_seconds"] = round(perf_counter() - started, 3)
    result["requested_model"] = payload["model"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("all", "chat", "embedding"), default="all")
    args = parser.parse_args()
    try:
        settings = load_settings()
        base_url = validate_settings(settings, args.operation)
        # Reuse our learning loader/splitter to select one actual source passage.
        ownership = next(chunk for chunk in split_by_section(load_markdown(ATLAS_PATH))
                         if chunk.metadata["section"] == "Ownership")
    except (ValueError, OSError, StopIteration):
        print("Local setup failed: check required settings, official URL, and Atlas Ownership section.")
        return 1

    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "diagnostic_version": 1, "base_url": base_url, "results": {}}
    # Use OS certificate trust with verification enabled. Construct the context
    # explicitly: create_default_context also honors SSLKEYLOGFILE, which may
    # point to an inaccessible debug device in a managed Windows environment.
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    tls.load_default_certs()
    # No redirects or automatic retries.
    with httpx.Client(base_url=base_url,
                      headers={"Authorization": f"Bearer {settings.nebius_api_key}"},
                      verify=tls, follow_redirects=False,
                      timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        if args.operation in {"all", "chat"}:
            print("Sending one bounded chat diagnostic...", flush=True)
            report["results"]["chat"] = run_probe(client, "chat/completions", {
                "model": settings.nebius_chat_model,
                "messages": [{"role": "user", "content": "Reply with exactly: Nebius connection works."}],
                "max_tokens": 128, "temperature": 0,
                # NVIDIA documents this direct-answer control. Hosted acceptance
                # is verified by this request, not assumed from the model card.
                "chat_template_kwargs": {"enable_thinking": False},
            }, summarize_chat)
        if args.operation in {"all", "embedding"}:
            print("Embedding one Atlas Ownership passage...", flush=True)
            report["results"]["embedding"] = {
                **run_probe(client, "embeddings", {
                    "model": settings.nebius_embedding_model,
                    "input": ownership.page_content, "encoding_format": "float",
                }, summarize_embedding),
                "source": ownership.metadata,
                "input_characters": len(ownership.page_content),
            }

    # Persist a small inspectable receipt, not keys, HTTP headers, or full vectors.
    folder = PROJECT_ROOT / "work" / "diagnostics"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    receipt = folder / f"nebius-{stamp}.json"
    receipt.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Local receipt: {receipt.relative_to(PROJECT_ROOT).as_posix()}")
    return 0 if all(result["status"] == "passed" for result in report["results"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
