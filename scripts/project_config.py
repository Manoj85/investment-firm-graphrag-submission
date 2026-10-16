"""Read local configuration without making service calls or printing secrets."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


# Resolve from this file, so commands do not depend on the terminal's directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "outputs" / "corpus"


@dataclass(frozen=True)
class Settings:
    # repr=False prevents accidental disclosure when someone prints Settings.
    # It is not encryption: the secrets still exist in the process's memory.
    nebius_api_key: str = field(repr=False)
    nebius_base_url: str
    nebius_chat_model: str
    nebius_embedding_model: str
    pinecone_api_key: str = field(repr=False)
    pinecone_index: str
    pinecone_namespace: str
    langsmith_api_key: str = field(repr=False)
    langsmith_project: str
    langsmith_tracing: bool
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    corpus_dir: str = "outputs/corpus"
    chunk_max_tokens: int = 256
    chunk_overlap: int = 32
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    embedding_document_context: bool = False

    def service_readiness(self) -> dict[str, bool]:
        """Report required settings' presence, not authentication or account access."""
        return {
            "Nebius": all((self.nebius_api_key, self.nebius_base_url,
                           self.nebius_chat_model, self.nebius_embedding_model)),
            "Pinecone": all((self.pinecone_api_key, self.pinecone_index,
                             self.pinecone_namespace)),
            "LangSmith": bool(self.langsmith_api_key and self.langsmith_project),
        }


def load_settings(
    env_path: Path = PROJECT_ROOT / ".env",
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Combine a local file and shell settings without mutating os.environ.

    Optional arguments let tests use temporary files and a controlled environment.
    A missing .env is allowed: deployed apps may use only shell variables.
    """
    file_values = dotenv_values(env_path, encoding="utf-8-sig", interpolate=False)
    shell_values = os.environ if environ is None else environ
    # Later dictionary entries win. Even an empty shell value overrides the file.
    values = {**file_values, **shell_values}

    def value(name: str, default: str = "") -> str:
        return (values.get(name) or default).strip()

    tracing = value("LANGSMITH_TRACING", "false").lower()
    if tracing not in {"true", "false"}:
        # Name the setting, but never include its supplied value in the error.
        raise ValueError("LANGSMITH_TRACING must be true or false")
    document_context = value("EMBEDDING_DOCUMENT_CONTEXT", "false").lower()
    if document_context not in {"true", "false"}:
        raise ValueError("EMBEDDING_DOCUMENT_CONTEXT must be true or false")

    try:
        chunk_max_tokens = int(value("CHUNK_MAX_TOKENS", "256"))
        chunk_overlap = int(value("CHUNK_OVERLAP", "32"))
    except ValueError:
        raise ValueError("CHUNK_MAX_TOKENS and CHUNK_OVERLAP must be integers") from None
    if not 0 <= chunk_overlap < chunk_max_tokens <= 2048:
        raise ValueError("Require 0 <= CHUNK_OVERLAP < CHUNK_MAX_TOKENS <= 2048")

    return Settings(
        nebius_api_key=value("NEBIUS_API_KEY"),
        nebius_base_url=value("NEBIUS_BASE_URL"),
        nebius_chat_model=value("NEBIUS_CHAT_MODEL"),
        nebius_embedding_model=value("NEBIUS_EMBEDDING_MODEL"),
        pinecone_api_key=value("PINECONE_API_KEY"),
        pinecone_index=value("PINECONE_INDEX"),
        pinecone_namespace=value("PINECONE_NAMESPACE"),
        langsmith_api_key=value("LANGSMITH_API_KEY"),
        langsmith_project=value("LANGSMITH_PROJECT"),
        langsmith_tracing=tracing == "true",
        pinecone_cloud=value("PINECONE_CLOUD", "aws"),
        pinecone_region=value("PINECONE_REGION", "us-east-1"),
        corpus_dir=value("CORPUS_DIR", "outputs/corpus"),
        chunk_max_tokens=chunk_max_tokens,
        chunk_overlap=chunk_overlap,
        langsmith_endpoint=value("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        embedding_document_context=document_context == "true",
    )
