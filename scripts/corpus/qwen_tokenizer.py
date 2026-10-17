"""Prepare/load pinned Qwen tokenizer assets. No model weights or provider calls."""

import hashlib
import ssl

import httpx
from tokenizers import Tokenizer

from scripts.config.project_config import PROJECT_ROOT


MODEL_ID = "Qwen/Qwen3-Embedding-8B"
# Pin both revision and bytes, so a later upstream edit cannot silently change IDs.
REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
TOKENIZER_SHA256 = "83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d"
TOKENIZER_PATH = PROJECT_ROOT / "data" / "cache" / "tokenizers" / REVISION / "tokenizer.json"
TOKENIZER_URL = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/tokenizer.json"


def load_tokenizer() -> Tokenizer:
    """Read the verified local file; never download implicitly during chunking."""
    if not TOKENIZER_PATH.is_file():
        raise ValueError("Tokenizer missing. Run python -m scripts.corpus.qwen_tokenizer once to download it.")
    if hashlib.sha256(TOKENIZER_PATH.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Tokenizer checksum mismatch; prepare the pinned file again.")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    # We need full counts, not silently truncated or padded encodings.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def count_tokens(tokenizer: Tokenizer, text: str) -> int:
    # Count literal passage text without applying a chat template or adding tokens.
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def main() -> None:
    if TOKENIZER_PATH.is_file():
        load_tokenizer()
        print("Pinned tokenizer already cached and checksum verified.")
        return
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    tls.load_default_certs()
    # Public artifact download only: no Nebius key or other credentials are sent.
    response = httpx.get(TOKENIZER_URL, verify=tls, follow_redirects=True, timeout=60)
    response.raise_for_status()
    content = response.content
    if hashlib.sha256(content).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Downloaded tokenizer checksum mismatch; nothing saved.")
    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKENIZER_PATH.write_bytes(content)
    load_tokenizer()
    print(f"Prepared tokenizer only: {len(content):,} bytes; revision {REVISION}.")


if __name__ == "__main__":
    main()
