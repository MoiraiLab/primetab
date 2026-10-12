"""Network settings used by model backends that may fetch weights."""

from __future__ import annotations

import os

PROXY_VARIABLES = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)


def configure_no_proxy_mirrors(huggingface_endpoint: str) -> None:
    """Disable environment proxies and select the configured Hugging Face mirror."""
    for name in PROXY_VARIABLES:
        os.environ.pop(name, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    os.environ["HF_ENDPOINT"] = huggingface_endpoint
