"""Shared repository paths with optional environment overrides."""

from __future__ import annotations

import os
from pathlib import Path


def _configured_path(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else default.resolve()


REPOSITORY_ROOT = _configured_path(
    "TFM_REPOSITORY_ROOT", Path(__file__).resolve().parents[1]
)
DATA_ROOT = _configured_path("TFM_DATA_ROOT", REPOSITORY_ROOT / "data")
BENCHMARK_WORKSPACE = _configured_path(
    "PRIMETAB_BENCHMARK_WORKSPACE", REPOSITORY_ROOT / "benchmark_workspace"
)


def repository_relative(path: Path) -> str:
    """Return a stable repository-relative display path when possible."""

    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)
