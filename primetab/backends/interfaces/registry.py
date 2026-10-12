"""Name-based construction for pipeline configuration files."""

from __future__ import annotations

from typing import Any

from .base import BaseTFM, TaskType
from .limix16m import LimiX16M
from .tabfm import TabFM
from .tabpfn3 import TabPFN3

_MODELS = {
    "tabpfn3": TabPFN3,
    "tabpfn-v3": TabPFN3,
    "tabfm": TabFM,
    "limix16m": LimiX16M,
    "limix-16m": LimiX16M,
}


def create_tfm(name: str, task: TaskType, **kwargs: Any) -> BaseTFM:
    """Create a TFM adapter from a stable configuration name."""
    normalized = name.strip().lower()
    try:
        model_type = _MODELS[normalized]
    except KeyError as exc:
        canonical = "tabpfn3, tabfm, limix16m"
        raise ValueError(f"unknown TFM {name!r}; choose one of: {canonical}") from exc
    return model_type(task=task, **kwargs)
