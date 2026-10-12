"""Validated TL-ANDI configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class TLANDIConfig:
    task: Literal["regression", "classification"] = "regression"
    context_budget: int = 1_000
    lambda_values: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    bandwidth_quantiles: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.40)
    calibration_fraction: float = 0.5
    bandwidth_sample_size: int | None = 2_000
    kernel_batch_size: int = 256
    standardize_features: bool = True
    random_state: int = 0

    def __post_init__(self) -> None:
        if self.task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'")
        if self.context_budget < 1:
            raise ValueError("context_budget must be positive")
        if not self.lambda_values or any(value < 0 for value in self.lambda_values):
            raise ValueError("lambda_values must be non-empty and non-negative")
        if not self.bandwidth_quantiles or any(
            value <= 0 or value >= 1 for value in self.bandwidth_quantiles
        ):
            raise ValueError("bandwidth_quantiles must lie strictly between zero and one")
        if not 0 < self.calibration_fraction < 1:
            raise ValueError("calibration_fraction must lie strictly between zero and one")
        if self.bandwidth_sample_size is not None and self.bandwidth_sample_size < 2:
            raise ValueError("bandwidth_sample_size must be at least two or null")
        if self.kernel_batch_size < 1:
            raise ValueError("kernel_batch_size must be positive")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TLANDIConfig:
        values = dict(payload)
        for key in ("lambda_values", "bandwidth_quantiles"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> TLANDIConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
