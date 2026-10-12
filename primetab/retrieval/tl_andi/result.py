"""Structured outputs returned by TL-ANDI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CandidateScore:
    method: str
    validation_mse: float
    bandwidth: float | None = None
    penalty: float | None = None
    anchor_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "validation_mse": self.validation_mse,
            "bandwidth": self.bandwidth,
            "penalty": self.penalty,
            "anchor_count": self.anchor_count,
        }

@dataclass(frozen=True)
class TLANDIResult:
    predictions: np.ndarray
    selected_method: str
    selected_bandwidth: float | None
    selected_penalty: float | None
    selected_anchor_indices: np.ndarray
    candidate_scores: tuple[CandidateScore, ...]
    calibration_indices: np.ndarray
    validation_indices: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "selected_method": self.selected_method,
            "selected_bandwidth": self.selected_bandwidth,
            "selected_penalty": self.selected_penalty,
            "selected_anchor_indices": self.selected_anchor_indices.tolist(),
            "candidate_scores": [score.to_dict() for score in self.candidate_scores],
            "calibration_size": len(self.calibration_indices),
            "validation_size": len(self.validation_indices),
        }
