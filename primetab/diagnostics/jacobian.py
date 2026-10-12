"""Jacobian feature and label sensitivity analysis for in-context TFMs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class JacobianReport:
    """Sensitivity analysis report of predictions with respect to context values."""

    jacobian_matrix: np.ndarray  # Shape: (N_queries, N_shots)
    mean_sensitivity: float
    max_sensitivity: float
    shot_importance: np.ndarray  # Shape: (N_shots,)
    query_vulnerability: np.ndarray  # Shape: (N_queries,)


def compute_jacobian_sensitivity(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    base_shot_y: np.ndarray,
    eps: float = 1e-3,
) -> JacobianReport:
    """Estimates the Jacobian d(y_query) / d(y_shot) via two-sided finite differences.
    
    predict_fn: A callable accepting perturbed shot_y of shape (k,) and returning query predictions of shape (q,).
    """
    k = len(base_shot_y)
    if k == 0:
        raise ValueError("shot_y must not be empty")

    base_pred = np.asarray(predict_fn(base_shot_y), dtype=float)
    q = len(base_pred)

    jacobian = np.zeros((q, k), dtype=np.float32)

    for i in range(k):
        y_plus = base_shot_y.copy()
        y_plus[i] += eps
        y_minus = base_shot_y.copy()
        y_minus[i] -= eps

        pred_plus = np.asarray(predict_fn(y_plus), dtype=float)
        pred_minus = np.asarray(predict_fn(y_minus), dtype=float)

        jacobian[:, i] = (pred_plus - pred_minus) / (2.0 * eps)

    abs_jac = np.abs(jacobian)
    shot_importance = np.mean(abs_jac, axis=0)
    query_vulnerability = np.mean(abs_jac, axis=1)

    return JacobianReport(
        jacobian_matrix=jacobian,
        mean_sensitivity=float(np.mean(abs_jac)),
        max_sensitivity=float(np.max(abs_jac)),
        shot_importance=shot_importance,
        query_vulnerability=query_vulnerability,
    )
