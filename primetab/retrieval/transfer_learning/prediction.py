"""Black-box prediction steps shared by transfer-learning pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np

from primetab.backends import ModelFactory


def continuous_prediction(model: Any, X: np.ndarray, task: str) -> np.ndarray:
    """Return regression values or positive-class probabilities."""

    if task == "regression":
        return np.asarray(model.predict(X), dtype=float).reshape(-1)
    if task != "classification":
        raise ValueError(f"unsupported task: {task!r}")
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if probabilities.ndim == 1:
        return probabilities
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("TL-ANDI currently supports binary classification probabilities only")
    return probabilities[:, 1]


def target_only_predict(
    model_factory: ModelFactory,
    task: str,
    X_target: np.ndarray,
    y_target: np.ndarray,
    X_query: np.ndarray,
) -> np.ndarray:
    operator_task = _operator_task(task)
    model = model_factory(operator_task)
    model.fit(X_target, y_target)
    prediction = continuous_prediction(model, X_query, operator_task)
    return np.clip(prediction, 0.0, 1.0) if task == "classification" else prediction


def vanilla_transfer_predict(
    model_factory: ModelFactory,
    task: str,
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    X_query: np.ndarray,
) -> np.ndarray:
    """Run the paper's source prediction plus target residual correction."""

    operator_task = _operator_task(task)
    source_model = model_factory(operator_task)
    source_model.fit(X_source, y_source)
    combined_query = np.vstack([X_target, X_query])
    combined_prediction = continuous_prediction(source_model, combined_query, operator_task)
    source_on_target = combined_prediction[: len(X_target)]
    source_on_query = combined_prediction[len(X_target) :]

    residual_model = model_factory("regression")
    residual_model.fit(X_target, np.asarray(y_target, dtype=float) - source_on_target)
    prediction = source_on_query + np.asarray(residual_model.predict(X_query), dtype=float).reshape(-1)
    if task == "classification":
        prediction = np.clip(prediction, 0.0, 1.0)
    return prediction


def _operator_task(task: str) -> str:
    if task not in {"regression", "classification"}:
        raise ValueError(f"unsupported task: {task!r}")
    # Distilled class labels and residuals are continuous probabilities. The
    # paper therefore treats binary classification as probability regression.
    return "regression"
