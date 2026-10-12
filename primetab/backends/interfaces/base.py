"""Small common contract shared by all TFM adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

TaskType = Literal["classification", "regression"]


class BackendUnavailableError(RuntimeError):
    """Raised when model source, weights, or a Python dependency is unavailable."""


class NotFittedError(RuntimeError):
    """Raised when inference is requested before fit."""


class BaseTFM(ABC):
    """A scikit-learn-shaped interface for a tabular foundation model."""

    def __init__(self, task: TaskType) -> None:
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        self.task = task
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: Any, y: Any) -> BaseTFM:
        """Register the labelled context table and return self."""

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Predict targets for new rows."""

    def predict_proba(self, X: Any) -> Any:
        raise AttributeError(f"{type(self).__name__} does not expose probabilities")

    def _mark_fitted(self) -> None:
        self._is_fitted = True

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError("call fit(X, y) before prediction")

    @staticmethod
    def _validate_xy(X: Any, y: Any) -> None:
        x_shape = getattr(X, "shape", None)
        y_shape = getattr(y, "shape", None)
        if x_shape is None or len(x_shape) != 2:
            raise ValueError("X must be a two-dimensional table")
        if y_shape is None:
            try:
                y_rows = len(y)
            except TypeError as exc:
                raise ValueError("y must be a one-dimensional sequence") from exc
        else:
            if len(y_shape) not in {1, 2} or (len(y_shape) == 2 and y_shape[1] != 1):
                raise ValueError("y must be one-dimensional or have shape (n, 1)")
            y_rows = y_shape[0]
        if x_shape[0] != y_rows:
            raise ValueError(f"X and y have different row counts: {x_shape[0]} != {y_rows}")

    @staticmethod
    def _validate_x(X: Any) -> None:
        shape = getattr(X, "shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError("X must be a two-dimensional table")
