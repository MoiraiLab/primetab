"""TabPFN v3 adapter backed by the weights already present on aietheia."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .base import BackendUnavailableError, BaseTFM, TaskType
from .settings import load_backend_settings


class TabPFN3(BaseTFM):
    """Expose local TabPFN v3 classifier/regressor through the common interface."""

    def __init__(
        self,
        task: TaskType,
        *,
        config_path: str | Path | None = None,
        device: str | Sequence[str] | None = None,
        model_path: str | Path | None = None,
        n_estimators: int = 8,
        categorical_features_indices: Sequence[int] | None = None,
        random_state: int = 0,
        estimator: Any | None = None,
        **estimator_kwargs: Any,
    ) -> None:
        super().__init__(task)
        settings, _ = load_backend_settings("tabpfn3", config_path)
        self.device = device if device is not None else settings["default_device"]
        selected_weight = model_path if model_path is not None else settings["weights"][task]
        self.model_path = Path(selected_weight)
        self.n_estimators = n_estimators
        self.categorical_features_indices = categorical_features_indices
        self.random_state = random_state
        self.estimator_kwargs = estimator_kwargs
        self.estimator_ = estimator

    def _build_estimator(self) -> Any:
        if not self.model_path.is_file():
            raise BackendUnavailableError(f"TabPFN3 weight not found: {self.model_path}")
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
        except ImportError as exc:
            raise BackendUnavailableError(
                "the 'tabpfn' package is not installed in this Python environment"
            ) from exc

        estimator_type = TabPFNClassifier if self.task == "classification" else TabPFNRegressor
        return estimator_type(
            model_path=self.model_path,
            device=self.device,
            n_estimators=self.n_estimators,
            categorical_features_indices=self.categorical_features_indices,
            random_state=self.random_state,
            show_progress_bar=False,
            **self.estimator_kwargs,
        )

    def fit(self, X: Any, y: Any) -> TabPFN3:
        self._validate_xy(X, y)
        if self.estimator_ is None:
            self.estimator_ = self._build_estimator()
        self.estimator_.fit(X, y)
        self._mark_fitted()
        return self

    def predict(self, X: Any) -> Any:
        self._require_fitted()
        self._validate_x(X)
        return self.estimator_.predict(X)

    def predict_proba(self, X: Any) -> Any:
        if self.task != "classification":
            return super().predict_proba(X)
        self._require_fitted()
        self._validate_x(X)
        return self.estimator_.predict_proba(X)
