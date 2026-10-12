"""Adapter for the server's existing LimiX-16M source tree and checkpoint."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .base import BackendUnavailableError, BaseTFM, TaskType
from .settings import load_backend_settings


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class LimiX16M(BaseTFM):
    """Turn LimiX's context-at-predict API into fit/predict semantics."""

    def __init__(
        self,
        task: TaskType,
        *,
        config_path: str | Path | None = None,
        device: str | None = None,
        limix_root: str | Path | None = None,
        local_python_packages: str | Path | None = None,
        model_path: str | Path | None = None,
        inference_config: str | Path | None = None,
        categorical_features_indices: Sequence[int] | None = None,
        mixed_precision: bool = True,
        random_state: int = 0,
        predictor: Any | None = None,
        **predictor_kwargs: Any,
    ) -> None:
        super().__init__(task)
        settings, _ = load_backend_settings("limix16m", config_path)
        self.device = device if device is not None else settings["default_device"]
        self.limix_root = Path(limix_root or settings["source_root"])
        self.local_python_packages = Path(local_python_packages or settings["python_packages"])
        self.model_path = Path(model_path or settings["checkpoint"])
        self.inference_config = (
            Path(inference_config)
            if inference_config is not None
            else Path(settings["inference_configs"][task])
        )
        self.categorical_features_indices = categorical_features_indices
        self.mixed_precision = mixed_precision
        self.random_state = random_state
        self.predictor_kwargs = predictor_kwargs
        self.predictor_ = predictor
        self.X_context_: np.ndarray | None = None
        self.y_context_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.y_mean_ = 0.0
        self.y_scale_ = 1.0

    def _build_predictor(self) -> Any:
        missing = [
            str(path)
            for path in (self.limix_root, self.model_path, self.inference_config)
            if not path.exists()
        ]
        if missing:
            raise BackendUnavailableError(f"LimiX resource not found: {', '.join(missing)}")

        root_string = str(self.limix_root)
        local_packages = str(self.local_python_packages)
        if self.local_python_packages.is_dir() and local_packages not in sys.path:
            sys.path.insert(0, local_packages)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        try:
            import torch
            from inference.predictor import LimiXPredictor
        except ImportError as exc:
            raise BackendUnavailableError(
                f"cannot import LimiX from {self.limix_root}: {exc}"
            ) from exc

        return LimiXPredictor(
            device=torch.device(self.device),
            model_path=str(self.model_path),
            inference_config=str(self.inference_config),
            mix_precision=self.mixed_precision,
            categorical_features_indices=self.categorical_features_indices,
            seed=self.random_state,
            **self.predictor_kwargs,
        )

    def fit(self, X: Any, y: Any) -> LimiX16M:
        self._validate_xy(X, y)
        self.X_context_ = np.asarray(X)
        raw_y = np.asarray(y).reshape(-1)
        if self.task == "classification":
            self.classes_, encoded = np.unique(raw_y, return_inverse=True)
            self.y_context_ = encoded
        else:
            numeric_y = raw_y.astype(np.float32)
            self.y_mean_ = float(numeric_y.mean())
            std = float(numeric_y.std())
            self.y_scale_ = std if std > 1e-12 else 1.0
            self.y_context_ = (numeric_y - self.y_mean_) / self.y_scale_
        if self.predictor_ is None:
            self.predictor_ = self._build_predictor()
        self._mark_fitted()
        return self

    def _raw_predict(self, X: Any) -> np.ndarray:
        self._require_fitted()
        self._validate_x(X)
        task_name = "Classification" if self.task == "classification" else "Regression"
        output = self.predictor_.predict(
            self.X_context_,
            self.y_context_,
            np.asarray(X),
            task_type=task_name,
        )
        return _as_numpy(output)

    def predict(self, X: Any) -> np.ndarray:
        output = self._raw_predict(X)
        if self.task == "classification":
            indices = output.argmax(axis=1)
            return self.classes_[indices]
        return output.reshape(-1) * self.y_scale_ + self.y_mean_

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.task != "classification":
            return super().predict_proba(X)
        return self._raw_predict(X)
