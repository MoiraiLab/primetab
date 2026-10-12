"""Adapter for Google Research TabFM's official PyTorch backend."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .base import BackendUnavailableError, BaseTFM, TaskType
from .network import configure_no_proxy_mirrors
from .settings import load_backend_settings


class TabFM(BaseTFM):
    """Expose the official TabFM estimator through the common interface."""

    def __init__(
        self,
        task: TaskType,
        *,
        config_path: str | Path | None = None,
        tabfm_root: str | Path | None = None,
        local_python_packages: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        huggingface_home: str | Path | None = None,
        device: str | None = None,
        dtype: Any | None = None,
        model: Any | None = None,
        estimator: Any | None = None,
        **estimator_kwargs: Any,
    ) -> None:
        super().__init__(task)
        settings, network_settings = load_backend_settings("tabfm", config_path)
        self.tabfm_root = Path(tabfm_root or settings["source_root"])
        self.local_python_packages = Path(local_python_packages or settings["python_packages"])
        self.checkpoint_path = Path(checkpoint_path or settings["checkpoint_root"])
        self.huggingface_home = Path(huggingface_home or settings["huggingface_home"])
        self.device = device if device is not None else settings["default_device"]
        self.dtype = dtype if dtype is not None else settings["default_dtype"]
        self.huggingface_endpoint = network_settings["huggingface_endpoint"]
        self.model_ = model
        self.estimator_ = estimator
        configured_estimator_kwargs = dict(settings.get("estimator_defaults", {}))
        configured_estimator_kwargs.update(estimator_kwargs)
        self.estimator_kwargs = configured_estimator_kwargs

    def _build_estimator(self) -> Any:
        configure_no_proxy_mirrors(self.huggingface_endpoint)
        os.environ["HF_HOME"] = str(self.huggingface_home)
        os.environ["HF_HUB_CACHE"] = str(self.huggingface_home / "hub")
        local_packages = str(self.local_python_packages)
        if self.local_python_packages.is_dir() and local_packages not in sys.path:
            sys.path.insert(0, local_packages)
        if self.tabfm_root.is_dir():
            root_string = str(self.tabfm_root)
            if root_string not in sys.path:
                sys.path.insert(0, root_string)
        try:
            # The official PyTorch-only import currently references jt.Array even
            # when JAX is absent. Supply the intended numpy annotation fallback.
            import jaxtyping as jt
            import numpy as np

            if not hasattr(jt, "Array"):
                jt.Array = np.ndarray
            import torch
            from tabfm import TabFMClassifier, TabFMRegressor, tabfm_v1_0_0_pytorch
        except ImportError as exc:
            raise BackendUnavailableError(
                "TabFM is unavailable; install the official google-research/tabfm source "
                f"or place it at {self.tabfm_root}"
            ) from exc

        if self.model_ is None:
            model_type = "classification" if self.task == "classification" else "regression"
            dtype = self.dtype
            if isinstance(dtype, str):
                try:
                    dtype = getattr(torch, dtype)
                except AttributeError as exc:
                    raise ValueError(f"unsupported torch dtype in config: {dtype}") from exc
            self.model_ = tabfm_v1_0_0_pytorch.load(
                model_type=model_type,
                checkpoint_path=str(self.checkpoint_path),
                device=self.device,
                dtype=dtype,
            )
        estimator_type = TabFMClassifier if self.task == "classification" else TabFMRegressor
        return estimator_type(model=self.model_, **self.estimator_kwargs)

    def fit(self, X: Any, y: Any) -> TabFM:
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
