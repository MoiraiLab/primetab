"""Ranking contracts and a shared-context TFM implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .feature_encoding import F1FeatureEncoder, require_f1_columns


class CandidateRanker(Protocol):
    """A replaceable scorer for generated candidates.

    Implementations receive candidate metadata and the aligned raw 73-column
    feature table.  The returned series must be indexed by ``design_id``.
    """

    def score(self, candidates: pd.DataFrame, features: pd.DataFrame) -> pd.Series: ...

    def metadata(self) -> Mapping[str, Any]: ...


class SharedContextTFMRanker:
    """Fit a regression backend on one labelled context and score all candidates.

    This is the reusable first ranking implementation.  It accepts any backend
    exposed by ``primetab.backends`` and also permits estimator injection for
    tests or lightweight local models.
    """

    def __init__(
        self,
        backend: str = "tabpfn3",
        *,
        backend_kwargs: Mapping[str, Any] | None = None,
        estimator: Any | None = None,
    ) -> None:
        self.backend = str(backend).strip().lower()
        if not self.backend:
            raise ValueError("backend must not be empty")
        self.backend_kwargs = dict(backend_kwargs or {})
        self.estimator = estimator
        self.encoder: F1FeatureEncoder | None = None
        self.context_rows = 0
        self.reference_rows = 0

    def fit(
        self,
        context_features: pd.DataFrame,
        labels: Any,
        *,
        preprocessing_reference: pd.DataFrame | None = None,
    ) -> SharedContextTFMRanker:
        require_f1_columns(context_features)
        targets = np.asarray(labels, dtype=np.float32).reshape(-1)
        if len(targets) != len(context_features):
            raise ValueError(
                f"context feature/label row mismatch: {len(context_features)} != {len(targets)}"
            )
        if not len(targets):
            raise ValueError("ranking context must contain at least one labelled row")
        if not np.isfinite(targets).all() or ((targets < 0) | (targets > 1)).any():
            raise ValueError("ranking labels must be finite editing-efficiency fractions in [0, 1]")
        reference = context_features if preprocessing_reference is None else preprocessing_reference
        require_f1_columns(reference)
        self.encoder = F1FeatureEncoder.fit(reference)
        if self.estimator is None:
            from primetab.backends import create_model_factory

            factory = create_model_factory(
                {"name": self.backend, "kwargs": dict(self.backend_kwargs)}
            )
            self.estimator = factory("regression")
        self.estimator.fit(self.encoder.transform(context_features), targets)
        self.context_rows = len(context_features)
        self.reference_rows = len(reference)
        return self

    def score(self, candidates: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        del candidates
        if self.encoder is None or self.estimator is None:
            raise RuntimeError("call fit(context_features, labels) before ranking")
        if features.index.name != "design_id":
            raise ValueError("ranking feature table must be indexed by design_id")
        prediction = np.asarray(
            self.estimator.predict(self.encoder.transform(features)),
            dtype=np.float64,
        ).reshape(-1)
        if len(prediction) != len(features):
            raise ValueError(f"ranker returned {len(prediction)} scores for {len(features)} rows")
        if not np.isfinite(prediction).all():
            raise FloatingPointError("ranker returned non-finite scores")
        return pd.Series(prediction, index=features.index, name="predicted_efficiency")

    def metadata(self) -> Mapping[str, Any]:
        return {
            "ranker": "shared_context_tfm",
            "backend": self.backend,
            "backend_kwargs": dict(self.backend_kwargs),
            "context_rows": self.context_rows,
            "preprocessing_reference_rows": self.reference_rows,
            "feature_encoding": None if self.encoder is None else self.encoder.summary(),
        }
