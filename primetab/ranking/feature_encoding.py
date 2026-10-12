"""Frozen F1 feature encoding shared by generator-to-ranker inference."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from primetab.features.features import F1_SEQUENCE_V1_COLUMNS

CATEGORICAL_DNA_WIDTHS = {
    "pam_before": 3,
    "pam_after": 3,
    "base_before_edit": 1,
    "base_after_edit": 1,
    "base_before_nick": 1,
    "base_after_nick": 1,
}


def require_f1_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in F1_SEQUENCE_V1_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"feature table is missing F1 columns: {missing}")


def encoded_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for column in F1_SEQUENCE_V1_COLUMNS:
        width = CATEGORICAL_DNA_WIDTHS.get(column)
        if width is None:
            names.append(column)
            continue
        names.extend(
            f"{column}__position_{position}__{base}"
            for position in range(width)
            for base in "ACGT"
        )
    return tuple(names)


def encode_raw_features(frame: pd.DataFrame) -> np.ndarray:
    """Encode the frozen 73 raw columns into the formal 107-column numeric layout."""

    require_f1_columns(frame)
    encoded: list[np.ndarray] = []
    for column in F1_SEQUENCE_V1_COLUMNS:
        width = CATEGORICAL_DNA_WIDTHS.get(column)
        if width is None:
            encoded.append(
                pd.to_numeric(frame[column], errors="coerce")
                .to_numpy(dtype=np.float32)
                .reshape(-1, 1)
            )
            continue
        values = frame[column].astype("string").fillna("").str.upper()
        for position in range(width):
            bases = values.str[position]
            encoded.extend(
                bases.eq(base).to_numpy(dtype=np.float32).reshape(-1, 1)
                for base in "ACGT"
            )
    return np.concatenate(encoded, axis=1)


@dataclass(frozen=True)
class F1FeatureEncoder:
    """Source-fitted imputation and scaling for generated F1 feature rows."""

    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, reference_features: pd.DataFrame) -> F1FeatureEncoder:
        if reference_features.empty:
            raise ValueError("preprocessing reference must contain at least one row")
        values = encode_raw_features(reference_features).astype(np.float64)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(values, axis=0)
        if not np.isfinite(median).all():
            names = encoded_feature_names()
            missing = [names[index] for index in np.flatnonzero(~np.isfinite(median))]
            raise ValueError(f"reference cannot impute entirely missing features: {missing}")
        imputed = np.where(np.isfinite(values), values, median)
        mean = imputed.mean(axis=0)
        scale = imputed.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return cls(
            median=median.astype(np.float32),
            mean=mean.astype(np.float32),
            scale=scale.astype(np.float32),
        )

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        values = encode_raw_features(features)
        if values.shape[1] != self.width:
            raise ValueError(f"encoded feature width changed: {values.shape[1]} != {self.width}")
        imputed = np.where(np.isfinite(values), values, self.median)
        transformed = (imputed - self.mean) / self.scale
        if not np.isfinite(transformed).all():
            raise ValueError("transformed features contain non-finite values")
        return transformed.astype(np.float32, copy=False)

    @property
    def width(self) -> int:
        return len(self.mean)

    def summary(self) -> dict[str, object]:
        return {
            "raw_feature_set": "F1_sequence_v1",
            "raw_width": len(F1_SEQUENCE_V1_COLUMNS),
            "encoded_width": self.width,
            "categorical_encoding": "fixed_position_dna_one_hot",
            "imputation": "reference_median",
            "scaling": "reference_standard",
        }
