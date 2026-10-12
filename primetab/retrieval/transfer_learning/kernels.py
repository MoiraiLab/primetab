"""Kernel distillation used by TL-ANDI."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.spatial.distance import cdist, pdist


def estimate_bandwidths(
    X: np.ndarray,
    quantiles: Iterable[float],
    *,
    sample_size: int | None = 2_000,
    random_state: int = 0,
) -> np.ndarray:
    """Estimate distance quantiles with an optional reproducible row sample."""

    features = _as_float_matrix(X, "X")
    requested = np.asarray(tuple(quantiles), dtype=float)
    if requested.ndim != 1 or requested.size == 0:
        raise ValueError("quantiles must be a non-empty one-dimensional sequence")
    if np.any((requested <= 0.0) | (requested >= 1.0)):
        raise ValueError("bandwidth quantiles must lie strictly between 0 and 1")
    if len(features) < 2:
        raise ValueError("at least two source rows are required to estimate bandwidths")

    sampled = features
    if sample_size is not None and len(features) > sample_size:
        if sample_size < 2:
            raise ValueError("sample_size must be at least two")
        rng = np.random.default_rng(random_state)
        sampled = features[rng.choice(len(features), size=sample_size, replace=False)]

    distances = pdist(sampled, metric="euclidean")
    positive = distances[distances > 0]
    if positive.size == 0:
        raise ValueError("all sampled source rows are identical")
    bandwidths = np.quantile(positive, requested)
    return np.maximum(bandwidths, np.finfo(float).eps)


def local_kernel_labels(
    X: np.ndarray,
    y: np.ndarray,
    bandwidths: Iterable[float],
    *,
    batch_size: int = 256,
) -> dict[float, np.ndarray]:
    """Distill one label per source row with an Epanechnikov kernel.

    Distances are computed in batches, so a 20,000-row source table does not
    require materialising its full pairwise distance matrix.
    """

    features = _as_float_matrix(X, "X")
    targets = np.asarray(y, dtype=float).reshape(-1)
    if len(features) != len(targets):
        raise ValueError("X and y must have the same number of rows")
    values = np.asarray(tuple(bandwidths), dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(values <= 0):
        raise ValueError("bandwidths must be a non-empty sequence of positive values")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    outputs = {float(bandwidth): np.empty(len(features), dtype=float) for bandwidth in values}
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        distances = cdist(features[start:stop], features, metric="euclidean")
        for bandwidth in values:
            ratio = distances / bandwidth
            weights = 0.75 * np.maximum(1.0 - ratio * ratio, 0.0)
            denominator = weights.sum(axis=1)
            # Every row is its own neighbour, so this guard only covers corrupt input.
            denominator = np.maximum(denominator, np.finfo(float).eps)
            outputs[float(bandwidth)][start:stop] = weights @ targets / denominator
    return outputs


def _as_float_matrix(X: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional numeric array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array
