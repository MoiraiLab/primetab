"""Query-aware greedy source-anchor selection for TL-ANDI."""

from __future__ import annotations

import heapq

import numpy as np
from scipy.spatial.distance import cdist


def build_anchor_cost(
    X_query: np.ndarray,
    X_source: np.ndarray,
    discrepancy: np.ndarray,
    penalty: float,
    *,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Return squared feature distance plus the posterior-discrepancy penalty."""

    query = np.asarray(X_query, dtype=float)
    source = np.asarray(X_source, dtype=float)
    delta = np.asarray(discrepancy, dtype=float).reshape(-1)
    if query.ndim != 2 or source.ndim != 2 or query.shape[1] != source.shape[1]:
        raise ValueError("query and source must be compatible two-dimensional arrays")
    if len(source) != len(delta):
        raise ValueError("one discrepancy value is required per source row")
    if penalty < 0:
        raise ValueError("penalty must be non-negative")
    spatial = cdist(query, source, metric="sqeuclidean")
    spatial += float(penalty) * np.square(delta)[None, :]
    return spatial.astype(dtype, copy=False)


def greedy_anchor_indices(cost: np.ndarray, budget: int) -> np.ndarray:
    """Minimise average query-to-nearest-anchor cost with lazy greedy selection.

    The objective and initial dummy cost follow the paper. Lazy gain updates are
    mathematically equivalent to the ordinary greedy loop but avoid rescanning
    every source point after every selection in typical inputs. ``budget`` is a
    maximum: selection stops once no remaining row improves the objective. This
    prevents arbitrary source-order filling when the budget exceeds the number
    of useful anchors.
    """

    matrix = np.asarray(cost)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("cost must have shape (n_query, n_source) with non-zero dimensions")
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError("cost must be numeric")
    if not np.isfinite(matrix).all():
        raise ValueError("cost contains NaN or infinite values")
    if budget < 1:
        raise ValueError("budget must be positive")
    budget = min(int(budget), matrix.shape[1])

    dummy_cost = float(matrix.max()) + max(1e-12, abs(float(matrix.max())) * 1e-12)
    current = np.full(matrix.shape[0], dummy_cost, dtype=float)
    initial_gains = (dummy_cost - matrix).sum(axis=0, dtype=np.float64)
    heap = [(-float(gain), int(index)) for index, gain in enumerate(initial_gains)]
    heapq.heapify(heap)
    selected: list[int] = []
    selected_mask = np.zeros(matrix.shape[1], dtype=bool)

    while len(selected) < budget:
        while heap:
            _negative_bound, index = heapq.heappop(heap)
            if selected_mask[index]:
                continue
            exact_gain = float(np.maximum(current - matrix[:, index], 0.0).sum())
            next_bound = -heap[0][0] if heap else -np.inf
            if exact_gain + 1e-12 >= next_bound:
                if exact_gain <= 0.0:
                    return np.asarray(selected, dtype=int)
                selected.append(index)
                selected_mask[index] = True
                current = np.minimum(current, matrix[:, index])
                break
            heapq.heappush(heap, (-exact_gain, index))
        else:  # pragma: no cover - protected by the validated dimensions
            break
    return np.asarray(selected, dtype=int)
