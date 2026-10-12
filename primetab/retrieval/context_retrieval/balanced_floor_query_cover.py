"""Regression adaptation of PolicyContext's strongest uncorrected selector.

The original ``balanced_floor_qct_ls_uncorrected`` method was developed for
classification.  This module preserves its two selection stages while making
the floor meaningful for regression: continuous training labels are divided
into quantile bins, a balanced random floor is drawn across those bins, and a
query-conditioned depthwise transfer fills the remaining budget.  Query labels
are never inspected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class BalancedFloorQueryCoverConfig:
    """Fully specified selector candidate."""

    budget: int
    floor_fraction: float
    representative_ratio: float
    label_bins: int = 8
    neighbor_depth: int = 2_048
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.budget < 1:
            raise ValueError("budget must be positive")
        if not 0.0 <= self.floor_fraction <= 1.0:
            raise ValueError("floor_fraction must lie in [0, 1]")
        if self.representative_ratio <= 0.0:
            raise ValueError("representative_ratio must be positive")
        if self.label_bins < 2:
            raise ValueError("label_bins must be at least two")
        if self.neighbor_depth < 1:
            raise ValueError("neighbor_depth must be positive")

    @property
    def candidate_id(self) -> str:
        return (
            f"bfqc__floor{self.floor_fraction:g}"
            f"__ratio{self.representative_ratio:g}__bins{self.label_bins}"
        )


@dataclass(frozen=True)
class ContextSelection:
    """Selected local row indices plus audit information."""

    indices: np.ndarray
    floor_indices: np.ndarray
    query_indices: np.ndarray
    representative_query_indices: np.ndarray
    maximum_neighbor_rank: int
    config: BalancedFloorQueryCoverConfig


def farthest_first_indices(
    X: np.ndarray,
    count: int,
    *,
    tie_break_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Deterministic centroid-start farthest-first traversal."""

    features = _matrix(X, "X")
    count = min(max(int(count), 0), len(features))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    ids = (
        np.arange(len(features), dtype=np.int64)
        if tie_break_ids is None
        else np.asarray(tie_break_ids, dtype=np.int64)
    )
    if ids.shape != (len(features),):
        raise ValueError("tie_break_ids must contain one value per row")

    centroid_distance = np.square(features - features.mean(axis=0)).sum(axis=1)
    first = int(np.lexsort((ids, centroid_distance))[0])
    selected = [first]
    minimum_distance = np.square(features - features[first]).sum(axis=1)
    minimum_distance[first] = -np.inf
    while len(selected) < count:
        available = np.flatnonzero(np.isfinite(minimum_distance))
        winner = int(available[np.lexsort((ids[available], -minimum_distance[available]))[0]])
        selected.append(winner)
        distance = np.square(features - features[winner]).sum(axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[np.asarray(selected, dtype=np.int64)] = -np.inf
    return np.asarray(selected, dtype=np.int64)


def select_balanced_floor_query_cover(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    config: BalancedFloorQueryCoverConfig,
    *,
    train_row_ids: np.ndarray | None = None,
    query_row_ids: np.ndarray | None = None,
) -> ContextSelection:
    """Select one exact-size context without observing query outcomes."""

    train = _matrix(X_train, "X_train")
    query = _matrix(X_query, "X_query")
    labels = np.asarray(y_train, dtype=float).reshape(-1)
    if train.shape[1] != query.shape[1]:
        raise ValueError("train and query feature counts differ")
    if len(labels) != len(train) or not np.isfinite(labels).all():
        raise ValueError("y_train must be finite and aligned with X_train")
    if config.budget > len(train):
        raise ValueError("budget exceeds the available training rows")

    train_ids = (
        np.arange(len(train), dtype=np.int64)
        if train_row_ids is None
        else np.asarray(train_row_ids, dtype=np.int64)
    )
    query_ids = (
        np.arange(len(query), dtype=np.int64)
        if query_row_ids is None
        else np.asarray(query_row_ids, dtype=np.int64)
    )
    if train_ids.shape != (len(train),) or query_ids.shape != (len(query),):
        raise ValueError("row ids must align with their feature matrices")

    scaler = StandardScaler().fit(train)
    train_scaled = scaler.transform(train).astype(np.float32, copy=False)
    query_scaled = scaler.transform(query).astype(np.float32, copy=False)

    floor_budget = round(config.floor_fraction * config.budget)
    floor = _balanced_quantile_floor(
        labels,
        floor_budget,
        bins=config.label_bins,
        random_state=config.random_state,
    )
    query_budget = config.budget - len(floor)
    if query_budget == 0:
        ordered = _canonical_order(floor, train_ids)
        return ContextSelection(
            indices=ordered,
            floor_indices=ordered.copy(),
            query_indices=np.empty(0, dtype=np.int64),
            representative_query_indices=np.empty(0, dtype=np.int64),
            maximum_neighbor_rank=0,
            config=config,
        )

    depth = min(
        len(train),
        max(config.neighbor_depth, config.budget + len(floor)),
    )
    neighbor_index = NearestNeighbors(
        n_neighbors=depth,
        algorithm="auto",
        metric="euclidean",
        n_jobs=-1,
    ).fit(train_scaled)
    distances, neighbors = neighbor_index.kneighbors(query_scaled, return_distance=True)
    for row in range(len(query)):
        order = np.lexsort((train_ids[neighbors[row]], distances[row]))
        neighbors[row] = neighbors[row, order]

    representative_count = min(
        len(query),
        max(1, round(config.representative_ratio * config.budget)),
    )
    representatives = farthest_first_indices(
        query_scaled,
        representative_count,
        tie_break_ids=query_ids,
    )
    transferred, maximum_rank = _depthwise_transfer(
        neighbors,
        representatives,
        query_budget,
        excluded=set(map(int, floor)),
    )
    combined = np.concatenate([floor, transferred])
    if len(combined) != config.budget or len(np.unique(combined)) != config.budget:
        raise RuntimeError("selector failed to produce an exact unique context")
    ordered = _canonical_order(combined, train_ids)
    floor_set = set(map(int, floor))
    return ContextSelection(
        indices=ordered,
        floor_indices=np.asarray(
            [index for index in ordered if int(index) in floor_set], dtype=np.int64
        ),
        query_indices=np.asarray(
            [index for index in ordered if int(index) not in floor_set], dtype=np.int64
        ),
        representative_query_indices=representatives,
        maximum_neighbor_rank=maximum_rank,
        config=config,
    )


def _balanced_quantile_floor(
    labels: np.ndarray,
    budget: int,
    *,
    bins: int,
    random_state: int,
) -> np.ndarray:
    if budget == 0:
        return np.empty(0, dtype=np.int64)
    edges = np.unique(np.quantile(labels, np.linspace(0.0, 1.0, bins + 1)))
    encoded = np.digitize(labels, edges[1:-1], right=True)
    groups = [np.flatnonzero(encoded == value) for value in np.unique(encoded)]
    rng = np.random.default_rng(random_state)
    pools = [rng.permutation(group) for group in groups]
    cursor = np.zeros(len(pools), dtype=np.int64)
    selected: list[int] = []
    while len(selected) < budget:
        progressed = False
        for position, pool in enumerate(pools):
            if cursor[position] >= len(pool):
                continue
            selected.append(int(pool[cursor[position]]))
            cursor[position] += 1
            progressed = True
            if len(selected) == budget:
                break
        if not progressed:
            raise RuntimeError("quantile floor exhausted before filling its budget")
    return np.asarray(selected, dtype=np.int64)


def _depthwise_transfer(
    neighbors: np.ndarray,
    representatives: np.ndarray,
    budget: int,
    *,
    excluded: set[int],
) -> tuple[np.ndarray, int]:
    chosen: list[int] = []
    taken = set(excluded)
    for depth in range(neighbors.shape[1]):
        for query_position in representatives:
            row = int(neighbors[int(query_position), depth])
            if row in taken:
                continue
            taken.add(row)
            chosen.append(row)
            if len(chosen) == budget:
                return np.asarray(chosen, dtype=np.int64), depth + 1
    raise RuntimeError("neighbor cache exhausted before filling the query-conditioned budget")


def _canonical_order(indices: np.ndarray, row_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    return values[np.argsort(row_ids[values], kind="stable")]


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array
