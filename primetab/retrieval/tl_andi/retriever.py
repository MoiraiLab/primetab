"""Use TL-ANDI's distillation and anchor objective as a context retriever."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from primetab.backends import ModelFactory
from primetab.retrieval.context_retrieval import farthest_first_indices
from primetab.retrieval.transfer_learning.anchoring import greedy_anchor_indices
from primetab.retrieval.transfer_learning.kernels import estimate_bandwidths, local_kernel_labels
from primetab.retrieval.transfer_learning.prediction import continuous_prediction
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RetrievalCandidate:
    bandwidth: float
    penalty: float
    indices: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class PreparedTLANDIRetrieval:
    """Reusable source distillation and target/source discrepancy."""

    source_scaled: np.ndarray
    query_scaler: StandardScaler
    distilled_labels: dict[float, np.ndarray]
    discrepancy: dict[float, np.ndarray]
    bandwidths: np.ndarray


def prepare_tl_andi_retrieval(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target_context: np.ndarray,
    y_target_context: np.ndarray,
    *,
    model_factory: ModelFactory,
    bandwidth_quantiles: tuple[float, ...],
    bandwidth_sample_size: int = 2_000,
    kernel_batch_size: int = 256,
    random_state: int = 42,
) -> PreparedTLANDIRetrieval:
    """Fit the target pilot and prepare every bandwidth without query labels."""

    source = np.asarray(X_source, dtype=float)
    target = np.asarray(X_target_context, dtype=float)
    source_y = np.asarray(y_source, dtype=float).reshape(-1)
    target_y = np.asarray(y_target_context, dtype=float).reshape(-1)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source and target contexts must have compatible feature matrices")
    if len(source_y) != len(source) or len(target_y) != len(target):
        raise ValueError("feature and label rows do not align")

    scaler = StandardScaler().fit(source)
    source_scaled = scaler.transform(source).astype(np.float32, copy=False)
    target_scaled = scaler.transform(target).astype(np.float32, copy=False)
    pilot = model_factory("regression")
    pilot.fit(target_scaled, target_y)
    target_on_source = continuous_prediction(pilot, source_scaled, "regression")

    bandwidths = estimate_bandwidths(
        source_scaled,
        bandwidth_quantiles,
        sample_size=bandwidth_sample_size,
        random_state=random_state,
    )
    distilled = local_kernel_labels(
        source_scaled,
        source_y,
        bandwidths,
        batch_size=kernel_batch_size,
    )
    discrepancy = {
        float(bandwidth): labels - target_on_source for bandwidth, labels in distilled.items()
    }
    return PreparedTLANDIRetrieval(
        source_scaled=source_scaled,
        query_scaler=scaler,
        distilled_labels=distilled,
        discrepancy=discrepancy,
        bandwidths=bandwidths,
    )


def retarget_tl_andi_retrieval(
    prepared: PreparedTLANDIRetrieval,
    X_target_context: np.ndarray,
    y_target_context: np.ndarray,
    *,
    model_factory: ModelFactory,
) -> PreparedTLANDIRetrieval:
    """Reuse source distillation while refitting the target pilot context."""

    target = np.asarray(X_target_context, dtype=float)
    target_y = np.asarray(y_target_context, dtype=float).reshape(-1)
    target_scaled = prepared.query_scaler.transform(target).astype(np.float32, copy=False)
    if len(target_y) != len(target_scaled):
        raise ValueError("target context features and labels do not align")
    pilot = model_factory("regression")
    pilot.fit(target_scaled, target_y)
    target_on_source = continuous_prediction(
        pilot,
        prepared.source_scaled,
        "regression",
    )
    discrepancy = {
        float(bandwidth): labels - target_on_source
        for bandwidth, labels in prepared.distilled_labels.items()
    }
    return PreparedTLANDIRetrieval(
        source_scaled=prepared.source_scaled,
        query_scaler=prepared.query_scaler,
        distilled_labels=prepared.distilled_labels,
        discrepancy=discrepancy,
        bandwidths=prepared.bandwidths,
    )


def retrieve_tl_andi_context(
    prepared: PreparedTLANDIRetrieval,
    X_query: np.ndarray,
    *,
    bandwidth: float,
    penalty: float,
    budget: int,
    query_representatives: int | None = 256,
    spatial_cost: np.ndarray | None = None,
) -> RetrievalCandidate:
    """Select a useful distilled context up to the supplied budget."""

    if spatial_cost is None:
        spatial_cost = prepare_tl_andi_query_cost(
            prepared,
            X_query,
            query_representatives=query_representatives,
        )
    selected_bandwidth = min(
        prepared.distilled_labels,
        key=lambda value: abs(float(value) - float(bandwidth)),
    )
    cost = np.asarray(spatial_cost, dtype=np.float32).copy()
    if cost.ndim != 2 or cost.shape[1] != len(prepared.source_scaled):
        raise ValueError("spatial_cost is incompatible with the prepared source")
    cost += (
        float(penalty)
        * np.square(prepared.discrepancy[selected_bandwidth], dtype=np.float32)[None, :]
    )
    indices = greedy_anchor_indices(cost, budget)
    return RetrievalCandidate(
        bandwidth=float(selected_bandwidth),
        penalty=float(penalty),
        indices=indices,
        labels=prepared.distilled_labels[selected_bandwidth][indices],
    )


def prepare_tl_andi_query_cost(
    prepared: PreparedTLANDIRetrieval,
    X_query: np.ndarray,
    *,
    query_representatives: int | None = None,
) -> np.ndarray:
    """Cache the parameter-independent query/source squared distances."""

    query = np.asarray(X_query, dtype=float)
    query_scaled = prepared.query_scaler.transform(query).astype(np.float32, copy=False)
    if query_representatives is not None and len(query_scaled) > query_representatives:
        representative_indices = farthest_first_indices(
            query_scaled,
            query_representatives,
        )
        query_scaled = query_scaled[representative_indices]
    return cdist(query_scaled, prepared.source_scaled, metric="sqeuclidean").astype(
        np.float32,
        copy=False,
    )
