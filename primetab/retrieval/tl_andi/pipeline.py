"""Paper-faithful, model-agnostic implementation of TL-ANDI."""

from __future__ import annotations

import numpy as np
from primetab.backends import ModelFactory
from primetab.retrieval.transfer_learning.anchoring import build_anchor_cost, greedy_anchor_indices
from primetab.retrieval.transfer_learning.kernels import estimate_bandwidths, local_kernel_labels
from primetab.retrieval.transfer_learning.prediction import (
    continuous_prediction,
    target_only_predict,
    vanilla_transfer_predict,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import TLANDIConfig
from .result import CandidateScore, TLANDIResult


class TLANDIPipeline:
    """Select distilled source anchors, validate transfer, and refit on all target rows."""

    def __init__(self, *, model_factory: ModelFactory, config: TLANDIConfig) -> None:
        self.model_factory = model_factory
        self.config = config

    def run(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        X_target: np.ndarray,
        y_target: np.ndarray,
        X_query: np.ndarray,
    ) -> TLANDIResult:
        X_source, y_source, X_target, y_target, X_query = self._validate_inputs(
            X_source, y_source, X_target, y_target, X_query
        )
        if self.config.task == "classification" and (
            not np.isin(y_source, (0.0, 1.0)).all()
            or not np.isin(y_target, (0.0, 1.0)).all()
        ):
            raise ValueError("binary classification labels must be encoded as 0 and 1")
        if self.config.standardize_features:
            scaler = StandardScaler().fit(X_source)
            X_source = scaler.transform(X_source)
            X_target = scaler.transform(X_target)
            X_query = scaler.transform(X_query)

        calibration_indices, validation_indices = self._split_target(y_target)
        X_calibration = X_target[calibration_indices]
        y_calibration = y_target[calibration_indices]
        X_validation = X_target[validation_indices]
        y_validation = y_target[validation_indices]

        # The method operates on conditional means. Binary classification uses
        # a regression TFM over 0/1 labels so smoothed pseudo-labels stay valid.
        target_model = self.model_factory("regression")
        target_model.fit(X_calibration, y_calibration)
        target_query = np.vstack([X_source, X_validation])
        target_prediction = continuous_prediction(target_model, target_query, "regression")
        target_on_source = target_prediction[: len(X_source)]
        target_on_validation = target_prediction[len(X_source) :]
        if self.config.task == "classification":
            target_on_source = np.clip(target_on_source, 0.0, 1.0)
            target_on_validation = np.clip(target_on_validation, 0.0, 1.0)

        scores: list[CandidateScore] = [
            CandidateScore(
                method="target_only",
                validation_mse=_mse(y_validation, target_on_validation),
            )
        ]
        best_score = scores[0].validation_mse
        best_bandwidth: float | None = None
        best_penalty: float | None = None
        best_anchors = np.empty(0, dtype=int)
        best_labels = np.empty(0, dtype=float)

        bandwidths = estimate_bandwidths(
            X_source,
            self.config.bandwidth_quantiles,
            sample_size=self.config.bandwidth_sample_size,
            random_state=self.config.random_state,
        )
        distilled_by_bandwidth = local_kernel_labels(
            X_source,
            y_source,
            bandwidths,
            batch_size=self.config.kernel_batch_size,
        )

        for bandwidth in bandwidths:
            distilled = distilled_by_bandwidth[float(bandwidth)]
            discrepancy = distilled - target_on_source
            for penalty in self.config.lambda_values:
                cost = build_anchor_cost(X_query, X_source, discrepancy, penalty)
                anchors = greedy_anchor_indices(cost, self.config.context_budget)
                validation_prediction = vanilla_transfer_predict(
                    self.model_factory,
                    self.config.task,
                    X_source[anchors],
                    distilled[anchors],
                    X_calibration,
                    y_calibration,
                    X_validation,
                )
                validation_mse = _mse(y_validation, validation_prediction)
                scores.append(
                    CandidateScore(
                        method="tl_andi",
                        validation_mse=validation_mse,
                        bandwidth=float(bandwidth),
                        penalty=float(penalty),
                        anchor_count=len(anchors),
                    )
                )
                if validation_mse < best_score:
                    best_score = validation_mse
                    best_bandwidth = float(bandwidth)
                    best_penalty = float(penalty)
                    best_anchors = anchors.copy()
                    best_labels = distilled[anchors].copy()

        if best_bandwidth is None:
            prediction = target_only_predict(
                self.model_factory,
                self.config.task,
                X_target,
                y_target,
                X_query,
            )
            selected_method = "target_only"
        else:
            prediction = vanilla_transfer_predict(
                self.model_factory,
                self.config.task,
                X_source[best_anchors],
                best_labels,
                X_target,
                y_target,
                X_query,
            )
            selected_method = "tl_andi"

        return TLANDIResult(
            predictions=np.asarray(prediction, dtype=float),
            selected_method=selected_method,
            selected_bandwidth=best_bandwidth,
            selected_penalty=best_penalty,
            selected_anchor_indices=best_anchors,
            candidate_scores=tuple(scores),
            calibration_indices=calibration_indices,
            validation_indices=validation_indices,
        )

    def _split_target(self, y_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        indices = np.arange(len(y_target))
        stratify = y_target if self.config.task == "classification" else None
        try:
            calibration, validation = train_test_split(
                indices,
                train_size=self.config.calibration_fraction,
                random_state=self.config.random_state,
                shuffle=True,
                stratify=stratify,
            )
        except ValueError as exc:
            if self.config.task == "classification":
                raise ValueError(
                    "target labels must contain enough examples of both classes for a stratified split"
                ) from exc
            raise
        return np.sort(calibration), np.sort(validation)

    @staticmethod
    def _validate_inputs(
        X_source: np.ndarray,
        y_source: np.ndarray,
        X_target: np.ndarray,
        y_target: np.ndarray,
        X_query: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        matrices = [np.asarray(value, dtype=float) for value in (X_source, X_target, X_query)]
        if any(value.ndim != 2 for value in matrices):
            raise ValueError("source, target, and query features must be two-dimensional")
        if len({value.shape[1] for value in matrices}) != 1:
            raise ValueError("source, target, and query must have the same feature count")
        if any(len(value) == 0 for value in matrices):
            raise ValueError("source, target, and query tables must be non-empty")
        if any(not np.isfinite(value).all() for value in matrices):
            raise ValueError("features contain NaN or infinite values")
        source_y = np.asarray(y_source, dtype=float).reshape(-1)
        target_y = np.asarray(y_target, dtype=float).reshape(-1)
        if len(source_y) != len(matrices[0]) or len(target_y) != len(matrices[1]):
            raise ValueError("feature and label row counts do not match")
        if len(target_y) < 4:
            raise ValueError("at least four target rows are required for calibration and validation")
        if not np.isfinite(source_y).all() or not np.isfinite(target_y).all():
            raise ValueError("labels contain NaN or infinite values")
        return matrices[0], source_y, matrices[1], target_y, matrices[2]


def _mse(y_true: np.ndarray, y_prediction: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(y_true, dtype=float) - y_prediction)))
