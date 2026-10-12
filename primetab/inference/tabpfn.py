"""A portable TabPFN inference engine with independent preprocessing per context.

The engine uses the standard ``tabpfn`` package. It fits one context and all its
queries together, with no fitted transforms crossing context boundaries, which is
what in-context, per-candidate scoring requires.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def default_weights() -> Path | None:
    """Resolve TabPFN weights from the environment. No machine-local default."""
    configured = os.environ.get('TFM_TABPFN_WEIGHTS')
    if configured:
        return Path(configured)
    cache = os.environ.get('TABPFN_MODEL_CACHE_DIR')
    if not cache:
        return None
    return Path(cache) / 'tabpfn-v3-regressor-v3_default.ckpt'


def constant_feature_prediction(X, y, query_count):
    """An explicit observed-only empirical predictor when no feature varies.

    TabPFN removes every constant input column and cannot process this context.
    No relation between candidate features and outcomes can be learned here.
    The interval is the empirical 10th--90th percentile of observed outcomes.
    """
    if not np.all(X == X[:1]):
        return None
    low, high = np.quantile(y, [.1, .9])
    return (np.full(query_count, float(np.mean(y, dtype=np.float64))),
            np.full(query_count, float((high - low) / 2)))


class TabPFNEngine:
    def __init__(self, device=None, model_path=None, n_estimators=8, seed=42):
        import torch
        from tabpfn import TabPFNRegressor

        if device is None:
            device = 'cpu'
        resolved = model_path or default_weights()
        if resolved is None:
            raise FileNotFoundError(
                'TabPFN weights missing: pass model_path or set TFM_TABPFN_WEIGHTS '
                'or TABPFN_MODEL_CACHE_DIR'
            )
        model_path = Path(resolved).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f'TabPFN weights missing: {model_path}')
        if str(device).startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is required when device is a CUDA device')
            torch.cuda.set_device(device)
        torch.set_num_threads(4)
        self.device = device
        self.last_prediction_kind = 'unfitted'
        self.model = TabPFNRegressor(
            model_path=model_path, device=device, n_estimators=n_estimators,
            auto_scale_n_estimators=False, random_state=seed, fit_mode='fit_preprocessors',
            inference_precision=torch.float32, memory_saving_mode=False,
            n_preprocessing_jobs=1, show_progress_bar=False,
        )

    def predict_context(self, X, y, queries, seed=42):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        queries = np.asarray(queries, dtype=np.float32)
        if X.ndim != 2 or queries.ndim != 2 or X.shape[1] != queries.shape[1]:
            raise ValueError('Context and query feature widths must match')
        if len(X) != len(y) or len(y) == 0:
            raise ValueError('Nonempty, aligned context is required')
        if not all(np.isfinite(v).all() for v in (X, y, queries)):
            raise ValueError('Inference requires finite numeric data')
        if len(queries) == 0:
            self.last_prediction_kind = 'empty_query'
            return np.empty(0), np.empty(0)
        if np.unique(y).size == 1:
            self.last_prediction_kind = 'constant_observed_target'
            return np.full(len(queries), float(y[0])), np.zeros(len(queries))
        constant = constant_feature_prediction(X, y, len(queries))
        if constant is not None:
            self.last_prediction_kind = 'constant_features_observed_empirical'
            return constant
        self.last_prediction_kind = 'tabpfn_public'
        self.model.random_state = seed
        self.model.fit(X, y)
        output = self.model.predict(queries, output_type='main', quantiles=[.1, .9])
        mean = np.asarray(output['mean'])
        width = (np.asarray(output['quantiles'][1]) - np.asarray(output['quantiles'][0])) / 2
        if not np.isfinite(mean).all() or not np.isfinite(width).all() or (width < 0).any():
            raise FloatingPointError('Invalid TabPFN prediction distribution')
        return mean, width


def with_domain(X, target: bool):
    return np.column_stack([X, np.full(len(X), int(target))]).astype(np.float32)
