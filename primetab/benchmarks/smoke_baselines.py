#!/usr/bin/env python3
"""Fit required conventional baselines on training cells and score only HeLa."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

from primetab.benchmarks.baselines import BASELINE_FEATURES, build_baseline_models
from primetab.benchmarks.split_policy import SplitPolicy
from primetab.paths import DATA_ROOT, BENCHMARK_WORKSPACE

DATA_DIR = DATA_ROOT / "deepprime/processed/pe2max"
OUTPUT = BENCHMARK_WORKSPACE / "reports/smoke/baselines_HeLa.json"


def main() -> int:
    policy = SplitPolicy.from_yaml(BENCHMARK_WORKSPACE / "configs/splits.yaml")
    train = pd.read_parquet(DATA_DIR / "train_HEK293T_A549.parquet")
    validation = pd.read_parquet(DATA_DIR / "cross_cell_validation_HeLa.parquet")
    policy.assert_fit_frame(train, purpose="conventional baseline smoke fit")
    if set(validation["cell_line"]) != {policy.validation_cell}:
        raise ValueError("Smoke validation data are not exclusively HeLa")
    if any(
        name in BASELINE_FEATURES
        for name in ("cell_line", "source_dataset_id", "upstream_fold")
    ):
        raise ValueError("Forbidden lineage or cell feature entered the baseline feature set")

    X_train = train[BASELINE_FEATURES]
    y_train = train["editing_efficiency"].to_numpy()
    X_validation = validation[BASELINE_FEATURES]
    y_validation = validation["editing_efficiency"].to_numpy()

    results = {}
    for name, estimator in build_baseline_models().items():
        fit_start = time.perf_counter()
        estimator.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_start
        predict_start = time.perf_counter()
        prediction = np.asarray(estimator.predict(X_validation))
        predict_seconds = time.perf_counter() - predict_start
        if not np.isfinite(prediction).all():
            raise ValueError(f"{name} produced non-finite predictions")
        results[name] = {
            "spearman_HeLa": float(spearmanr(y_validation, prediction).statistic),
            "mae_HeLa": float(mean_absolute_error(y_validation, prediction)),
            "fit_seconds": round(fit_seconds, 4),
            "predict_seconds": round(predict_seconds, 4),
        }
        print(name, results[name])

    report = {
        "status": "PASS",
        "purpose": "smoke test only; no hyperparameter search",
        "training_cells": list(policy.train_cells),
        "validation_cell": policy.validation_cell,
        "sealed_test_opened": False,
        "rows": {"train": len(train), "HeLa": len(validation)},
        "features": BASELINE_FEATURES,
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
