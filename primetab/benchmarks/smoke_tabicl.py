#!/usr/bin/env python3
"""Run the primary TFM on all training rows and HeLa only."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tabicl import TabICLRegressor

from primetab.benchmarks.baselines import BASELINE_FEATURES
from primetab.benchmarks.split_policy import SplitPolicy
from primetab.paths import DATA_ROOT, BENCHMARK_WORKSPACE

DATA_DIR = DATA_ROOT / "deepprime/processed/pe2max"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    policy = SplitPolicy.from_yaml(BENCHMARK_WORKSPACE / "configs/splits.yaml")
    train = pd.read_parquet(DATA_DIR / "train_HEK293T_A549.parquet")
    validation = pd.read_parquet(DATA_DIR / "cross_cell_validation_HeLa.parquet")
    policy.assert_fit_frame(train, purpose="TabICL smoke fit")
    if set(validation["cell_line"]) != {policy.validation_cell}:
        raise ValueError("TabICL smoke validation is not exclusively HeLa")

    model = TabICLRegressor(
        n_estimators=args.n_estimators,
        batch_size=min(args.n_estimators, 4),
        model_path=BENCHMARK_WORKSPACE
        / "models/checkpoints/tabicl-regressor-v2-20260212.ckpt",
        allow_auto_download=False,
        device=args.device,
        use_amp=False if args.device == "cpu" else "auto",
        random_state=42,
        n_jobs=1,
    )
    fit_start = time.perf_counter()
    model.fit(train[BASELINE_FEATURES], train["editing_efficiency"])
    fit_seconds = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    prediction = np.asarray(model.predict(validation[BASELINE_FEATURES]))
    predict_seconds = time.perf_counter() - predict_start
    if not np.isfinite(prediction).all():
        raise ValueError("TabICL produced non-finite HeLa predictions")

    report = {
        "status": "PASS",
        "model": "TabICLRegressor",
        "checkpoint": "tabicl-regressor-v2-20260212.ckpt",
        "n_estimators": args.n_estimators,
        "device": args.device,
        "training_cells": list(policy.train_cells),
        "validation_cell": policy.validation_cell,
        "sealed_test_opened": False,
        "rows": {"train": len(train), "HeLa": len(validation)},
        "feature_count": len(BASELINE_FEATURES),
        "spearman_HeLa": float(spearmanr(validation["editing_efficiency"], prediction).statistic),
        "fit_seconds": round(fit_seconds, 4),
        "predict_seconds": round(predict_seconds, 4),
    }
    output = (
        BENCHMARK_WORKSPACE
        / f"reports/smoke/tabicl_{args.n_estimators}est_HeLa.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
