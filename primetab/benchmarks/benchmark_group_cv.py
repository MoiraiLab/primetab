#!/usr/bin/env python3
"""Benchmark TabICL and conventional baselines using training-cell group CV only."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tabicl import TabICLRegressor

from primetab.benchmarks.baselines import BASELINE_FEATURES, build_baseline_models
from primetab.benchmarks.split_policy import SplitPolicy
from primetab.paths import DATA_ROOT, BENCHMARK_WORKSPACE

DATA_DIR = DATA_ROOT / "deepprime/processed/pe2max"
OUTPUT = BENCHMARK_WORKSPACE / "reports/benchmarks/group_cv_training_cells.json"


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> int:
    policy = SplitPolicy.from_yaml(BENCHMARK_WORKSPACE / "configs/splits.yaml")
    table = pd.read_parquet(DATA_DIR / "train_HEK293T_A549.parquet")
    folds = pd.read_csv(DATA_DIR / "train_group_cv_folds.csv")
    policy.assert_fit_frame(table, purpose="training-cell group CV benchmark")
    policy.assert_group_column(table)

    identity_columns = ["source_dataset_id", "source_row_id", "cell_line", "design_id"]
    if not table[identity_columns].reset_index(drop=True).equals(
        folds[identity_columns].reset_index(drop=True)
    ):
        raise ValueError("Fold manifest order or identity differs from the training table")
    fold_id = folds["group_cv_fold"].to_numpy()
    if set(table["cell_line"]) != set(policy.train_cells):
        raise ValueError("Held-out cell entered training-cell group CV")

    estimators = build_baseline_models()
    estimators["TabICL"] = None
    per_model: dict[str, dict] = {}

    for name, estimator_template in estimators.items():
        fold_scores: list[float] = []
        fold_times: list[float] = []
        for fold in sorted(np.unique(fold_id)):
            train_mask = fold_id != fold
            validation_mask = fold_id == fold
            train_groups = set(table.loc[train_mask, policy.group_column])
            validation_groups = set(table.loc[validation_mask, policy.group_column])
            if train_groups & validation_groups:
                raise ValueError(f"Target leakage detected in fold {fold}")

            if name == "TabICL":
                estimator = TabICLRegressor(
                    n_estimators=1,
                    batch_size=1,
                    model_path=BENCHMARK_WORKSPACE
                    / "models/checkpoints/tabicl-regressor-v2-20260212.ckpt",
                    allow_auto_download=False,
                    device="cpu",
                    use_amp=False,
                    random_state=42,
                    n_jobs=1,
                )
            else:
                from sklearn.base import clone

                estimator = clone(estimator_template)
                single_thread_params = {
                    key: 1
                    for key in estimator.get_params()
                    if key == "n_jobs" or key.endswith("__n_jobs")
                }
                estimator.set_params(**single_thread_params)

            started = time.perf_counter()
            estimator.fit(
                table.loc[train_mask, BASELINE_FEATURES],
                table.loc[train_mask, "editing_efficiency"],
            )
            prediction = np.asarray(
                estimator.predict(table.loc[validation_mask, BASELINE_FEATURES])
            )
            elapsed = time.perf_counter() - started
            if not np.isfinite(prediction).all():
                raise ValueError(f"{name} produced non-finite predictions in fold {fold}")
            score = float(
                spearmanr(
                    table.loc[validation_mask, "editing_efficiency"], prediction
                ).statistic
            )
            fold_scores.append(score)
            fold_times.append(elapsed)
            print(name, "fold", int(fold), "spearman", round(score, 6), flush=True)

        per_model[name] = {
            "spearman_by_fold": fold_scores,
            "spearman_summary": summarize(fold_scores),
            "fit_plus_predict_seconds_by_fold": fold_times,
            "total_seconds": float(sum(fold_times)),
        }

    report = {
        "status": "PASS",
        "purpose": "initial feasibility benchmark; fixed smoke hyperparameters",
        "training_cells_only": list(policy.train_cells),
        "cross_cell_validation_used": False,
        "sealed_test_opened": False,
        "split_method": "5-fold GroupKFold by target_locus_id",
        "rows": len(table),
        "unique_target_loci": int(table[policy.group_column].nunique()),
        "feature_count": len(BASELINE_FEATURES),
        "models": per_model,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
