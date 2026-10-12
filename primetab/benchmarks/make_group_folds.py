#!/usr/bin/env python3
"""Create five leakage-safe target-group folds within HEK293T+A549 only."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from primetab.benchmarks.split_policy import SplitPolicy
from primetab.paths import DATA_ROOT, BENCHMARK_WORKSPACE

DATA_DIR = DATA_ROOT / "deepprime/processed/pe2max"


def main() -> int:
    policy = SplitPolicy.from_yaml(BENCHMARK_WORKSPACE / "configs/splits.yaml")
    train = pd.read_parquet(DATA_DIR / "train_HEK293T_A549.parquet")
    policy.assert_fit_frame(train, purpose="internal grouped CV manifest")
    policy.assert_group_column(train)

    fold_id = np.full(len(train), -1, dtype=np.int8)
    splitter = GroupKFold(n_splits=5)
    for fold, (_, validation_index) in enumerate(
        splitter.split(train, groups=train[policy.group_column])
    ):
        fold_id[validation_index] = fold
    if (fold_id < 0).any():
        raise ValueError("One or more training rows did not receive a CV fold")

    manifest = train[
        ["source_dataset_id", "source_row_id", "cell_line", "target_locus_id", "design_id"]
    ].copy()
    manifest["group_cv_fold"] = fold_id
    if manifest.groupby("target_locus_id")["group_cv_fold"].nunique().max() != 1:
        raise ValueError("Target-locus leakage detected across internal CV folds")
    if set(manifest["cell_line"]) != set(policy.train_cells):
        raise ValueError("Held-out cell entered the internal CV manifest")

    output_csv = DATA_DIR / "train_group_cv_folds.csv"
    manifest.to_csv(output_csv, index=False)
    counts = (
        manifest.groupby(["group_cv_fold", "cell_line"], observed=True)
        .size()
        .unstack(fill_value=0)
        .astype(int)
    )
    report = {
        "status": "PASS",
        "method": "GroupKFold",
        "n_splits": 5,
        "group_column": policy.group_column,
        "allowed_cells": list(policy.train_cells),
        "held_out_cells_present": False,
        "rows": len(manifest),
        "unique_targets": int(manifest["target_locus_id"].nunique()),
        "fold_cell_counts": {
            str(fold): {cell: int(value) for cell, value in row.items()}
            for fold, row in counts.iterrows()
        },
    }
    (DATA_DIR / "train_group_cv_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
