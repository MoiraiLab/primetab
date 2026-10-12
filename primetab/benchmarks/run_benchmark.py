#!/usr/bin/env python3
"""Run synthetic distribution-shift benchmarks for the retained TabPFN path.

Usage:
    python -m primetab.benchmarks.run_benchmark --output results/benchmarks/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

# Disable telemetry to avoid PostHog network timeouts offline
os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"

import numpy as np
import pandas as pd

from primetab.benchmarks.shift_baselines import TabPFNWithMarkers

DEFAULT_METHODS = [
    "tabpfn_naive",
]


def generate_benchmark_data(
    n_source: int = 500,
    n_target_pool: int = 200,
    n_features: int = 10,
    shift_type: str = "mixed",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic multi-source distribution shift tabular data.

    Simulates:
    - Covariate shift: source has P(X) ~ N(0, 1), target has P(X) ~ N(1, 1.5).
    - Concept shift: target function includes non-linear interaction and sign inversion on key feature.
    """
    rng = np.random.default_rng(seed)

    # Source features & targets
    X_source = rng.normal(0.0, 1.0, size=(n_source, n_features)).astype(np.float32)
    # y = 3.0 * x0 - 2.0 * x1 + 1.5 * x2 + noise
    y_source = (3.0 * X_source[:, 0] - 2.0 * X_source[:, 1] + 1.5 * X_source[:, 2] + rng.normal(0, 0.2, n_source)).astype(np.float32)

    # Target features
    if shift_type in ("covariate_only", "mixed"):
        X_target = rng.normal(1.2, 1.3, size=(n_target_pool, n_features)).astype(np.float32)
    else:
        X_target = rng.normal(0.0, 1.0, size=(n_target_pool, n_features)).astype(np.float32)

    # Target labels
    if shift_type in ("concept_only", "mixed"):
        # Concept shift: coefficient of x0 inverts (Cheap Prior Trap analogy), plus scale shift
        y_target = (-2.5 * X_target[:, 0] - 2.0 * X_target[:, 1] + 1.5 * X_target[:, 2] + 5.0 + rng.normal(0, 0.2, n_target_pool)).astype(np.float32)
    else:
        y_target = (3.0 * X_target[:, 0] - 2.0 * X_target[:, 1] + 1.5 * X_target[:, 2] + rng.normal(0, 0.2, n_target_pool)).astype(np.float32)

    return X_source, y_source, X_target, y_target


def evaluate_model(
    model_name: str,
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_shot: np.ndarray,
    y_shot: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k_shot: int,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluates a single model under a specific shot budget k."""
    start_time = time.time()

    if model_name == "tabpfn_naive":
        # Native TabPFN without domain markers (blind concatenation)
        model = TabPFNWithMarkers(device=device)
        X_train = np.vstack([X_source[:100], X_shot]) if k_shot > 0 else X_source[:100]
        y_train = np.concatenate([y_source[:100], y_shot]) if k_shot > 0 else y_source[:100]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    elapsed = time.time() - start_time
    mae = float(np.mean(np.abs(y_test - preds)))
    rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))

    # Spearman correlation
    if np.std(preds) > 1e-6 and np.std(y_test) > 1e-6:
        corr = float(np.corrcoef(preds, y_test)[0, 1])
    else:
        corr = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "spearman_rho": corr,
        "runtime_sec": elapsed,
    }


def run_benchmark(
    output_dir: Path,
    shift_types: list[str] | None = None,
    shot_regimes: list[int] | None = None,
    models: list[str] | None = None,
    n_reps: int = 3,
    device: str = "cpu",
) -> pd.DataFrame:
    """Executes the synthetic benchmark matrix for retained models."""
    shot_regimes = shot_regimes or [1, 2, 4, 8, 16, 32]
    models = models or DEFAULT_METHODS
    shift_types = shift_types or ["covariate_only", "concept_only", "mixed"]

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    print(f"================================================================")
    print(f" Starting distribution-shift benchmark suite")
    print(f" Shifts: {shift_types}")
    print(f" Shot Regimes: {shot_regimes}")
    print(f" Models: {models}")
    print(f" Repetitions: {n_reps}")
    print(f" Device: {device}")
    print(f" Output Directory: {output_dir}")
    print(f"================================================================\n")

    for shift in shift_types:
        print(f"\n>>> Running Shift Condition: [{shift.upper()}]")
        for rep in range(n_reps):
            X_src, y_src, X_tgt, y_tgt = generate_benchmark_data(
                shift_type=shift,
                seed=42 + rep * 100,
            )

            # Hold out test set
            n_test = 80
            X_test, y_test = X_tgt[:n_test], y_tgt[:n_test]
            X_tgt_pool, y_tgt_pool = X_tgt[n_test:], y_tgt[n_test:]

            for k in shot_regimes:
                # Subsample k target shots
                X_shot = X_tgt_pool[:k]
                y_shot = y_tgt_pool[:k]

                for model_name in models:
                    metrics = evaluate_model(
                        model_name=model_name,
                        X_source=X_src,
                        y_source=y_src,
                        X_shot=X_shot,
                        y_shot=y_shot,
                        X_test=X_test,
                        y_test=y_test,
                        k_shot=k,
                        device=device,
                    )

                    record = {
                        "dataset": shift,
                        "shift_type": shift,
                        "rep": rep,
                        "seed": 42 + rep * 100,
                        "k_shot": k,
                        "model": model_name,
                        "method": model_name,
                        **metrics,
                    }
                    records.append(record)

                # Progress summary for this shot
                k_records = [r for r in records if r["shift_type"] == shift and r["k_shot"] == k and r["rep"] == rep]
                best_k = min(k_records, key=lambda x: x["mae"])
                print(f"  [Shift: {shift:14s} | Rep {rep+1}/{n_reps} | k={k:2d}] Best Model: {best_k['model']:14s} | MAE: {best_k['mae']:.4f}")

    df = pd.DataFrame(records)

    summary_path = output_dir / "benchmark_results.csv"
    df.to_csv(summary_path, index=False)
    print(f"\n[SUCCESS] Raw results saved to: {summary_path}")

    # Generate aggregate pivot table
    group_col = "dataset" if "dataset" in df.columns else "shift_type"
    pivot_mae = df.pivot_table(
        index=[group_col, "k_shot"],
        columns="model",
        values="mae",
        aggfunc=["mean", "std"],
    )
    pivot_path = output_dir / "mae_pivot.csv"
    pivot_mae.to_csv(pivot_path)
    print(f"[SUCCESS] MAE aggregation pivot saved to: {pivot_path}")

    # Save metadata
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "shift_types": shift_types,
        "datasets": None,
        "shot_regimes": shot_regimes,
        "models": models,
        "total_runs": len(df),
    }
    with open(output_dir / "benchmark_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return df


def main():
    parser = argparse.ArgumentParser(description="Synthetic TabPFN distribution-shift benchmark")
    parser.add_argument("--output", type=str, default="results/benchmarks", help="Output directory")
    parser.add_argument("--shifts", nargs="+", default=["covariate_only", "concept_only", "mixed"], help="Shift types to evaluate")
    parser.add_argument(
        "--shots", "--k-shot", "--k-shots",
        dest="shots",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16],
        help="Few-shot regimes (k)",
    )
    parser.add_argument(
        "--methods", "--models",
        dest="methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Models to evaluate. Retained default is tabpfn_naive.",
    )
    parser.add_argument("--reps", type=int, default=2, help="Number of repetitions per condition")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--quick", action="store_true", help="Run fast dry-run for smoke testing")
    args = parser.parse_args()

    out_dir = Path(args.output)

    if args.quick:
        # Fast smoke configuration: 1 shot, 1 rep, retained TabPFN path
        shifts = ["covariate_only"]
        shots = [args.shots[0]] if args.shots else [4]
        methods = [m for m in ["tabpfn_naive"] if m in args.methods] or ["tabpfn_naive"]
        reps = 1
    else:
        shifts = args.shifts
        shots = args.shots
        methods = args.methods
        reps = args.reps

    run_benchmark(
        output_dir=out_dir,
        shift_types=shifts,
        shot_regimes=shots,
        models=methods,
        n_reps=reps,
        device=args.device,
    )


if __name__ == "__main__":
    main()

