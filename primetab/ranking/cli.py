"""Command-line entry point for candidate generation and shared-context ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from primetab.features.features import F1_SEQUENCE_V1_COLUMNS
from primetab.generation import generate
from .pipeline import GeneratorRankerPipeline
from .rankers import SharedContextTFMRanker
from .task_csv import load_generator_tasks, task_parameter_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate PRIDICT2-compatible pegRNAs and rank them with a TFM context.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="CSV with one independently configured generator task per row",
    )
    parser.add_argument(
        "--context",
        type=Path,
        help="Preselected labelled context with the frozen 73 F1 columns; required for ranking",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate candidates without loading a ranking context or producing ranks",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--backend", default="tabpfn3")
    parser.add_argument("--label-column", default="editing_efficiency_fraction")
    parser.add_argument(
        "--preprocessing-reference",
        type=Path,
        help="Optional source-only table used to fit imputation and scaling",
    )
    parser.add_argument("--context-budget", type=int, default=8023)
    parser.add_argument(
        "--device",
        default=None,
        help="Backend device, for example cpu. Omit to leave the backend default.",
    )
    parser.add_argument("--model-path", type=Path, help="Optional backend weight path")
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.context_budget < 1:
        raise ValueError("context-budget must be positive")
    tasks = load_generator_tasks(args.input)
    if not args.generate_only and args.context is None:
        raise ValueError("--context is required unless --generate-only is set")
    _prepare_output_directory(args.output_dir)
    ranker = None if args.generate_only else _fit_ranker(args)
    table_parts: dict[str, list[pd.DataFrame]] = {
        "requests": [],
        "candidates": [],
        "features": [],
        "nicking_guides": [],
        "primers": [],
        "errors": [],
    }
    if ranker is not None:
        table_parts["ranked_candidates"] = []
    pipeline_metadata: dict[str, Any] | None = None
    for task in tasks:
        if ranker is None:
            generation = generate(task.request, config=task.config)
            ranked_candidates = None
        else:
            pipeline = GeneratorRankerPipeline(ranker, generation_config=task.config)
            result = pipeline.run([task.request])
            generation = result.generation
            ranked_candidates = result.ranked_candidates
            pipeline_metadata = pipeline.metadata()
        generation_tables = {
            "requests": generation.requests,
            "candidates": generation.candidates,
            "features": generation.features,
            "nicking_guides": generation.nicking_guides,
            "primers": generation.primers,
            "errors": generation.errors,
        }
        for name, frame in generation_tables.items():
            table_parts[name].append(_with_task_row(frame, task.task_row))
        if ranked_candidates is not None:
            table_parts["ranked_candidates"].append(
                _with_task_row(ranked_candidates, task.task_row)
            )

    tables = {name: pd.concat(parts, ignore_index=True) for name, parts in table_parts.items()}
    tables["task_parameters"] = task_parameter_frame(tasks)
    for name, frame in tables.items():
        _write_table(frame, args.output_dir / f"{name}.{args.format}", args.format)
    manifest = {
        "format": "primetab.generator_ranker.v2",
        "mode": "generate_only" if args.generate_only else "generate_and_rank",
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "task_count": len(tasks),
        "context": None if args.context is None else str(args.context),
        "context_sha256": None if args.context is None else _sha256(args.context),
        "preprocessing_reference": (
            None if args.preprocessing_reference is None else str(args.preprocessing_reference)
        ),
        "generation_parameters": "one configuration per input CSV row",
        "pipeline": pipeline_metadata,
        "rows": {name: len(frame) for name, frame in tables.items()},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return 0


def _fit_ranker(args: argparse.Namespace) -> SharedContextTFMRanker:
    context_columns = [*F1_SEQUENCE_V1_COLUMNS, args.label_column]
    context = _read_table(args.context, columns=context_columns)
    if context.empty:
        raise ValueError("ranking context is empty")
    if len(context) > args.context_budget:
        raise ValueError(
            f"ranking context has {len(context)} rows, above the {args.context_budget}-row "
            "budget; pass an already selected context"
        )
    reference = (
        None
        if args.preprocessing_reference is None
        else _read_table(
            args.preprocessing_reference,
            columns=list(F1_SEQUENCE_V1_COLUMNS),
        )
    )
    backend_kwargs: dict[str, Any] = {}
    if args.device:
        backend_kwargs["device"] = args.device
    if args.model_path:
        backend_kwargs["model_path"] = args.model_path
    if args.backend.strip().lower() in {"tabpfn3", "tabpfn-v3"}:
        backend_kwargs["n_estimators"] = args.n_estimators
    return SharedContextTFMRanker(args.backend, backend_kwargs=backend_kwargs).fit(
        context.loc[:, list(F1_SEQUENCE_V1_COLUMNS)],
        context[args.label_column],
        preprocessing_reference=reference,
    )


def _with_task_row(frame: pd.DataFrame, task_row: int) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "task_row", task_row)
    return result


def _read_table(path: Path, *, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=columns)
    raise ValueError(f"unsupported table format: {path.suffix}")


def _prepare_output_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _write_table(frame: pd.DataFrame, path: Path, output_format: str) -> None:
    if output_format == "parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
