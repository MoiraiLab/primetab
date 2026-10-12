"""CLI for the model-free PRIDICT2-compatible generator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from primetab.generation.api import (
    EditRequest,
    GenerationConfig,
    GenerationResult,
    config_record,
    generate,
    generate_batch,
)
from primetab.generation.core import UPSTREAM_SOURCE_SHA256


def _range_values(value: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"Invalid descending range: {token}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    if not values:
        raise argparse.ArgumentTypeError("At least one length is required")
    return tuple(dict.fromkeys(values))


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    parser.add_argument("--pbs-lengths", type=_range_values, default=tuple(range(7, 16)))
    parser.add_argument(
        "--rt-overhang-lengths",
        type=_range_values,
        default=tuple(range(3, 20)),
    )
    parser.add_argument("--max-edit-distance", type=int, default=25)
    parser.add_argument("--max-edit-length", type=int, default=40)
    parser.add_argument("--nicking", action="store_true")
    parser.add_argument("--ngs-primers", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m primetab.generation",
        description="Generate PRIDICT2-compatible PE2/PE3 designs without model prediction.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("single")
    single.add_argument("--sequence-name", required=True)
    single.add_argument("--sequence", required=True)
    _add_common_arguments(single)

    batch = commands.add_parser("batch")
    batch.add_argument("--input", type=Path, required=True)
    _add_common_arguments(batch)
    return parser


def _config(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        pbs_lengths=args.pbs_lengths,
        rt_overhang_lengths=args.rt_overhang_lengths,
        max_edit_distance=args.max_edit_distance,
        max_edit_length=args.max_edit_length,
        include_nicking=args.nicking,
        include_ngs_primers=args.ngs_primers,
    )


def _input_digest(requests: Sequence[EditRequest]) -> str:
    payload = json.dumps(
        [[request.sequence_name, request.editseq] for request in requests],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _feature_type(column: str) -> pa.DataType:
    integer_prefixes = (
        "edit_position",
        "edit_type_",
        "gc_count_",
        "max_run_",
        "poly_t_max_run",
        "poly_t_risk_",
        "pam_length",
        "pam_disrupted",
    )
    string_columns = {
        "design_id",
        "feature_status",
        "feature_failure_codes",
        "pam_before",
        "pam_after",
        "base_before_edit",
        "base_after_edit",
        "base_before_nick",
        "base_after_nick",
    }
    if column in string_columns:
        return pa.string()
    if (
        column.endswith("_length")
        or column.startswith(integer_prefixes)
        or column == "right_homology_arm_length"
    ):
        return pa.int64()
    return pa.float64()


def _table_schema(name: str, columns: Sequence[str]) -> pa.Schema:
    integer_columns = {
        "edit_length",
        "edit_position",
        "left_flank_length",
        "right_flank_length",
        "pam_relative_to_edit",
        "nick_to_edit_distance",
        "right_homology_arm_length",
        "pbs_length",
        "rt_overhang_length",
        "rtt_length",
        "rt_overhang_matches",
        "product_size",
    }
    boolean_columns = {"outside_validated_range", "pam_disrupted"}
    float_columns = {"primer_pair_penalty"}
    fields: list[pa.Field] = []
    for column in columns:
        if name == "features":
            dtype = _feature_type(column)
        elif column in integer_columns:
            dtype = pa.int64()
        elif column in boolean_columns:
            dtype = pa.bool_()
        elif column in float_columns:
            dtype = pa.float64()
        else:
            dtype = pa.string()
        fields.append(pa.field(column, dtype))
    return pa.schema(fields)


def write_result(
    result: GenerationResult,
    output_dir: Path,
    output_format: str,
    requests: Sequence[EditRequest],
) -> dict:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Output path already exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty output directory: {output_dir}"
            )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    tables = {
        "requests": result.requests,
        "candidates": result.candidates,
        "features": result.features,
        "nicking_guides": result.nicking_guides,
        "primers": result.primers,
        "errors": result.errors,
    }
    try:
        for name, frame in tables.items():
            path = staging_dir / f"{name}.{output_format}"
            if output_format == "parquet":
                table = pa.Table.from_pandas(
                    frame,
                    schema=_table_schema(name, frame.columns.tolist()),
                    preserve_index=False,
                    safe=True,
                )
                pq.write_table(table, path, compression="zstd")
            else:
                frame.to_csv(path, index=False)
        manifest = {
            "schema_version": 1,
            "generator": "primetab.generation",
            "package_version": _package_version("tfm-prime-editing"),
            "upstream": "uzh-dqbm-cmi/PRIDICT2",
            "upstream_source_sha256": UPSTREAM_SOURCE_SHA256,
            "input_sha256": _input_digest(requests),
            "config": config_record(result.config),
            "dependencies": {
                "python_package_primer3_py": _package_version("primer3-py"),
                "biopython": _package_version("biopython"),
                "viennarna": _package_version("ViennaRNA"),
            },
            "output_format": output_format,
            "row_counts": {name: len(frame) for name, frame in tables.items()},
        }
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            output_dir.rmdir()
        staging_dir.replace(output_dir)
        return manifest
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config(args)
    if args.command == "single":
        requests = [EditRequest(args.sequence_name, args.sequence)]
        result = generate(requests[0], config)
    else:
        source = pd.read_csv(args.input)
        required = {"sequence_name", "editseq"}
        if not required <= set(source.columns):
            missing = sorted(required - set(source.columns))
            raise ValueError(f"Batch input is missing columns: {missing}")
        if source.empty:
            raise ValueError("Batch input must contain at least one request")
        null_columns = [column for column in sorted(required) if source[column].isna().any()]
        if null_columns:
            raise ValueError(f"Batch input contains null values in columns: {null_columns}")
        requests = [
            EditRequest(str(row.sequence_name), str(row.editseq))
            for row in source.loc[:, ["sequence_name", "editseq"]].itertuples(index=False)
        ]
        result = generate_batch(requests, config)
    manifest = write_result(result, args.output_dir, args.format, requests)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if len(result.errors) else 0
