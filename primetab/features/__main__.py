"""Command-line interface for the unified prime-editing data package."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from primetab.features.harmonization import (
    audit_ena_run_manifest,
    count_exact_read_pairs,
    harmonize_count_frame,
    iter_paired_fastq,
    load_deepprime_table_s1_designs,
    load_harmonization_contract,
    load_pridict2_sheet2_designs,
)
from primetab.features.pipeline import audit_sources, build_package, load_contract, verify_package
from primetab.features.raw_registry import build_raw_registry_package
from primetab.features.raw_runner import count_registered_head, run_registered_batch
from primetab.paths import REPOSITORY_ROOT


def _default_input(source: str) -> Path:
    relative = load_contract()["inputs"][source]["default_path"]
    return REPOSITORY_ROOT / relative


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m primetab.features",
        description="Audit and build the versioned DeepPrime + PRIDICT2 unified package.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Validate immutable source workbooks.")
    audit.add_argument("--deepprime", type=Path, default=_default_input("deepprime"))
    audit.add_argument("--pridict2", type=Path, default=_default_input("pridict2"))

    build = subparsers.add_parser("build", help="Build a new versioned package.")
    build.add_argument("--deepprime", type=Path, default=_default_input("deepprime"))
    build.add_argument("--pridict2", type=Path, default=_default_input("pridict2"))
    build.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "prime_editing_unified",
    )
    build.add_argument("--version", default=load_contract()["package_version"])

    verify = subparsers.add_parser("verify", help="Verify a completed package.")
    verify.add_argument("--package", type=Path, required=True)

    raw_audit = subparsers.add_parser(
        "raw-audit",
        help="Validate ENA run manifests needed for raw-read label harmonization.",
    )
    raw_audit.add_argument("--deepprime-runs", type=Path, required=True)
    raw_audit.add_argument("--pridict2-runs", type=Path, required=True)

    raw_registry = subparsers.add_parser(
        "raw-registry",
        help="Build the ENA run registry and an exact DeepPrime pilot design registry.",
    )
    raw_registry.add_argument("--deepprime-runs", type=Path, required=True)
    raw_registry.add_argument("--pridict2-runs", type=Path, required=True)
    raw_registry.add_argument("--deepprime-table-s1", type=Path, required=True)
    raw_registry.add_argument("--pridict2-workbook", type=Path, default=_default_input("pridict2"))
    raw_registry.add_argument("--unified-package", type=Path, required=True)
    raw_registry.add_argument("--output", type=Path, required=True)
    raw_registry.add_argument("--pilot-source-sheet", default="3")
    raw_registry.add_argument("--pilot-design-sheet", default="Lib-Small")

    raw_batch = subparsers.add_parser(
        "raw-batch",
        help="Download, verify, count, and checkpoint runs with complete design registries.",
    )
    raw_batch.add_argument("--registry-package", type=Path, required=True)
    raw_batch.add_argument("--run-accession", action="append", required=True)
    raw_batch.add_argument("--fastq-directory", type=Path, required=True)
    raw_batch.add_argument("--checkpoint-directory", type=Path, required=True)
    raw_batch.add_argument("--maximum-pairs", type=int)
    raw_batch.add_argument("--no-download", action="store_true")
    raw_batch.add_argument("--discard-raw", action="store_true")

    raw_head_count = subparsers.add_parser(
        "raw-head-count",
        help="Count a bounded, possibly truncated FASTQ head for layout validation only.",
    )
    raw_head_count.add_argument("--registry-package", type=Path, required=True)
    raw_head_count.add_argument("--run-accession", required=True)
    raw_head_count.add_argument("--read1", type=Path, required=True)
    raw_head_count.add_argument("--read2", type=Path, required=True)
    raw_head_count.add_argument("--maximum-pairs", type=int, required=True)
    raw_head_count.add_argument("--output", type=Path, required=True)

    raw_count = subparsers.add_parser(
        "raw-count",
        help="Count one local paired FASTQ run with the exact reference matcher.",
    )
    raw_count.add_argument("--source", choices=("deepprime", "pridict2"), required=True)
    raw_count.add_argument("--read1", type=Path, required=True)
    raw_count.add_argument("--read2", type=Path, required=True)
    raw_count.add_argument("--design-workbook", type=Path, required=True)
    raw_count.add_argument(
        "--design-sheet",
        help="DeepPrime Table S1 sheet, for example Lib-Small.",
    )
    raw_count.add_argument("--run-accession")
    raw_count.add_argument("--maximum-pairs", type=int)
    raw_count.add_argument("--output", type=Path, required=True)

    harmonize_counts = subparsers.add_parser(
        "harmonize-counts",
        help="Create the unified raw-read label from replicate-level read counts.",
    )
    harmonize_counts.add_argument("--counts", type=Path, required=True)
    harmonize_counts.add_argument("--output", type=Path, required=True)
    return parser


def _write_parquet_atomic(frame: pd.DataFrame, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd", engine="pyarrow")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _read_count_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported count-table suffix: {path.suffix}")


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "audit":
        result = audit_sources(args.deepprime.resolve(), args.pridict2.resolve())
    elif args.command == "build":
        package = build_package(
            args.deepprime.resolve(),
            args.pridict2.resolve(),
            args.output.resolve(),
            args.version,
        )
        result = verify_package(package)
    elif args.command == "verify":
        result = verify_package(args.package.resolve())
    elif args.command == "raw-audit":
        definition = load_harmonization_contract()["raw_sources"]
        deepprime = audit_ena_run_manifest(
            args.deepprime_runs.resolve(),
            expected_bioproject=definition["deepprime"]["bioproject"],
            expected_run_count=int(definition["deepprime"]["expected_run_count"]),
        )
        pridict2 = audit_ena_run_manifest(
            args.pridict2_runs.resolve(),
            expected_bioproject=definition["pridict2"]["bioproject"],
            expected_run_count=int(definition["pridict2"]["expected_run_count"]),
        )
        result = {
            "status": "pass" if deepprime["status"] == pridict2["status"] == "pass" else "fail",
            "deepprime": deepprime,
            "pridict2": pridict2,
            "combined_fastq_bytes": (
                deepprime["total_fastq_bytes"] + pridict2["total_fastq_bytes"]
            ),
        }
    elif args.command == "raw-registry":
        result = build_raw_registry_package(
            deepprime_manifest=args.deepprime_runs.resolve(),
            pridict2_manifest=args.pridict2_runs.resolve(),
            deepprime_table_s1=args.deepprime_table_s1.resolve(),
            pridict2_workbook=args.pridict2_workbook.resolve(),
            unified_package=args.unified_package.resolve(),
            output=args.output.resolve(),
            pilot_source_sheet=str(args.pilot_source_sheet),
            pilot_design_sheet=args.pilot_design_sheet,
        )
    elif args.command == "raw-batch":
        result = run_registered_batch(
            registry_package=args.registry_package.resolve(),
            run_accessions=args.run_accession,
            fastq_directory=args.fastq_directory.resolve(),
            checkpoint_directory=args.checkpoint_directory.resolve(),
            download=not args.no_download,
            maximum_pairs=args.maximum_pairs,
            keep_raw=not args.discard_raw,
        )
    elif args.command == "raw-head-count":
        counts, result = count_registered_head(
            registry_package=args.registry_package.resolve(),
            run_accession=args.run_accession,
            read1=args.read1.resolve(),
            read2=args.read2.resolve(),
            maximum_pairs=args.maximum_pairs,
        )
        _write_parquet_atomic(counts, args.output.resolve())
        result["output"] = str(args.output.resolve())
    elif args.command == "raw-count":
        if args.source == "deepprime":
            if not args.design_sheet:
                raise ValueError("--design-sheet is required for DeepPrime")
            designs = load_deepprime_table_s1_designs(
                args.design_workbook.resolve(),
                sheet_name=args.design_sheet,
            )
        else:
            if args.design_sheet:
                raise ValueError("--design-sheet is not used for PRIDICT2")
            designs = load_pridict2_sheet2_designs(args.design_workbook.resolve())
        counts, audit = count_exact_read_pairs(
            iter_paired_fastq(args.read1.resolve(), args.read2.resolve()),
            designs,
            maximum_pairs=args.maximum_pairs,
        )
        counts.insert(0, "source_study", "DeepPrime" if args.source == "deepprime" else "PRIDICT2")
        counts.insert(1, "run_accession", args.run_accession)
        _write_parquet_atomic(counts, args.output.resolve())
        result = {
            "status": "pass",
            "output": str(args.output.resolve()),
            "designs": len(designs),
            "nonzero_designs": int(counts["n_total"].gt(0).sum()),
            "audit": audit,
        }
    else:
        labels = harmonize_count_frame(_read_count_table(args.counts.resolve()))
        _write_parquet_atomic(labels, args.output.resolve())
        result = {
            "status": "pass",
            "output": str(args.output.resolve()),
            "observations": len(labels),
            "training_eligible": int(labels["training_eligible"].sum()),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
