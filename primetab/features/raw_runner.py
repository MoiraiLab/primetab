"""Resumable FASTQ staging and per-run exact-count checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from primetab.features.features import stable_hash
from primetab.features.harmonization import (
    ExactReadDesign,
    count_exact_read_pairs,
    iter_paired_fastq,
    load_harmonization_contract,
)


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional(value: Any) -> Any:
    return None if pd.isna(value) else value


def load_exact_design_registry(path: Path) -> list[ExactReadDesign]:
    """Load the deterministic Parquet representation of exact read designs."""

    frame = pd.read_parquet(path)
    required = {
        "design_key",
        "primary_identity",
        "identity_tokens",
        "wt_window",
        "intended_window",
        "primary_read",
        "allele_read",
        "identity_allow_reverse_complement",
        "allele_allow_reverse_complement",
        "preferred_identity_starts",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"raw_design_registry_columns_missing:{missing}")
    designs: list[ExactReadDesign] = []
    for record in frame.to_dict(orient="records"):
        designs.append(
            ExactReadDesign(
                design_key=str(record["design_key"]),
                primary_identity=str(record["primary_identity"]),
                identity_tokens=tuple(json.loads(record["identity_tokens"])),
                wt_window=str(record["wt_window"]),
                intended_window=str(record["intended_window"]),
                primary_read=int(record["primary_read"]),
                allele_read=int(record["allele_read"]),
                identity_allow_reverse_complement=bool(
                    record["identity_allow_reverse_complement"]
                ),
                allele_allow_reverse_complement=bool(
                    record["allele_allow_reverse_complement"]
                ),
                design_id=_optional(record.get("design_id")),
                target_locus_id=_optional(record.get("target_locus_id")),
                observation_id=_optional(record.get("observation_id")),
                control_match_id=_optional(record.get("control_match_id")),
                source_study=_optional(record.get("source_study")),
                source_sheet=_optional(record.get("source_sheet")),
                source_excel_row=(
                    int(record["source_excel_row"])
                    if not pd.isna(record.get("source_excel_row"))
                    else None
                ),
                source_design_key=_optional(record.get("source_design_key")),
                source_wide_target=_optional(record.get("source_wide_target")),
                preferred_identity_starts=tuple(
                    int(value) for value in json.loads(record["preferred_identity_starts"])
                ),
            )
        )
    if len({design.design_key for design in designs}) != len(designs):
        raise ValueError("raw_design_registry_duplicate_key")
    return designs


def download_resumable(
    url: str,
    destination: Path,
    *,
    expected_bytes: int,
    expected_md5: str,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Resume one HTTP download and require exact ENA size and MD5."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == expected_bytes:
            observed_md5 = file_digest(destination, "md5")
            if observed_md5 == expected_md5:
                return {
                    "status": "already_verified",
                    "path": str(destination),
                    "bytes": expected_bytes,
                    "md5": observed_md5,
                }
        raise FileExistsError(f"Unverified final FASTQ already exists: {destination}")

    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    if existing > expected_bytes:
        raise ValueError(f"partial_fastq_larger_than_manifest:{partial}:{existing}")
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=timeout_seconds) as response:
        if existing and response.status_code != 206:
            raise ValueError(f"server_did_not_honor_range:{url}:{response.status_code}")
        response.raise_for_status()
        mode = "ab" if existing else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())

    observed_bytes = partial.stat().st_size
    if observed_bytes != expected_bytes:
        raise ValueError(
            f"fastq_byte_count_mismatch:{partial}:{observed_bytes}:{expected_bytes}"
        )
    observed_md5 = file_digest(partial, "md5")
    if observed_md5 != expected_md5:
        raise ValueError(f"fastq_md5_mismatch:{partial}:{observed_md5}:{expected_md5}")
    os.replace(partial, destination)
    return {
        "status": "downloaded_and_verified",
        "path": str(destination),
        "bytes": observed_bytes,
        "md5": observed_md5,
    }


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _registry_run(registry_package: Path, run_accession: str) -> pd.Series:
    runs = pd.read_parquet(registry_package / "run_registry.parquet")
    selected = runs.loc[runs["run_accession"].eq(run_accession)]
    if len(selected) != 1:
        raise ValueError(f"run_registry_selection:{run_accession}:{len(selected)}")
    row = selected.iloc[0]
    if row["sample_role"] not in {"treated", "control"}:
        raise ValueError(f"run_not_countable:{run_accession}:{row['sample_role']}")
    return row


def _registry_assets(
    registry_package: Path,
    row: pd.Series,
) -> tuple[Path, Path, str, str | None]:
    source_study = str(row["source_study"])
    source_sheets = str(row["source_sheet"]).split(";")
    if source_study == "DeepPrime" and "3" in source_sheets:
        return (
            registry_package / "deepprime_sheet3_raw_designs.parquet",
            registry_package / "deepprime_sheet3_observation_links.parquet",
            "3",
            None,
        )
    if source_study == "PRIDICT2" and source_sheets == ["2"]:
        return (
            registry_package / "pridict2_sheet2_raw_designs.parquet",
            registry_package / "pridict2_sheet2_observation_links.parquet",
            "2",
            str(row["outcome_source_column"]),
        )
    raise ValueError(
        f"run_outside_available_design_scope:{row['run_accession']}:"
        f"{source_study}:{row['source_sheet']}"
    )


def _link_observation_counts(
    raw_counts: pd.DataFrame,
    *,
    links_path: Path,
    source_sheet: str,
    outcome_source_column: str | None,
) -> pd.DataFrame:
    links = pd.read_parquet(links_path)
    selected = links.loc[links["source_sheet"].astype(str).eq(source_sheet)].copy()
    if outcome_source_column is not None:
        selected = selected.loc[
            selected["outcome_source_column"].eq(outcome_source_column)
        ]
    if selected.empty or selected["read_design_id"].duplicated().any():
        raise ValueError(
            f"invalid_observation_link_scope:{source_sheet}:{outcome_source_column}:"
            f"{len(selected)}"
        )
    selected = selected.rename(columns={"read_design_id": "design_key"})[
        [
            "design_key",
            "observation_id",
            "design_id",
            "target_locus_id",
            "control_match_id",
        ]
    ]
    linked = raw_counts.merge(
        selected,
        on="design_key",
        how="inner",
        validate="one_to_one",
        suffixes=("_raw", ""),
    )
    for column in ("observation_id", "design_id", "target_locus_id", "control_match_id"):
        raw_column = f"{column}_raw"
        if raw_column not in linked:
            continue
        comparable = linked[raw_column].notna()
        if not linked.loc[comparable, raw_column].eq(linked.loc[comparable, column]).all():
            raise ValueError(f"raw_observation_link_mismatch:{column}")
        linked = linked.drop(columns=raw_column)
    return linked


def _add_run_metadata(counts: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    result = counts.copy()
    result.insert(0, "registry_version", str(row["registry_version"]))
    result.insert(
        1,
        "harmonization_version",
        load_harmonization_contract()["harmonization_version"],
    )
    result.insert(2, "run_accession", str(row["run_accession"]))
    result["experiment_accession"] = row["experiment_accession"]
    result["sample_accession"] = row["sample_accession"]
    result["sample_role"] = row["sample_role"]
    result["biological_replicate"] = row["biological_replicate"]
    result["technical_replicate"] = row["technical_replicate"]
    result["control_group_id"] = row["control_group_id"]
    result["outcome_source_column"] = row["outcome_source_column"]
    result["run_mapping_status"] = row["mapping_status"]
    return result


def process_registered_run(
    *,
    registry_package: Path,
    run_accession: str,
    fastq_directory: Path,
    checkpoint_directory: Path,
    download: bool = True,
    maximum_pairs: int | None = None,
    keep_raw: bool = True,
) -> dict[str, Any]:
    """Stage, verify, count, and atomically checkpoint one registered run."""

    row = _registry_run(registry_package, run_accession)
    design_path, links_path, source_sheet, outcome_column = _registry_assets(
        registry_package, row
    )
    raw_counts_path = checkpoint_directory / f"{run_accession}.raw_counts.parquet"
    counts_path = checkpoint_directory / f"{run_accession}.observation_counts.parquet"
    checkpoint_path = checkpoint_directory / f"{run_accession}.checkpoint.json"
    checkpoint_files = (raw_counts_path, counts_path, checkpoint_path)
    if any(path.exists() for path in checkpoint_files):
        if not all(path.exists() for path in checkpoint_files):
            raise ValueError(f"incomplete_run_checkpoint:{run_accession}")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("counts_sha256") != file_digest(counts_path):
            raise ValueError(f"run_checkpoint_hash_mismatch:{run_accession}")
        if checkpoint.get("raw_counts_sha256") != file_digest(raw_counts_path):
            raise ValueError(f"raw_run_checkpoint_hash_mismatch:{run_accession}")
        return {"status": "already_checkpointed", **checkpoint}

    fastq_directory.mkdir(parents=True, exist_ok=True)
    read_paths = (
        fastq_directory / f"{run_accession}_1.fastq.gz",
        fastq_directory / f"{run_accession}_2.fastq.gz",
    )
    downloads = []
    for read_number, path in enumerate(read_paths, start=1):
        if download:
            downloads.append(
                download_resumable(
                    str(row[f"fastq_{read_number}_url"]),
                    path,
                    expected_bytes=int(row[f"fastq_{read_number}_bytes"]),
                    expected_md5=str(row[f"fastq_{read_number}_md5"]),
                )
            )
        else:
            expected_bytes = int(row[f"fastq_{read_number}_bytes"])
            expected_md5 = str(row[f"fastq_{read_number}_md5"])
            if path.stat().st_size != expected_bytes or file_digest(path, "md5") != expected_md5:
                raise ValueError(f"local_fastq_verification_failed:{path}")
            downloads.append(
                {
                    "status": "local_verified",
                    "path": str(path),
                    "bytes": expected_bytes,
                    "md5": expected_md5,
                }
            )

    designs = load_exact_design_registry(design_path)
    raw_counts, audit = count_exact_read_pairs(
        iter_paired_fastq(*read_paths),
        designs,
        maximum_pairs=maximum_pairs,
    )
    if maximum_pairs is None and audit["read_pairs"] != int(row["read_pairs"]):
        raise ValueError(
            f"run_read_pair_count_mismatch:{run_accession}:"
            f"{audit['read_pairs']}:{int(row['read_pairs'])}"
        )
    if audit["assigned"] != int(raw_counts["n_total"].sum()):
        raise ValueError(f"run_assigned_count_conservation:{run_accession}")
    observation_counts = _link_observation_counts(
        raw_counts,
        links_path=links_path,
        source_sheet=source_sheet,
        outcome_source_column=outcome_column,
    )
    raw_counts = _add_run_metadata(raw_counts, row)
    observation_counts = _add_run_metadata(observation_counts, row)
    _atomic_parquet(raw_counts, raw_counts_path)
    _atomic_parquet(observation_counts, counts_path)

    checkpoint_key = stable_hash(
        "raw_count_checkpoint",
        {
            "harmonization_version": load_harmonization_contract()["harmonization_version"],
            "registry_version": str(row["registry_version"]),
            "run_accession": run_accession,
            "fastq_md5": [row["fastq_1_md5"], row["fastq_2_md5"]],
            "design_registry_sha256": file_digest(design_path),
            "observation_links_sha256": file_digest(links_path),
            "maximum_pairs": maximum_pairs,
        },
    )
    checkpoint = {
        "status": "pass",
        "checkpoint_key": checkpoint_key,
        "run_accession": run_accession,
        "maximum_pairs": maximum_pairs,
        "design_count": len(designs),
        "linked_read_design_count": int(observation_counts["design_key"].nunique()),
        "observation_count": len(observation_counts),
        "design_registry_sha256": file_digest(design_path),
        "observation_links_sha256": file_digest(links_path),
        "download_verification": downloads,
        "count_audit": audit,
        "raw_counts_path": str(raw_counts_path.resolve()),
        "raw_counts_sha256": file_digest(raw_counts_path),
        "counts_path": str(counts_path.resolve()),
        "counts_sha256": file_digest(counts_path),
    }
    _atomic_json(checkpoint, checkpoint_path)
    if not keep_raw:
        for path in read_paths:
            path.unlink()
        checkpoint["raw_fastq_deleted_after_checkpoint"] = True
        _atomic_json(checkpoint, checkpoint_path)
    return checkpoint


def count_registered_head(
    *,
    registry_package: Path,
    run_accession: str,
    read1: Path,
    read2: Path,
    maximum_pairs: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Count a bounded FASTQ head for layout validation only.

    The inputs may be deliberately truncated gzip streams.  The result is
    therefore marked as non-training evidence and can never satisfy a full-run
    checkpoint.
    """

    if maximum_pairs <= 0:
        raise ValueError("maximum_pairs_must_be_positive")
    row = _registry_run(registry_package, run_accession)
    design_path, links_path, source_sheet, outcome_column = _registry_assets(
        registry_package, row
    )
    designs = load_exact_design_registry(design_path)
    raw_counts, audit = count_exact_read_pairs(
        iter_paired_fastq(read1, read2),
        designs,
        maximum_pairs=maximum_pairs,
    )
    if audit["read_pairs"] != maximum_pairs:
        raise ValueError(
            f"partial_head_too_short:{run_accession}:{audit['read_pairs']}:{maximum_pairs}"
        )
    observation_counts = _link_observation_counts(
        raw_counts,
        links_path=links_path,
        source_sheet=source_sheet,
        outcome_source_column=outcome_column,
    )
    observation_counts = _add_run_metadata(observation_counts, row)
    observation_counts["input_verification_status"] = "partial_head_not_checksum_verified"
    observation_counts["training_authorized"] = False
    return observation_counts, {
        "status": "pass",
        "purpose": "read_layout_validation_only",
        "training_authorized": False,
        "run_accession": run_accession,
        "maximum_pairs": maximum_pairs,
        "design_count": len(designs),
        "linked_observation_count": len(observation_counts),
        "raw_assigned_read_pairs": int(raw_counts["n_total"].sum()),
        "linked_assigned_read_pairs": int(observation_counts["n_total"].sum()),
        "count_audit": audit,
    }


def run_registered_batch(
    *,
    registry_package: Path,
    run_accessions: list[str],
    fastq_directory: Path,
    checkpoint_directory: Path,
    download: bool = True,
    maximum_pairs: int | None = None,
    keep_raw: bool = True,
) -> dict[str, Any]:
    """Process a deterministic run list and resume from completed checkpoints."""

    if not run_accessions or len(set(run_accessions)) != len(run_accessions):
        raise ValueError("run_accessions_must_be_nonempty_and_unique")
    results = [
        process_registered_run(
            registry_package=registry_package,
            run_accession=run_accession,
            fastq_directory=fastq_directory,
            checkpoint_directory=checkpoint_directory,
            download=download,
            maximum_pairs=maximum_pairs,
            keep_raw=keep_raw,
        )
        for run_accession in run_accessions
    ]
    return {
        "status": "pass",
        "runs": len(results),
        "run_accessions": run_accessions,
        "results": results,
    }
