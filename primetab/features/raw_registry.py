"""Run, raw-design, and observation registries for raw-read harmonization."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

from primetab.features.features import stable_hash
from primetab.features.harmonization import (
    ExactReadDesign,
    load_deepprime_table_s1_designs,
    load_pridict2_sheet2_designs,
)

RAW_REGISTRY_VERSION = "raw_registry_v1"


def _semicolon_values(value: Any) -> list[str]:
    return [part for part in str(value).split(";") if part and part.lower() != "nan"]


def _replicate_from_name(name: str) -> str | None:
    match = re.search(r"(?:rep|Rep)(\d+)$", name)
    return match.group(1) if match else None


def _control_group(source_study: str, scope: str) -> str:
    return stable_hash(
        "control_group",
        {"source_study": source_study, "scope": scope},
    )


def _manifest_fields(record: Any, source_study: str) -> dict[str, Any]:
    urls = _semicolon_values(record.fastq_ftp)
    md5_values = _semicolon_values(record.fastq_md5)
    byte_values = [int(value) for value in _semicolon_values(record.fastq_bytes)]
    if not (len(urls) == len(md5_values) == len(byte_values) == 2):
        raise ValueError(f"invalid_paired_fastq_manifest:{record.run_accession}")
    return {
        "registry_version": RAW_REGISTRY_VERSION,
        "source_study": source_study,
        "run_accession": record.run_accession,
        "experiment_accession": record.experiment_accession,
        "sample_accession": record.sample_accession,
        "library_name": record.library_name,
        "read_pairs": int(record.read_count),
        "fastq_1_bytes": byte_values[0],
        "fastq_2_bytes": byte_values[1],
        "fastq_total_bytes": sum(byte_values),
        "fastq_1_md5": md5_values[0],
        "fastq_2_md5": md5_values[1],
        "fastq_1_url": f"https://{urls[0]}",
        "fastq_2_url": f"https://{urls[1]}",
    }


DEEPPRIME_EXACT_TREATED_RULES: tuple[tuple[str, str, str, str, str], ...] = (
    (
        r"^Yu2022_Library-ClinVar\+Profiling_HEK293T_PE2_rep\d+$",
        "1",
        "pegRNA_on_target",
        "Lib-ClinVar",
        "clinvar_profiling_conventional",
    ),
    (
        r"^Yu2022_Library-Small_HEK293T_PE2max_rep\d+$",
        "3",
        "pegRNA_on_target",
        "Lib-Small",
        "small_optimized",
    ),
    (
        r"^Yu2022_Library-Small_HEK293T_PE4max-LM_rep\d+$",
        "4",
        "pegRNA_on_target",
        "Lib-Small",
        "small_optimized",
    ),
    (
        r"^Yu2022_Library-Small_HEK293T_NRCH-PE2_rep\d+$",
        "5",
        "pegRNA_on_target",
        "Lib-Small",
        "small_optimized",
    ),
    (
        r"^Yu2022_Library-Small_HEK293T_NRCH-PE2max_rep\d+$",
        "6",
        "pegRNA_on_target",
        "Lib-Small",
        "small_optimized",
    ),
    (
        r"^Yu2022_Library-epegRNA_HEK293T_PE2max_rep\d+$",
        "7",
        "pegRNA_on_target",
        "Lib-epegRNA",
        "small_optimized",
    ),
    (
        r"^Yu2022_Library-epegRNA_HEK293T_PE4max-LM_rep\d+$",
        "8",
        "pegRNA_on_target",
        "Lib-epegRNA",
        "small_optimized",
    ),
    (
        r"^Yu2022_Library-Small_DLD1_PE2max_rep\d+$",
        "10",
        "pegRNA_on_target",
        "Lib-Small",
        "small_optimized",
    ),
    (
        r"^Yu2022_Library-Small_DLD1_PE4max-LM_rep\d+$",
        "11",
        "pegRNA_on_target",
        "Lib-Small",
        "small_optimized",
    ),
    (
        r"^Yu2022_Library-Small_DLD1_NRCH-PE4max-LM_rep\d+$",
        "12",
        "pegRNA_on_target",
        "Lib-Small",
        "small_optimized",
    ),
    (
        r"^Yu2022_Library-Small_A549_PE2max_rep\d+$",
        "14",
        "pegRNA_on_target",
        "Lib-Small",
        "small_optimized",
    ),
    (
        r"^Yu2022_Library-Small_A549_PE4max-LM_rep\d+$",
        "15",
        "pegRNA_on_target",
        "Lib-Small",
        "small_optimized",
    ),
    (
        r"^Yu2022_Library-epegRNA_A549_PE2max_rep\d+$",
        "16",
        "pegRNA_on_target",
        "Lib-epegRNA",
        "small_optimized",
    ),
    (
        r"^Yu2022_Library-epegRNA_A549_PE4max-LM_rep\d+$",
        "17",
        "pegRNA_on_target",
        "Lib-epegRNA",
        "small_optimized",
    ),
    (
        r"^Yu2022_Library-Small_HeLa_PE2max_rep\d+$",
        "18",
        "pegRNA_on_target",
        "Lib-Small",
        "small_optimized",
    ),
    (
        r"^Yu2022_Library-Small_NIH3T3_NRCH-PE4max-LM_rep\d+$",
        "19",
        "pegRNA_on_target",
        "Lib-Small",
        "small_optimized",
    ),
    (
        r"^Yu2022_Library-Off-50K_HEK293T_PE2_rep\d+$",
        "20",
        "pegRNA_on_target",
        "Lib-Off",
        "off_library",
    ),
    (
        r"^Yu2022_Library-Off-7K_HEK293T_PE2max_rep\d+$",
        "21",
        "mismatch_off_target",
        "Lib-Off",
        "off_library",
    ),
    (
        r"^Yu2022_Library-Off-7K_HEK293T_PE4max-VM_rep\d+$",
        "22",
        "mismatch_off_target",
        "Lib-Off",
        "off_library",
    ),
    (
        r"^Yu2022_Library-Off-50K_HEK293T_PE4max-LM_rep\d+$",
        "23",
        "mismatch_off_target",
        "Lib-Off",
        "off_library",
    ),
    (
        r"^Yu2022_Library-Off-7K_A549_PE2max_rep\d+$",
        "24",
        "mismatch_off_target",
        "Lib-Off",
        "off_library",
    ),
    (
        r"^Yu2022_Library-Off-7K_A549_PE4max-VM_rep\d+$",
        "25",
        "mismatch_off_target",
        "Lib-Off",
        "off_library",
    ),
    (
        r"^Yu2022_Library-Off-7K_A549_PE4max-LM_rep\d+$",
        "26",
        "mismatch_off_target",
        "Lib-Off",
        "off_library",
    ),
    (
        r"^Yu2022_Library-Off-7K_DLD1_PE2max_rep\d+$",
        "27",
        "mismatch_off_target",
        "Lib-Off",
        "off_library",
    ),
    (
        r"^Yu2022_Library-Off-7K_DLD1_PE4max-VM_rep\d+$",
        "28",
        "mismatch_off_target",
        "Lib-Off",
        "off_library",
    ),
    (
        r"^Yu2022_Library-Off-7K_DLD1_PE4max-LM_rep\d+$",
        "29",
        "mismatch_off_target",
        "Lib-Off",
        "off_library",
    ),
    (
        r"^Yu2022_Library-Off-7K_NIH3T3_PE4max-VM_rep\d+$",
        "30",
        "mismatch_off_target",
        "Lib-Off",
        "off_library",
    ),
)


DEEPPRIME_AMBIGUOUS_TREATED_RULES: tuple[tuple[str, str, str], ...] = (
    (r"^Yu2022_small_HEK293T_PE2_ConventionalScaff_rep\d+$", "2", "small_conventional"),
    (r"^Yu2022_Library-Small_HEK293T_PE2-ConvScaffold_rep\d+$", "2", "small_conventional"),
    (r"^Yu2022_small_HCT116_rep\d+$", "9", "small_optimized"),
    (r"^Yu2022_Library-Small_HCT116_PE2_rep\d+$", "9", "small_optimized"),
    (r"^Yu2022_small_MDA-MB-231_PE2_rep\d+$", "13", "small_optimized"),
    (r"^Yu2022_Library-Small_MDA-MB-231_PE2_rep\d+$", "13", "small_optimized"),
)


def _classify_deepprime_run(record: Any) -> dict[str, Any]:
    name = str(record.library_name)
    base = _manifest_fields(record, "DeepPrime")
    biological_replicate = _replicate_from_name(name)

    controls = (
        (
            r"^Yu2022_Library-ClinVar\+Profiling_HEK293T_BG_rep\d+$",
            "1",
            "Lib-ClinVar",
            "clinvar_profiling_conventional",
        ),
        (
            r"^Yu2022_Library-Small_BG-ConvScaffold_rep\d+$",
            "2",
            "Lib-Small",
            "small_conventional",
        ),
        (
            r"^Yu2022_Library-Small_BG-OptiScaffold_rep\d+$",
            ";".join(str(value) for value in range(3, 20)),
            "Lib-Small;Lib-epegRNA",
            "small_optimized",
        ),
        (
            r"^Yu2022_Library-Off-50K_HEK293T_BG_rep\d+$",
            ";".join(str(value) for value in range(20, 31)),
            "Lib-Off",
            "off_library",
        ),
    )
    for pattern, sheets, design_scope, control_scope in controls:
        if re.fullmatch(pattern, name):
            return {
                **base,
                "task_scope": "shared_background_control",
                "source_sheet": sheets,
                "outcome_source_column": None,
                "sample_role": "control",
                "biological_replicate": biological_replicate,
                "technical_replicate": "1",
                "control_group_id": _control_group("DeepPrime", control_scope),
                "design_registry_scope": design_scope,
                "source_design_key": None,
                "mapping_status": "exact_shared_control_scope",
                "failure_code": None,
            }

    for pattern, sheet, task_scope, design_scope, control_scope in DEEPPRIME_EXACT_TREATED_RULES:
        if re.fullmatch(pattern, name):
            return {
                **base,
                "task_scope": task_scope,
                "source_sheet": sheet,
                "outcome_source_column": (
                    "Measured_PE_efficiency" if task_scope == "pegRNA_on_target" else None
                ),
                "sample_role": "treated",
                "biological_replicate": biological_replicate,
                "technical_replicate": "1",
                "control_group_id": _control_group("DeepPrime", control_scope),
                "design_registry_scope": design_scope,
                "source_design_key": None,
                "mapping_status": "exact_run_scope",
                "failure_code": None,
            }

    for pattern, sheet, control_scope in DEEPPRIME_AMBIGUOUS_TREATED_RULES:
        if re.fullmatch(pattern, name):
            return {
                **base,
                "task_scope": "pegRNA_on_target",
                "source_sheet": sheet,
                "outcome_source_column": "Measured_PE_efficiency",
                "sample_role": "treated",
                "biological_replicate": biological_replicate,
                "technical_replicate": "1",
                "control_group_id": _control_group("DeepPrime", control_scope),
                "design_registry_scope": "Lib-Small",
                "source_design_key": None,
                "mapping_status": "ambiguous_source_run_set",
                "failure_code": "ambiguous_source_run_set",
            }

    return {
        **base,
        "task_scope": "supporting_raw_unlinked",
        "source_sheet": None,
        "outcome_source_column": None,
        "sample_role": "excluded",
        "biological_replicate": biological_replicate,
        "technical_replicate": "1",
        "control_group_id": None,
        "design_registry_scope": None,
        "source_design_key": None,
        "mapping_status": "no_registered_label_scope",
        "failure_code": "missing_raw_sample_mapping",
    }


def _pridict2_library_diverse(record: Any) -> dict[str, Any] | None:
    name = str(record.library_name)
    definitions = (
        (
            r"^lib_diverse_\d+-451_HEK_(PE2|Ctr)_(\d+)$",
            "HEK293T",
            "HEKaverageedited",
            "library_diverse_hek",
        ),
        (
            r"^lib_diverse_\d+-451_K562_(PE2|Ctr)_(\d+)$",
            "K562",
            "K562averageedited",
            "library_diverse_k562",
        ),
        (
            r"^lib_diverse_\d+-451_K562_MLH_(Rep|Ctr)(\d+)$",
            "K562-MLH1dn",
            "K562MLH1dnaverageedited",
            "library_diverse_k562_mlh1dn",
        ),
        (
            r"^Mouse_Hepatocytes_lib_diverse_\d+-451NM_invivo_(AdV|Ctr)_(\d+)$",
            "mouse_hepatocytes",
            "AdVaverageedited",
            "library_diverse_mouse_adv",
        ),
    )
    for pattern, cell_line, outcome_column, control_scope in definitions:
        match = re.fullmatch(pattern, name)
        if not match:
            continue
        treatment_token, replicate = match.groups()
        sample_role = "control" if treatment_token == "Ctr" else "treated"
        return {
            "task_scope": "pegRNA_on_target",
            "source_sheet": "2",
            "outcome_source_column": outcome_column,
            "sample_role": sample_role,
            "cell_line": cell_line,
            "editor_system": None if sample_role == "control" else treatment_token,
            "biological_replicate": replicate,
            "technical_replicate": "1",
            "control_group_id": _control_group("PRIDICT2", control_scope),
            "design_registry_scope": "PRIDICT2:2",
            "source_design_key": None,
            "mapping_status": "exact_run_scope",
            "failure_code": None,
        }
    return None


def _arrayed_shortname(name: str, assay: str) -> str | None:
    prefix = f"arrayed_{assay}_"
    value = name.removeprefix(prefix)
    for delimiter in ("-End-", "-Rep-", "-PE2-", "-PE-"):
        if delimiter in value:
            candidate = value.split(delimiter, 1)[0]
            break
    else:
        candidate = value.split("-R", 1)[0] if "-R" in value else ""
    candidate = candidate.removeprefix("hepg2_")
    match = re.fullmatch(r"NM(\d+)", candidate, flags=re.IGNORECASE)
    if match:
        candidate = f"{match.group(1)}NM"
    return candidate or None


def _classify_pridict2_run(record: Any) -> dict[str, Any]:
    name = str(record.library_name)
    base = _manifest_fields(record, "PRIDICT2")
    library_diverse = _pridict2_library_diverse(record)
    if library_diverse is not None:
        return {**base, **library_diverse}

    if name.startswith("trip_mapping_"):
        return {
            **base,
            "task_scope": "supporting_metadata",
            "source_sheet": None,
            "outcome_source_column": None,
            "sample_role": "mapping",
            "cell_line": "K562",
            "editor_system": None,
            "biological_replicate": None,
            "technical_replicate": None,
            "control_group_id": None,
            "design_registry_scope": "PRIDICT2:12:TRIP",
            "source_design_key": None,
            "mapping_status": "trip_mapping_library",
            "failure_code": None,
        }

    trip = re.fullmatch(r"trip_editing_\d+-NM_TRIP_(Ctrl|PE|Cas9|ABE8e|BE4max)_([ABC])", name)
    if trip:
        treatment, replicate_letter = trip.groups()
        included = treatment in {"Ctrl", "PE"}
        return {
            **base,
            "task_scope": "chromatin_context" if included else "excluded_non_prime_editor",
            "source_sheet": "12" if included else None,
            "outcome_source_column": "PE_editing_efficiency" if included else None,
            "sample_role": (
                "control" if treatment == "Ctrl" else "treated" if treatment == "PE" else "excluded"
            ),
            "cell_line": "K562",
            "editor_system": None if treatment == "Ctrl" else treatment,
            "biological_replicate": str("ABC".index(replicate_letter) + 1),
            "technical_replicate": "1",
            "control_group_id": (
                _control_group("PRIDICT2", "trip_prime_editing") if included else None
            ),
            "design_registry_scope": "PRIDICT2:12:TRIP",
            "source_design_key": None,
            "mapping_status": (
                "design_registry_pending" if included else "excluded_non_prime_editor"
            ),
            "failure_code": "missing_raw_sample_mapping" if included else None,
        }

    if name.startswith(("arrayed_endogenous_", "arrayed_reporter_")):
        assay = "endogenous" if name.startswith("arrayed_endogenous_") else "reporter"
        lower = name.lower()
        non_prime = next(
            (editor for editor in ("ABE8e", "BE4max", "Cas9") if editor.lower() in lower),
            None,
        )
        if non_prime:
            return {
                **base,
                "task_scope": "excluded_non_prime_editor",
                "source_sheet": None,
                "outcome_source_column": None,
                "sample_role": "excluded",
                "cell_line": None,
                "editor_system": non_prime,
                "biological_replicate": None,
                "technical_replicate": None,
                "control_group_id": None,
                "design_registry_scope": None,
                "source_design_key": None,
                "mapping_status": "excluded_non_prime_editor",
                "failure_code": None,
            }

        is_control = any(token in lower for token in ("ctrl", "control", "ctr-", "gblock"))
        cell_line = "HepG2" if "hepg2" in lower else "HEK293T" if "hek" in lower else "K562"
        outcome_column = f"{cell_line}_edited_percentage_{assay}"
        suffix = re.search(r"-(\d+)-(\d+)$", name)
        r_suffix = re.search(r"-R(\d+)$", name, flags=re.IGNORECASE)
        biological_replicate = suffix.group(1) if suffix else r_suffix.group(1) if r_suffix else None
        technical_replicate = suffix.group(2) if suffix else "1" if r_suffix else None
        shortname = None if is_control else _arrayed_shortname(name, assay)
        return {
            **base,
            "task_scope": "pegRNA_on_target",
            "source_sheet": "7",
            "outcome_source_column": outcome_column,
            "sample_role": "control" if is_control else "treated",
            "cell_line": cell_line,
            "editor_system": None if is_control else "PE2",
            "biological_replicate": biological_replicate,
            "technical_replicate": technical_replicate,
            "control_group_id": _control_group(
                "PRIDICT2", f"arrayed_{assay}_{cell_line}"
            ),
            "design_registry_scope": "PRIDICT2:7",
            "source_design_key": shortname,
            "mapping_status": (
                "target_scoped_control_pending"
                if is_control
                else "exact_shortname_pending_allele_adapter"
                if shortname
                else "design_registry_pending"
            ),
            "failure_code": "missing_raw_sample_mapping",
        }

    return {
        **base,
        "task_scope": "unclassified",
        "source_sheet": None,
        "outcome_source_column": None,
        "sample_role": "excluded",
        "cell_line": None,
        "editor_system": None,
        "biological_replicate": None,
        "technical_replicate": None,
        "control_group_id": None,
        "design_registry_scope": None,
        "source_design_key": None,
        "mapping_status": "unclassified_run_name",
        "failure_code": "missing_raw_sample_mapping",
    }


def build_run_registry(deepprime_manifest: Path, pridict2_manifest: Path) -> pd.DataFrame:
    """Classify every ENA run while retaining unresolved mappings explicitly."""

    deepprime = pd.read_csv(deepprime_manifest, sep="\t", dtype=str)
    pridict2 = pd.read_csv(pridict2_manifest, sep="\t", dtype=str)
    rows = [
        *(_classify_deepprime_run(record) for record in deepprime.itertuples(index=False)),
        *(_classify_pridict2_run(record) for record in pridict2.itertuples(index=False)),
    ]
    result = pd.DataFrame(rows).sort_values(["source_study", "run_accession"]).reset_index(drop=True)
    if result["run_accession"].duplicated().any():
        raise ValueError("duplicate_run_accession_in_registry")
    if len(result) != len(deepprime) + len(pridict2):
        raise ValueError("run_registry_row_conservation_failure")
    return result


def _design_record(design: ExactReadDesign) -> dict[str, Any]:
    record = asdict(design)
    for column in ("identity_tokens", "preferred_identity_starts"):
        record[column] = json.dumps(record[column], separators=(",", ":"))
    record["observation_mapping_status"] = (
        "ordered_exact_subsequence" if design.observation_id else "not_in_source_observation_scope"
    )
    return record


def link_deepprime_observations(
    designs: Iterable[ExactReadDesign],
    *,
    unified_package: Path,
    source_sheet: str,
) -> tuple[list[ExactReadDesign], pd.DataFrame, dict[str, Any]]:
    """Link Table S1 entries to Table S2 observations without using labels.

    DeepPrime Table S2 rows are an order-preserving exact sequence subset of
    Table S1.  The link requires equal canonical ``design_id`` and exact
    containment of the Table S1 target evidence in the canonical 74-nt target.
    """

    task = pd.read_parquet(unified_package / "tasks" / "pegRNA_on_target.parquet")
    task = task.loc[
        task["source_study"].eq("DeepPrime")
        & task["source_sheet"].astype(str).eq(str(source_sheet))
    ].sort_values("source_excel_row")
    targets = pd.read_parquet(unified_package / "tables" / "target_loci.parquet")
    target_sequences = targets.set_index("target_locus_id")["oriented_74nt_window"].to_dict()
    task_rows = list(task.itertuples(index=False))
    pointer = 0
    linked_designs: list[ExactReadDesign] = []
    links: list[dict[str, Any]] = []

    for design in designs:
        linked = design
        if pointer < len(task_rows):
            observation = task_rows[pointer]
            target_sequence = target_sequences[observation.target_locus_id]
            exact_match = (
                design.design_id == observation.design_id
                and design.source_wide_target is not None
                and design.source_wide_target in target_sequence
            )
            if exact_match:
                linked = replace(
                    design,
                    target_locus_id=observation.target_locus_id,
                    observation_id=observation.observation_id,
                )
                links.append(
                    {
                        "registry_version": RAW_REGISTRY_VERSION,
                        "source_study": "DeepPrime",
                        "source_sheet": str(source_sheet),
                        "source_excel_row": int(observation.source_excel_row),
                        "source_record_id": observation.source_record_id,
                        "observation_id": observation.observation_id,
                        "design_id": observation.design_id,
                        "target_locus_id": observation.target_locus_id,
                        "read_design_id": design.design_key,
                        "control_match_id": design.control_match_id,
                        "table_s1_sheet": design.source_sheet,
                        "table_s1_excel_row": design.source_excel_row,
                        "table_s1_design_key": design.source_design_key,
                        "mapping_status": "ordered_exact_subsequence",
                    }
                )
                pointer += 1
        linked_designs.append(linked)

    if pointer != len(task_rows):
        next_row = task_rows[pointer]
        raise ValueError(
            "deepprime_observation_subsequence_incomplete:"
            f"{source_sheet}:{pointer}:{len(task_rows)}:{next_row.source_excel_row}"
        )
    link_frame = pd.DataFrame(links).sort_values("source_excel_row").reset_index(drop=True)
    audit = {
        "status": "pass",
        "source_sheet": str(source_sheet),
        "raw_design_count": len(linked_designs),
        "linked_observation_count": len(link_frame),
        "expected_observation_count": len(task_rows),
        "unique_observation_count": int(link_frame["observation_id"].nunique()),
        "unlinked_raw_design_count": sum(
            design.observation_id is None for design in linked_designs
        ),
        "source_label_used_for_mapping": False,
    }
    return linked_designs, link_frame, audit


def link_pridict2_sheet2_observations(
    designs: Iterable[ExactReadDesign],
    *,
    unified_package: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Link every non-null PRIDICT2 sheet-2 outcome to its exact raw design."""

    design_list = list(designs)
    design_by_row = {design.source_excel_row: design for design in design_list}
    if None in design_by_row or len(design_by_row) != len(design_list):
        raise ValueError("pridict2_design_registry_source_row_failure")
    task = pd.read_parquet(unified_package / "tasks" / "pegRNA_on_target.parquet")
    task = task.loc[
        task["source_study"].eq("PRIDICT2") & task["source_sheet"].astype(str).eq("2")
    ].sort_values(["source_excel_row", "outcome_source_column"])
    links: list[dict[str, Any]] = []
    for observation in task.itertuples(index=False):
        design = design_by_row.get(int(observation.source_excel_row))
        if design is None:
            raise ValueError(
                f"pridict2_observation_without_design:{observation.source_excel_row}"
            )
        if (
            design.design_id != observation.design_id
            or design.target_locus_id != observation.target_locus_id
        ):
            raise ValueError(
                f"pridict2_observation_design_mismatch:{observation.source_excel_row}:"
                f"{observation.outcome_source_column}"
            )
        links.append(
            {
                "registry_version": RAW_REGISTRY_VERSION,
                "source_study": "PRIDICT2",
                "source_sheet": "2",
                "source_excel_row": int(observation.source_excel_row),
                "source_record_id": observation.source_record_id,
                "outcome_source_column": observation.outcome_source_column,
                "observation_id": observation.observation_id,
                "design_id": observation.design_id,
                "target_locus_id": observation.target_locus_id,
                "read_design_id": design.design_key,
                "control_match_id": design.control_match_id,
                "mapping_status": "exact_source_row_design_target",
            }
        )
    link_frame = pd.DataFrame(links).reset_index(drop=True)
    audit = {
        "status": "pass",
        "source_sheet": "2",
        "raw_design_count": len(design_by_row),
        "linked_observation_count": len(link_frame),
        "expected_observation_count": len(task),
        "unique_observation_count": int(link_frame["observation_id"].nunique()),
        "source_label_used_for_mapping": False,
    }
    return link_frame, audit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_raw_registry_package(
    *,
    deepprime_manifest: Path,
    pridict2_manifest: Path,
    deepprime_table_s1: Path,
    pridict2_workbook: Path,
    unified_package: Path,
    output: Path,
    pilot_source_sheet: str = "3",
    pilot_design_sheet: str = "Lib-Small",
) -> dict[str, Any]:
    """Build and atomically publish the v1 run registry plus pilot design links."""

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        runs = build_run_registry(deepprime_manifest, pridict2_manifest)
        designs = load_deepprime_table_s1_designs(
            deepprime_table_s1,
            sheet_name=pilot_design_sheet,
        )
        linked, links, design_audit = link_deepprime_observations(
            designs,
            unified_package=unified_package,
            source_sheet=pilot_source_sheet,
        )
        design_frame = pd.DataFrame(_design_record(design) for design in linked)
        pridict2_designs = load_pridict2_sheet2_designs(pridict2_workbook)
        pridict2_links, pridict2_design_audit = link_pridict2_sheet2_observations(
            pridict2_designs,
            unified_package=unified_package,
        )
        pridict2_design_frame = pd.DataFrame(
            _design_record(design) for design in pridict2_designs
        )

        files = {
            "run_registry": staging / "run_registry.parquet",
            "pilot_raw_design_registry": staging / "deepprime_sheet3_raw_designs.parquet",
            "pilot_observation_links": staging / "deepprime_sheet3_observation_links.parquet",
            "pridict2_sheet2_raw_design_registry": (
                staging / "pridict2_sheet2_raw_designs.parquet"
            ),
            "pridict2_sheet2_observation_links": (
                staging / "pridict2_sheet2_observation_links.parquet"
            ),
        }
        runs.to_parquet(files["run_registry"], index=False, compression="zstd")
        design_frame.to_parquet(
            files["pilot_raw_design_registry"], index=False, compression="zstd"
        )
        links.to_parquet(files["pilot_observation_links"], index=False, compression="zstd")
        pridict2_design_frame.to_parquet(
            files["pridict2_sheet2_raw_design_registry"],
            index=False,
            compression="zstd",
        )
        pridict2_links.to_parquet(
            files["pridict2_sheet2_observation_links"],
            index=False,
            compression="zstd",
        )

        audit = {
            "status": "pass",
            "registry_version": RAW_REGISTRY_VERSION,
            "run_counts": {
                "DeepPrime": int(runs["source_study"].eq("DeepPrime").sum()),
                "PRIDICT2": int(runs["source_study"].eq("PRIDICT2").sum()),
            },
            "unclassified_run_count": int(runs["task_scope"].eq("unclassified").sum()),
            "mapping_status_counts": {
                str(key): int(value)
                for key, value in runs["mapping_status"].value_counts().sort_index().items()
            },
            "pilot_design_link": design_audit,
            "pridict2_sheet2_design_link": pridict2_design_audit,
        }
        audit_path = staging / "audit.json"
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "registry_version": RAW_REGISTRY_VERSION,
            "inputs": {
                "deepprime_manifest": {
                    "path": str(deepprime_manifest.resolve()),
                    "sha256": _sha256(deepprime_manifest),
                },
                "pridict2_manifest": {
                    "path": str(pridict2_manifest.resolve()),
                    "sha256": _sha256(pridict2_manifest),
                },
                "deepprime_table_s1": {
                    "path": str(deepprime_table_s1.resolve()),
                    "sha256": _sha256(deepprime_table_s1),
                },
                "pridict2_workbook": {
                    "path": str(pridict2_workbook.resolve()),
                    "sha256": _sha256(pridict2_workbook),
                },
                "unified_package_manifest": {
                    "path": str((unified_package / "manifest.json").resolve()),
                    "sha256": _sha256(unified_package / "manifest.json"),
                },
            },
            "outputs": {
                name: {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for name, path in files.items()
            },
            "audit_sha256": _sha256(audit_path),
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
        return {"status": "pass", "output": str(output), "audit": audit}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
