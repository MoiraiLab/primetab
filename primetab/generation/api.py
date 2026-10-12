"""Public, model-free PRIDICT2 generation and TFM feature interfaces."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field

import pandas as pd

from primetab.features.features import (
    F1_SEQUENCE_V1_COLUMNS,
    compute_f1_sequence,
    stable_hash,
)
from primetab.generation.core import GenerationError, enumerate_designs, parse_editseq


@dataclass(frozen=True)
class EditRequest:
    sequence_name: str
    editseq: str

    def __post_init__(self) -> None:
        if not str(self.sequence_name).strip():
            raise ValueError("sequence_name must not be empty")


@dataclass(frozen=True)
class GenerationConfig:
    pbs_lengths: tuple[int, ...] = field(default_factory=lambda: tuple(range(7, 16)))
    rt_overhang_lengths: tuple[int, ...] = field(default_factory=lambda: tuple(range(3, 20)))
    max_edit_distance: int = 25
    max_edit_length: int = 40
    include_nicking: bool = False
    include_ngs_primers: bool = False

    def __post_init__(self) -> None:
        pbs_lengths = tuple(dict.fromkeys(int(value) for value in self.pbs_lengths))
        rt_overhang_lengths = tuple(
            dict.fromkeys(int(value) for value in self.rt_overhang_lengths)
        )
        if not pbs_lengths or min(pbs_lengths) < 1:
            raise ValueError("pbs_lengths must contain positive integers")
        if not rt_overhang_lengths or min(rt_overhang_lengths) < 1:
            raise ValueError("rt_overhang_lengths must contain positive integers")
        if int(self.max_edit_distance) < 1:
            raise ValueError("max_edit_distance must be positive")
        if int(self.max_edit_length) < 1:
            raise ValueError("max_edit_length must be positive")
        object.__setattr__(self, "pbs_lengths", pbs_lengths)
        object.__setattr__(
            self,
            "rt_overhang_lengths",
            rt_overhang_lengths,
        )
        object.__setattr__(self, "max_edit_distance", int(self.max_edit_distance))
        object.__setattr__(self, "max_edit_length", int(self.max_edit_length))


REQUEST_COLUMNS = [
    "request_id",
    "sequence_name",
    "editseq",
    "normalized_editseq",
    "edit_type",
    "edit_length",
    "left_flank_length",
    "right_flank_length",
    "outside_validated_range",
    "warnings",
    "status",
]
CANDIDATE_COLUMNS = [
    "request_id",
    "sequence_name",
    "design_id",
    "target_strand",
    "editor_system",
    "mutation_type",
    "edit_type",
    "edit_length",
    "edit_position",
    "pam_relative_to_edit",
    "nick_to_edit_distance",
    "right_homology_arm_length",
    "pbs_length",
    "rt_overhang_length",
    "rtt_length",
    "original_allele",
    "edited_allele",
    "spacer",
    "spacer_u6",
    "pbs",
    "rtt",
    "rtt_original",
    "rtt_overhang",
    "extension",
    "pam_before",
    "pam_after",
    "pam_disrupted",
    "scaffold",
    "pegrna",
    "spacer_oligo_forward",
    "spacer_oligo_reverse",
    "extension_oligo_forward",
    "extension_oligo_reverse",
    "pcr_gg_oligo1",
    "pcr_gg_oligo2",
    "original_sequence_oriented",
    "edited_sequence_oriented",
    "wide_initial_target",
    "wide_mutated_target",
    "canonical_target_74",
    "base_before_edit",
    "base_after_edit",
    "base_before_nick",
    "base_after_nick",
    "rt_overhang_matches",
    "outside_validated_range",
]
NICKING_COLUMNS = [
    "request_id",
    "nicking_guide_id",
    "target_strand",
    "spacer",
    "spacer_u6",
    "pridict2_spacer_19",
    "pam",
    "nick_to_edit_distance",
    "pe3_class",
    "pam_disrupted",
    "context_30nt",
    "oligo_forward",
    "oligo_reverse",
]
PRIMER_COLUMNS = [
    "request_id",
    "primer_set_id",
    "left_primer",
    "right_primer",
    "product_size",
    "primer_pair_penalty",
    "status",
]
ERROR_COLUMNS = ["request_id", "sequence_name", "stage", "code", "detail"]


def _frame(records: list[dict], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records, columns=columns)


@dataclass
class GenerationResult:
    requests: pd.DataFrame
    candidates: pd.DataFrame
    features: pd.DataFrame
    nicking_guides: pd.DataFrame
    primers: pd.DataFrame
    errors: pd.DataFrame
    config: GenerationConfig

    def feature_frame(self) -> pd.DataFrame:
        return tfm_feature_frame(self)


def _request_id(request: EditRequest, normalized_editseq: str | None = None) -> str:
    return stable_hash(
        "request",
        {
            "sequence_name": str(request.sequence_name).strip(),
            "editseq": normalized_editseq or "".join(str(request.editseq).split()).upper(),
        },
    )


def generate(request: EditRequest, config: GenerationConfig | None = None) -> GenerationResult:
    """Generate one request while returning failures as structured result rows."""

    return generate_batch([request], config=config)


def generate_batch(
    requests: Sequence[EditRequest] | Iterable[EditRequest],
    config: GenerationConfig | None = None,
) -> GenerationResult:
    """Generate a deterministic batch; invalid biological inputs do not abort peers."""

    config = config or GenerationConfig()
    request_list = list(requests)
    if not request_list:
        raise ValueError("requests must contain at least one EditRequest")
    names = [str(request.sequence_name).strip() for request in request_list]
    if len(names) != len(set(names)):
        raise ValueError("sequence_name values must be unique within a batch")

    request_rows: list[dict] = []
    candidate_rows: list[dict] = []
    feature_rows_by_id: dict[str, dict] = {}
    nicking_rows: list[dict] = []
    primer_rows: list[dict] = []
    error_rows: list[dict] = []
    f1_columns = list(F1_SEQUENCE_V1_COLUMNS)

    for request in request_list:
        request_name = str(request.sequence_name).strip()
        initial_id = _request_id(request)
        request_id = initial_id
        parsed = None
        try:
            parsed = parse_editseq(request.editseq, max_edit_length=config.max_edit_length)
            request_id = _request_id(request, parsed.normalized_editseq)
            core = enumerate_designs(
                request_id=request_id,
                sequence_name=request_name,
                editseq=request.editseq,
                pbs_lengths=config.pbs_lengths,
                rt_overhang_lengths=config.rt_overhang_lengths,
                max_edit_distance=config.max_edit_distance,
                max_edit_length=config.max_edit_length,
                include_nicking=config.include_nicking,
                include_ngs_primers=config.include_ngs_primers,
            )
            request_features: dict[str, dict] = {}
            for design in core.designs:
                values, failures = compute_f1_sequence(design)
                if list(values) != f1_columns:
                    raise GenerationError(
                        "feature_schema_mismatch",
                        "Computed feature order differs from F1_sequence_v1.",
                        stage="feature",
                    )
                request_features[design.design_id] = {
                    "design_id": design.design_id,
                    "feature_status": (
                        "complete" if not failures else "complete_with_null_features"
                    ),
                    "feature_failure_codes": "|".join(sorted(failures)) if failures else None,
                    **values,
                }

            # Commit one request atomically only after every PE2 feature row is valid.
            request_rows.append(
                {
                    "request_id": request_id,
                    "sequence_name": request_name,
                    "editseq": request.editseq,
                    "normalized_editseq": parsed.normalized_editseq,
                    "edit_type": parsed.edit_type,
                    "edit_length": parsed.edit_length,
                    "left_flank_length": parsed.left_flank_length,
                    "right_flank_length": parsed.right_flank_length,
                    "outside_validated_range": parsed.outside_validated_range,
                    "warnings": ";".join(parsed.warnings),
                    "status": "ok" if core.candidates else "no_candidates",
                }
            )
            candidate_rows.extend(core.candidates)
            feature_rows_by_id.update(request_features)
            nicking_rows.extend(core.nicking_guides)
            primer_rows.extend(core.primers)
            request_errors = [
                {"request_id": request_id, "sequence_name": request_name, **error}
                for error in core.errors
            ]
            if not core.candidates:
                request_errors.append(
                    {
                        "request_id": request_id,
                        "sequence_name": request_name,
                        "stage": "candidate",
                        "code": "no_candidate",
                        "detail": "No PE2-NGG candidate satisfies the configured design window.",
                    }
                )
            error_rows.extend(request_errors)
        except MemoryError:
            raise
        # The batch contract isolates one failed request, including unexpected
        # third-party feature-library errors, while BaseException still propagates.
        except Exception as exc:  # noqa: BLE001
            is_generation_error = isinstance(exc, GenerationError)
            request_rows.append(
                {
                    "request_id": request_id,
                    "sequence_name": request_name,
                    "editseq": request.editseq,
                    "normalized_editseq": (
                        parsed.normalized_editseq if parsed is not None else None
                    ),
                    "edit_type": parsed.edit_type if parsed is not None else None,
                    "edit_length": parsed.edit_length if parsed is not None else None,
                    "left_flank_length": (
                        parsed.left_flank_length if parsed is not None else None
                    ),
                    "right_flank_length": (
                        parsed.right_flank_length if parsed is not None else None
                    ),
                    "outside_validated_range": (
                        parsed.outside_validated_range if parsed is not None else None
                    ),
                    "warnings": ";".join(parsed.warnings) if parsed is not None else "",
                    "status": "error",
                }
            )
            error_rows.append(
                {
                    "request_id": request_id,
                    "sequence_name": request_name,
                    "stage": exc.stage if is_generation_error else "internal",
                    "code": exc.code if is_generation_error else "unexpected_generation_error",
                    "detail": (
                        str(exc)
                        if is_generation_error
                        else f"{type(exc).__name__}: {exc}"
                    ),
                }
            )

    candidate_frame = _frame(candidate_rows, CANDIDATE_COLUMNS)
    if not candidate_frame.empty:
        candidate_frame = candidate_frame.sort_values(
            ["request_id", "target_strand", "spacer", "pbs_length", "rtt_length"]
        ).reset_index(drop=True)
    features = pd.DataFrame.from_records(
        [feature_rows_by_id[key] for key in sorted(feature_rows_by_id)],
        columns=["design_id", "feature_status", "feature_failure_codes", *f1_columns],
    )
    nicking = _frame(nicking_rows, NICKING_COLUMNS)
    if not nicking.empty:
        nicking = nicking.sort_values(
            ["request_id", "target_strand", "nick_to_edit_distance", "spacer"]
        ).drop_duplicates("nicking_guide_id").reset_index(drop=True)
    primers = _frame(primer_rows, PRIMER_COLUMNS)
    if not primers.empty:
        primers = primers.sort_values(["request_id", "primer_set_id"]).reset_index(drop=True)
    errors = _frame(error_rows, ERROR_COLUMNS)
    if not errors.empty:
        errors = errors.sort_values(["request_id", "stage", "code"]).reset_index(drop=True)
    return GenerationResult(
        requests=_frame(request_rows, REQUEST_COLUMNS),
        candidates=candidate_frame,
        features=features,
        nicking_guides=nicking,
        primers=primers,
        errors=errors,
        config=config,
    )


def tfm_feature_frame(result: GenerationResult) -> pd.DataFrame:
    """Return only the frozen 73 sequence features, indexed by design_id."""

    columns = list(F1_SEQUENCE_V1_COLUMNS)
    if result.features.empty:
        return pd.DataFrame(columns=columns, index=pd.Index([], name="design_id"))
    if result.features["design_id"].duplicated().any():
        raise ValueError("features contains duplicate design_id values")
    return result.features.set_index("design_id").loc[:, columns]


def config_record(config: GenerationConfig) -> dict:
    return asdict(config)
