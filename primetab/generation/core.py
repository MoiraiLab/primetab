"""Model-free PRIDICT2-compatible prime-editing design routines.

The candidate-enumeration, nicking-guide, oligo, and primer-design behavior in
this module is derived from the PRIDICT2 command-line implementation:
https://github.com/uzh-dqbm-cmi/PRIDICT2

Copyright (c) 2023 Krauthammer Lab & Schwank Lab & orisenbazuru.
The upstream code is distributed under the MIT License.  This module contains
no PRIDICT/PRIDICT2 or DeepCas9 model loading, inference, score, rank, or
percentile logic.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from primetab.features.features import (
    CanonicalDesign,
    local_bases,
    normalize_dna,
    reverse_complement,
    stable_hash,
    validate_edit_alignment,
)

OPTIMIZED_SCAFFOLD = (
    "GTTTCAGAGCTATGCTGGAAACAGCATAGCAAGTTGAAATAAGGCTAGTCCGTTATCAACTTG"
    "AAAAAGTGGCACCGAGTCGGTGC"
)
UPSTREAM_SOURCE_SHA256 = "b9e13ec84b20a41c3f8ca36c605dd5706582437ac24db179fcef0848d20142ea"


class GenerationError(ValueError):
    """A stable, stage-specific failure suitable for batch error reporting."""

    def __init__(self, code: str, message: str, *, stage: str = "input") -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage


@dataclass(frozen=True)
class ParsedEdit:
    normalized_editseq: str
    original_sequence: str
    edited_sequence: str
    original_allele: str
    edited_allele: str
    edit_type: str
    mutation_type: str
    edit_length: int
    left_flank_length: int
    right_flank_length: int
    outside_validated_range: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class CoreGeneration:
    parsed: ParsedEdit
    candidates: tuple[dict[str, Any], ...]
    designs: tuple[CanonicalDesign, ...]
    nicking_guides: tuple[dict[str, Any], ...]
    primers: tuple[dict[str, Any], ...]
    errors: tuple[dict[str, str], ...]


def _clean_marked_sequence(value: str) -> str:
    sequence = "".join(str(value).split()).upper()
    if sequence.count("(") != 1 or sequence.count(")") != 1:
        raise GenerationError(
            "invalid_edit_brackets",
            "Exactly one parenthesized edit is required.",
        )
    left = sequence.index("(")
    right = sequence.index(")")
    if right <= left + 1 or "(" in sequence[left + 1 : right] or ")" in sequence[:right]:
        raise GenerationError("invalid_edit_brackets", "The edit brackets are malformed.")
    return sequence


def _require_dna(sequence: str, field: str, *, allow_empty: bool = False) -> str:
    if not sequence and allow_empty:
        return sequence
    try:
        return normalize_dna(sequence)
    except ValueError as exc:
        raise GenerationError("invalid_dna", f"{field}: {exc}") from exc


def parse_editseq(editseq: str, *, max_edit_length: int = 40) -> ParsedEdit:
    """Parse the PRIDICT marked-edit syntax into immutable oriented sequences."""

    sequence = _clean_marked_sequence(editseq)
    left_index = sequence.index("(")
    right_index = sequence.index(")")
    five_prime = _require_dna(sequence[:left_index], "five_prime_flank")
    three_prime = _require_dna(sequence[right_index + 1 :], "three_prime_flank")
    token = sequence[left_index + 1 : right_index]

    original_allele: str
    edited_allele: str
    edit_type: str
    mutation_type: str
    if token.startswith("+") and token.count("+") == 1:
        original_allele = ""
        edited_allele = _require_dna(token[1:], "inserted_allele")
        edit_type = "insertion"
        mutation_type = "Insertion"
    elif token.startswith("-") and token.count("-") == 1:
        original_allele = _require_dna(token[1:], "deleted_allele")
        edited_allele = ""
        edit_type = "deletion"
        mutation_type = "Deletion"
    elif token.count("/") == 1:
        original_token, edited_token = token.split("/")
        if original_token == "-":
            original_allele = ""
            edited_allele = _require_dna(edited_token, "inserted_allele")
            edit_type = "insertion"
            mutation_type = "Insertion"
        elif edited_token == "-":
            original_allele = _require_dna(original_token, "deleted_allele")
            edited_allele = ""
            edit_type = "deletion"
            mutation_type = "Deletion"
        else:
            original_allele = _require_dna(original_token, "original_allele")
            edited_allele = _require_dna(edited_token, "edited_allele")
            if len(original_allele) != len(edited_allele):
                raise GenerationError(
                    "unequal_replacement_lengths",
                    "Replacement alleles must have equal lengths; use insertion/deletion syntax otherwise.",
                )
            if original_allele == edited_allele:
                raise GenerationError("declared_edit_absent", "The replacement does not change sequence.")
            if (
                original_allele[0] == edited_allele[0]
                or original_allele[-1] == edited_allele[-1]
            ):
                raise GenerationError(
                    "unchanged_edit_flank_inside_brackets",
                    "Unchanged flanking bases must be placed outside the edit brackets.",
                )
            edit_type = "substitution"
            mutation_type = (
                "1bpReplacement" if len(original_allele) == 1 else "MultibpReplacement"
            )
    else:
        raise GenerationError(
            "invalid_edit_token",
            "Use (A/G), (+SEQ), or (-SEQ) edit notation.",
        )

    edit_length = len(edited_allele) if edit_type == "insertion" else len(original_allele)
    if edit_length < 1:
        raise GenerationError("empty_edit", "The edit allele must not be empty.")
    if edit_length > max_edit_length:
        raise GenerationError(
            "edit_too_long",
            f"Edit length {edit_length} exceeds the configured maximum {max_edit_length}.",
        )

    warnings: list[str] = []
    # PRIDICT2 documents 100 nt but its released code and official example accept 99 nt.
    if len(five_prime) < 99 or len(three_prime) < 99:
        raise GenerationError(
            "insufficient_flanking_sequence",
            "At least 99 nt on each side are required for PRIDICT2 compatibility.",
        )
    if len(five_prime) < 100 or len(three_prime) < 100:
        warnings.append("flank_below_documented_100")
    if edit_length > 15:
        warnings.append("outside_pridict2_validated_edit_range")

    original_sequence = five_prime + original_allele + three_prime
    edited_sequence = five_prime + edited_allele + three_prime
    normalized_token = {
        "substitution": f"{original_allele}/{edited_allele}",
        "insertion": f"+{edited_allele}",
        "deletion": f"-{original_allele}",
    }[edit_type]
    normalized = f"{five_prime}({normalized_token}){three_prime}"
    return ParsedEdit(
        normalized_editseq=normalized,
        original_sequence=original_sequence,
        edited_sequence=edited_sequence,
        original_allele=original_allele,
        edited_allele=edited_allele,
        edit_type=edit_type,
        mutation_type=mutation_type,
        edit_length=edit_length,
        left_flank_length=len(five_prime),
        right_flank_length=len(three_prime),
        outside_validated_range=edit_length > 15,
        warnings=tuple(warnings),
    )


def _pam_gg_positions(sequence: str) -> list[int]:
    """Return PRIDICT2 regex positions (the first G in each overlapping GG)."""

    return [
        index
        for index in range(25, len(sequence) - 4)
        if sequence[index : index + 2] == "GG"
    ]


def _count_overlapping(sequence: str, subsequence: str) -> int:
    return sum(
        sequence.startswith(subsequence, index)
        for index in range(max(0, len(sequence) - len(subsequence) + 1))
    )


def _oriented_edit(parsed: ParsedEdit, target_strand: str) -> tuple[str, str, str, str, int]:
    if target_strand == "Fw":
        return (
            parsed.original_sequence,
            parsed.edited_sequence,
            parsed.original_allele,
            parsed.edited_allele,
            parsed.left_flank_length,
        )
    return (
        reverse_complement(parsed.original_sequence),
        reverse_complement(parsed.edited_sequence),
        reverse_complement(parsed.original_allele) if parsed.original_allele else "",
        reverse_complement(parsed.edited_allele) if parsed.edited_allele else "",
        parsed.right_flank_length,
    )


def classify_nicking_guide(nick_to_edit_distance: int) -> tuple[str, bool]:
    """Return the upstream PRIDICT2 PE3 class and PAM-disruption flag."""

    is_pe3b = nick_to_edit_distance in {-5, -4} or -2 <= nick_to_edit_distance <= 16
    return ("PE3b" if is_pe3b else "PE3", nick_to_edit_distance in {-5, -4})


def _nicking_records(
    request_id: str,
    parsed: ParsedEdit,
    target_strand: str,
    original_sequence: str,
    edited_sequence: str,
    edit_index: int,
) -> list[dict[str, Any]]:
    # Upstream emits nicking guides for a strand only when that strand has a primary PAM.
    if not _pam_gg_positions(original_sequence):
        return []
    records: list[dict[str, Any]] = []
    for gg_index in _pam_gg_positions(edited_sequence):
        pam_start = gg_index - 1
        canonical_spacer = edited_sequence[pam_start - 20 : pam_start]
        upstream_spacer_19 = edited_sequence[pam_start - 19 : pam_start]
        if len(canonical_spacer) != 20 or len(upstream_spacer_19) != 19:
            continue
        spacer_u6 = (
            upstream_spacer_19
            if upstream_spacer_19.startswith("G")
            else "G" + upstream_spacer_19
        )
        nick_to_edit_distance = pam_start - 3 - edit_index
        pe3_class, pam_disrupted = classify_nicking_guide(nick_to_edit_distance)
        context_30nt = edited_sequence[pam_start - 24 : pam_start + 6]
        guide_id = stable_hash(
            "nicking",
            {
                "request_id": request_id,
                "target_strand": target_strand,
                "spacer": canonical_spacer,
                "nick_to_edit_distance": nick_to_edit_distance,
            },
        )
        records.append(
            {
                "request_id": request_id,
                "nicking_guide_id": guide_id,
                "target_strand": target_strand,
                "spacer": canonical_spacer,
                "spacer_u6": spacer_u6,
                "pridict2_spacer_19": upstream_spacer_19,
                "pam": edited_sequence[pam_start : pam_start + 3],
                "nick_to_edit_distance": nick_to_edit_distance,
                "pe3_class": pe3_class,
                "pam_disrupted": pam_disrupted,
                "context_30nt": context_30nt,
                "oligo_forward": "CACC" + spacer_u6,
                "oligo_reverse": "AAAC" + reverse_complement(spacer_u6),
            }
        )
    return records


def _primer_records(request_id: str, parsed: ParsedEdit) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    try:
        import primer3  # type: ignore[import-not-found]
    except ImportError as exc:
        return [], [
            {
                "stage": "primer",
                "code": "primer3_unavailable",
                "detail": str(exc),
            }
        ]

    sequence_target = [parsed.left_flank_length - 20, 40 + len(parsed.original_allele)]
    sequence_args = {
        "SEQUENCE_TEMPLATE": parsed.original_sequence,
        "SEQUENCE_TARGET": sequence_target,
    }
    global_args = {
        "PRIMER_OPT_SIZE": 20,
        "PRIMER_PICK_INTERNAL_OLIGO": 0,
        "PRIMER_MIN_SIZE": 18,
        "PRIMER_MAX_SIZE": 25,
        "PRIMER_OPT_TM": 60.0,
        "PRIMER_MIN_TM": 56.0,
        "PRIMER_MAX_TM": 64.0,
        "PRIMER_MIN_GC": 20.0,
        "PRIMER_MAX_GC": 80.0,
        "PRIMER_MAX_POLY_X": 100,
        "PRIMER_SALT_MONOVALENT": 50.0,
        "PRIMER_DNA_CONC": 50.0,
        "PRIMER_MAX_NS_ACCEPTED": 0,
        "PRIMER_MAX_SELF_ANY": 12,
        "PRIMER_MAX_SELF_END": 8,
        "PRIMER_PAIR_MAX_COMPL_ANY": 12,
        "PRIMER_PAIR_MAX_COMPL_END": 8,
        "PRIMER_PRODUCT_SIZE_RANGE": [[130, 200]],
    }
    try:
        outcome = primer3.bindings.design_primers(sequence_args, global_args)
        left = outcome.get("PRIMER_LEFT_0_SEQUENCE")
        right = outcome.get("PRIMER_RIGHT_0_SEQUENCE")
        if not left or not right:
            raise GenerationError(
                "primer_pair_unavailable",
                str(outcome.get("PRIMER_ERROR") or outcome.get("PRIMER_WARNING") or "No primer pair"),
                stage="primer",
            )
        primer_set_id = stable_hash(
            "primer",
            {"request_id": request_id, "left": left, "right": right},
        )
        return [
            {
                "request_id": request_id,
                "primer_set_id": primer_set_id,
                "left_primer": str(left).upper(),
                "right_primer": str(right).upper(),
                "product_size": int(outcome["PRIMER_PAIR_0_PRODUCT_SIZE"]),
                "primer_pair_penalty": float(outcome["PRIMER_PAIR_0_PENALTY"]),
                "status": "ok",
            }
        ], []
    except (GenerationError, KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
        code = exc.code if isinstance(exc, GenerationError) else "primer_design_failed"
        return [], [{"stage": "primer", "code": code, "detail": str(exc)}]


def enumerate_designs(
    *,
    request_id: str,
    sequence_name: str,
    editseq: str,
    pbs_lengths: Iterable[int],
    rt_overhang_lengths: Iterable[int],
    max_edit_distance: int,
    max_edit_length: int,
    include_nicking: bool,
    include_ngs_primers: bool,
) -> CoreGeneration:
    """Enumerate PRIDICT2-compatible designs without any predictive model."""

    parsed = parse_editseq(editseq, max_edit_length=max_edit_length)
    pbs_values = tuple(int(value) for value in pbs_lengths)
    overhang_values = tuple(int(value) for value in rt_overhang_lengths)
    if not pbs_values or min(pbs_values) < 1:
        raise GenerationError("invalid_pbs_lengths", "PBS lengths must be positive.")
    if not overhang_values or min(overhang_values) < 1:
        raise GenerationError("invalid_rt_overhang_lengths", "RT overhang lengths must be positive.")
    if max_edit_distance < 1:
        raise GenerationError("invalid_max_edit_distance", "Maximum edit distance must be positive.")

    candidates: list[dict[str, Any]] = []
    designs: list[CanonicalDesign] = []
    nicking_guides: list[dict[str, Any]] = []

    for target_strand in ("Fw", "Rv"):
        original, edited, original_allele, edited_allele, edit_index = _oriented_edit(
            parsed, target_strand
        )
        gg_positions = _pam_gg_positions(original)
        if include_nicking:
            nicking_guides.extend(
                _nicking_records(
                    request_id,
                    parsed,
                    target_strand,
                    original,
                    edited,
                    edit_index,
                )
            )
        for gg_index in gg_positions:
            pam_start = gg_index - 1
            pam_relative_to_edit = gg_index - edit_index - 1
            if pam_relative_to_edit not in range(-max_edit_distance + 3, 4):
                continue
            nick_index = pam_start - 3
            spacer_start = pam_start - 20
            canonical_spacer = original[spacer_start:pam_start]
            upstream_spacer_19 = original[pam_start - 19 : pam_start]
            if len(canonical_spacer) != 20 or len(upstream_spacer_19) != 19:
                continue
            spacer_u6 = "G" + upstream_spacer_19
            pam_before = original[pam_start : pam_start + 3]
            pam_after = edited[pam_start : pam_start + 3]

            for rt_overhang_length in overhang_values:
                stop = edit_index + len(edited_allele) + rt_overhang_length
                rtt_target = edited[nick_index:stop]
                rt_overhang_target = edited[stop - rt_overhang_length : stop]
                if not rtt_target or len(rt_overhang_target) != rt_overhang_length:
                    continue
                rtt = reverse_complement(rtt_target).upper()
                rtt_overhang = reverse_complement(rt_overhang_target).upper()
                edit_position = edit_index - nick_index + 1
                original_rtt_length = len(rtt)
                if parsed.edit_type == "insertion":
                    original_rtt_length -= parsed.edit_length
                elif parsed.edit_type == "deletion":
                    original_rtt_length += parsed.edit_length
                rtt_original_target = original[nick_index : nick_index + original_rtt_length]
                if len(rtt_original_target) != original_rtt_length:
                    continue
                try:
                    validate_edit_alignment(
                        rtt_original_target,
                        rtt_target.upper(),
                        parsed.edit_type,
                        edit_position,
                        parsed.edit_length,
                    )
                except ValueError as exc:
                    raise GenerationError(
                        "candidate_edit_alignment_failed",
                        str(exc),
                        stage="candidate",
                    ) from exc
                rtt_original = reverse_complement(rtt_original_target).upper()
                rha = (
                    len(rtt) - edit_position + 1
                    if parsed.edit_type == "deletion"
                    else len(rtt) - edit_position - parsed.edit_length + 1
                )
                if rha != rt_overhang_length:
                    raise GenerationError(
                        "rha_mismatch",
                        f"Calculated RHA {rha} != configured overhang {rt_overhang_length}.",
                        stage="candidate",
                    )

                bases = local_bases(
                    original,
                    nick_index,
                    edit_position,
                    parsed.edit_type,
                    parsed.edit_length,
                )
                for pbs_length in pbs_values:
                    pbs_target = original[nick_index - pbs_length : nick_index]
                    if len(pbs_target) != pbs_length:
                        continue
                    pbs = reverse_complement(pbs_target).upper()
                    design = CanonicalDesign(
                        spacer=canonical_spacer,
                        pbs=pbs,
                        rtt=rtt,
                        pam_before=pam_before,
                        pam_after=pam_after,
                        edit_type=parsed.edit_type,
                        edit_position=edit_position,
                        edit_length=parsed.edit_length,
                        right_homology_arm_length=rha,
                        rtt_original=rtt_original,
                        **bases,
                    )
                    extension = rtt + pbs
                    if extension != design.extension:
                        raise GenerationError(
                            "extension_invariant_failed",
                            "Extension is not RTT+PBS.",
                            stage="candidate",
                        )
                    canonical_target_74 = original[spacer_start - 4 : spacer_start + 70]
                    wide_start = pam_start - 29
                    wide_initial = original[wide_start : wide_start + 99]
                    wide_mutated = edited[wide_start : wide_start + 99]
                    overhang_window = edited[stop - rt_overhang_length : stop + 15]
                    record = {
                        "request_id": request_id,
                        "sequence_name": sequence_name,
                        "design_id": design.design_id,
                        "target_strand": target_strand,
                        "editor_system": "PE2-NGG",
                        "mutation_type": parsed.mutation_type,
                        "edit_type": parsed.edit_type,
                        "edit_length": parsed.edit_length,
                        "edit_position": edit_position,
                        "pam_relative_to_edit": pam_relative_to_edit,
                        "nick_to_edit_distance": pam_relative_to_edit - 3,
                        "right_homology_arm_length": rha,
                        "pbs_length": pbs_length,
                        "rt_overhang_length": rt_overhang_length,
                        "rtt_length": len(rtt),
                        "original_allele": original_allele,
                        "edited_allele": edited_allele,
                        "spacer": canonical_spacer,
                        "spacer_u6": spacer_u6,
                        "pbs": pbs,
                        "rtt": rtt,
                        "rtt_original": rtt_original,
                        "rtt_overhang": rtt_overhang,
                        "extension": extension,
                        "pam_before": pam_before,
                        "pam_after": pam_after,
                        "pam_disrupted": pam_before != pam_after,
                        "scaffold": OPTIMIZED_SCAFFOLD,
                        "pegrna": spacer_u6 + OPTIMIZED_SCAFFOLD + extension,
                        "spacer_oligo_forward": "CACC" + spacer_u6 + OPTIMIZED_SCAFFOLD[:5],
                        "spacer_oligo_reverse": reverse_complement(OPTIMIZED_SCAFFOLD[:9])
                        + reverse_complement(spacer_u6),
                        "extension_oligo_forward": OPTIMIZED_SCAFFOLD[-4:] + extension,
                        "extension_oligo_reverse": "AAAA" + pbs_target + rtt_target.upper(),
                        "pcr_gg_oligo1": "ATGGTCTCGCACC"
                        + spacer_u6
                        + "GTTTCAGAGCTATGCTGGAAACAGC",
                        "pcr_gg_oligo2": "ATGGTCTCGCGCG"
                        + pbs_target
                        + rtt_target.upper()
                        + "GCACCGACTCGGTGCCAC",
                        "original_sequence_oriented": original,
                        "edited_sequence_oriented": edited,
                        "wide_initial_target": wide_initial,
                        "wide_mutated_target": wide_mutated,
                        "canonical_target_74": canonical_target_74,
                        "base_before_edit": bases["base_before_edit"],
                        "base_after_edit": bases["base_after_edit"],
                        "base_before_nick": bases["base_before_nick"],
                        "base_after_nick": bases["base_after_nick"],
                        "rt_overhang_matches": _count_overlapping(
                            overhang_window, rt_overhang_target
                        ),
                        "outside_validated_range": parsed.outside_validated_range,
                    }
                    candidates.append(record)
                    designs.append(design)

    primers: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if include_ngs_primers:
        primers, primer_errors = _primer_records(request_id, parsed)
        errors.extend(primer_errors)

    # IDs are content hashes; dropping duplicates makes repeated PAM regex paths harmless.
    candidate_by_id = {record["design_id"]: record for record in candidates}
    design_by_id = {design.design_id: design for design in designs}
    guide_by_id = {record["nicking_guide_id"]: record for record in nicking_guides}
    return CoreGeneration(
        parsed=parsed,
        candidates=tuple(candidate_by_id[key] for key in sorted(candidate_by_id)),
        designs=tuple(design_by_id[key] for key in sorted(design_by_id)),
        nicking_guides=tuple(guide_by_id[key] for key in sorted(guide_by_id)),
        primers=tuple(primers),
        errors=tuple(errors),
    )
