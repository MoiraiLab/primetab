"""Source-independent sequence normalization and F0/F1 feature computation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from Bio.SeqUtils import MeltingTemp as mt

DNA_ALPHABET = frozenset("ACGT")
BASES = "ACGT"
REGIONS = ("spacer", "pbs", "rtt", "extension")

F0_CORE_V1_COLUMNS = (
    "pbs_length",
    "rtt_length",
    "extension_length",
    "edit_position",
    "edit_length",
    "right_homology_arm_length",
    "edit_type_substitution",
    "edit_type_insertion",
    "edit_type_deletion",
    *(f"gc_count_{region}" for region in REGIONS),
    *(f"gc_fraction_{region}" for region in REGIONS),
)
F1_SEQUENCE_V1_COLUMNS = (
    *F0_CORE_V1_COLUMNS,
    *(f"base_fraction_{base.lower()}_{region}" for region in REGIONS for base in BASES),
    *(f"max_run_{base.lower()}_{region}" for region in REGIONS for base in BASES),
    "poly_t_max_run_spacer_extension",
    "poly_t_risk_ge4",
    "poly_t_risk_ge5",
    "poly_t_risk_ge6",
    "pam_before",
    "pam_after",
    "pam_length",
    "pam_disrupted",
    "base_before_edit",
    "base_after_edit",
    "base_before_nick",
    "base_after_nick",
    "tm_pbs_rna_dna",
    "tm_rtt_dna_dna_original",
    "tm_rtt_dna_dna_edited",
    "tm_rtt_dna_dna_mismatch",
    "tm_rtt_rna_dna",
    "delta_tm_edited_original",
    "mfe_extension_plain",
    "mfe_extension_deepprime_poly_u6",
    "mfe_spacer_canonical",
    "mfe_spacer_u6",
    "mfe_rtt",
    "mfe_pbs",
)


def stable_hash(prefix: str, payload: Any) -> str:
    """Return a deterministic full SHA-256 identifier."""

    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()}"


def normalize_dna(value: Any) -> str:
    """Normalize a DNA value without guessing or repairing invalid bases."""

    if value is None:
        raise ValueError("missing_sequence")
    sequence = str(value).strip().upper().replace(" ", "")
    if not sequence:
        raise ValueError("empty_sequence")
    invalid = sorted(set(sequence) - DNA_ALPHABET)
    if invalid:
        raise ValueError(f"invalid_bases:{''.join(invalid)}")
    return sequence


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def complement(sequence: str) -> str:
    """Return the 3'-to-5' complement expected by Biopython ``c_seq``."""

    return sequence.translate(str.maketrans("ACGT", "TGCA"))


def rna_reverse_complement(sequence: str) -> str:
    return reverse_complement(sequence).replace("T", "U")


def parse_edit_type(value: Any) -> str:
    normalized = str(value).strip().lower()
    mapping = {
        "sub": "substitution",
        "substitution": "substitution",
        "replacement": "substitution",
        "ins": "insertion",
        "insertion": "insertion",
        "del": "deletion",
        "deletion": "deletion",
    }
    if normalized not in mapping:
        raise ValueError(f"unknown_edit_type:{value}")
    return mapping[normalized]


def infer_rtt_original_length(rtt_length: int, edit_type: str, edit_length: int) -> int:
    if edit_type == "insertion":
        return rtt_length - edit_length
    if edit_type == "deletion":
        return rtt_length + edit_length
    return rtt_length


def reconstruct_edited_target(
    wide_target: str,
    nick_index: int,
    rtt: str,
    edit_type: str,
    edit_length: int,
) -> tuple[str, str]:
    """Patch the edited RTT into an oriented target and return its original RTT segment."""

    original_length = infer_rtt_original_length(len(rtt), edit_type, edit_length)
    if original_length <= 0 or nick_index < 0 or nick_index + original_length > len(wide_target):
        raise ValueError("rtt_original_out_of_bounds")
    original = wide_target[nick_index : nick_index + original_length]
    edited = wide_target[:nick_index] + rtt + wide_target[nick_index + original_length :]
    return edited, original


def validate_edit_alignment(
    original_target: str,
    edited_target: str,
    edit_type: str,
    edit_position: int,
    edit_length: int,
    *,
    allow_additional_mismatches: bool = False,
) -> None:
    """Validate one exact edit event in target orientation without fuzzy matching."""

    start = edit_position - 1
    if start < 0 or edit_length <= 0:
        raise ValueError("invalid_edit_coordinates")
    if edit_type == "substitution":
        end = start + edit_length
        if (
            len(original_target) == len(edited_target)
            and end <= len(original_target)
            and original_target[start:end] == edited_target[start:end]
        ):
            raise ValueError("declared_edit_absent")
        valid = (
            len(original_target) == len(edited_target)
            and end <= len(original_target)
            and original_target[start:end] != edited_target[start:end]
        )
        if not allow_additional_mismatches:
            valid = (
                valid
                and original_target[:start] == edited_target[:start]
                and original_target[end:] == edited_target[end:]
            )
    elif edit_type == "insertion":
        end = start + edit_length
        valid = (
            len(edited_target) == len(original_target) + edit_length
            and start <= len(original_target)
        )
        if not allow_additional_mismatches:
            valid = (
                valid
                and edited_target[:start] == original_target[:start]
                and edited_target[end:] == original_target[start:]
            )
    elif edit_type == "deletion":
        end = start + edit_length
        valid = (
            len(original_target) == len(edited_target) + edit_length
            and end <= len(original_target)
        )
        if not allow_additional_mismatches:
            valid = (
                valid
                and original_target[:start] == edited_target[:start]
                and original_target[end:] == edited_target[start:]
            )
    else:
        raise ValueError(f"unknown_edit_type:{edit_type}")
    if not valid:
        raise ValueError("edit_alignment_mismatch")


def local_bases(
    original_target: str,
    nick_index: int,
    edit_position: int,
    edit_type: str,
    edit_length: int,
) -> dict[str, str | None]:
    """Extract bases adjacent to the nick and edit using the frozen coordinate convention."""

    edit_index = nick_index + edit_position - 1
    original_span = 0 if edit_type == "insertion" else edit_length

    def at(index: int) -> str | None:
        return original_target[index] if 0 <= index < len(original_target) else None

    return {
        "base_before_edit": at(edit_index - 1),
        "base_after_edit": at(edit_index + original_span),
        "base_before_nick": at(nick_index - 1),
        "base_after_nick": at(nick_index),
    }


@dataclass(frozen=True)
class CanonicalDesign:
    spacer: str
    pbs: str
    rtt: str
    pam_before: str
    pam_after: str
    edit_type: str
    edit_position: int
    edit_length: int
    right_homology_arm_length: int
    rtt_original: str
    base_before_edit: str | None
    base_after_edit: str | None
    base_before_nick: str | None
    base_after_nick: str | None

    @property
    def extension(self) -> str:
        return self.rtt + self.pbs

    @property
    def design_id(self) -> str:
        return stable_hash(
            "design",
            {
                "spacer": self.spacer,
                "pbs": self.pbs,
                "rtt": self.rtt,
                "rtt_original": self.rtt_original,
                "pam_before": self.pam_before,
                "pam_after": self.pam_after,
                "edit_type": self.edit_type,
                "edit_position": self.edit_position,
                "edit_length": self.edit_length,
            },
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "design_id": self.design_id,
            "spacer": self.spacer,
            "pbs": self.pbs,
            "rtt": self.rtt,
            "extension": self.extension,
            "pam_before": self.pam_before,
            "pam_after": self.pam_after,
            "edit_type": self.edit_type,
            "edit_position": self.edit_position,
            "edit_length": self.edit_length,
            "right_homology_arm_length": self.right_homology_arm_length,
            "rtt_original": self.rtt_original,
            "base_before_edit": self.base_before_edit,
            "base_after_edit": self.base_after_edit,
            "base_before_nick": self.base_before_nick,
            "base_after_nick": self.base_after_nick,
        }


def gc_count(sequence: str) -> int:
    return sequence.count("G") + sequence.count("C")


def fraction(sequence: str, base: str) -> float:
    return sequence.count(base) / len(sequence)


def max_run(sequence: str, base: str) -> int:
    longest = current = 0
    for char in sequence:
        if char == base:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def compute_f0(design: CanonicalDesign) -> dict[str, Any]:
    regions = {
        "spacer": design.spacer,
        "pbs": design.pbs,
        "rtt": design.rtt,
        "extension": design.extension,
    }
    result: dict[str, Any] = {
        "pbs_length": len(design.pbs),
        "rtt_length": len(design.rtt),
        "extension_length": len(design.extension),
        "edit_position": design.edit_position,
        "edit_length": design.edit_length,
        "right_homology_arm_length": design.right_homology_arm_length,
        "edit_type_substitution": int(design.edit_type == "substitution"),
        "edit_type_insertion": int(design.edit_type == "insertion"),
        "edit_type_deletion": int(design.edit_type == "deletion"),
    }
    for region, sequence in regions.items():
        result[f"gc_count_{region}"] = gc_count(sequence)
    for region, sequence in regions.items():
        result[f"gc_fraction_{region}"] = gc_count(sequence) / len(sequence)
    return result


TM_PARAMETERS = {
    "dnac1": 25,
    "dnac2": 25,
    "Na": 50,
    "K": 0,
    "Tris": 0,
    "Mg": 0,
    "dNTPs": 0,
    "saltcorr": 5,
    "selfcomp": False,
    "strict": True,
}


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


@lru_cache(maxsize=300_000)
def tm_dna(sequence: str) -> float | None:
    try:
        return _finite_or_none(mt.Tm_NN(sequence, nn_table=mt.DNA_NN3, **TM_PARAMETERS))
    except (ValueError, KeyError, IndexError, ZeroDivisionError):
        return None


@lru_cache(maxsize=300_000)
def tm_rna_dna(transcript_sequence: str) -> float | None:
    try:
        # Canonical PBS/RTT sequences already use the 5'-to-3' pegRNA
        # transcript direction.  Biopython infers the complementary DNA
        # strand when c_seq is omitted, matching the DeepPrime calculation.
        rna = transcript_sequence.replace("T", "U")
        return _finite_or_none(mt.Tm_NN(rna, nn_table=mt.R_DNA_NN1, **TM_PARAMETERS))
    except (ValueError, KeyError, IndexError, ZeroDivisionError):
        return None


@lru_cache(maxsize=300_000)
def tm_mismatch(edited_rtt: str, original_rtt: str) -> float | None:
    if len(edited_rtt) != len(original_rtt):
        return None
    try:
        template = complement(original_rtt)
        value = mt.Tm_NN(
            edited_rtt,
            c_seq=template,
            nn_table=mt.DNA_NN3,
            **TM_PARAMETERS,
        )
        return _finite_or_none(value)
    except (ValueError, KeyError, IndexError, ZeroDivisionError):
        return None


@lru_cache(maxsize=500_000)
def rna_mfe(dna_sequence: str) -> float | None:
    try:
        import RNA  # type: ignore[import-not-found]

        rna = dna_sequence.replace("T", "U")
        model = RNA.md()
        model.temperature = 37.0
        model.dangles = 2
        model.noLP = 0
        _structure, energy = RNA.fold_compound(rna, model).mfe()
        return _finite_or_none(energy)
    except (ImportError, ValueError, RuntimeError):
        return None


def compute_f1_sequence(design: CanonicalDesign) -> tuple[dict[str, Any], list[str]]:
    result = compute_f0(design)
    failures: list[str] = []
    regions = {
        "spacer": design.spacer,
        "pbs": design.pbs,
        "rtt": design.rtt,
        "extension": design.extension,
    }
    for region, sequence in regions.items():
        for base in BASES:
            result[f"base_fraction_{base.lower()}_{region}"] = fraction(sequence, base)
    for region, sequence in regions.items():
        for base in BASES:
            result[f"max_run_{base.lower()}_{region}"] = max_run(sequence, base)

    transcribed = design.spacer + design.extension
    poly_t = max_run(transcribed, "T")
    result.update(
        {
            "poly_t_max_run_spacer_extension": poly_t,
            "poly_t_risk_ge4": int(poly_t >= 4),
            "poly_t_risk_ge5": int(poly_t >= 5),
            "poly_t_risk_ge6": int(poly_t >= 6),
            "pam_before": design.pam_before,
            "pam_after": design.pam_after,
            "pam_length": len(design.pam_before),
            "pam_disrupted": int(design.pam_before != design.pam_after),
            "base_before_edit": design.base_before_edit,
            "base_after_edit": design.base_after_edit,
            "base_before_nick": design.base_before_nick,
            "base_after_nick": design.base_after_nick,
        }
    )

    tm_values = {
        "tm_pbs_rna_dna": tm_rna_dna(design.pbs),
        "tm_rtt_dna_dna_original": tm_dna(design.rtt_original),
        "tm_rtt_dna_dna_edited": tm_dna(design.rtt),
        "tm_rtt_dna_dna_mismatch": tm_mismatch(design.rtt, design.rtt_original),
        "tm_rtt_rna_dna": tm_rna_dna(design.rtt),
    }
    original_tm = tm_values["tm_rtt_dna_dna_original"]
    edited_tm = tm_values["tm_rtt_dna_dna_edited"]
    tm_values["delta_tm_edited_original"] = (
        edited_tm - original_tm
        if edited_tm is not None and original_tm is not None
        else None
    )
    for name, value in tm_values.items():
        if value is None:
            failures.append(f"{name}:unavailable")
    result.update(tm_values)

    mfe_inputs = {
        "mfe_extension_plain": design.extension,
        # DeepPrime starts from target-oriented PBS+RTT and folds its reverse
        # complement.  In the unified transcript orientation that is exactly
        # RTT+PBS, i.e. the canonical extension.
        "mfe_extension_deepprime_poly_u6": design.extension + "TTTTTT",
        "mfe_spacer_canonical": design.spacer,
        "mfe_spacer_u6": "G" + design.spacer[1:],
        "mfe_rtt": design.rtt,
        "mfe_pbs": design.pbs,
    }
    for name, sequence in mfe_inputs.items():
        result[name] = rna_mfe(sequence)
        if result[name] is None:
            failures.append(f"{name}:unavailable")
    return result, failures
