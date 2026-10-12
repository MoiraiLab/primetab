"""Hard constraints for the cross-cell experiment.

This module intentionally contains no model-specific logic. Every trainable
model and learned preprocessor must call ``assert_fit_frame`` before fitting.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

_CELL_ALIASES = {
    "hek293t": "HEK293T",
    "hek293": "HEK293T",
    "hek-293t": "HEK293T",
    "a549": "A549",
    "hela": "HeLa",
    "hela-s3": "HeLa",
    "dld1": "DLD1",
    "dld-1": "DLD1",
}


def canonicalize_cell_line(value: object) -> str:
    """Return a stable cell-line label without guessing unknown values."""

    raw = str(value).strip()
    key = raw.lower().replace(" ", "")
    return _CELL_ALIASES.get(key, raw)


def canonicalize_editor(value: object) -> str:
    raw = str(value).strip()
    return raw.replace(" ", "")


@dataclass(frozen=True)
class SplitPolicy:
    editor: str
    train_cells: tuple[str, ...]
    validation_cell: str
    test_cell: str
    group_column: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> SplitPolicy:
        with Path(path).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        strict = config["strict_track"]
        policy = cls(
            editor=canonicalize_editor(strict["editor"]),
            train_cells=tuple(canonicalize_cell_line(x) for x in strict["train_cells"]),
            validation_cell=canonicalize_cell_line(strict["cross_cell_validation_cell"]),
            test_cell=canonicalize_cell_line(strict["sealed_test_cell"]),
            group_column=strict["group_column"],
        )
        policy.assert_policy_is_disjoint()
        return policy

    def assert_policy_is_disjoint(self) -> None:
        train = set(self.train_cells)
        held_out = {self.validation_cell, self.test_cell}
        overlap = train & held_out
        if overlap or self.validation_cell == self.test_cell:
            raise ValueError(
                "Invalid split policy: training, cross-cell validation and sealed test "
                f"must be distinct. Overlap={sorted(overlap)}"
            )

    def assign_split(
        self,
        frame: pd.DataFrame,
        *,
        cell_column: str = "cell_line",
        editor_column: str = "editor_system",
    ) -> pd.Series:
        """Assign strict-track rows; non-PE2max or other cells are excluded."""

        self._require_columns(frame, [cell_column, editor_column])
        cells = frame[cell_column].map(canonicalize_cell_line)
        editors = frame[editor_column].map(canonicalize_editor)
        labels = pd.Series("excluded", index=frame.index, dtype="string")
        matched = editors.eq(self.editor)
        labels.loc[matched & cells.isin(self.train_cells)] = "train"
        labels.loc[matched & cells.eq(self.validation_cell)] = "cross_cell_validation"
        labels.loc[matched & cells.eq(self.test_cell)] = "sealed_test"
        return labels

    def assert_fit_frame(
        self,
        frame: pd.DataFrame,
        *,
        cell_column: str = "cell_line",
        editor_column: str = "editor_system",
        purpose: str = "model or preprocessing fit",
    ) -> None:
        """Fail if a fitted object could see HeLa, DLD1, or another editor."""

        self._require_columns(frame, [cell_column, editor_column])
        cells = set(frame[cell_column].map(canonicalize_cell_line).dropna().unique())
        forbidden = cells - set(self.train_cells)
        if forbidden:
            raise ValueError(
                f"Leakage blocked during {purpose}: fit data contain non-training cells "
                f"{sorted(forbidden)}; allowed={list(self.train_cells)}"
            )
        editors = set(frame[editor_column].map(canonicalize_editor).dropna().unique())
        if editors != {self.editor}:
            raise ValueError(
                f"Strict-track mismatch during {purpose}: editors={sorted(editors)}, "
                f"required={[self.editor]}"
            )

    def assert_group_column(self, frame: pd.DataFrame) -> None:
        self._require_columns(frame, [self.group_column])
        if frame[self.group_column].isna().any():
            raise ValueError(f"Missing group IDs in {self.group_column}; grouped CV is unsafe")

    @staticmethod
    def assert_source_rows_disjoint(
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        *,
        id_columns: Iterable[str] = ("source_dataset_id", "source_row_id"),
    ) -> None:
        columns = list(id_columns)
        for name, frame in (("train", train), ("validation", validation), ("test", test)):
            missing = [column for column in columns if column not in frame.columns]
            if missing:
                raise ValueError(f"{name} lacks lineage columns: {missing}")
        keys = {
            "train": set(map(tuple, train[columns].astype(str).to_numpy())),
            "validation": set(map(tuple, validation[columns].astype(str).to_numpy())),
            "test": set(map(tuple, test[columns].astype(str).to_numpy())),
        }
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = keys[left] & keys[right]
            if overlap:
                preview = sorted(overlap)[:5]
                raise ValueError(f"Source-row overlap between {left} and {right}: {preview}")

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
