"""CSV contract for independently configured generator tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from primetab.generation import EditRequest, GenerationConfig

REQUIRED_TASK_COLUMNS = ("editseq",)
OPTIONAL_TASK_COLUMNS = (
    "pbs_lengths",
    "rt_overhang_lengths",
    "max_edit_distance",
    "max_edit_length",
    "include_nicking",
    "include_ngs_primers",
)
TASK_COLUMNS = (*REQUIRED_TASK_COLUMNS, *OPTIONAL_TASK_COLUMNS)


@dataclass(frozen=True)
class GeneratorTask:
    """One CSV row resolved to a generator request and its own configuration."""

    task_row: int
    request: EditRequest
    config: GenerationConfig


def load_generator_tasks(path: Path) -> list[GeneratorTask]:
    """Read one generator task per CSV row.

    Integer lists accept either an inclusive range (``7-15``) or a semicolon
    list (``7;9;11``). Empty optional cells use ``GenerationConfig`` defaults.
    """

    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"generator task input must be CSV, got: {path.suffix}")
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    missing = [column for column in REQUIRED_TASK_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"generator task CSV is missing required columns: {missing}")
    unexpected = [column for column in frame if column not in TASK_COLUMNS]
    if unexpected:
        raise ValueError(f"generator task CSV has unsupported columns: {unexpected}")
    if frame.empty:
        raise ValueError("generator task CSV is empty")

    defaults = GenerationConfig()
    tasks: list[GeneratorTask] = []
    for zero_based_row, row in frame.iterrows():
        task_row = int(zero_based_row) + 1
        editseq = str(row["editseq"]).strip()
        if not editseq:
            raise ValueError(f"task row {task_row}: editseq must not be empty")
        config = GenerationConfig(
            pbs_lengths=_parse_int_tuple(
                row.get("pbs_lengths", ""),
                default=defaults.pbs_lengths,
                field="pbs_lengths",
                task_row=task_row,
            ),
            rt_overhang_lengths=_parse_int_tuple(
                row.get("rt_overhang_lengths", ""),
                default=defaults.rt_overhang_lengths,
                field="rt_overhang_lengths",
                task_row=task_row,
            ),
            max_edit_distance=_parse_positive_int(
                row.get("max_edit_distance", ""),
                default=defaults.max_edit_distance,
                field="max_edit_distance",
                task_row=task_row,
            ),
            max_edit_length=_parse_positive_int(
                row.get("max_edit_length", ""),
                default=defaults.max_edit_length,
                field="max_edit_length",
                task_row=task_row,
            ),
            include_nicking=_parse_bool(
                row.get("include_nicking", ""),
                default=defaults.include_nicking,
                field="include_nicking",
                task_row=task_row,
            ),
            include_ngs_primers=_parse_bool(
                row.get("include_ngs_primers", ""),
                default=defaults.include_ngs_primers,
                field="include_ngs_primers",
                task_row=task_row,
            ),
        )
        tasks.append(
            GeneratorTask(
                task_row=task_row,
                request=EditRequest(f"task_{task_row:06d}", editseq),
                config=config,
            )
        )
    return tasks


def task_parameter_frame(tasks: list[GeneratorTask]) -> pd.DataFrame:
    """Return the normalized task parameters written beside generated tables."""

    rows: list[dict[str, object]] = []
    for task in tasks:
        config = asdict(task.config)
        rows.append(
            {
                "task_row": task.task_row,
                "editseq": task.request.editseq,
                "pbs_lengths": ";".join(str(value) for value in config["pbs_lengths"]),
                "rt_overhang_lengths": ";".join(
                    str(value) for value in config["rt_overhang_lengths"]
                ),
                "max_edit_distance": config["max_edit_distance"],
                "max_edit_length": config["max_edit_length"],
                "include_nicking": config["include_nicking"],
                "include_ngs_primers": config["include_ngs_primers"],
            }
        )
    return pd.DataFrame(rows, columns=["task_row", *TASK_COLUMNS])


def _parse_int_tuple(
    value: object,
    *,
    default: tuple[int, ...],
    field: str,
    task_row: int,
) -> tuple[int, ...]:
    text = str(value).strip()
    if not text:
        return default
    try:
        if ";" in text:
            values = tuple(int(part.strip()) for part in text.split(";") if part.strip())
        elif "-" in text:
            parts = [part.strip() for part in text.split("-")]
            if len(parts) != 2:
                raise ValueError
            start, stop = (int(part) for part in parts)
            if start > stop:
                raise ValueError
            values = tuple(range(start, stop + 1))
        else:
            values = (int(text),)
    except ValueError as exc:
        raise ValueError(
            f"task row {task_row}: {field} must be an integer, an inclusive range "
            "such as 7-15, or a semicolon list such as 7;9;11"
        ) from exc
    if not values or min(values) < 1:
        raise ValueError(f"task row {task_row}: {field} must contain positive integers")
    return values


def _parse_positive_int(
    value: object,
    *,
    default: int,
    field: str,
    task_row: int,
) -> int:
    text = str(value).strip()
    if not text:
        return default
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(f"task row {task_row}: {field} must be an integer") from exc
    if parsed < 1:
        raise ValueError(f"task row {task_row}: {field} must be positive")
    return parsed


def _parse_bool(
    value: object,
    *,
    default: bool,
    field: str,
    task_row: int,
) -> bool:
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"task row {task_row}: {field} must be true or false")
