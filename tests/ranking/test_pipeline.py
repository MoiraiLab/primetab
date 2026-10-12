from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from primetab.features.features import F1_SEQUENCE_V1_COLUMNS
from primetab.ranking import EditRequest, GeneratorRankerPipeline
from primetab.ranking.cli import main as cli_main
from primetab.ranking.feature_encoding import (
    CATEGORICAL_DNA_WIDTHS,
    F1FeatureEncoder,
)
from primetab.ranking.rankers import SharedContextTFMRanker
from primetab.ranking.task_csv import load_generator_tasks

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PATH = REPO_ROOT / "projects/prime-editing-tfm/examples/pridict2_batch_example.csv"
TASK_EXAMPLE_PATH = (
    REPO_ROOT / "projects/prime-editing-tfm/examples/pe2_generator_tasks_example.csv"
)


class PBSLengthRanker:
    def score(self, candidates: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        del candidates
        return pd.to_numeric(features["pbs_length"]).rename("score")

    def metadata(self) -> dict[str, str]:
        return {"ranker": "pbs_length_test_double"}


class NonFiniteRanker(PBSLengthRanker):
    def score(self, candidates: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        result = super().score(candidates, features).astype(float)
        result.iloc[0] = np.nan
        return result


class DummyEstimator:
    def fit(self, features: np.ndarray, labels: np.ndarray) -> DummyEstimator:
        self.fit_shape = features.shape
        self.label_rows = len(labels)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return features[:, 0]


def _example_request(name: str = "example") -> EditRequest:
    row = pd.read_csv(EXAMPLE_PATH).iloc[0]
    return EditRequest(name, str(row["editseq"]))


def _synthetic_features(rows: int) -> pd.DataFrame:
    values = {
        column: np.arange(1, rows + 1, dtype=np.float32)
        for column in F1_SEQUENCE_V1_COLUMNS
    }
    frame = pd.DataFrame(values)
    for column, width in CATEGORICAL_DNA_WIDTHS.items():
        frame[column] = "AGG"[:width]
    return frame


def test_generate_and_rank_candidates_within_request() -> None:
    result = GeneratorRankerPipeline(PBSLengthRanker()).run([_example_request()])

    ranked = result.ranked_candidates
    assert not ranked.empty
    assert len(ranked) == len(result.generation.features)
    assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))
    assert ranked["predicted_efficiency"].is_monotonic_decreasing
    assert ranked.iloc[0]["pbs_length"] == ranked["pbs_length"].max()
    assert result.successful_requests == 1


def test_invalid_request_does_not_abort_valid_peer() -> None:
    result = GeneratorRankerPipeline(PBSLengthRanker()).run(
        [_example_request("valid"), EditRequest("invalid", "A(A/G)T")]
    )

    status = result.generation.requests.set_index("sequence_name")["status"]
    assert status["valid"] == "ok"
    assert status["invalid"] == "error"
    assert set(result.ranked_candidates["sequence_name"]) == {"valid"}
    assert set(result.generation.errors["sequence_name"]) == {"invalid"}


def test_pipeline_rejects_non_finite_scores() -> None:
    with pytest.raises(FloatingPointError, match="non-finite"):
        GeneratorRankerPipeline(NonFiniteRanker()).run([_example_request()])


def test_shared_context_ranker_encodes_fits_and_scores() -> None:
    context = _synthetic_features(4)
    labels = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    estimator = DummyEstimator()
    ranker = SharedContextTFMRanker("dummy", estimator=estimator).fit(context, labels)
    query = context.iloc[:2].copy()
    query.index = pd.Index(["design-a", "design-b"], name="design_id")

    scores = ranker.score(pd.DataFrame(index=query.index), query)

    assert estimator.fit_shape == (4, 107)
    assert estimator.label_rows == 4
    assert scores.index.tolist() == ["design-a", "design-b"]
    assert ranker.metadata()["feature_encoding"]["raw_width"] == 73


def test_encoder_rejects_missing_f1_columns() -> None:
    with pytest.raises(ValueError, match="missing F1 columns"):
        F1FeatureEncoder.fit(pd.DataFrame({"pbs_length": [10]}))


def test_task_csv_resolves_per_row_generation_parameters(tmp_path: Path) -> None:
    editseq = str(pd.read_csv(EXAMPLE_PATH).iloc[0]["editseq"])
    path = tmp_path / "tasks.csv"
    pd.DataFrame(
        [
            {
                "editseq": editseq,
                "pbs_lengths": "7-9",
                "rt_overhang_lengths": "3;5;7",
                "max_edit_distance": "20",
                "max_edit_length": "30",
                "include_nicking": "false",
                "include_ngs_primers": "true",
            }
        ]
    ).to_csv(path, index=False)

    tasks = load_generator_tasks(path)

    assert len(tasks) == 1
    assert tasks[0].request.sequence_name == "task_000001"
    assert tasks[0].config.pbs_lengths == (7, 8, 9)
    assert tasks[0].config.rt_overhang_lengths == (3, 5, 7)
    assert tasks[0].config.max_edit_distance == 20
    assert tasks[0].config.max_edit_length == 30
    assert tasks[0].config.include_nicking is False
    assert tasks[0].config.include_ngs_primers is True


def test_generate_only_cli_runs_task_csv(tmp_path: Path) -> None:
    output_dir = tmp_path / "generated"

    exit_code = cli_main(
        [
            "--input",
            str(TASK_EXAMPLE_PATH),
            "--generate-only",
            "--format",
            "csv",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    requests = pd.read_csv(output_dir / "requests.csv")
    candidates = pd.read_csv(output_dir / "candidates.csv")
    primers = pd.read_csv(output_dir / "primers.csv")
    parameters = pd.read_csv(output_dir / "task_parameters.csv")
    assert requests[["task_row", "sequence_name", "status"]].iloc[0].tolist() == [
        1,
        "task_000001",
        "ok",
    ]
    assert not candidates.empty
    assert set(candidates["editor_system"]) == {"PE2-NGG"}
    assert candidates["pbs_length"].between(7, 15).all()
    assert len(primers) == 1
    assert parameters.iloc[0]["include_nicking"] in {False, "False"}
