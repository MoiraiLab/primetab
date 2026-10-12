"""Composable candidate-generation and ranking pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from primetab.generation import (
    EditRequest,
    GenerationConfig,
    GenerationResult,
    generate_batch,
    tfm_feature_frame,
)
from .rankers import CandidateRanker


@dataclass(frozen=True)
class GeneratorRankerResult:
    generation: GenerationResult
    ranked_candidates: pd.DataFrame

    @property
    def successful_requests(self) -> int:
        return int(self.generation.requests["status"].eq("ok").sum())


class GeneratorRankerPipeline:
    """Generate legal pegRNAs, score them, and rank within each edit request."""

    def __init__(
        self,
        ranker: CandidateRanker,
        *,
        generation_config: GenerationConfig | None = None,
        score_column: str = "predicted_efficiency",
        higher_is_better: bool = True,
    ) -> None:
        self.ranker = ranker
        self.generation_config = generation_config or GenerationConfig()
        self.score_column = str(score_column).strip()
        if not self.score_column:
            raise ValueError("score_column must not be empty")
        self.higher_is_better = bool(higher_is_better)

    def run(self, requests: list[EditRequest]) -> GeneratorRankerResult:
        generation = generate_batch(requests, config=self.generation_config)
        candidates = generation.candidates.copy()
        if candidates.empty:
            return GeneratorRankerResult(
                generation=generation,
                ranked_candidates=self._empty_ranked_candidates(candidates),
            )
        if candidates["design_id"].isna().any() or candidates["design_id"].duplicated().any():
            raise ValueError("generated candidates must have unique non-null design_id values")
        features = tfm_feature_frame(generation)
        if features.index.isna().any() or features.index.duplicated().any():
            raise ValueError("generated feature rows must have unique non-null design_id values")
        candidate_ids = pd.Index(candidates["design_id"], dtype="object")
        if set(candidate_ids) != set(features.index):
            raise ValueError("generated candidate and feature design_id sets differ")

        raw_scores = self.ranker.score(candidates.copy(), features.copy())
        scores = self._normalize_scores(raw_scores, features.index)
        score_frame = scores.rename(self.score_column).rename_axis("design_id").reset_index()
        ranked = candidates.merge(score_frame, on="design_id", how="left", validate="one_to_one")
        ranked = ranked.sort_values(
            ["request_id", self.score_column, "design_id"],
            ascending=[True, not self.higher_is_better, True],
            kind="mergesort",
        ).reset_index(drop=True)
        ranked["rank"] = ranked.groupby("request_id", sort=False).cumcount() + 1
        ranked["candidate_count"] = ranked.groupby("request_id", sort=False)[
            "design_id"
        ].transform("size")
        return GeneratorRankerResult(generation=generation, ranked_candidates=ranked)

    def metadata(self) -> dict[str, Any]:
        ranker_metadata = getattr(self.ranker, "metadata", None)
        return {
            "architecture": "generator_ranker",
            "generator": "workspace_local_pridict2_model_free_snapshot",
            "score_column": self.score_column,
            "higher_is_better": self.higher_is_better,
            "ranker": dict(ranker_metadata()) if callable(ranker_metadata) else {},
        }

    def _normalize_scores(self, raw_scores: Any, design_ids: pd.Index) -> pd.Series:
        if isinstance(raw_scores, pd.Series):
            if raw_scores.index.has_duplicates:
                raise ValueError("ranker returned duplicate design_id values")
            missing = design_ids.difference(raw_scores.index)
            extra = raw_scores.index.difference(design_ids)
            if len(missing) or len(extra):
                raise ValueError(
                    f"ranker design_id mismatch: missing={len(missing)}, extra={len(extra)}"
                )
            scores = raw_scores.reindex(design_ids)
        else:
            values = np.asarray(raw_scores).reshape(-1)
            if len(values) != len(design_ids):
                raise ValueError(
                    f"ranker returned {len(values)} scores for {len(design_ids)} candidates"
                )
            scores = pd.Series(values, index=design_ids)
        values = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise FloatingPointError("ranker returned missing or non-finite scores")
        return pd.Series(values, index=design_ids)

    def _empty_ranked_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        ranked = candidates.copy()
        ranked[self.score_column] = pd.Series(dtype="float64")
        ranked["rank"] = pd.Series(dtype="int64")
        ranked["candidate_count"] = pd.Series(dtype="int64")
        return ranked
