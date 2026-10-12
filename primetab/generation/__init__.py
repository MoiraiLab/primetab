"""PRIDICT2-compatible candidate design without predictive models.

Public API: :class:`EditRequest`, :class:`GenerationConfig`,
:class:`GenerationResult`, :func:`generate`, :func:`generate_batch`, and
:func:`tfm_feature_frame`.
"""

from primetab.generation.api import (
    EditRequest,
    GenerationConfig,
    GenerationResult,
    generate,
    generate_batch,
    tfm_feature_frame,
)

__all__ = [
    "EditRequest",
    "GenerationConfig",
    "GenerationResult",
    "generate",
    "generate_batch",
    "tfm_feature_frame",
]
