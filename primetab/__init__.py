"""PrimeTab: cross-domain, in-context-learning driven interpretable pegRNA design.

PrimeTab represents prime-editing design as tabular features, retrieves reference
samples per candidate from historical experiments, and scores editing efficiency
with a tabular foundation model (TabPFN) under in-context learning. Model weights
stay fixed; newly added experimental data only enters the context, enabling
training-free adaptation to new cells and datasets.

Subpackages
-----------
- ``generation``   : model-free PE2-NGG pegRNA candidate generation.
- ``features``     : unified structured feature engineering (73 -> 107 channels).
- ``backends``     : tabular foundation model backends (TabPFN, etc.).
- ``retrieval``    : per-candidate reference retrieval and transfer-learning context.
- ``ranking``      : one-click generator -> ranker pipeline.
- ``inference``    : TabPFN inference helpers.
- ``benchmarks``   : cross-cell / cross-domain evaluation.
- ``diagnostics``  : feature-perturbation interpretability.
"""

__version__ = "0.1.0"
