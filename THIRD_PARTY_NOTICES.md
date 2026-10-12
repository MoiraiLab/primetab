# Third-Party Notices

PrimeTab depends on and/or adapts the following third-party components. Each is
governed by its own license. Before publishing, verify every item below.

## Adapted source code
| Component | Use in PrimeTab | Upstream | License |
|---|---|---|---|
| PRIDICT2 | pegRNA candidate-enumeration logic (`primetab/generation/`) | https://github.com/uzh-dqbm-cmi/PRIDICT2 | MIT (redistribution and modification permitted) |

The upstream copyright and full MIT license for the adapted generator are included
in `primetab/generation/NOTICE`.

## Runtime dependencies
| Package | Purpose | License |
|---|---|---|
| TabPFN (Prior Labs) | tabular foundation model backend | Apache-2.0 (verify) |
| PyTorch | TabPFN runtime | BSD-3-Clause |
| ViennaRNA | RNA minimum free energy features | ViennaRNA license (verify) |
| Biopython | nearest-neighbor melting temperature | Biopython License / BSD-like |
| NumPy, SciPy, scikit-learn, pandas, pyarrow | arrays, modeling, tables | BSD-3-Clause / Apache-2.0 |
| openpyxl, PyYAML, requests | IO | MIT / MIT / Apache-2.0 |
| xgboost (optional) | baseline model | Apache-2.0 |

## Data
Public experimental data (DeepPrime, PRIDICT2 supplementary tables) is **not**
redistributed in this repository. See `data/README.md` for sources and retrieval.
Model weights (TabPFN v3) are obtained from the official release; SHA-256 and
retrieval instructions are documented in `docs/method.md`.

> Open-source checklist: confirm ViennaRNA/TabPFN license terms and the data
> redistribution rights of the public datasets before relying on them in a release.
> PRIDICT2 (the only adapted source code) is MIT and already cleared.
