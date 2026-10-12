"""Feature-perturbation diagnostics."""

from .jacobian import JacobianReport, compute_jacobian_sensitivity

__all__ = ["JacobianReport", "compute_jacobian_sensitivity"]
