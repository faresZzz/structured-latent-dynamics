"""Offline audit instrument: what gets displayed vs. what gets measured.

``spectral`` carries the display-only objects (EDMD baseline, operator
spectra for the fixed-operator regimes, fields of local spectra,
pseudospectra); ``jacobian`` carries the quantitative objects (total
Jacobian of the augmented state, QR-stabilized products, free-running
Lyapunov exponents, Lip(pi), drift curves). A displayed spectrum is never a
stability object -- only the Jacobian side supports quantitative claims,
and the pair is read together (REGLES sections 3, 5).

Doctrine: the audit runs offline, on CPU, in float64, on checkpoints --
never on MPS, never in float32, and never through the stochastic training
estimator of ``objectives.jac_reg``.
"""

from sld.audit.jacobian import (
    ExcludedAugmentedJacobian,
    augmented_jacobian_excluding_identity,
    augmented_sigma_max_normalized,
    conditional_jacobian,
    fr_drift_curve,
    fr_lyapunov_exponents,
    jacobian_products,
    lipschitz_pi,
    total_jacobian,
)
from sld.audit.invariance import (
    GaugeCheckResult,
    assert_diagonal_gauge_invariant,
    check_diagonal_gauge_invariance,
    check_shear_breaks_invariance,
    diagonal_gauges,
)
from sld.audit.rollout import VPTResult, vpt
from sld.audit.spectral import (
    edmd_fit,
    local_spectra_field,
    operator_norm,
    operator_spectrum,
    pseudospectrum_grid,
)

__all__ = [
    "ExcludedAugmentedJacobian",
    "GaugeCheckResult",
    "VPTResult",
    "assert_diagonal_gauge_invariant",
    "augmented_jacobian_excluding_identity",
    "augmented_sigma_max_normalized",
    "check_diagonal_gauge_invariance",
    "check_shear_breaks_invariance",
    "conditional_jacobian",
    "diagonal_gauges",
    "edmd_fit",
    "fr_drift_curve",
    "fr_lyapunov_exponents",
    "jacobian_products",
    "lipschitz_pi",
    "local_spectra_field",
    "operator_norm",
    "operator_spectrum",
    "pseudospectrum_grid",
    "total_jacobian",
    "vpt",
]
