"""Fig. 1 -- free-running latent drift vs. horizon (v0, D1 pendulum) --
RETIRED pending phase D.

This figure used to decode the rollout prediction into true-state
coordinates via a linear readout ``W`` fit by OLS INDEPENDENTLY PER RUN
(val split, encoder embedding -> dataset ``z_true``). That per-run fit is
a free parameter PER ARM: a bad encoder can be rescued by a
better-fitting readout, so the resulting cross-arm comparison is not at
arms' length -- a defect of comparison METHOD, not a numerical bug (it is
not what B3's F1 gauge-invariance fix addressed; that fix concerned how a
single run's OWN readout is measured, not whether different arms may use
different readouts at all).

The replacement is a SHARED, FROZEN encoder/readout built once in phase D
(``_private/plans/MASTER-plan.md``); it does not exist yet, and this
module does not anticipate it (phase B does not anticipate phase D).
``make_fig1`` therefore raises a clear error naming phase D instead of
plotting on the superseded method -- a wrong figure is worse than no
figure.

``_aggregate_curves`` (seed-mean +/-1 SE) and ``_expected_grid``
(pre-registered-seed completeness) are kept: both are generic to any
per-cell, per-seed aggregation and will be reused unchanged once phase D's
frozen readout replaces the retired per-run OLS fit. Neither is currently
called by ``make_fig1``; both are exercised directly by their own tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from sld.viz.style import CONSTRAINT_LINESTYLES, FAMILY_COLORS

# pre-registered 5-seed law (PROTOCOL.md section 3; pendulum_core.yaml's
# `seeds: [0, 1, 2, 3, 4]`) -- the reference set a "complete" cell must
# cover in strict/final-grid mode. Not exposed as a knob (ponytail): if the
# protocol's seed count ever changes, this constant changes with it.
_REQUIRED_SEEDS: frozenset[int] = frozenset(range(5))


def _aggregate_curves(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and +/-1 SE over the seed axis (``curves``, ``(n_seeds, H)``).

    Kept standalone (module docstring: not currently called by
    ``make_fig1``, which is retired pending phase D) because phase D's
    frozen-readout figure will still need this identical seed-aggregation
    arithmetic; the numeric-lock test (M6/m12) verifies this function
    directly rather than through a render. ``se`` is
    ``std(ddof=1) / sqrt(n)``; ``0`` when ``n == 1`` (ddof=1 needs
    ``n >= 2``) -- the honest "no spread observed" value, never a silent
    NaN. Mean, never the median (REGLES section 4: "+/-1 SE" bands -- SE
    is defined around the mean, not the median).
    """
    n = curves.shape[0]
    mean_curve = curves.mean(axis=0)
    se = curves.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean_curve)
    return mean_curve, se


def _expected_grid(exp_yaml: str) -> tuple[list[tuple[str, str, str]], frozenset[int]]:
    """The FULL ``(regime, family, constraint)`` grid + seeds ``exp_yaml`` promises.

    Derived, not hardcoded (M4 + m7, round 3): regimes come from
    ``exp_yaml``'s ``datasets`` block; cells and seeds come from the yaml's
    OWN ``arms``/``seeds`` declarations (the factorial schema) when
    present -- so ``smoke.yaml``'s declared 2 arms x 1 seed are what strict
    mode requires there, not an invented 8 cells x 5 seeds. Fallback when
    the yaml declares no ``arms`` (viz-test fixtures): the style modules'
    ``FAMILY_COLORS`` x ``CONSTRAINT_LINESTYLES`` with the pre-registered
    seeds 0-4 -- declared, not silent. For ``pendulum_core.yaml`` (which
    declares all 8 arms and seeds 0-4) both paths agree: 16 cells, 5 seeds.

    Kept standalone (module docstring): not currently called by
    ``make_fig1``, which is retired pending phase D; exercised directly by
    its own test.
    """
    with open(exp_yaml) as f:
        exp_config = yaml.safe_load(f)
    regimes = sorted(exp_config.get("datasets", {}))
    arms = exp_config.get("arms")
    if arms:
        cells = [
            (regime, arm["family"], arm["constraint"])
            for regime in regimes
            for arm in arms
        ]
        seeds = frozenset(int(s) for s in exp_config.get("seeds", sorted(_REQUIRED_SEEDS)))
    else:
        cells = [
            (regime, family, constraint)
            for regime in regimes
            for family in sorted(FAMILY_COLORS)
            for constraint in sorted(CONSTRAINT_LINESTYLES)
        ]
        seeds = _REQUIRED_SEEDS
    return cells, seeds


def make_fig1(
    results_root: str = "results",
    exp_name: str = "pendulum_core",
    exp_yaml: str = "src/sld/experiments/configs/pendulum_core.yaml",
    out_path: str = "assets/fig1_rollout.png",
    horizon: int = 50,
    require_full_cells: bool = True,
) -> Path:
    """Retired pending phase D (module docstring): refuses to render rather
    than plot on the superseded per-run OLS readout. Arguments are kept for
    interface stability; none are consulted before raising.
    """
    raise NotImplementedError(
        "fig1: retired pending phase D's shared, frozen encoder/readout. "
        "The per-run OLS readout this figure used to fit independently for "
        "each run is a free parameter per arm, not an at-arms-length "
        "cross-arm comparison method -- see _private/plans/MASTER-plan.md, "
        "phase D. A wrong figure is worse than no figure."
    )


def main() -> None:
    make_fig1()


if __name__ == "__main__":
    main()
