"""Fig. 2 -- spectral display/screening across families (v0, D1 pendulum).

STATUS (jalon B2, demotion explicite) -- SECONDARY -- not primary thesis.
Fig. 2 is a secondary regression/screening display, tied to the retired
mu_J line's diagnostic tooling; it blocks NO gate of phases C-E and is
never a stability certificate (see the doctrine paragraph below, which
already says so). Remains available and tested (tests/viz/test_viz.py).

REGLES sections 2/3/5 discipline, restated at the point of use: a displayed
spectrum is a SCREENING tool, NEVER a stability certificate. Only
``koopman_ssm`` in its ``constant``/``bilinear`` regimes exposes a genuine
fixed linear operator (Koopman regimes (i)/(ii)) -- its eigenvalues are
plotted as POINTS via ``sld.audit.spectral.operator_spectrum``. Every other
case (``mlp``, ``transformer``, ``min_ssm``, and ``koopman_ssm`` itself in
its ``selective`` regime -- an LPV ablation, explicitly NOT a Koopman
operator, REGLES section 3) has no single fixed operator;
``sld.audit.spectral.local_spectra_field`` instead samples a FIELD of local
(state-dependent) spectra along a validation trajectory, plotted as a
semi-transparent cloud.

Every panel that renders shows its provenance (family, constraint, seed) in
its title, and an ``on``-run missing (falling back to ``constraint='off'``)
is declared IN the title, never a silent substitution (Wave 5 review
finding F12). In final-grid mode (``require_full_cells=True``, the
default, same flag as ``fig1_rollout.make_fig1``) a family with no cell run
at all, or whose spectrum computation raises, ABORTS the whole render with
a clear error naming the family -- it never ships a PNG with a blank panel
next to real ones. A checkpoint/dataset mismatch
(``sld.experiments.loading.check_run_matches_dataset``, finding F6) is
never caught, in either mode.

M5 (Wave 5 review round 2): the ``on``-run-missing fallback above is a
LENIENT-mode-only allowance. In strict/final-grid mode
(``require_full_cells=True``) a family whose ``constraint='on'`` run is
missing -- even when its ``constraint='off'`` run IS present -- ABORTS the
render naming the family, exactly like the "no run at all" case; it never
silently substitutes ``'off'`` for a missing ``'on'``, which would
contradict the very completeness guarantee strict mode promises.
"""

from __future__ import annotations

import copy
import math
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from sld.audit.spectral import local_spectra_field, operator_spectrum
from sld.experiments.loading import (
    build_dataset_from_expconfig,
    check_run_matches_dataset,
    load_run,
    read_index,
)
from sld.systems.damped_pendulum import DampedPendulum
from sld.viz.style import FAMILY_COLORS, apply_style, save_fig

# ponytail: fixed constants, not exposed as make_fig2 kwargs -- v0 only
# needs "enough probes to see the cloud", not a tunable axis
_N_PROBES = 8
_SAMPLE_EVERY = 5


def _select_cell_run(rows: list[dict], family: str) -> tuple[dict | None, str | None]:
    """The (regime='ou', constraint, seed=0) cell run for ``family``:
    ``constraint='on'`` preferred, ``'off'`` as a logged fallback."""
    for constraint in ("on", "off"):
        for row in rows:
            if (
                row["family"] == family
                and row["regime"] == "ou"
                and row["structure"] == constraint  # index.csv column: "structure" (phase B4, legacy "constraint" aliased by read_index)
                and row["seed"] == 0
            ):
                return row, constraint
    return None, None


def _family_spectrum(run: dict, dataset: dict) -> tuple[np.ndarray, str]:
    """Eigenvalues for one run: ``"points"`` for a fixed Koopman operator
    (regimes (i)/(ii)), ``"field"`` for everything else (autograd fallback
    included). Audit doctrine (REGLES section 5): CPU float64 only, so the
    loaded float32 modules are cast on throwaway deep copies."""
    encoder = copy.deepcopy(run["encoder"]).double().eval()
    predictor = copy.deepcopy(run["predictor"]).double().eval()

    x_val = torch.as_tensor(dataset["val"]["x"], dtype=torch.float64).transpose(0, 1)
    u_val = torch.as_tensor(dataset["val"]["u"], dtype=torch.float64).transpose(0, 1)
    n_probes = min(_N_PROBES, x_val.shape[1])

    lifted = hasattr(predictor, "lift")
    lift_fn = predictor.lift if lifted else (lambda z: z)
    with torch.no_grad():
        z0 = encoder(x_val[0, :n_probes])
        h0 = lift_fn(z0)

    # only the FIXED-operator koopman regimes (i)/(ii) get a single
    # displayed spectrum; "selective" is LPV (REGLES section 3), routed
    # through the generic field path like every other family
    if getattr(predictor, "regime", None) in ("constant", "bilinear"):
        u_const = torch.zeros(predictor.d_u, dtype=torch.float64)
        return operator_spectrum(predictor, h0, u_const), "points"

    u_seq = u_val[:, :n_probes, :]
    _, eigvals_field = local_spectra_field(predictor, h0, u_seq, sample_every=_SAMPLE_EVERY)
    return eigvals_field.reshape(-1), "field"


def _blank_panel(ax) -> None:
    """Shared "not available" rendering for a family panel (lenient mode only)."""
    ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_family_panel(
    ax, family: str, rows: list[dict], exp_yaml: str, require_full_cells: bool
) -> None:
    row, constraint_used = _select_cell_run(rows, family)
    if row is None:
        if require_full_cells:
            raise ValueError(
                f"fig2 strict mode: family {family!r} has no regime='ou'/seed=0 run "
                "(neither constraint='on' nor 'off' available) -- refusing to render "
                "a blank panel; pass require_full_cells=False to override explicitly"
            )
        ax.set_title(f"{family}\nno ou/seed=0 run available")
        _blank_panel(ax)
        return

    # M5: the on->off fallback below is LENIENT-mode only -- strict mode
    # must refuse it (a missing 'on' run with 'off' present used to survive
    # require_full_cells=True, contradicting the completeness guarantee)
    if constraint_used == "off" and require_full_cells:
        raise ValueError(
            f"fig2 strict mode: family {family!r} has no regime='ou'/constraint='on'/seed=0 "
            "run (only constraint='off' is available) -- refusing to silently substitute "
            "'off' for a missing 'on' run; pass require_full_cells=False to override "
            "explicitly"
        )

    # provenance shown on every panel that reaches here: family, constraint, seed
    title = f"{family}\nconstraint={constraint_used}, seed={row['seed']}"
    if constraint_used == "off":
        title += " [off: no on-run]"  # explicit in the panel, not a silent substitution
        warnings.warn(
            f"fig2: family {family!r} has no regime='ou'/constraint='on'/seed=0 "
            "run; falling back to constraint='off' (declared in the panel title)"
        )

    try:
        run = load_run(row["run_dir"])
        dataset = build_dataset_from_expconfig(exp_yaml, "ou")
    except Exception as e:
        if require_full_cells:
            raise ValueError(
                f"fig2 strict mode: family {family!r} run/dataset failed to load "
                f"(constraint={constraint_used}, seed={row['seed']}): {e}"
            ) from e
        warnings.warn(f"fig2: family {family!r} spectrum unavailable: {e}")
        ax.set_title(title + " (compute failed)")
        _blank_panel(ax)
        return

    # F6: a checkpoint trained on a dataset spec the CURRENT yaml no longer
    # produces is never silently evaluated -- always aborts, in both modes
    check_run_matches_dataset(row["run_dir"], dataset["meta"])

    try:
        eigvals, kind = _family_spectrum(run, dataset)
    except Exception as e:
        if require_full_cells:
            raise ValueError(
                f"fig2 strict mode: family {family!r} spectrum computation failed "
                f"(constraint={constraint_used}, seed={row['seed']}): {e}"
            ) from e
        warnings.warn(f"fig2: family {family!r} spectrum unavailable: {e}")
        ax.set_title(title + " (compute failed)")
        _blank_panel(ax)
        return

    ax.set_title(title)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="0.6", linewidth=1, linestyle=":", label="unit circle")
    if kind == "points":
        ax.scatter(eigvals.real, eigvals.imag, color=FAMILY_COLORS[family], s=35, label="operator spectrum")
    else:
        ax.scatter(
            eigvals.real, eigvals.imag, color=FAMILY_COLORS[family], s=8, alpha=0.25, label="local spectra field"
        )

    dt = dataset["meta"]["dt"]
    analytic = np.exp(dt * DampedPendulum().analytic_eigenvalues())
    ax.scatter(analytic.real, analytic.imag, color="black", marker="*", s=140, label="analytic (linearized)", zorder=5)

    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=7, loc="best")


def make_fig2(
    results_root: str = "results",
    exp_name: str = "pendulum_core",
    exp_yaml: str = "src/sld/experiments/configs/pendulum_core.yaml",
    out_path: str = "assets/fig2_spectrum.png",
    require_full_cells: bool = True,
) -> Path:
    """Regenerate Fig. 2 (2x2, one panel per predictor family) from
    ``<results_root>/<exp_name>`` and save to ``out_path``.

    Per family, the cell run is ``regime='ou'``, ``constraint='on'``,
    ``seed=0``. In LENIENT mode this falls back to ``constraint='off'``
    with a logged warning AND an explicit title annotation if ``'on'`` is
    absent; in STRICT mode (``require_full_cells=True``, the default) that
    fallback never happens -- a missing ``'on'`` run aborts the render
    naming the family, even when ``'off'`` is present (M5).

    Args:
        require_full_cells: when ``True`` (default, "final grid" mode), a
            family with no cell run, or whose spectrum computation raises,
            aborts the whole figure with a clear error naming the family
            (finding F12) instead of rendering a blank panel. Pass
            ``False`` explicitly to relax this for exploratory/partial
            grids -- the panel then reads "not available"/"compute
            failed" instead of crashing.
    """
    apply_style()
    try:
        rows = read_index(results_root, exp_name)
    except FileNotFoundError as e:
        print(f"fig2: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    complete = [r for r in rows if r["status"] == "complete"]
    if not complete:
        print(
            f"fig2: no complete runs in {results_root}/{exp_name}/index.csv -- nothing to plot",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # The grid is SIZED FROM the family list, never fixed at 2x2. It used to
    # be, and `zip(axes.ravel(), families)` truncates silently: the day
    # FAMILY_COLORS grew past 4 entries the extra families would simply have
    # vanished from the figure with every gate still green. Every family gets
    # a panel or this raises; the leftover cells of the rectangle are turned
    # off so they render blank rather than as empty axes.
    families = sorted(FAMILY_COLORS)
    ncols = math.ceil(math.sqrt(len(families)))
    nrows = math.ceil(len(families) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows), squeeze=False)
    panels = axes.ravel()
    if len(panels) < len(families):  # unreachable by the arithmetic above
        raise AssertionError(f"fig2: {len(panels)} panels for {len(families)} families")
    for ax, family in zip(panels, families):
        _plot_family_panel(ax, family, complete, exp_yaml, require_full_cells)
    for ax in panels[len(families) :]:
        ax.axis("off")

    fig.suptitle(
        "Spectral DISPLAY/screening across families -- never a stability certificate "
        "(v0, D1 pendulum)",
        fontsize=11,
    )
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    return Path(out_path)


def main() -> None:
    make_fig2()


if __name__ == "__main__":
    main()
