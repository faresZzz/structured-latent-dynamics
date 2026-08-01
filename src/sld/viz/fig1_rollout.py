"""Fig. 1 -- free-running latent drift vs. horizon (v0, D1 pendulum).

Regenerates from ``results/`` only (REGLES section 2: ``viz/`` never
retrains). For each completed run, the drift curve
(``sld.audit.jacobian.fr_drift_curve``) is measured in TRUE-STATE
coordinates: a per-run linear readout ``W`` is fit by ordinary least
squares (val split, encoder embedding -> dataset ``z_true``, with a bias
term) and applied to BOTH the rollout prediction and the encoder's own
target embedding before the drift is computed.

This closes the F1 blocker from the Wave 5 adversarial review: measuring
drift in a run's raw encoder coordinates is NOT gauge invariant -- a
``S=diag(100,1)`` relabeling of the latent axes can reverse both the
curve's height and its ON/OFF ordering with no physical change (the
reviewer's exact reproduction is now
``tests/viz/test_viz.py::test_readout_drift_invariant_to_linear_reparam``,
which must pass for this fix to be considered closed). Invariance
argument: under any invertible LINEAR reparameterization ``z' = S z``,
OLS fit on the transformed regressors reproduces EXACTLY the same fitted
state estimates -- it is an exact change of basis (``W' = S^-1``-composed
for the linear block, bias unchanged), not an approximation, whenever the
design matrix has full column rank. So the readout-composed drift curve
is identical whether measured before or after ``S``. This exactness holds
ONLY for invertible linear reparameterizations; a nonlinear
reparameterization is only APPROXIMATELY absorbed by a linear readout --
a declared residual limit, not a hidden one. The readout's R^2 (val-split
fit quality) is reported alongside each curve: a poor readout means a
weak coordinate change, and the drift reading should be trusted less.

B1 (Wave 5 review rounds 2-4): the exactness argument above holds ONLY
when the readout's design matrix ``[z_enc | 1]`` has full column rank --
a rank-deficient design (e.g. a collapsed latent coordinate ``z2 = z1``,
a LEGAL state since latent collapse is itself an object of study in this
repo) can still reach R^2 = 1 while its Moore-Penrose solution is NOT
equivariant outside the design's span. Round 3 established (two
independent adversarial probes) that an ELIGIBILITY verdict catching
NEAR-collapse cannot itself be gauge-invariant: exact rank is the only
GL(d)-invariant degeneracy notion, so any continuous criterion can be
flipped by an adversarial invertible gauge -- round 3's own probe
(``S=diag(1e9,1)``) was one such flip, taken at the time as an
unavoidable, declared cost.

Round 4 found that flip was avoidable, not fundamental. The round-3
criterion thresholded the RAW design's cond, where the intercept column
stays fixed at 1 while a diagonal gauge -- ``S=diag(1e9,1)``, or even a
UNIFORM per-coordinate rescale like ``2**20`` (exactly representable,
bit-exact round-trip through float32, ``cond(S)=1``, no unit singled
out) -- rescales only the z columns. That inflates the raw cond from ~1
to ~1e6 purely because of the UNIT mismatch between the (fixed-scale)
intercept column and the (rescaled) z columns, not because of any new
degeneracy: neither gauge destroys anything, both only relabel. The fix
(``_fit_readout``): measure cond on the COLUMN-EQUILIBRATED design
instead -- each column of ``[z_enc | 1]`` divided by its own RMS, which
is exactly invariant under any diagonal rescaling of the z columns (a
zero-RMS column is rank-deficient by construction and short-circuits
straight to "indeterminate" rather than dividing by zero). Storage noise
at float32 is relative PER COORDINATE (~1.19e-7 of each coordinate's own
magnitude), so it is the equilibrated design's conditioning -- not the
raw one's -- that actually governs how that noise propagates into the
OLS solution.

The resulting verdict is STABLE under the entire diagonal-rescaling
class (any per-coordinate unit change: ``diag(1e9,1)``, uniform
``2**20``, or any other ``diag(s_1, ..., s_d)``) -- strictly stronger
than round 3's declaration. It still catches genuine near-collinearity
between latent coordinates (e.g. a near-collapse ``z2 ~ z1`` up to tiny
noise: full column rank, so not exact rank deficiency, but nearly
duplicate once equilibrated) -- the equilibrated cond still blows up.
The residual limitation is honestly QUALIFIED, not silently hidden: this
invariance holds for per-coordinate unit changes (diagonal gauges) ONLY
-- a general shear (any non-diagonal invertible gauge) can still create
or destroy near-collinearity between coordinates, and no continuous
criterion can be invariant to that (exact rank remains the only
GL(d)-invariant degeneracy notion -- round 3's impossibility result
still holds, just for a narrower gauge class than round 3 assumed).
Operationally: this is a componentwise sensitivity measure of the OLS
SOLUTION's exposure to per-coordinate relative float32 storage noise,
NOT a claim that any bit-representable rescaling destroys information --
``2**20`` and ``diag(1e9,1)`` round-trip float32 exactly and destroy
nothing; round 3's raw-cond criterion was simply the wrong quantity to
threshold on. An indeterminate run is never plotted silently: in strict
mode (``require_full_cells=True``) rendering FAILS naming the run and
the measured rank/cond_equilibrated/sensitivity (after the per-run CSV
is written -- M5); in lenient mode the run is excluded with an explicit
warning, and its classification still lands in the per-run CSV artifact
(see M7/M9 below). A run whose readout fit succeeds but whose DRIFT
computation fails downstream (e.g. a decoded target ~0 trips
``fr_drift_curve``'s own zero-target guard) gets a named
``drift-failed: <reason>`` CSV status instead of vanishing before any
record of its (perfectly good) readout verdict is written (M8).

In final-grid mode (``require_full_cells=True``, the default) a cell
missing any of the 5 pre-registered seeds (PROTOCOL.md section 3;
``pendulum_core.yaml``'s ``seeds: [0, 1, 2, 3, 4]``; REGLES section 4:
"5 seeds minimum") is refused with a clear error naming the cell and the
missing seeds, rather than silently plotted at a smaller n -- this check
now runs over the FULL expected (regime, family, constraint) grid and
seed set derived from the experiment YAML's OWN ``arms``/``seeds``
declarations when present (m7 round 3; style-domain fallback otherwise --
see ``_expected_grid``), not just the cells that happen to have a
surviving run, so a cell with ZERO usable runs is named too (M4 round-2
review: an entirely-missing cell used to render silently). An
UNEXPECTED seed outside the required set is refused too, not just a
missing one (M10 round 4) -- a stray extra run must not silently enter
the mean. Curves show the MEAN over seeds with a +/-1 SE band (not the
median -- SE is defined around the mean); the on-figure R^2 per cell is
the MIN over included seeds (not the median) so one catastrophic seed
cannot hide behind the aggregate -- the full per-run breakdown (every
run's R^2, rank, equilibrated cond, sensitivity, and readout status) is
always written to ``<out_path's dir>/fig1_readout_r2.csv`` alongside the
figure (M9: a verdict must be auditable from the artifact alone).
"""

from __future__ import annotations

import csv
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import matplotlib
import yaml

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from sld.audit.jacobian import fr_drift_curve
from sld.experiments.loading import (
    build_dataset_from_expconfig,
    check_run_matches_dataset,
    load_run,
    read_index,
)
from sld.viz.style import CONSTRAINT_LINESTYLES, FAMILY_COLORS, apply_style, save_fig

# preferred panel order when both are present; anything else appends after
_PANEL_ORDER = ("ou", "action_conditioned")

# pre-registered 5-seed law (PROTOCOL.md section 3; pendulum_core.yaml's
# `seeds: [0, 1, 2, 3, 4]`) -- the reference set a "complete" cell must
# cover in strict/final-grid mode. Not exposed as a knob (ponytail): if the
# protocol's seed count ever changes, this constant changes with it.
_REQUIRED_SEEDS: frozenset[int] = frozenset(range(5))

# B1 (rounds 3-4): the eligibility verdict CANNOT be gauge-invariant under
# the FULL general-linear group, and this module stops pretending it is.
# Impossibility (round 3, two independent adversarial probes): exact rank
# is the ONLY GL(d)-invariant notion of degeneracy -- any continuous
# near-collapse criterion (cond thresholds, relative-eigenvalue
# truncation, whitening cutoffs) can be flipped by a general invertible
# gauge (a SHEAR) that mixes near-collinearity into or out of existence,
# so an eligibility rule that catches near-collapse is necessarily
# gauge-dependent under shears -- this residual limitation is real and
# stays declared.
#
# Round 4 narrowed that gauge-dependence to where it is actually
# unavoidable: round 3's own probe (S=diag(1e9, 1)) was NOT a shear, it
# was a per-coordinate unit change, and its flip came from thresholding
# the RAW design's cond, where the fixed-scale intercept column made a
# unit change look like new degeneracy. Measuring cond on the
# COLUMN-EQUILIBRATED design (each column divided by its own RMS) instead
# makes the verdict STABLE under the entire diagonal-rescaling class
# (any per-coordinate unit change, bounded or not: diag(1e9,1), a uniform
# 2**20, ...) while still catching genuine near-collinearity between
# coordinates. Neither diag(1e9,1) nor 2**20 destroys information at
# float32 (both round-trip exactly) -- they only relabel units, and the
# equilibrated criterion now recognizes that. The drift CURVE, by
# contrast, has always been gauge-invariant on the full-column-rank
# domain (unique OLS solution, W' = W S^{-1}) up to conditioning-amplified
# rounding, locked by tests for bounded-conditioning rescalings and
# shears -- unaffected by this eligibility-side fix.
_LATENT_EPS: float = float(np.finfo(np.float32).eps)  # latents are stored float32
# the decoded drift is trusted only when conditioning amplifies latent
# storage noise by less than this relative amount (1%%):
_INDETERMINATE_SENSITIVITY: float = 1e-2
assert np.isfinite(_LATENT_EPS) and np.isfinite(_INDETERMINATE_SENSITIVITY)


class ReadoutFit(NamedTuple):
    """What ``_fit_readout`` measured about one run's OLS readout (B1).

    Attributes:
        W: fitted readout, ``(d_z + 1, d_state)`` (last row is the bias
            term, fit via a ones column).
        r2: coefficient of determination on ``z_gt`` -- reported, never
            hidden, so a bad coordinate change is visible rather than
            silently trusted (REGLES section 5). Gauge-invariant (fitted
            values are a projection onto the design's column span).
        rank: numerical rank of the COLUMN-EQUILIBRATED ``[z_enc | 1]``
            design, using the same scale-invariant matrix as ``cond``.
            This is the rank used by the determinability verdict; the raw
            least-squares rank is deliberately not used because its
            floating-point cutoff changes under a harmless unit rescale.
        cond: condition number (largest singular value / smallest) of the
            COLUMN-EQUILIBRATED design (B1-v3, round 4: each column of
            ``[z_enc | 1]`` divided by its own RMS before the SVD) --
            ``inf`` when the equilibrated design is rank-deficient or any column
            has zero RMS. This internal undefined value is exported as an empty
            CSV field, never as a non-finite public metric. The OLS solve is
            performed on this same equilibrated design; ``W`` is transformed
            back so application to raw coordinates is unchanged.
        sensitivity: ``cond * _LATENT_EPS`` -- the relative indeterminacy
            of the decoded output induced by float32 latent storage noise,
            now measured on the equilibrated design so a per-coordinate
            unit change cannot masquerade as sensitivity.
        indeterminate: ``True`` when the equilibrated design is
            column-rank-deficient or ``sensitivity >
            _INDETERMINATE_SENSITIVITY`` --
            the decoded drift is then not numerically determined by the
            data at the latents' storage precision. STABLE under the
            entire diagonal-rescaling class (any per-coordinate unit
            change); DECLARED gauge-dependent only under general shears
            (see the impossibility note above) -- this is a verdict about
            the run's observed representation, not about the underlying
            dynamics.
    """

    W: np.ndarray
    r2: float
    rank: int
    cond: float
    sensitivity: float
    indeterminate: bool


def _fit_readout(z_enc: np.ndarray, z_gt: np.ndarray) -> ReadoutFit:
    """OLS-fit a linear+bias readout from encoded latent to true state.

    ``z_enc``/``z_gt`` are ``(N, ·)`` float64 -- flattened over time and
    batch. Fit on the ENCODED ground truth (``z_enc``), never on the
    predictor's rollout -- this is an estimated change of coordinates, not
    a fit to the drift metric itself.

    B1-v4 (round 5): the OLS solve, rank, conditioning, and R^2 are all
    computed on the COLUMN-EQUILIBRATED affine design (each column of
    ``[z_enc | 1]`` divided by its own RMS). The resulting coefficients are
    transformed back by the inverse column scale, so applying ``W`` to the
    raw design is exactly unchanged. A
    zero-RMS column is rank-deficient by construction and short-circuits
    straight to "indeterminate" rather than dividing by zero. This is what
    makes the verdict stable under the entire diagonal-rescaling class
    (e.g. ``diag(1e9,1)`` or a uniform ``2**20`` -- round 4's exact
    reproduction: multiplying every latent coordinate by ``2**20`` is
    exactly representable and round-trips bit-exact through float32, so it
    destroys no information, yet used to inflate the RAW design's cond
    from ~1 to ~1e6 purely because the intercept column stays fixed at 1
    while the z columns are rescaled -- a unit mismatch, not degeneracy).
    Genuine near-collinearity between latent coordinates (e.g. a
    near-collapse ``z2 ~ z1``) is still caught: equilibrated columns that
    are nearly identical still blow up the equilibrated cond. See the
    module docstring's B1 section and the ``_LATENT_EPS`` block above for
    the full argument, including the residual (general-shear) limitation.

    Returns:
        ``ReadoutFit`` (see its docstring) -- ``rank``, ``cond``, and
        ``sensitivity`` all come from the same equilibrated design, never
        recomputed elsewhere, so the indeterminacy classification cannot
        drift under a diagonal unit rescale.

    Raises:
        ValueError: ``z_gt`` has zero variance across the split (R^2
            undefined), or the fit produces a non-finite R^2.
    """
    design = np.concatenate([z_enc, np.ones((z_enc.shape[0], 1))], axis=1)

    # B1-v4 (round 5): solve on the COLUMN-EQUILIBRATED design too, not
    # merely its eligibility verdict. This prevents lstsq's raw-coordinate
    # rank cutoff from corrupting an otherwise determinate affine readout.
    # RMS-normalize each column so a per-coordinate unit change (the
    # intercept column staying fixed at 1 while z columns are rescaled)
    # cannot masquerade as degeneracy. Compute each RMS after max-scaling
    # its column: direct ``mean(design**2)`` would overflow for a valid
    # large unit rescale and underflow for a valid small one.
    col_max = np.abs(design).max(axis=0)
    scaled = design / np.where(col_max == 0.0, 1.0, col_max)
    col_rms = col_max * np.sqrt(np.mean(scaled**2, axis=0))
    # A zero column is retained as zero (rather than divided by zero), so
    # lstsq can report its rank deficiency on the same solved design.
    column_scale = np.where(col_rms == 0.0, 1.0, col_rms)
    design_bal = design / column_scale
    W_eq, _, rank, singular_values_bal = np.linalg.lstsq(design_bal, z_gt, rcond=None)
    W = W_eq / column_scale[:, None]
    fitted = design_bal @ W_eq
    ss_res = float(np.sum((z_gt - fitted) ** 2))
    ss_tot = float(np.sum((z_gt - z_gt.mean(axis=0, keepdims=True)) ** 2))
    if ss_tot <= 0.0:
        raise ValueError("z_gt has zero variance across the val split; readout R^2 is undefined")
    r2 = 1.0 - ss_res / ss_tot
    if not np.isfinite(r2):
        raise ValueError(f"readout R^2 is non-finite ({r2}) -- bug, not a result")
    rank = int(rank)
    if np.any(col_rms == 0.0) or rank < design.shape[1] or singular_values_bal[-1] == 0.0:
        cond = float("inf")
        sensitivity = float("inf")
    else:
        cond = float(singular_values_bal[0] / singular_values_bal[-1])
        sensitivity = cond * _LATENT_EPS
    indeterminate = rank < design.shape[1] or sensitivity > _INDETERMINATE_SENSITIVITY
    return ReadoutFit(
        W=W, r2=r2, rank=int(rank), cond=cond, sensitivity=sensitivity, indeterminate=indeterminate
    )


def _apply_readout(z: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply the fitted readout ``W`` (``(d_z+1, d_state)``, bias last row)
    to ``z`` (``(..., d_z)``), returning ``(..., d_state)``."""
    ones = np.ones((*z.shape[:-1], 1), dtype=z.dtype)
    return np.concatenate([z, ones], axis=-1) @ W


class _RunProjection(NamedTuple):
    """Everything ``fr_drift_curve`` needs, computed BEFORE it is ever
    called (M8): a run's readout fit plus both trajectories decoded into
    true-state coordinates. Split out of the old monolithic
    ``_run_drift_curve`` so ``make_fig1`` can record ``fit`` -- a verdict
    that is already fully known at this point -- even when the downstream
    drift computation itself fails (e.g. a decoded target ~0 trips
    ``fr_drift_curve``'s own zero-target guard); a drift failure must
    never make an already-computed readout verdict vanish before it is
    written to the CSV.
    """

    fit: ReadoutFit
    pred_state: np.ndarray
    target_state: np.ndarray


def _fit_run_readout(run: dict, dataset: dict, horizon: int) -> _RunProjection:
    """Fit one run's readout and decode both the rollout prediction and the
    encoder's own target embedding into true-state coordinates (M8: the
    half of the old ``_run_drift_curve`` that can succeed independently of
    ``fr_drift_curve``). This function only MEASURES ``ReadoutFit``;
    deciding what an indeterminate readout means for strict vs. lenient
    rendering, or what a downstream drift failure means, is
    ``make_fig1``'s job (B1/M8), not this one's.
    """
    encoder, predictor = run["encoder"], run["predictor"]
    x_val = torch.as_tensor(dataset["val"]["x"], dtype=torch.float32).transpose(0, 1)
    u_val = torch.as_tensor(dataset["val"]["u"], dtype=torch.float32).transpose(0, 1)
    # dataset's actual ground-truth physical state -- distinct from the
    # encoder's OWN embedding below (the old code called that "z_true",
    # which it is not; renamed to z_enc to keep the two apart)
    z_gt = np.asarray(dataset["val"]["z_true"], dtype=np.float64).transpose(1, 0, 2)  # (T+1, B, d_state)
    H = min(horizon, u_val.shape[0])
    lifted = hasattr(predictor, "lift")
    lift_fn = predictor.lift if lifted else (lambda z: z)
    project_fn = predictor.project if lifted else (lambda h: h)
    with torch.no_grad():
        z_enc = encoder(x_val)  # (T+1, B, d_z) -- the run's OWN embedding of x_val
        h0 = lift_fn(z_enc[0])
        h_hat = predictor.rollout(h0, u_val[:H])  # (H, B, d_h)
        pred_z = project_fn(h_hat)  # (H, B, d_z)

    z_enc_np = z_enc.double().numpy()
    fit = _fit_readout(
        z_enc_np.reshape(-1, z_enc_np.shape[-1]),
        z_gt.reshape(-1, z_gt.shape[-1]),
    )
    pred_state = _apply_readout(pred_z.double().numpy(), fit.W)
    target_state = _apply_readout(z_enc_np[1 : H + 1], fit.W)
    return _RunProjection(fit=fit, pred_state=pred_state, target_state=target_state)


def _run_drift_curve(run: dict, dataset: dict, horizon: int) -> tuple[np.ndarray, ReadoutFit]:
    """One run's FR drift curve in TRUE-STATE coordinates, plus the
    readout's full ``ReadoutFit`` (module docstring: the F1 gauge-invariance
    fix, B1's indeterminacy classification).

    ``fr_drift_curve`` already averages over the batch (episode) axis
    internally (``err_t = mean_B ||pred_t - target_t||``), so this returns
    ONE ``(H,)`` curve per run -- the "mean over 5 seeds" aggregation in
    ``make_fig1`` is a mean over these per-run curves, not over episodes.
    Thin composition of ``_fit_run_readout`` + ``fr_drift_curve`` (M8):
    kept for callers (and the gauge-invariance test) that want the fit AND
    the curve unconditionally, raising if either half fails.
    """
    proj = _fit_run_readout(run, dataset, horizon)
    return fr_drift_curve(proj.pred_state, proj.target_state), proj.fit


def _aggregate_curves(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and +/-1 SE over the seed axis (``curves``, ``(n_seeds, H)``).

    The ONE place fig1's seed-aggregation arithmetic lives: ``make_fig1``
    plots exactly what this returns, so the numeric-lock test (M6/m12)
    verifies the actual rendered values rather than a parallel
    reimplementation. ``se`` is ``std(ddof=1) / sqrt(n)``; ``0`` when
    ``n == 1`` (ddof=1 needs ``n >= 2``) -- the honest "no spread observed"
    value, never a silent NaN. Mean, never the median (REGLES section 4:
    "+/-1 SE" bands -- SE is defined around the mean, not the median).
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


# M9: every column the per-run CSV artifact carries, in on-disk order --
# the single source both the writer and _r2_record key off, so the two
# can never drift apart.
_R2_CSV_COLUMNS: tuple[str, ...] = (
    "run_dir",
    "regime",
    "family",
    "constraint",
    "seed",
    "r2",
    "rank",
    "n_design_cols",
    "cond_equilibrated",
    "sensitivity",
    "sensitivity_threshold",
    "readout_status",
    "drift_status",
)


def _finite_csv_metric(value: float) -> float | str:
    """Return a public numeric metric, or categorical absence if undefined."""
    return value if np.isfinite(value) else ""


def _r2_record(row: dict, regime: str, fit: ReadoutFit, status: str) -> dict:
    """One ``fig1_readout_r2.csv`` row (M7/M9): the full numeric evidence
    behind a verdict -- rank, equilibrated cond, sensitivity, and the
    threshold applied -- not just its label, so an "indeterminate" (or
    "drift-failed") verdict is auditable from the artifact alone without
    re-deriving it from the run.
    """
    return {
        "run_dir": row["run_dir"],
        "regime": regime,
        "family": row["family"],
        "constraint": row["constraint"],
        "seed": row["seed"],
        "r2": fit.r2,
        "rank": fit.rank,
        "n_design_cols": fit.W.shape[0],
        # REGLES §5 finite-metrics law: undefined conditioning is a
        # categorical absence in the public CSV, never an inf/NaN number.
        "cond_equilibrated": _finite_csv_metric(fit.cond),
        "sensitivity": _finite_csv_metric(fit.sensitivity),
        "sensitivity_threshold": _INDETERMINATE_SENSITIVITY,
        "readout_status": status,
        "drift_status": "ok",
    }


def make_fig1(
    results_root: str = "results",
    exp_name: str = "pendulum_core",
    exp_yaml: str = "src/sld/experiments/configs/pendulum_core.yaml",
    out_path: str = "assets/fig1_rollout.png",
    horizon: int = 50,
    require_full_cells: bool = True,
) -> Path:
    """Regenerate Fig. 1 from ``<results_root>/<exp_name>`` and save to ``out_path``.

    Aggregates per ``(regime, family, constraint)`` cell as the MEAN over
    the per-run curves with a +/-1 SE band (``_aggregate_curves``); the
    on-figure R^2 per cell is the MIN over its included runs (M7 -- a
    median would hide a single catastrophic seed). Missing/unloadable
    individual runs are warned about and skipped; a checkpoint/dataset
    mismatch (``sld.experiments.loading.check_run_matches_dataset`` --
    finding F6) is never caught here, it always aborts the figure. Zero
    usable complete runs is a clean ``SystemExit(1)``, never a raw
    traceback. A run whose readout is "indeterminate" (B1, module docstring)
    is never plotted -- excluded with a warning in lenient mode, a hard
    ``ValueError`` naming the run in strict mode.

    Every run this function manages to fit a readout for -- including ones
    excluded from the plot for being indeterminate, and ones whose
    downstream drift computation itself failed (M8) -- is recorded in
    ``<out_path's parent>/fig1_readout_r2.csv`` (columns: ``run_dir,
    regime, family, constraint, seed, r2, rank, n_design_cols,
    cond_equilibrated, sensitivity, sensitivity_threshold, readout_status,
    drift_status``; M9: full numeric evidence for every verdict, not
    just its label), written on every successful render (M7): the full
    per-run breakdown is never reduced away, even though the figure itself
    only shows a compact per-cell summary. ``readout_status`` is ``"ok"``
    or ``"indeterminate"``; the independent ``drift_status`` is ``"ok"``,
    ``"drift-failed: zero-or-negligible-decoded-target-signal"``, or
    ``"drift-failed: <reason>"`` (M8) -- a run
    whose readout fit succeeds but whose decoded drift curve computation
    raises (e.g. a decoded target ~0 trips ``fr_drift_curve``'s own
    zero-target guard) gets this last, named status instead of silently
    disappearing before any record of its (perfectly good) fit is
    written.

    Args:
        require_full_cells: when ``True`` (default, "final grid" mode), a
            cell missing any of the 5 pre-registered seeds raises
            ``ValueError`` naming the cell and the missing seeds instead
            of silently plotting at a smaller n (REGLES section 4). This
            check runs over the FULL expected grid (``_expected_grid``),
            so a cell with ZERO surviving runs is named exactly like a
            partially-filled one (M4), and a cell with an UNEXPECTED seed
            outside the required set is refused too, not just a missing
            one (M10) -- a stray extra run must not silently enter the
            mean. Pass ``False`` explicitly to relax this -- e.g. small
            smoke tests that only have 1-2 seeds. A missing seed whose run
            is a ``divergence`` (index status, PROTOCOL.md section 11.7) is
            named by run_dir, distinct from a seed that was never run at
            all.
    """
    apply_style()
    try:
        rows = read_index(results_root, exp_name)
    except FileNotFoundError as e:
        print(f"fig1: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    complete = [r for r in rows if r["status"] == "complete"]
    if not complete:
        print(
            f"fig1: no complete runs in {results_root}/{exp_name}/index.csv -- nothing to plot",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # (regime, family, constraint) -> [(seed, curve, r2), ...]
    cells: dict[tuple[str, str, str], list[tuple[int, np.ndarray, float]]] = defaultdict(list)
    dataset_cache: dict[str, dict | None] = {}
    r2_records: list[dict] = []  # M7: per-run artifact, every run that got a fit
    strict_violations: list[str] = []  # M5: collected, raised AFTER the CSV is durable
    n_ok = 0
    for row in complete:
        regime = row["regime"]
        if regime not in dataset_cache:
            try:
                dataset_cache[regime] = build_dataset_from_expconfig(exp_yaml, regime)
            except Exception as e:
                warnings.warn(f"fig1: could not build dataset for regime {regime!r}: {e}")
                dataset_cache[regime] = None
        dataset = dataset_cache[regime]
        if dataset is None:
            continue

        # F6: a checkpoint trained on a dataset spec the CURRENT yaml no
        # longer produces is never silently evaluated -- this is not
        # caught below, it aborts the whole figure
        check_run_matches_dataset(row["run_dir"], dataset["meta"])

        try:
            run = load_run(row["run_dir"])
        except Exception as e:
            warnings.warn(
                f"fig1: skipping run {row['run_dir']} "
                f"({row['family']}/{row['constraint']}/seed{row['seed']}): {e}"
            )
            continue

        # M8: the readout fit is computed and recorded BEFORE fr_drift_curve
        # is ever called, so a downstream drift-computation failure cannot
        # make a run's (already fully known) readout verdict disappear
        # before it is written to the CSV.
        try:
            proj = _fit_run_readout(run, dataset, horizon)
        except Exception as e:
            warnings.warn(
                f"fig1: skipping run {row['run_dir']} "
                f"({row['family']}/{row['constraint']}/seed{row['seed']}) -- readout fit "
                f"failed: {e}"
            )
            continue

        fit = proj.fit
        # counts every run a readout fit was actually computed for -- an
        # indeterminate classification, or a downstream drift failure, is a
        # validity verdict, not a load failure, so it still counts here
        # (only load/fit exceptions, caught above, do not)
        n_ok += 1

        # M8: append the durable fit record BEFORE calling the downstream
        # drift metric. A failure below only updates this named status; it
        # can never erase the already measured ReadoutFit.
        readout_status = "indeterminate" if fit.indeterminate else "ok"
        r2_record = _r2_record(row, regime, fit, readout_status)
        r2_records.append(r2_record)
        indeterminate_reason = (
            f"rank={fit.rank}/{fit.W.shape[0]} readout design columns, "
            f"cond_equilibrated={fit.cond:.3e}, sensitivity={fit.sensitivity:.3e} "
            f"(threshold {_INDETERMINATE_SENSITIVITY:.0e} at float32 latent precision)"
        )

        try:
            curve = fr_drift_curve(proj.pred_state, proj.target_state)
        except Exception as e:
            # M8: the fit succeeded (its ReadoutFit is fully known) but the
            # decoded drift curve itself could not be computed (e.g. a
            # decoded target ~0 trips fr_drift_curve's own zero-target
            # guard) -- a NAMED CSV status, never a silent skip that would
            # lose the readout's own verdict along with the curve.
            reason = str(e).strip().splitlines()[0]
            if "target is identically zero" in reason:
                r2_record["drift_status"] = (
                    "drift-failed: zero-or-negligible-decoded-target-signal"
                )
            else:
                r2_record["drift_status"] = f"drift-failed: {reason}"
            warnings.warn(
                f"fig1: run {row['run_dir']} "
                f"({row['family']}/{row['constraint']}/seed{row['seed']}) readout fit complete but "
                f"drift curve computation failed: {reason}"
            )
            if fit.indeterminate and require_full_cells:
                strict_violations.append(
                    f"run {row['run_dir']} "
                    f"({row['family']}/{row['constraint']}/seed{row['seed']}) has an "
                    f"indeterminate readout ({indeterminate_reason})"
                )
            continue

        if fit.indeterminate:
            # B1: the decoded drift is not numerically determined by the
            # data at float32 latent precision (rank-deficient design, or
            # equilibrated conditioning amplifies storage noise beyond the
            # declared 1%%) -- this run is NEVER plotted silently. Strict
            # mode collects the violation and raises AFTER the per-run CSV
            # is written (M5, round 3: the verdict must be durable even
            # when the render fails).
            if require_full_cells:
                strict_violations.append(
                    f"run {row['run_dir']} "
                    f"({row['family']}/{row['constraint']}/seed{row['seed']}) has an "
                    f"indeterminate readout ({indeterminate_reason})"
                )
            else:
                warnings.warn(
                    f"fig1: excluding run {row['run_dir']} "
                    f"({row['family']}/{row['constraint']}/seed{row['seed']}) -- readout "
                    f"indeterminate ({indeterminate_reason})"
                )
            continue

        cells[(regime, row["family"], row["constraint"])].append((row["seed"], curve, fit.r2))

    if n_ok == 0:
        print("fig1: every complete run failed to load/rollout -- nothing to plot", file=sys.stderr)
        raise SystemExit(1)

    if require_full_cells:
        # M4: the FULL expected grid, not just the cells that happen to
        # have a surviving run -- a cell with zero usable runs is named
        # exactly like a partially-filled one. All violations are
        # collected, not just the first (M5, round 3). Cells and seeds
        # come from the yaml's own declarations when present (m7).
        #
        # PROTOCOL.md section 11.7: a missing seed whose run diverged is
        # NOT the same thing as a seed that was never run -- "on the same
        # footing as a readout indeterminate verdict", so it must be named
        # by run_dir, exactly like the indeterminate-readout violation
        # above, not folded into the generic "missing seeds" message.
        # Looked up from ``rows`` (the FULL, unfiltered index -- ``complete``
        # only ever contains status=="complete" rows, so a diverged run
        # never reaches ``cells`` above and would otherwise be silently
        # indistinguishable from one that was never attempted at all).
        status_by_cell_seed = {
            (r["regime"], r["family"], r["constraint"], r["seed"]): r for r in rows
        }
        expected_cells, required_seeds = _expected_grid(exp_yaml)
        for regime, family, constraint in expected_cells:
            entries = cells.get((regime, family, constraint), [])
            present_seeds = {seed for seed, _, _ in entries}
            missing = sorted(required_seeds - present_seeds)
            diverged = [
                seed
                for seed in missing
                if status_by_cell_seed.get((regime, family, constraint, seed), {}).get("status")
                == "divergence"
            ]
            for seed in diverged:
                diverged_row = status_by_cell_seed[(regime, family, constraint, seed)]
                strict_violations.append(
                    f"run {diverged_row['run_dir']} "
                    f"({family}/{constraint}/seed{seed}) is a divergence -- non-finite "
                    "training loss (PROTOCOL.md section 11.7), never silently excluded"
                )
            still_missing = [seed for seed in missing if seed not in diverged]
            if still_missing:
                strict_violations.append(
                    f"cell (regime={regime!r}, family={family!r}, "
                    f"constraint={constraint!r}) is missing seeds {still_missing} "
                    f"(has {sorted(present_seeds)}, requires "
                    f"{sorted(required_seeds)})"
                )
            # M10: a cell only ever checked for MISSING seeds -- any
            # complete row was appended regardless of whether its seed was
            # actually one of the pre-registered ones (:433's ancestor),
            # so a stray extra seed silently entered the mean. An
            # unexpected seed is named as its own violation.
            extra = sorted(present_seeds - required_seeds)
            if extra:
                strict_violations.append(
                    f"cell (regime={regime!r}, family={family!r}, "
                    f"constraint={constraint!r}) has unexpected seed(s) {extra} "
                    f"outside the required set {sorted(required_seeds)} "
                    f"(has {sorted(present_seeds)})"
                )

    # M7 + M5: every run a fit was computed for -- including ones excluded
    # above as indeterminate -- is recorded next to the figure BEFORE any
    # strict failure is raised, so the per-run verdict is durable even when
    # the render aborts (round 3: a collapsed seed must leave a record).
    r2_csv_path = Path(out_path).parent / "fig1_readout_r2.csv"
    r2_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(r2_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_R2_CSV_COLUMNS)  # M9: full numeric evidence, not just r2/status
        for rec in sorted(
            r2_records, key=lambda r: (r["regime"], r["family"], r["constraint"], r["seed"])
        ):
            writer.writerow([rec[col] for col in _R2_CSV_COLUMNS])

    if strict_violations:
        # M5: raised only now, with the CSV already on disk and listing
        # EVERY violation, not the first one encountered
        raise ValueError(
            "fig1 strict mode: refusing to render -- "
            f"{len(strict_violations)} violation(s):\n  - "
            + "\n  - ".join(strict_violations)
            + "\n(pass require_full_cells=False to exclude/override explicitly; "
            "REGLES section 4: 5 seeds minimum, no silently non-determined curve)"
        )

    if not cells:
        # m8: every computed readout was excluded (all indeterminate in
        # lenient mode) -- an explicit verdict, never a bare matplotlib
        # crash on a zero-column subplot
        print(
            "fig1: every run's readout was excluded (indeterminate) -- nothing to plot",
            file=sys.stderr,
        )
        raise SystemExit(1)

    regimes_present = sorted(
        {k[0] for k in cells},
        key=lambda r: (_PANEL_ORDER.index(r) if r in _PANEL_ORDER else len(_PANEL_ORDER), r),
    )
    fig, axes = plt.subplots(
        1, len(regimes_present), figsize=(5.5 * len(regimes_present), 4.5), sharey=True, squeeze=False
    )
    axes = axes[0]

    for ax, regime in zip(axes, regimes_present):
        ax.set_title(regime)
        regime_cells = sorted((k, v) for k, v in cells.items() if k[0] == regime)
        for (_, family, constraint), entries in regime_cells:
            entries = sorted(entries, key=lambda e: e[0])
            curves = np.stack([c for _, c, _ in entries], axis=0)  # (n_seeds, H)
            n = curves.shape[0]
            mean_curve, se = _aggregate_curves(curves)
            # M7: MIN, never the median -- a median hides a single
            # catastrophic seed (e.g. [-100, .99, .99, .99, .99] -> .99);
            # the full per-run breakdown always lives in fig1_readout_r2.csv
            r2_min = float(np.min([r2 for _, _, r2 in entries]))
            x = np.arange(1, len(mean_curve) + 1)
            color = FAMILY_COLORS.get(family, "gray")
            ax.plot(
                x,
                mean_curve,
                color=color,
                linestyle=CONSTRAINT_LINESTYLES.get(constraint, "-"),
                label=f"{family}/{constraint} (n={n}, R2min={r2_min:.2f})",
            )
            lower = np.clip(mean_curve - se, 1e-12, None)  # log-scale floor
            ax.fill_between(x, lower, mean_curve + se, color=color, alpha=0.2, linewidth=0)
        ax.set_xlabel("rollout horizon (steps)")
        ax.set_yscale("log")
    axes[0].set_ylabel("normalized FR drift (true-state coords, per-run readout)")

    handles_labels: dict[str, object] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles_labels[label] = handle
    fig.legend(
        handles_labels.values(),
        handles_labels.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(4, max(1, len(handles_labels))),
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(
        "Latent FR drift in TRUE-STATE coordinates (per-run OLS readout; v0 D1 pendulum)\n"
        "mean over N seeds +/-1 SE (never the median); R2 = per-cell MIN readout fit "
        "quality across included seeds (exact under linear reparam, approximate under "
        "nonlinear; indeterminate readouts excluded -- see fig1_readout_r2.csv)",
        fontsize=10,
    )
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    return Path(out_path)


def main() -> None:
    make_fig1()


if __name__ == "__main__":
    main()
