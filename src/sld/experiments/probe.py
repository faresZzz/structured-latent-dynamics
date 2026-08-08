"""The calibration probe harness (PROTOCOL.md sections 14.3, 14.11).

A budget chosen by assertion is a guess. This module turns the four entry
values of sections 14.1/14.2 -- plus the two invariance/fidelity axes of
14.11 -- from asserted numbers into MEASURED ones, on one stratum, one seed,
the cheapest arm (``mlp_k``), with a confirmation pass on the costliest arm
(``transformer_k``). One axis moves at a time from a declared baseline;
NEVER a full cross-product (14.3's own wording).

This module does not decide anything. It writes one record per cell to
``results_probe_d/report.json`` and prints a human table; the project owner
reads the numbers and moves 14.1/14.2's entry values (or does not) in a
dated amendment. The gate's thresholds (``H_GATE``, ``GAMMA``, ``TAU_VPT``,
and ``GAMMA_1`` of section 15.1) are CONSUMED here, imported from
``sld.experiments.protocol``, with the gate arithmetic itself imported from
``sld.experiments.gate`` -- never redefined or moved (14.3: "what the probe
may not move").

Reused, never reimplemented (R2):

- ``sld.experiments.gate.evaluate_run_free_rollout`` for ``e_h`` (hence
  ``e_25``/``e_1``) and the positive-control verdict/margins -- this module
  never recomputes a normalized error or the gate arithmetic itself.
- ``sld.encoders.pretrain.pretrain_representation`` /
  ``save_frozen_representation`` / ``load_frozen_representation`` for the
  representation stage, and ``sld.experiments.factorial._load_or_build_representation``
  for its idempotent build-if-absent/load-otherwise discipline (atomic
  write, digest revalidation on every load) -- reused via import, not
  copied, so the two callers can never drift.
- ``sld.training.trainer.Trainer`` for the predictor stage, and
  ``sld.experiments.factorial._validate_completion_sentinel`` for the SAME
  "is this run genuinely complete" check the factorial runner uses to
  decide resume-vs-retrain, so a killed probe resumes exactly like a killed
  factorial run does.

Not reused, because nothing existing computes it: ``cond(Sigma))`` of the
frozen representation's latent on the held-out split (14.11 -- this project
was damaged once by a badly-conditioned latent, PROTOCOL.md's own mu_J
numbers: 62.7, 1.5e7, 1.3e5 across families). Computed once per distinct
representation build, cached, never per cell.

Namespace discipline (14.3, 14.6): every artifact this module writes --
representations, run directories, the report -- lands under
``PROBE_ROOT`` (``results_probe_d/`` by default) and nowhere else. Writes
this module owns directly (the representation path, a run's ``run_dir``,
the report) are checked by ``_require_under_root`` before use; a bug that
constructs a path outside the root is refused loudly, not written quietly.

Declared, not derived (this probe's own infrastructure choices, not a
14.3/14.11 axis and not a 12.3/12.4 threshold): the representation
pretraining budget (``_REP_EPOCHS=50``, matching the repo's existing
"selection budget" convention, ``trainer._DEFAULTS["epochs"]``/PROTOCOL.md
section 2 step 3) and its ``lr``/``tau_ema``/VICReg weights (the
``pretrain_representation`` module defaults) are held FIXED across every
cell except where ``w_align``/``view_offset`` themselves are the axis being
probed. This is an engineering tolerance (14.4's second kind), not a
pre-registered number -- it exists so the w_align/view_offset axis measures
ONE thing changing, not two.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from sld.audit._common import require_cpu_float64_module
from sld.data.generate import build_dataset
from sld.data.regimes import OURegime
from sld.encoders.frame_mlp import FrameMLPEncoder
from sld.encoders.pretrain import representation_digest as _representation_digest
from sld.experiments import factorial as _factorial
from sld.experiments import protocol
from sld.experiments.gate import RolloutDivergence, evaluate_run_free_rollout
from sld.experiments.loading import load_run
from sld.experiments.protocol import GAMMA as _GAMMA
from sld.experiments.protocol import H_GATE as _H_GATE
from sld.experiments.protocol import TAU_VPT as _TAU_VPT
from sld.predictors.base import PredictorState
from sld.systems.damped_pendulum import DampedPendulum
from sld.training.trainer import _FAMILY_CONFIG_KEYS, FAILURE_FILENAME, Trainer, data_fingerprint

# --------------------------------------------------------------------------
# Namespace root (PROTOCOL.md 14.3/14.6): a module-level Path, read
# dynamically by every function below (never captured as a default
# argument) so a test can redirect it with monkeypatch.setattr without this
# module ever gaining a CLI flag that could point production runs elsewhere.
# --------------------------------------------------------------------------
PROBE_ROOT = Path("results_probe_d")


def _require_under_root(path: Path) -> Path:
    """Refuse ``path`` unless it resolves to ``PROBE_ROOT`` or a descendant.

    The one guard every write this module makes directly goes through --
    the representation cache path, a cell's ``run_dir``, and the report.
    Raises ``ValueError`` rather than writing; never coerces the path into
    the root.
    """
    root = PROBE_ROOT.resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(
            f"refusing to write outside {PROBE_ROOT}: {path} resolves to {resolved}, "
            f"not under {root} (PROTOCOL.md section 14.3/14.6)"
        )
    return path


# --------------------------------------------------------------------------
# Fixed stratum / probe infrastructure (declared, see module docstring)
# --------------------------------------------------------------------------
_STRATUM_SYSTEM = "DampedPendulum"
_STRATUM_REGIME = "OURegime"
_D_X = 2
_D_Z = 4
_D_U = 1
_T = 100
_DT = 0.02
_DATA_SEED = 0
_MODEL_SEED = 0
_ENCODER_HIDDEN = (64, 64)
_DELTA = _DT  # structured arms' ZOH step -- the system's own physical step
_BATCH_EPISODES = 8  # PROTOCOL.md 14.1, never an axis

_REP_SEED = 0
_REP_EPOCHS = 50
_REP_LR = 1e-3
_REP_TAU_EMA = 0.99
_REP_W_VAR = 1.0
_REP_W_COV = 1.0
_REP_W_RECON = 1.0

# PROTOCOL.md 12.3/12.2 -- consumed from sld.experiments.protocol (R4), the
# single source of truth (CORR-2/2a); names kept as _H_GATE/_GAMMA/_TAU_VPT
# because the existing tests reference them.

# --------------------------------------------------------------------------
# The axis table (PROTOCOL.md 14.3, 14.11)
# --------------------------------------------------------------------------
_BASELINE = {
    "epochs": 1000,  # predictor training updates, 14.1
    "h_fr": 5,
    "target_params": 10_000,
    "n_episodes": 256,
    "lr": 1e-4,  # 14.2: mlp/mlp_k AND transformer/transformer_k both 1e-4
    "w_align": 1.0,
    "view_offset": 1,
}

# (axis name, knob key, non-baseline values) -- mlp_k only, one axis moves
# at a time off _BASELINE (14.3's own wording: "no cross-product")
_MLP_AXES: list[tuple[str, str, list]] = [
    ("updates", "epochs", [500, 2000, 4000]),
    ("h_fr", "h_fr", [10, 15, 25]),  # 25 == H_gate: le rollout entraine sur l'horizon note
    ("target_params", "target_params", [40_000]),
    ("n_episodes", "n_episodes", [64]),
    ("lr", "lr", [3e-4]),
    ("w_align", "w_align", [0.0, 10.0]),
    ("view_offset", "view_offset", [2]),
]
# transformer_k confirmation: updates only, "the cost driver" (task brief)
_TRANSFORMER_UPDATES = [2000]

# PROTOCOL.md section 15.4(b) -- probe round 2, a CLOSED list of nine cells,
# reproduced here VERBATIM from the committed table, never built via a
# cross-product generator: several rows move more than one knob at once off
# _BASELINE (e.g. row 4), unlike round 1's strict one-axis-at-a-time table,
# so a generator would either under- or over-produce this list.
_ROUND2_CELLS: list[dict] = [
    {
        "cell_id": "mlp_k__hfr25_updates_2000", "arm": "mlp_k", "axis": "round2",
        "value": "hfr25_updates_2000", "knobs": dict(_BASELINE, h_fr=25, epochs=2000),
    },
    {
        "cell_id": "mlp_k__hfr25_updates_4000", "arm": "mlp_k", "axis": "round2",
        "value": "hfr25_updates_4000", "knobs": dict(_BASELINE, h_fr=25, epochs=4000),
    },
    {
        "cell_id": "mlp_k__hfr25_lr_3e-4", "arm": "mlp_k", "axis": "round2",
        "value": "hfr25_lr_3e-4", "knobs": dict(_BASELINE, h_fr=25, lr=3e-4),
    },
    {
        "cell_id": "mlp_k__hfr25_updates_4000_lr_3e-4", "arm": "mlp_k", "axis": "round2",
        "value": "hfr25_updates_4000_lr_3e-4",
        "knobs": dict(_BASELINE, h_fr=25, epochs=4000, lr=3e-4),
    },
    {
        # pure autoencoder encoder (w_align=0) -- reuses round 1's existing
        # w_align=0 representation artifact, same caching key (_rep_cfg
        # doesn't depend on h_fr/epochs/lr/target_params, only w_align/
        # view_offset/n_episodes)
        "cell_id": "mlp_k__hfr25_walign_0", "arm": "mlp_k", "axis": "round2",
        "value": "hfr25_walign_0", "knobs": dict(_BASELINE, h_fr=25, w_align=0.0),
    },
    {
        "cell_id": "transformer_k__hfr25", "arm": "transformer_k", "axis": "round2",
        "value": "hfr25", "knobs": dict(_BASELINE, h_fr=25),
    },
    {
        "cell_id": "transformer_k__hfr25_updates_2000", "arm": "transformer_k", "axis": "round2",
        "value": "hfr25_updates_2000", "knobs": dict(_BASELINE, h_fr=25, epochs=2000),
    },
    {
        # the missing ssm comparison point (14.8 names ssm/ssm_k the most
        # likely divergence candidate; round 1 never touched this family)
        # lr=3e-4, NOT _BASELINE's 1e-4: 14.2 pins the LR per backbone (ssm/ssm_k at 3e-4)
        "cell_id": "ssm_k__baseline", "arm": "ssm_k", "axis": "round2",
        "value": "baseline", "knobs": dict(_BASELINE, lr=3e-4),
    },
    {
        "cell_id": "ssm_k__hfr25", "arm": "ssm_k", "axis": "round2",
        "value": "hfr25", "knobs": dict(_BASELINE, h_fr=25, lr=3e-4),
    },
]

# PROTOCOL.md section 16 (Amendment 7) -- probe round 3: the learning rate is
# re-selected at the D regime, uniformly per backbone, by 16.2's declared
# procedure. Ascending order is itself part of the contract: "the next lower
# listed rate" (16.2(3)) means the PREVIOUS element of this list.
_ROUND3_LR_LIST: list[float] = [1e-4, 3e-4, 1e-3]

# Label used only to spell a runtime lr float into a cell_id/value string --
# never fed back into arithmetic, so this never has to agree with
# _fmt_value's "%g" formatting (which would render 1e-4 as "0.0001", not
# PROTOCOL.md's own "1e-4" spelling).
_LR_LABEL: dict[float, str] = {1e-4: "1e-4", 3e-4: "3e-4", 1e-3: "1e-3"}

# PROTOCOL.md 16.2(1)/16.4 stage 1 -- nine runs, a CLOSED list, reproduced
# VERBATIM (same precedent as _ROUND2_CELLS above: literal cell_ids/values,
# never a generator). ``arm`` is the OFF arm's FAMILY name (registry names,
# matching factorial._FAMILY_BY_PAIR's ("mlp","off")/("transformer","off")/
# ("ssm","off") -> "mlp"/"transformer"/"min_ssm") -- _trainer_config passes
# it straight through as the Trainer's ``family``.
_ROUND3_SWEEP: list[dict] = [
    {
        "cell_id": "mlp__r3_lr_1e-4", "arm": "mlp", "axis": "round3",
        "value": "r3_lr_1e-4", "knobs": dict(_BASELINE, epochs=4000, h_fr=25, lr=1e-4),
    },
    {
        "cell_id": "mlp__r3_lr_3e-4", "arm": "mlp", "axis": "round3",
        "value": "r3_lr_3e-4", "knobs": dict(_BASELINE, epochs=4000, h_fr=25, lr=3e-4),
    },
    {
        "cell_id": "mlp__r3_lr_1e-3", "arm": "mlp", "axis": "round3",
        "value": "r3_lr_1e-3", "knobs": dict(_BASELINE, epochs=4000, h_fr=25, lr=1e-3),
    },
    {
        "cell_id": "transformer__r3_lr_1e-4", "arm": "transformer", "axis": "round3",
        "value": "r3_lr_1e-4", "knobs": dict(_BASELINE, epochs=4000, h_fr=25, lr=1e-4),
    },
    {
        "cell_id": "transformer__r3_lr_3e-4", "arm": "transformer", "axis": "round3",
        "value": "r3_lr_3e-4", "knobs": dict(_BASELINE, epochs=4000, h_fr=25, lr=3e-4),
    },
    {
        "cell_id": "transformer__r3_lr_1e-3", "arm": "transformer", "axis": "round3",
        "value": "r3_lr_1e-3", "knobs": dict(_BASELINE, epochs=4000, h_fr=25, lr=1e-3),
    },
    {
        "cell_id": "min_ssm__r3_lr_1e-4", "arm": "min_ssm", "axis": "round3",
        "value": "r3_lr_1e-4", "knobs": dict(_BASELINE, epochs=4000, h_fr=25, lr=1e-4),
    },
    {
        "cell_id": "min_ssm__r3_lr_3e-4", "arm": "min_ssm", "axis": "round3",
        "value": "r3_lr_3e-4", "knobs": dict(_BASELINE, epochs=4000, h_fr=25, lr=3e-4),
    },
    {
        "cell_id": "min_ssm__r3_lr_1e-3", "arm": "min_ssm", "axis": "round3",
        "value": "r3_lr_1e-3", "knobs": dict(_BASELINE, epochs=4000, h_fr=25, lr=1e-3),
    },
]

# PROTOCOL.md 16.2(5): screen cells whose exact (ON arm, lr) config already
# ran to completion as a round-2 cell reuse that cell_id VERBATIM -- the
# mechanical map from (arm, lr) to the EXISTING round-2 cell_id; _screen_cell
# falls back to a fresh "{arm}__r3_screen_lr_{...}" id otherwise.
# transformer_k/ssm_k never had a round-2 cell at 4000 updates, so they have
# no entries here (16.2(5)'s own wording).
_ROUND2_SCREEN_TWINS: dict[tuple[str, float], str] = {
    ("mlp_k", 3e-4): "mlp_k__hfr25_updates_4000_lr_3e-4",
    ("mlp_k", 1e-4): "mlp_k__hfr25_updates_4000",
}

# OFF-arm family name (the sweep's own "arm", 16.2(1)) -> its ON-arm twin --
# 16.2's shared-rate rule: the SAME backbone, OFF and ON, never a per-arm
# tuning.
_ROUND3_ON_ARM: dict[str, str] = {"mlp": "mlp_k", "transformer": "transformer_k", "min_ssm": "ssm_k"}


def _fmt_value(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def expand_axes(*, include_round2: bool = False) -> list[dict]:
    """The declared cell list: mlp_k baseline + one axis at a time, then
    transformer_k baseline + its updates confirmation. Deterministic order,
    pure (no I/O, no dataset built, no model constructed) -- mirrors
    ``factorial.expand``'s own contract.

    ``include_round2=True`` appends PROTOCOL.md section 15.4(b)'s nine
    declared cells (``_ROUND2_CELLS``) after the fifteen above. Default is
    ``False``: round 2 requires an explicit ``--round2`` flag (``main``)
    even though the amendment authorizing it is now committed in this tree
    -- defense in depth against launching it by accident.
    """
    cells: list[dict] = []

    def _cell(idx: int, arm: str, axis: str, value, knobs: dict) -> dict:
        cell_id = f"{arm}__{axis}" if axis == "baseline" else f"{arm}__{axis}_{_fmt_value(value)}"
        return {"index": idx, "cell_id": cell_id, "arm": arm, "axis": axis, "value": value, "knobs": knobs}

    idx = 0
    cells.append(_cell(idx, "mlp_k", "baseline", "baseline", dict(_BASELINE)))
    idx += 1
    for axis, key, values in _MLP_AXES:
        for v in values:
            knobs = dict(_BASELINE, **{key: v})
            cells.append(_cell(idx, "mlp_k", axis, v, knobs))
            idx += 1

    tr_baseline = dict(_BASELINE)
    cells.append(_cell(idx, "transformer_k", "baseline", "baseline", dict(tr_baseline)))
    idx += 1
    for v in _TRANSFORMER_UPDATES:
        knobs = dict(tr_baseline, epochs=v)
        cells.append(_cell(idx, "transformer_k", "updates", v, knobs))
        idx += 1

    if include_round2:
        # NOT routed through _cell(): its cell_id is DERIVED from
        # (arm, axis, value), and round 2's cell_ids are the table's own
        # literal names (e.g. "mlp_k__hfr25_updates_2000"), not of the
        # single-axis form that derivation assumes.
        for r2 in _ROUND2_CELLS:
            cells.append({"index": idx, **r2, "knobs": dict(r2["knobs"])})
            idx += 1

    return cells


# --------------------------------------------------------------------------
# Representation stage -- shared across every cell with the same
# (dataset, w_align, view_offset); everything else (updates, h_fr,
# target_params, lr) leaves the representation untouched.
# --------------------------------------------------------------------------


@dataclass
class ProbeContext:
    """Per-invocation caches: one dataset per ``n_episodes`` value, one
    representation (+ its ``cond(Sigma)``) per (dataset, w_align,
    view_offset) -- never rebuilt twice within one ``main()`` call, and
    build-if-absent/load-otherwise across separate invocations (resume).
    """

    datasets: dict[int, dict] = field(default_factory=dict)
    representations: dict[str, tuple] = field(default_factory=dict)  # key -> (artifact, data, cond_sigma)


def _build_dataset(n_episodes: int) -> dict:
    system = DampedPendulum()
    regime = OURegime(theta_ou=1.0)
    return build_dataset(system, regime, None, n_episodes, _T, _DT, _DATA_SEED)


def _get_dataset(ctx: ProbeContext, n_episodes: int) -> dict:
    if n_episodes not in ctx.datasets:
        ctx.datasets[n_episodes] = _build_dataset(n_episodes)
    return ctx.datasets[n_episodes]


def _rep_cfg(w_align: float, view_offset: int) -> dict:
    return {
        "d_z": _D_Z,
        "encoder_hidden": list(_ENCODER_HIDDEN),
        "seed": _REP_SEED,
        "epochs": _REP_EPOCHS,
        "lr": _REP_LR,
        "tau_ema": _REP_TAU_EMA,
        "view_offset": int(view_offset),
        "w_align": float(w_align),
        "w_var": _REP_W_VAR,
        "w_cov": _REP_W_COV,
        "w_recon": _REP_W_RECON,
    }


def _rep_path(data_hash: str, rep_cfg: dict) -> Path:
    rep_hash = hashlib.sha1(json.dumps(rep_cfg, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    path = PROBE_ROOT / "representations" / f"{_STRATUM_SYSTEM}__{_STRATUM_REGIME}__{data_hash}__{rep_hash}.pt"
    return _require_under_root(path)


def _latent_cond_sigma(artifact, data: dict) -> float:
    """``cond(Sigma))`` of the frozen encoder's latent on the held-out
    (test) split -- PROTOCOL.md 14.11. No existing function computes this
    (gate.py's job is the free-rollout error, not this), so it lives here,
    computed once per representation build/load and cached by the caller.
    """
    encoder = FrameMLPEncoder(_D_X, _D_Z, _ENCODER_HIDDEN)
    encoder.load_state_dict(artifact.encoder_state_dict)
    encoder = encoder.double().eval()
    require_cpu_float64_module(encoder, "encoder")
    x = torch.as_tensor(data["test"]["x"], dtype=torch.float64).reshape(-1, _D_X)
    with torch.no_grad():
        z = encoder(x).numpy()
    sigma = np.cov(z, rowvar=False)
    return float(np.linalg.cond(sigma))


def _get_representation(ctx: ProbeContext, n_episodes: int, w_align: float, view_offset: int) -> tuple:
    data = _get_dataset(ctx, n_episodes)
    data_hash = data_fingerprint(data)
    rep_cfg = _rep_cfg(w_align, view_offset)
    key = data_hash + "|" + json.dumps(rep_cfg, sort_keys=True)
    if key not in ctx.representations:
        path = _rep_path(data_hash, rep_cfg)
        artifact = _factorial._load_or_build_representation(
            path, system=_STRATUM_SYSTEM, regime=_STRATUM_REGIME, data=data, rep_cfg=rep_cfg
        )
        cond_sigma = _latent_cond_sigma(artifact, data)
        ctx.representations[key] = (artifact, data, cond_sigma)
    return ctx.representations[key]


# --------------------------------------------------------------------------
# Operator screening (PROTOCOL.md 15.2(iii)/15.4(b)) -- DISPLAY/SCREENING
# ONLY, never a stability certificate (REGLES sections 3, 5). Deliberately
# independent of gate.py: it must not recompute e_h or any gate quantity, so
# a bug here can never silently move a gate verdict.
# --------------------------------------------------------------------------


def _operator_screening(
    run: dict, artifact, data: dict, horizon: int
) -> tuple[float | None, float | None, int | None]:
    """Field of local spectra along a genuine free rollout on the held-out
    (test) split: at every step, the EXACT local operator
    ``predictor.local_operator(h_t, u_t, state_t)`` (never the total
    Jacobian, never a stability object -- REGLES section 3's "field of local
    spectra", never "the spectrum"), read off the SAME family-space
    recurrence ``step`` advances.

    Manual step loop (not ``sld.audit.spectral.local_spectra_field``, which
    raises on a non-finite reading instead of stopping and reporting where):
    stops at the first step whose local operator is non-finite, and never
    touches ``e_h``, a baseline, or the gate arithmetic -- it measures
    something else, full stop.

    Returns:
        ``(rho_max_rollout, phi_norm_at_h, last_finite_step)``:

        - ``rho_max_rollout``: the maximum, over the rollout (worst case
          over episodes), of the spectral radius (``max |eigenvalue|``) of
          the local operator field.
        - ``phi_norm_at_h``: the 2-norm (largest singular value) of the
          composed propagator ``A_bar_k @ ... @ A_bar_1`` at the last finite
          step (worst case over episodes).
        - ``last_finite_step``: the last free-running step (1-indexed,
          matching ``e_h``'s own ``h = 1..H`` convention) whose local
          operator was finite.

        ``(None, None, None)`` for a family with no structured operator
        (``local_operator`` returns ``None`` -- ``mlp``/``transformer``).
    """
    encoder64 = copy.deepcopy(run["encoder"]).double().eval()
    predictor64 = copy.deepcopy(run["predictor"]).double().eval()
    require_cpu_float64_module(encoder64, "encoder")
    require_cpu_float64_module(predictor64, "predictor")

    test = data["test"]
    x0_obs = torch.as_tensor(np.asarray(test["x"])[:, 1, :], dtype=torch.float64)  # index 1 == x_0, gate.py's _T0
    u = torch.as_tensor(np.asarray(test["u"]), dtype=torch.float64)
    u_seq = u[:, 1 : 1 + horizon, :].transpose(0, 1).contiguous()  # (H, B, d_u), time-major

    with torch.no_grad():
        z0 = encoder64(x0_obs)
        h = predictor64.lift(z0) if hasattr(predictor64, "lift") else z0

        state = PredictorState()
        B, d_h = h.shape
        phi = torch.eye(d_h, dtype=torch.float64).unsqueeze(0).repeat(B, 1, 1)
        rho_max: float | None = None
        last_finite_step = 0
        for t in range(horizon):
            op = predictor64.local_operator(h, u_seq[t], state)
            if op is None:
                return None, None, None
            A_bar, _ = op
            if not torch.isfinite(A_bar).all():
                break
            eig = np.linalg.eigvals(A_bar.numpy())
            step_rho = float(np.max(np.abs(eig)))
            rho_max = step_rho if rho_max is None else max(rho_max, step_rho)
            phi = torch.matmul(A_bar, phi)
            last_finite_step = t + 1
            h, state = predictor64.step(h, u_seq[t], state)

    phi_np = phi.numpy()
    phi_norm = max(float(np.linalg.norm(phi_np[b], ord=2)) for b in range(B))
    return rho_max, phi_norm, last_finite_step


# --------------------------------------------------------------------------
# Predictor stage + gate evaluation
# --------------------------------------------------------------------------


def _trainer_config(cell: dict) -> dict:
    knobs = cell["knobs"]
    config = {
        "exp_name": cell["cell_id"],
        "seed": _MODEL_SEED,
        "family": cell["arm"],
        "d_x": _D_X,
        "d_z": _D_Z,
        "d_u": _D_U,
        "target_params": knobs["target_params"],
        "epochs": knobs["epochs"],
        "batch_episodes": _BATCH_EPISODES,
        "lr": knobs["lr"],
        "h_fr": knobs["h_fr"],
        "results_root": str(PROBE_ROOT / "runs"),
        "stratum_system": _STRATUM_SYSTEM,
        "stratum_regime": _STRATUM_REGIME,
    }
    # PROTOCOL.md section 16 round-3 OFF arms (mlp/transformer/min_ssm):
    # `delta` is a structured-arm-only knob -- trainer._FAMILY_CONFIG_KEYS is
    # the single source of truth for which family accepts it (never a second,
    # drifting list here); an OFF family carrying it is refused loudly by
    # Trainer.__init__ ("applied or rejected, never silently ignored").
    # Every family this module ran BEFORE round 3 (mlp_k/transformer_k/
    # ssm_k) accepts it unconditionally, so this branch is a no-op for them
    # -- the dict is byte-for-byte what it was before this fix, preserving
    # config_hash (sort_keys=True downstream, so key order here is
    # irrelevant) and therefore resume for the 24 existing cells.
    if "delta" in _FAMILY_CONFIG_KEYS[cell["arm"]]:
        config["delta"] = _DELTA
    # PROTOCOL.md 17.1/17.2 (Amendment 8): grad_clip is an OPTIONAL knob on
    # BOTH Trainer and this cell's own knobs dict -- only round-3c cells
    # carry it, so no existing cell (whose knobs never have this key) moves
    # its config or its config_hash.
    if "grad_clip" in knobs:
        config["grad_clip"] = knobs["grad_clip"]
    return config


def _write_eval_failure(
    run_dir: Path,
    *,
    config_hash: str,
    data_hash: str,
    representation_digest: str,
    horizon: int,
    first_nonfinite_step: int | None,
    reason: str,
) -> Path:
    """PROTOCOL.md section 15.3's durable ``eval_failure.json``: atomic write
    (tmp file + ``os.replace``, the same pattern ``_write_report`` already
    uses for ``report.json``), into the run directory. Idempotent -- re-
    evaluating the same checkpoint against the same split overwrites this
    file with the SAME content, never a second, drifting copy (evaluation is
    a pure function of a fixed checkpoint, section 15.3).
    """
    path = _require_under_root(run_dir / "eval_failure.json")
    payload = {
        "config_hash": config_hash,
        "data_hash": data_hash,
        "representation_digest": representation_digest,
        "horizon": horizon,
        "first_nonfinite_step": first_nonfinite_step,
        "reason": reason,
        "rule": "15.3",
    }
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp_path, path)
    return path


def _run_cell(ctx: ProbeContext, cell: dict) -> dict:
    """Resolve one cell's representation + predictor, train-or-skip
    (factorial's own sentinel discipline -- ``_factorial._validate_completion_sentinel``),
    and score it through ``gate.py``. Every value under ``e_*``/``m_*``/
    ``verdict``/``near_miss`` below is READ from ``evaluate_run_free_rollout``'s
    return value, never recomputed (R1/R2).

    A training-time divergence (PROTOCOL.md 11.7) is classified the SAME way
    as an evaluation-time one (15.3) below: by the existence of the
    trainer's own terminal artifact (``failure.json``), never by matching
    ``fit``'s exception message. ``load_run``/``_operator_screening``/
    ``evaluate_run_free_rollout`` all need a genuine checkpoint, so none of
    them run on that path -- 11.7's own wording, "aborts without success
    artifacts".
    """
    knobs = cell["knobs"]
    artifact, data, cond_sigma = _get_representation(
        ctx, knobs["n_episodes"], knobs["w_align"], knobs["view_offset"]
    )

    trainer_cfg = _trainer_config(cell)
    trainer = Trainer(trainer_cfg, representation=artifact)
    run_dir = trainer.resolve_run_dir(data)
    _require_under_root(run_dir)
    failure_path = run_dir / FAILURE_FILENAME

    t0 = time.perf_counter()
    sentinel = _factorial._validate_completion_sentinel(run_dir)
    if sentinel is None and failure_path.exists():
        # PROTOCOL.md 11.7: "resume preserves it and does not retrain" -- a
        # terminal failure.json from a PRIOR invocation of this exact cell,
        # classified by its existence alone (same object-not-message
        # doctrine as the RolloutDivergence handling below, 15.3). fit() is
        # never called.
        status = "divergence_training"
    elif sentinel is None:
        try:
            trainer.fit(data)
        except RuntimeError:
            if not failure_path.exists():
                # No terminal artifact from the trainer: an infrastructure
                # defect (OOM, a bad config, ...), not a diagnosed
                # divergence -- re-raised as-is so the campaign dies loudly
                # instead of mislabeling it (mirrors 15.3's corruption/
                # divergence split, on the training side).
                raise
            status = "divergence_training"
        else:
            status = "trained"
    else:
        status = "skipped_resume"
    wall_clock_s = time.perf_counter() - t0

    # PROTOCOL.md section 15.1: tau_1 = e_1_pers / gamma_1, superseding 12.3's
    # 3*e_recon for D3/D4 onward. e_1_pers is the artifact's own (immutable,
    # digest-pinned) D2 measurement -- this rule is computed by the CONSUMER
    # (this module), never by regenerating the artifact. tau1_condition_holds
    # below stays exactly as it was: the historical trace of the OLD (12.3)
    # condition, recorded on the artifact at D2, never rewritten here.
    # These are all properties of the REPRESENTATION artifact (built before
    # the predictor stage, shared across cells), so they are computable and
    # filled in even when the predictor's own training diverged below.
    e_1_pers = float(artifact.manifest["e_1_pers"])
    e_recon = float(artifact.e_recon)
    tau_1 = protocol.tau_1(e_1_pers)
    tau1_validity = protocol.tau1_validity(e_recon, e_1_pers)

    common = {
        "cell_id": cell["cell_id"],
        "arm": cell["arm"],
        "axis": cell["axis"],
        "value": cell["value"],
        "knobs": knobs,
        "config_hash": trainer.config_hash,
        "data_hash": trainer.data_hash,
        "run_dir": str(run_dir),
        "wall_clock_s": wall_clock_s,
        "e_recon": e_recon,
        "e_1_pers": e_1_pers,
        "tau1_condition_holds": bool(artifact.manifest["gated"]),  # jamais .get avec defaut
        "tau_1": tau_1,
        "tau1_rule": "15.1",
        "tau1_validity": tau1_validity,
        "readout_r2": float(artifact.readout.r2),
        "readout_rank": int(artifact.readout.rank),
        "readout_cond": float(artifact.readout.cond),
        "readout_indeterminate": bool(artifact.readout.indeterminate),
        "latent_cond_sigma": float(cond_sigma),
    }

    if status == "divergence_training":
        # 11.7's own durable artifact, read (never rewritten, never
        # replaced) for its detail string. This is NOT 15.3's
        # eval_failure.json -- that governs evaluation only, and is never
        # written on this path: the trainer's failure.json already IS the
        # terminal artifact 11.7 requires.
        failure_payload = json.loads(failure_path.read_text())
        return {
            **common,
            "status": status,
            "divergence_detail": (
                f"training diverged (11.7): {failure_payload['reason']} "
                f"at epoch {failure_payload['epoch']}"
            ),
            "first_nonfinite_step": None,
            "e_1": None, "e_25": None, "e_25_persistence": None, "e_25_linear": None,
            "m_low": None, "m_high": None, "verdict": "divergence", "near_miss": None,
            "rho_max_rollout": None, "phi_norm_at_h": None,
        }

    # PROTOCOL.md 15.2(iii)/15.4(b): the field-of-local-spectra screening,
    # independent of gate.py, computed unconditionally on a genuinely
    # trained/resumed run (unlike the e_*/m_* gate quantities below, this is
    # not fabricated by a divergence -- it is informative EVEN WHEN the
    # model's decoded rollout diverges). Never reached on the
    # divergence_training path above: there is no checkpoint to load.
    run = load_run(run_dir)
    rho_max_rollout, phi_norm_at_h, _last_finite_step = _operator_screening(run, artifact, data, _H_GATE)
    common["rho_max_rollout"] = rho_max_rollout
    common["phi_norm_at_h"] = phi_norm_at_h

    try:
        result = evaluate_run_free_rollout(
            run_dir, artifact, data["test"], horizon=_H_GATE, tau=_TAU_VPT, h_gate=_H_GATE,
            gamma=_GAMMA, tau_1=tau_1,
        )
    except RolloutDivergence as exc:
        # PROTOCOL.md section 15.3: `divergence (evaluation)`, distinct from
        # `ok`/`missing`/section 11.7's training-time divergence -- terminal
        # at the CELL level, published with its numbers, never replaced,
        # never averaged, licenses NO inference about the others. Typed now
        # (gate.py's RolloutDivergence, raised ONLY when the tensor that came
        # from the MODEL went non-finite): a corrupted target/sigma_train is
        # a real defect, a plain ValueError, and still propagates uncaught --
        # the campaign dies loudly instead of silently mislabeling it.
        _write_eval_failure(
            run_dir,
            config_hash=trainer.config_hash,
            data_hash=trainer.data_hash,
            representation_digest=_representation_digest(artifact),
            horizon=_H_GATE,
            first_nonfinite_step=exc.first_step,
            reason=str(exc),
        )
        return {
            **common,
            "status": "divergence_rollout",
            "divergence_detail": str(exc),
            "first_nonfinite_step": exc.first_step,
            "e_1": None, "e_25": None, "e_25_persistence": None, "e_25_linear": None,
            "m_low": None, "m_high": None, "verdict": "divergence", "near_miss": None,
        }
    pc = result.positive_control

    return {
        **common,
        "status": status,
        "first_nonfinite_step": None,
        "e_1": float(result.e_h[0]),
        "e_25": float(result.e_h[_H_GATE - 1]),
        "e_25_persistence": float(result.persistence_e_h[_H_GATE - 1]),
        "e_25_linear": float(result.linear_e_h[_H_GATE - 1]),
        "m_low": float(pc.m_low),
        "m_high": float(pc.m_high),
        "verdict": pc.verdict,
        "near_miss": bool(pc.near_miss),
    }


def _val_reading(run_dir: Path, artifact, data: dict) -> dict:
    """PROTOCOL.md 16.4's reading rule: every round-3 selection/retention
    reading is taken on the val split, never test -- the SAME
    ``evaluate_run_free_rollout`` call ``_run_cell`` makes for its test-split
    gate reading above, split swapped, same ``tau_1`` (the representation
    artifact's own 15.1 value, recomputed here the same way ``_run_cell``
    does -- never a second definition of ``tau_1``).

    On ``RolloutDivergence``: the four val fields go ``None``, and the
    divergence is reported in the two extra keys below. UNLIKE 15.3's
    ``eval_failure.json`` (which belongs to the test-split gate reading
    ``_run_cell`` owns), no durable artifact is written here -- a val-split
    divergence is published in the round-3 report only (supervisor decision:
    15.3's artifact governs the GATE reading on test, a decision this
    amendment does not touch).
    """
    tau_1 = protocol.tau_1(float(artifact.manifest["e_1_pers"]))
    try:
        result = evaluate_run_free_rollout(
            run_dir, artifact, data["val"], horizon=_H_GATE, tau=_TAU_VPT, h_gate=_H_GATE,
            gamma=_GAMMA, tau_1=tau_1,
        )
    except RolloutDivergence as exc:
        return {
            "e_1_val": None, "e_25_val": None, "m_low_val": None, "m_high_val": None,
            "val_divergence": str(exc), "first_nonfinite_step_val": exc.first_step,
        }
    pc = result.positive_control
    return {
        "e_1_val": float(result.e_h[0]),
        "e_25_val": float(result.e_h[_H_GATE - 1]),
        "m_low_val": float(pc.m_low),
        "m_high_val": float(pc.m_high),
    }


def _run_cell_with_val(ctx: ProbeContext, cell: dict) -> dict:
    """Round-3 scope only (PROTOCOL.md 16.4): ``_run_cell``'s existing
    test-split gate reading, PLUS a val-split reading (``_val_reading``).
    Round 1/2 cells never go through this -- only the sweep/screen/stage-3
    cells 16.4 lists -- so val fields exist ONLY where a round-3 decision
    actually reads them.

    A training-time divergence (status ``"divergence_training"``) leaves no
    checkpoint to roll out (11.7): the val fields are filled with ``None``
    directly, ``_val_reading`` is never called. Every other status --
    including ``"divergence_rollout"`` (the TEST-split rollout diverged, not
    necessarily the val one) -- has a genuine checkpoint, so the val split
    gets its own, independent reading.
    """
    record = _run_cell(ctx, cell)
    if record["status"] == "divergence_training":
        return {**record, "e_1_val": None, "e_25_val": None, "m_low_val": None, "m_high_val": None}
    knobs = cell["knobs"]
    artifact, data, _cond_sigma = _get_representation(
        ctx, knobs["n_episodes"], knobs["w_align"], knobs["view_offset"]
    )
    val = _val_reading(Path(record["run_dir"]), artifact, data)
    return {**record, **val}


# --------------------------------------------------------------------------
# Round 3 (PROTOCOL.md section 16, Amendment 7): LR re-selection, the ON-arm
# finitude screen, and the two stage-3 cells. Every "which comes next"
# decision below is a PURE function of already-computed records -- no
# training, no I/O -- so it is unit-testable without running anything; only
# main() wires the pure decisions to the live run_cell_fn.
# --------------------------------------------------------------------------

_ROUND3_BACKBONES: tuple[str, ...] = ("mlp", "transformer", "min_ssm")


def _select_lr(sweep_records: list[dict], backbones: tuple[str, ...] = _ROUND3_BACKBONES) -> dict:
    """PROTOCOL.md 16.2(2): per backbone, the listed rate minimizing the OFF
    arm's ``e_25_val`` among the sweep cells that (a) did not fail to train
    or to evaluate (status not ``divergence_training``/``divergence_rollout``)
    and (b) evaluated finitely on val (no ``"val_divergence"`` key). A
    diverged cell is excluded from the argmin even if its ``e_25_val`` would
    be minimal -- "a diverged run cannot win" (16.2(2)). Exhaustion (no
    eligible cell for a backbone): ``selected_lr`` is ``None``, published,
    never inferred.

    Tiebreak (mechanical, NOT specified by 16.2 -- declared here, published
    in the report): on an exact ``e_25_val`` tie, the LOWEST listed rate
    wins.

    ``backbones`` defaults to round 3's three (regression-safe for existing
    callers); PROTOCOL.md 17.2 restricts round 3c's ssm-only re-selection to
    a single backbone by passing ``("min_ssm",)`` here -- same mechanics, no
    duplicated argmin logic.
    """
    selection: dict[str, dict] = {}
    for backbone in backbones:
        backbone_records = [r for r in sweep_records if r["arm"] == backbone]
        basis = {r["knobs"]["lr"]: r["e_25_val"] for r in backbone_records}
        eligible = [
            r for r in backbone_records
            if r["status"] not in ("divergence_training", "divergence_rollout")
            and "val_divergence" not in r
            and r["e_25_val"] is not None
        ]
        if not eligible:
            selection[backbone] = {"selected_lr": None, "basis": basis}
            continue
        best = min(eligible, key=lambda r: (r["e_25_val"], r["knobs"]["lr"]))
        selection[backbone] = {"selected_lr": best["knobs"]["lr"], "basis": basis}
    return selection


def _next_lower_lr(lr: float) -> float | None:
    """The previous element of ``_ROUND3_LR_LIST``, or ``None`` at the
    lowest rate (16.2(4)'s exhaustion boundary)."""
    idx = _ROUND3_LR_LIST.index(lr)
    return _ROUND3_LR_LIST[idx - 1] if idx > 0 else None


def _screen_cell(arm: str, lr: float) -> dict:
    """PROTOCOL.md 16.2(3)/16.4 stage 2: the ON-arm finitude screen cell for
    ``arm`` (e.g. ``mlp_k``) at ``lr``. Dedup (16.2(5)): if this exact (arm,
    lr) already ran to completion as a round-2 cell, its cell_id is reused
    VERBATIM -- ``Trainer.resolve_run_dir`` then resolves to the SAME
    run_dir, and the existing sentinel/resume discipline (``_run_cell``)
    does the skip (status ``"skipped_resume"``), never a special case here.
    """
    cell_id = _ROUND2_SCREEN_TWINS.get((arm, lr), f"{arm}__r3_screen_lr_{_LR_LABEL[lr]}")
    knobs = dict(_BASELINE, epochs=4000, h_fr=25, lr=lr)
    return {"cell_id": cell_id, "arm": arm, "axis": "round3_screen", "value": _LR_LABEL[lr], "knobs": knobs}


def _run_screen(
    arm: str,
    start_lr: float,
    run_cell_fn: Callable[[dict], dict],
    cell_builder: Callable[[str, float], dict] = _screen_cell,
) -> dict:
    """PROTOCOL.md 16.2(3)/(4): the ON-arm finitude screen loop, decision
    separated from execution. Only training COMPLETION gates the loop --
    11.7's ``failure.json`` absence -- an evaluation-time (rollout)
    divergence is NOT a screen failure ("No metric of the screen run is
    consulted for selection; only completion is", 16.2(3)): any status other
    than ``"divergence_training"`` ends the loop successfully at that rate.

    ``run_cell_fn`` is the injected execution step (cell -> record) -- a fake
    in tests, ``_run_cell_with_val`` bound to a live ctx in production -- so
    this decision loop is unit-testable with canned records, never a real
    ``Trainer.fit``. ``cell_builder`` defaults to ``_screen_cell`` (round
    2/3's dedup-aware builder, regression-safe for existing callers);
    PROTOCOL.md 17.2's round-3c screen (row 5) injects ``_r3c_screen_cell``
    instead -- same descent/exhaustion mechanics, a different cell shape.

    Returns ``{"screened_lr": float | None, "trail": [record, ...]}``:
    ``trail`` holds every attempted screen record in attempt order;
    ``screened_lr`` is ``None`` on exhaustion (fails even at the lowest
    listed rate, 16.2(4)).
    """
    lr = start_lr
    trail: list[dict] = []
    while True:
        record = run_cell_fn(cell_builder(arm, lr))
        trail.append(record)
        if record["status"] != "divergence_training":
            return {"screened_lr": lr, "trail": trail}
        next_lr = _next_lower_lr(lr)
        if next_lr is None:
            return {"screened_lr": None, "trail": trail}
        lr = next_lr


def _decide_budget(rung_rec: dict, twin_rec: dict) -> int:
    """PROTOCOL.md 16.1's rung criterion, pure: 8000 is retained iff the rung
    (mlp_k, 8000 updates, h_fr=25, the screened rate) completed training and
    evaluated finitely on val AND its ``e_25_val`` is at least 10% better
    (relative) than the twin's -- exact equality at 0.9 counts as retained
    (16.1: "exactly 10% ... retained"). Any divergence anywhere in the
    comparison -- the rung's or the twin's -- keeps 4000, never inferred.
    """
    if rung_rec["status"] in ("divergence_training", "divergence_rollout"):
        return 4000
    if rung_rec.get("e_25_val") is None or "val_divergence" in rung_rec:
        return 4000
    if twin_rec.get("e_25_val") is None or "val_divergence" in twin_rec:
        return 4000
    if rung_rec["e_25_val"] <= 0.9 * twin_rec["e_25_val"]:
        return 8000
    return 4000


def _decide_walign(cell_rec: dict, twin_rec: dict) -> bool:
    """PROTOCOL.md 16.3's two-sided retention rule, pure: w_align=0 is
    retained iff the cell (mlp_k, 4000 updates, h_fr=25, the screened rate,
    the w_align=0 encoder) completed (training AND evaluation both finite)
    AND strictly improves ``e_25_val`` over the twin AND does not worsen
    ``m_high_val``. Mixed evidence -- any divergence, a partial improvement
    -- keeps the shipped default (w_align=1), never inferred.
    """
    if cell_rec["status"] in ("divergence_training", "divergence_rollout"):
        return False
    if cell_rec.get("e_25_val") is None or "val_divergence" in cell_rec:
        return False
    if twin_rec.get("e_25_val") is None or "val_divergence" in twin_rec:
        return False
    return cell_rec["e_25_val"] < twin_rec["e_25_val"] and cell_rec["m_high_val"] <= twin_rec["m_high_val"]


def _upsert_record(records: list[dict], record: dict) -> None:
    """Insert ``record`` by ``cell_id`` identity: replaces an existing entry
    for the SAME cell_id in place (position preserved) instead of appending
    a duplicate; appends when the cell_id is new. _REPORT_SCHEMA's own
    contract: "cell_id: arm__axis[_value], UNIQUE per cell".

    PROTOCOL.md 16.2(5)'s dedup means a round-3 screen cell can carry a
    cell_id that ALREADY has a record in ``records`` (the round-2 twin,
    appended by the round1/2 loop above, with no val fields) -- that
    artifact is cited, not re-executed, and this call site's own
    ``_run_cell_with_val`` result is the SAME cell with the val-split
    reading added, so it REPLACES the round-2 entry rather than sitting next
    to it as a second row for the same cell.
    """
    for i, existing in enumerate(records):
        if existing["cell_id"] == record["cell_id"]:
            records[i] = record
            return
    records.append(record)


_ROUND3_SCHEMA = {
    "lr_list": "PROTOCOL.md 16.2(1)'s ordered rate list, ascending -- 'the next lower listed rate' "
    "(16.2(3)) means the previous element of this list",
    "tie_rule": "the mechanical tiebreak this module applies on an exact e_25_val tie in the "
    "stage-1 sweep argmin (NOT specified by 16.2 itself): the lowest listed rate wins",
    "selected_lr": "per backbone (mlp/transformer/min_ssm): {selected_lr (16.2(2)'s argmin over "
    "the sweep, null on exhaustion), basis (e_25_val per swept lr, including diverged/null "
    "entries), screen_trail (16.2(3)'s attempted screen cells in order, each "
    "{lr, cell_id, status}), screened_lr (16.2(3)/(4)'s outcome -- the lr the ON arm's screen "
    "actually completed training at, null on exhaustion at the lowest listed rate)}",
    "budget": "16.1's rung decision: {entry_value (4000 or 8000), criterion, e_25_val_rung, "
    "e_25_val_twin} -- null when mlp has no screened lr (stage 3 does not run, 16.4 stage 3)",
    "w_align": "16.3's retention decision: {retained (bool), criterion, e_25_val, e_25_val_twin, "
    "m_high_val, m_high_val_twin} -- null when mlp has no screened lr",
}


# --------------------------------------------------------------------------
# Round 3c (PROTOCOL.md section 17, Amendment 8): the gradient guard's no-op
# verification cell, the clipped ssm sweep, and the ssm ON screen -- same
# pure-decision-vs-injected-execution discipline as round 3 above.
# --------------------------------------------------------------------------

_R3C_GRAD_CLIP: float = 100.0  # PROTOCOL.md 17.1's uniform clip norm

# PROTOCOL.md 17.2 table, row 1: the no-op gate cell -- mlp__r3_lr_1e-3's
# EXACT knobs plus grad_clip, so a clip that never binds reproduces that
# twin exactly (multiplying gradients by 1.0 changes nothing).
_ROUND3C_NOOP_CELL: dict = {
    "cell_id": "mlp__r3c_lr_1e-3", "arm": "mlp", "axis": "round3c",
    "value": "r3c_lr_1e-3",
    "knobs": dict(_BASELINE, epochs=4000, h_fr=25, lr=1e-3, grad_clip=_R3C_GRAD_CLIP),
}

# PROTOCOL.md 17.2 table, rows 2-4: the clipped ssm sweep -- a CLOSED list,
# reproduced VERBATIM (same precedent as _ROUND2_CELLS/_ROUND3_SWEEP).
_ROUND3C_SWEEP: list[dict] = [
    {
        "cell_id": "min_ssm__r3c_lr_1e-4", "arm": "min_ssm", "axis": "round3c",
        "value": "r3c_lr_1e-4",
        "knobs": dict(_BASELINE, epochs=4000, h_fr=25, lr=1e-4, grad_clip=_R3C_GRAD_CLIP),
    },
    {
        "cell_id": "min_ssm__r3c_lr_3e-4", "arm": "min_ssm", "axis": "round3c",
        "value": "r3c_lr_3e-4",
        "knobs": dict(_BASELINE, epochs=4000, h_fr=25, lr=3e-4, grad_clip=_R3C_GRAD_CLIP),
    },
    {
        "cell_id": "min_ssm__r3c_lr_1e-3", "arm": "min_ssm", "axis": "round3c",
        "value": "r3c_lr_1e-3",
        "knobs": dict(_BASELINE, epochs=4000, h_fr=25, lr=1e-3, grad_clip=_R3C_GRAD_CLIP),
    },
]


def _r3c_screen_cell(arm: str, lr: float) -> dict:
    """PROTOCOL.md 17.2 table, row 5: the ssm ON screen cell for ``arm`` at
    ``lr``, clipped. NO round-2 dedup (unlike ``_screen_cell``) -- a clipped
    config never existed before this amendment, so every round-3c screen
    cell is genuinely executed, never cited as an existing artifact.
    """
    knobs = dict(_BASELINE, epochs=4000, h_fr=25, lr=lr, grad_clip=_R3C_GRAD_CLIP)
    cell_id = f"{arm}__r3c_screen_lr_{_LR_LABEL[lr]}"
    return {"cell_id": cell_id, "arm": arm, "axis": "round3c_screen", "value": _LR_LABEL[lr], "knobs": knobs}


def _read_metrics_columns(run_dir, columns: tuple[str, ...]) -> dict[str, list[float]]:
    """``columns`` of ``run_dir``/metrics.csv, as float lists, in row order.
    Minimal and scoped: ``_noop_gate_check`` below is the only caller."""
    with open(Path(run_dir) / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return {col: [float(row[col]) for row in rows] for col in columns}


def _noop_gate_check(r3c_record: dict, r3_record: dict) -> dict:
    """PROTOCOL.md 17.2's pre-declared no-op gate (table row 1): a clip that
    never binds multiplies gradients by exactly 1.0, so the r3c cell must
    reproduce its r3 twin EXACTLY -- the SAME ``train_loss``/``val_loss``
    series in metrics.csv (the r3c run's own extra ``grad_norm`` column is
    never compared -- "colonnes communes seulement") AND the same
    ``e_25_val``. Equal -> every section-16 retained value stands
    unconditionally. Not equal -> STOP (main(), 17.2): cells 2-5 do not run.
    No third path.
    """
    r3c_losses = _read_metrics_columns(r3c_record["run_dir"], ("train_loss", "val_loss"))
    r3_losses = _read_metrics_columns(r3_record["run_dir"], ("train_loss", "val_loss"))
    losses_equal = r3c_losses == r3_losses
    e25_equal = r3c_record["e_25_val"] == r3_record["e_25_val"]
    equal = losses_equal and e25_equal
    return {
        "equal": equal,
        "losses_equal": losses_equal,
        "e_25_val_equal": e25_equal,
        "e_25_val_r3c": r3c_record["e_25_val"],
        "e_25_val_r3": r3_record["e_25_val"],
    }


_ROUND3C_SCHEMA = {
    "grad_clip": "PROTOCOL.md 17.1's uniform clip norm (100.0) applied to every round-3c cell",
    "noop": "17.2's pre-declared no-op gate on cell 1 (mlp__r3c_lr_1e-3 vs its mlp__r3_lr_1e-3 "
    "twin): {equal, losses_equal (train_loss/val_loss series, metrics.csv), e_25_val_equal, "
    "e_25_val_r3c, e_25_val_r3} -- equal=False STOPS cells 2-5 (17.2, no third path)",
    "selected_lr_ssm": "null while the noop gate has not passed; otherwise {basis (min_ssm's "
    "e_25_val per swept lr, 17.2 rows 2-4), screen_trail (the ssm_k ON screen's attempted "
    "cells, each {lr, cell_id, status}), screened_lr (the screen's outcome, null on exhaustion)}",
}


# --------------------------------------------------------------------------
# Report + CLI
# --------------------------------------------------------------------------

_REPORT_SCHEMA = {
    "cell_id": "arm__axis[_value], unique per cell",
    "arm": "mlp_k or transformer_k",
    "axis": "which axis this cell varies off _BASELINE ('baseline' for the anchor cell)",
    "value": "the axis's value at this cell",
    "knobs": "the full resolved {epochs,h_fr,target_params,n_episodes,lr,w_align,view_offset}",
    "config_hash": "Trainer.config_hash for this cell's predictor run",
    "data_hash": "fingerprint of the dataset this cell trained/evaluated on",
    "run_dir": "where the predictor run's artifacts live, under results_probe_d/runs/",
    "status": "'trained' (this invocation ran it), 'skipped_resume' (already complete on disk), "
    "'divergence_rollout' (15.3, evaluation-time divergence -- durable artifact = this module's own "
    "eval_failure.json) or 'divergence_training' (11.7, training-time divergence -- durable artifact "
    "= the trainer's own failure.json, classified by its existence, never retrained on resume)",
    "wall_clock_s": "seconds spent in the train-or-skip decision this invocation",
    "e_1": "gate.py's e_h at h=1 (free-running)",
    "e_25": "gate.py's e_h at h=H_gate=25 (free-running) -- PROTOCOL.md 12.3's primary score",
    "e_25_persistence": "persistence baseline's e_h at h=25, same coordinates/episodes/sigma",
    "e_25_linear": "linear-extrapolation baseline's e_h at h=25",
    "m_low": "12.4 margin ratio, lower bound (< 1 is a pass)",
    "m_high": "12.4 margin ratio, upper bound (< 1 is a pass)",
    "verdict": "gate.py's positive-control verdict: 'learns' or 'does_not_learn'",
    "near_miss": "14.4's near-miss flag",
    "e_recon": "the stratum's frozen readout reconstruction error (12.3's OLD tau_1 floor)",
    "e_1_pers": "persistence baseline's 1-step error on the representation's own eval split",
    "tau1_condition_holds": "12.3's OLD pre-registered sanity check: 3*e_recon < e_1_pers -- "
    "historical trace recorded on the artifact at D2, never rewritten by this rule's own "
    "15.1 replacement below",
    "tau_1": "the upper bound ACTUALLY used to gate this cell -- PROTOCOL.md 15.1's "
    "e_1_pers/gamma_1, supersedes 12.3's 3*e_recon for D3/D4 onward",
    "tau1_rule": "which rule produced tau_1 above; always '15.1' from CORR-2 onward",
    "tau1_validity": "15.1's two validity conditions, {'attainable': tau_1 > e_recon, "
    "'discriminant': tau_1 < e_1_pers}, both strict -- checked at D2, published, never adjusted",
    "readout_r2": "frozen linear readout's R^2",
    "readout_rank": "frozen linear readout's numerical rank",
    "readout_cond": "frozen linear readout's condition number",
    "readout_indeterminate": "frozen linear readout's indeterminacy verdict",
    "latent_cond_sigma": "cond(Sigma) of the frozen encoder's latent on the held-out split (14.11)",
    "first_nonfinite_step": "RolloutDivergence.first_step (15.3) when status == "
    "'divergence_rollout', else null",
    "rho_max_rollout": "15.4(b) SCREENING ONLY, never a stability certificate (REGLES 3/5): max "
    "over the rollout of the field-of-local-spectra's spectral radius, worst case over episodes; "
    "null for a family with no structured operator",
    "phi_norm_at_h": "15.4(b) SCREENING ONLY, never a stability certificate (REGLES 3/5): 2-norm "
    "of the composed local-operator propagator at the last finite step, worst case over episodes; "
    "null for a family with no structured operator",
    "e_1_val": "round-3 scope only (PROTOCOL.md 16.4): gate.py's e_h at h=1 on the VAL split -- "
    "every round-3 selection/retention reading is taken on val, never test (16.4)",
    "e_25_val": "round-3 scope only: e_h at h=H_gate=25 on the VAL split -- the value 16.2's "
    "argmin, 16.1's rung criterion, and 16.3's retention rule all read",
    "m_low_val": "round-3 scope only: 12.4's margin ratio, lower bound, on the VAL split",
    "m_high_val": "round-3 scope only: 12.4's margin ratio, upper bound, on the VAL split -- the "
    "value 16.3's retention rule reads",
    "val_divergence": "round-3 scope only: present (str(RolloutDivergence)) iff the VAL-split "
    "rollout diverged -- unlike 15.3's eval_failure.json (test split), no durable artifact is "
    "written for this; published in the report only",
    "first_nonfinite_step_val": "round-3 scope only: RolloutDivergence.first_step for the VAL-split "
    "rollout when val_divergence is present, else absent",
}


def _fmt(value, spec: str) -> str:
    """Format a probe field, or ``DIVERGED`` when it is absent.

    A diverged cell carries ``None`` for every numeric field ON PURPOSE:
    PROTOCOL.md section 11.7 forbids fabricating a number for a rollout that
    left the finite range. Three separate printers in this file crashed on
    that ``None`` before this helper existed -- an output path that cannot
    render a DECLARED terminal state is a reporting bug, not a data problem.
    """
    return "DIVERGED" if value is None else format(value, spec)


def _print_table(records: list[dict]) -> None:
    header = ("cell_id", "e_1", "e_25", "verdict", "m_low", "m_high", "e_recon", "cond_sigma", "r2", "wall_s")
    rows = [header]
    for r in records:
        rows.append(
            (
                r["cell_id"],
                _fmt(r["e_1"], ".4f"),
                _fmt(r["e_25"], ".4f"),
                r["verdict"],
                _fmt(r["m_low"], ".3f"),
                _fmt(r["m_high"], ".3f"),
                _fmt(r["e_recon"], ".4f"),
                _fmt(r["latent_cond_sigma"], ".3g"),
                _fmt(r["readout_r2"], ".4f"),
                _fmt(r["wall_clock_s"], ".1f"),
            )
        )
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(header))]
    for row in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))


def _write_report(
    records: list[dict], *, round3: dict | None = None, round3c: dict | None = None
) -> Path:
    report_path = _require_under_root(PROBE_ROOT / "report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = report_path.with_name(report_path.name + ".tmp")
    payload = {"schema": _REPORT_SCHEMA, "cells": records}
    if round3 is not None:
        payload["round3"] = round3
    if round3c is not None:
        payload["round3c"] = round3c
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp_path, report_path)
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sld-probe-d")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--round2", action="store_true",
        help="also run PROTOCOL.md section 15.4(b)'s nine declared round-2 cells",
    )
    parser.add_argument(
        "--round3", action="store_true",
        help="also run PROTOCOL.md section 16 (Amendment 7)'s probe round 3 -- the LR "
        "re-selection sweep, the ON-arm finitude screen, and the budget/w_align stage-3 cells. "
        "Implies round 1+2's scope (report.json stays the single report of the whole campaign) "
        "even though the amendment authorizing it is committed in this tree -- defense in depth "
        "against launching it by accident, same doctrine as --round2",
    )
    parser.add_argument(
        "--round3c", action="store_true",
        help="also run PROTOCOL.md section 17 (Amendment 8)'s probe round 3c -- the gradient-"
        "clip no-op gate, the clipped ssm sweep, and the ssm ON screen. Implies round 1+2+3's "
        "scope (report.json stays the single report of the whole campaign) even though the "
        "amendment authorizing it is committed in this tree -- defense in depth against "
        "launching it by accident, same doctrine as --round2/--round3",
    )
    args = parser.parse_args(argv)

    cells = expand_axes(include_round2=args.round2 or args.round3 or args.round3c)

    if args.dry_run:
        for c in cells:
            print(f"[{c['index']}] {c['cell_id']} arm={c['arm']} axis={c['axis']} value={c['value']} knobs={c['knobs']}")
        print(f"total: {len(cells)} cells")
        return 0

    ctx = ProbeContext()
    records = []
    for c in cells:
        record = _run_cell(ctx, c)
        # a diverged cell carries e_25 = None on purpose (section 11.7: no
        # number is fabricated for a rollout that left the finite range)
        e25 = record["e_25"]
        e25_str = "DIVERGED" if e25 is None else f"{e25:.4f}"
        print(
            f"[{c['index'] + 1}/{len(cells)}] {c['cell_id']} -> status={record['status']} "
            f"e_25={e25_str} verdict={record['verdict']} ({record['wall_clock_s']:.1f}s)"
        )
        records.append(record)

    round3_block = None
    sweep_records: list[dict] = []
    if args.round3 or args.round3c:
        print("\n-- probe round 3 (PROTOCOL.md section 16) -- stage 1: LR sweep --\n")
        sweep_records = []
        for c in _ROUND3_SWEEP:
            record = _run_cell_with_val(ctx, c)
            e25v = record["e_25_val"]
            e25v_str = "DIVERGED" if e25v is None else f"{e25v:.4f}"
            print(f"[round3 sweep] {c['cell_id']} -> status={record['status']} e_25_val={e25v_str}")
            sweep_records.append(record)
            _upsert_record(records, record)  # 16.2(5)/_REPORT_SCHEMA: cell_id unique per cell

        selection = _select_lr(sweep_records)

        print("\n-- probe round 3 -- stage 2: ON-arm finitude screen --\n")
        screen_results: dict[str, dict] = {}
        for backbone in _ROUND3_BACKBONES:
            on_arm = _ROUND3_ON_ARM[backbone]
            lr = selection[backbone]["selected_lr"]
            if lr is None:
                print(f"[round3 screen] {on_arm} -> no eligible lr from the sweep, screen not attempted")
                screen_results[on_arm] = {"screened_lr": None, "trail": []}
                continue
            result = _run_screen(on_arm, lr, lambda cell: _run_cell_with_val(ctx, cell))
            for rec in result["trail"]:
                # 16.2(5): a dedup'd screen cell reuses a round-2 cell_id
                # VERBATIM -- that record already exists in `records` (the
                # round1/2 loop above, no val fields). The artifact is
                # cited, not re-executed: this call's own val-enriched
                # record REPLACES it, never a second row for the same
                # cell_id (_REPORT_SCHEMA: "unique per cell").
                _upsert_record(records, rec)
            print(
                f"[round3 screen] {on_arm} -> screened_lr={result['screened_lr']} "
                f"({len(result['trail'])} attempt(s))"
            )
            screen_results[on_arm] = result

        budget_block = None
        walign_block = None
        mlp_screened_lr = screen_results["mlp_k"]["screened_lr"]
        if mlp_screened_lr is not None:
            print("\n-- probe round 3 -- stage 3: budget rung + w_align --\n")
            twin_rec = screen_results["mlp_k"]["trail"][-1]

            rung_cell = {
                "cell_id": "mlp_k__r3_updates_8000", "arm": "mlp_k", "axis": "round3_stage3",
                "value": "r3_updates_8000",
                "knobs": dict(_BASELINE, epochs=8000, h_fr=25, lr=mlp_screened_lr),
            }
            rung_rec = _run_cell_with_val(ctx, rung_cell)
            _upsert_record(records, rung_rec)  # 16.2(5)/_REPORT_SCHEMA: cell_id unique per cell
            entry_value = _decide_budget(rung_rec, twin_rec)
            print(
                f"[round3 stage3] {rung_cell['cell_id']} -> status={rung_rec['status']} "
                f"entry_value={entry_value}"
            )

            walign_cell = {
                "cell_id": "mlp_k__r3_walign0", "arm": "mlp_k", "axis": "round3_stage3",
                "value": "r3_walign0",
                "knobs": dict(_BASELINE, epochs=4000, h_fr=25, lr=mlp_screened_lr, w_align=0.0),
            }
            walign_rec = _run_cell_with_val(ctx, walign_cell)
            _upsert_record(records, walign_rec)  # 16.2(5)/_REPORT_SCHEMA: cell_id unique per cell
            walign_retained = _decide_walign(walign_rec, twin_rec)
            print(
                f"[round3 stage3] {walign_cell['cell_id']} -> status={walign_rec['status']} "
                f"retained={walign_retained}"
            )

            budget_block = {
                "entry_value": entry_value,
                "criterion": "e_25_val(8000) <= 0.9 * e_25_val(4000 twin)",
                "e_25_val_rung": rung_rec["e_25_val"],
                "e_25_val_twin": twin_rec["e_25_val"],
            }
            walign_block = {
                "retained": walign_retained,
                "criterion": "e_25_val(w_align=0) < e_25_val(twin) AND m_high_val(w_align=0) <= m_high_val(twin)",
                "e_25_val": walign_rec["e_25_val"],
                "e_25_val_twin": twin_rec["e_25_val"],
                "m_high_val": walign_rec["m_high_val"],
                "m_high_val_twin": twin_rec["m_high_val"],
            }
        else:
            print("\n-- probe round 3 -- stage 3 skipped: mlp has no screened lr --\n")

        round3_block = {
            "lr_list": list(_ROUND3_LR_LIST),
            "tie_rule": "lowest lr wins an exact e_25_val tie",
            "selected_lr": {
                backbone: {
                    "selected_lr": selection[backbone]["selected_lr"],
                    "basis": selection[backbone]["basis"],
                    "screen_trail": [
                        {"lr": r["knobs"]["lr"], "cell_id": r["cell_id"], "status": r["status"]}
                        for r in screen_results[_ROUND3_ON_ARM[backbone]]["trail"]
                    ],
                    "screened_lr": screen_results[_ROUND3_ON_ARM[backbone]]["screened_lr"],
                }
                for backbone in _ROUND3_BACKBONES
            },
            "budget": budget_block,
            "w_align": walign_block,
        }

    round3c_block = None
    if args.round3c:
        print("\n-- probe round 3c (PROTOCOL.md section 17) -- no-op gate --\n")
        noop_rec = _run_cell_with_val(ctx, _ROUND3C_NOOP_CELL)
        _upsert_record(records, noop_rec)  # new cell_id -> appends (_REPORT_SCHEMA: unique per cell)
        r3_twin = next(r for r in sweep_records if r["cell_id"] == "mlp__r3_lr_1e-3")
        noop_result = _noop_gate_check(noop_rec, r3_twin)
        print(f"[round3c noop] equal={noop_result['equal']}")

        round3c_block = {
            "grad_clip": _R3C_GRAD_CLIP,
            "noop": noop_result,
            "selected_lr_ssm": None,
        }

        if not noop_result["equal"]:
            # PROTOCOL.md 17.2: STOP, not an error -- exit 0, the report
            # carries the discrepancy, cells 2-5 never run (no third path).
            print(
                "\n-- probe round 3c STOPPED: no-op gate did not reproduce its round-3 twin "
                "-- cells 2-5 (clipped ssm sweep, ssm screen) do NOT run --\n"
            )
        else:
            print("\n-- probe round 3c -- clipped ssm sweep --\n")
            r3c_sweep_records = []
            for c in _ROUND3C_SWEEP:
                rec = _run_cell_with_val(ctx, c)
                _upsert_record(records, rec)
                r3c_sweep_records.append(rec)
                print(f"[round3c sweep] {c['cell_id']} -> status={rec['status']}")

            ssm_selection = _select_lr(r3c_sweep_records, backbones=("min_ssm",))
            ssm_lr = ssm_selection["min_ssm"]["selected_lr"]

            screen_trail: list[dict] = []
            screened_lr = None
            if ssm_lr is not None:
                print("\n-- probe round 3c -- ssm ON screen --\n")
                screen_result = _run_screen(
                    "ssm_k", ssm_lr, lambda cell: _run_cell_with_val(ctx, cell),
                    cell_builder=_r3c_screen_cell,
                )
                for rec in screen_result["trail"]:
                    _upsert_record(records, rec)
                screen_trail = [
                    {"lr": r["knobs"]["lr"], "cell_id": r["cell_id"], "status": r["status"]}
                    for r in screen_result["trail"]
                ]
                screened_lr = screen_result["screened_lr"]
                print(f"[round3c screen] ssm_k -> screened_lr={screened_lr}")
            else:
                print("\n-- probe round 3c -- ssm sweep exhausted, screen not attempted --\n")

            round3c_block["selected_lr_ssm"] = {
                "basis": ssm_selection["min_ssm"]["basis"],
                "screen_trail": screen_trail,
                "screened_lr": screened_lr,
            }

    report_path = _write_report(records, round3=round3_block, round3c=round3c_block)
    print(f"\nreport written to {report_path}\n")
    _print_table(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
