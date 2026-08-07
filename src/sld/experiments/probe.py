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
import hashlib
import json
import os
import time
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
from sld.training.trainer import Trainer, data_fingerprint

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
    return {
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
        "delta": _DELTA,
        "results_root": str(PROBE_ROOT / "runs"),
        "stratum_system": _STRATUM_SYSTEM,
        "stratum_regime": _STRATUM_REGIME,
    }


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
    return value, never recomputed (R1/R2)."""
    knobs = cell["knobs"]
    artifact, data, cond_sigma = _get_representation(
        ctx, knobs["n_episodes"], knobs["w_align"], knobs["view_offset"]
    )

    trainer_cfg = _trainer_config(cell)
    trainer = Trainer(trainer_cfg, representation=artifact)
    run_dir = trainer.resolve_run_dir(data)
    _require_under_root(run_dir)

    t0 = time.perf_counter()
    sentinel = _factorial._validate_completion_sentinel(run_dir)
    if sentinel is None:
        trainer.fit(data)
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
    e_1_pers = float(artifact.manifest["e_1_pers"])
    e_recon = float(artifact.e_recon)
    tau_1 = protocol.tau_1(e_1_pers)
    tau1_validity = protocol.tau1_validity(e_recon, e_1_pers)

    # PROTOCOL.md 15.2(iii)/15.4(b): the field-of-local-spectra screening,
    # independent of gate.py, computed unconditionally (unlike the e_*/m_*
    # gate quantities below, this is not fabricated by a divergence -- it is
    # informative EVEN WHEN the model's decoded rollout diverges).
    run = load_run(run_dir)
    rho_max_rollout, phi_norm_at_h, _last_finite_step = _operator_screening(run, artifact, data, _H_GATE)

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
        "rho_max_rollout": rho_max_rollout,
        "phi_norm_at_h": phi_norm_at_h,
    }

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
    "status": "'trained' (this invocation ran it) or 'skipped_resume' (already complete on disk)",
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


def _write_report(records: list[dict]) -> Path:
    report_path = _require_under_root(PROBE_ROOT / "report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = report_path.with_name(report_path.name + ".tmp")
    tmp_path.write_text(json.dumps({"schema": _REPORT_SCHEMA, "cells": records}, indent=2, sort_keys=True))
    os.replace(tmp_path, report_path)
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sld-probe-d")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--round2", action="store_true",
        help="also run PROTOCOL.md section 15.4(b)'s nine declared round-2 cells",
    )
    args = parser.parse_args(argv)

    cells = expand_axes(include_round2=args.round2)

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

    report_path = _write_report(records)
    print(f"\nreport written to {report_path}\n")
    _print_table(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
