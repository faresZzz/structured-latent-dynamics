"""D3's spectral and action bounds (PROTOCOL.md section 19, amendment 10 as
rewritten -- section 19.2).

This file gained a second half (the PERSISTENCE section below, same
discipline as ``gate.py``'s own): the four bounds this module computes
(state error at H=100, the spectral/action bounds above, and the gain
against the two trivial baselines) were, until then, always calculated and
never persisted -- no ``d3_gold.json``, no per-run verdict, no campaign,
nothing indexed. The split is structural: everything ABOVE the persistence
section is pure computation (no ``json``/``os``/``csv``/``yaml``/``argparse``
touched), everything BELOW reads or writes disk. The persistence section
also carries PROTOCOL.md section 19.3's non-gating diagnostic
(``measure_frozen_chain_cost``), clearly separated at the end of the file.

D3 asks whether the structured arms (``mlp_k``/``transformer_k``/``ssm_k``),
trained on coupled oscillators, find the true discrete generator under the
declared optimizer and budget -- at the configuration where the question
has a truth value: ``r = 2`` (``2r = d = 4``, the phase-C gold's own proven
configuration; kept here as a choice, not a precondition -- see below).

Both bounds are taken in STATE coordinates, on the EFFECTIVE one-step map

    x_hat_{t+1} = pi( A_bar . psi(x_t) + B_bar . u_t )

never on ``A_bar``/``B_bar`` read directly out of the block basis
``local_operator`` returns. ``local_jacobians`` builds that map at one probe
(``A_bar``/``B_bar`` FROZEN -- ``local_operator``'s own "frozen-context"
contract, base.py) and differentiates it by autograd; ``oscillator_bounds``
screens the resulting ``(J_x, J_u)`` field against the physical reference
built by ``discrete_reference``.

WHY STATE COORDINATES, NOT THE BLOCK BASIS (found while building the first
version of this module, which compared ``B_bar`` directly against ``Phi B``
-- kept here as the reasoning, not just the conclusion, so nobody has to
reconstruct it from a diff):

1. For any invertible ``T``, ``(psi, A_bar, B_bar, pi)`` and
   ``(T psi, T A_bar T^-1, T B_bar, pi T^-1)`` produce IDENTICAL predictions
   (the ``T``s cancel exactly), so the training objective is exactly flat
   along that family. ``A_bar``'s raw spectrum is invariant under it, but
   ``B_bar -> T B_bar`` is NOT: a model that has genuinely learned the
   physics can fail a ``B_bar``-against-``Phi B`` bound by basis luck alone.
   ``J_u`` (via the chain rule, ``d pi/d h`` at the frozen ``B_bar``) IS
   invariant under this family -- it is the physically meaningful quantity,
   the state-space response to an input.
2. The block-basis spectrum cannot see a rotation-sign flip either: a
   conjugate pair's eigenvalue SET is invariant under ``omega -> -omega``
   (flipping it only swaps which root is called "+"), so
   ``eig_relative_error`` on ``A_bar`` alone reports ~0 even when the model
   is a different dynamical system with the same latent spectrum. Moving to
   state coordinates does not repair this for the SPECTRAL bound
   specifically -- ``J_x``'s eigenvalues inherit the SAME sign-invariance
   (eigenvalues are a similarity invariant, and ``omega -> -omega`` is
   invariant regardless of which similarity class), so a pure rotation-sign
   flip stays invisible to ``eig_relative_error(J_x, ...)`` even here
   (measured in this module's test file: ~1e-16, unchanged). It IS caught
   by the action bound, since ``J_u`` is not spectrum-only.

``J_x`` is ``d x d`` whatever ``2r`` is, so the bound's well-posedness no
longer depends on ``r`` at all; ``r = 2`` is a choice (phase-C's own proven
configuration), not a precondition.

R2 (REGLES section 7.1). ``discrete_reference`` is the independent oracle: a
Van Loan augmented-matrix exponential, never ``rotation_scaling_zoh`` /
``discretize_zoh`` (this repo's closed-form ``(I, J)`` coefficient algorithm
for 2x2 rotation-scaling blocks, blocks2x2.py) and never phase A's RK4
stability polynomial. ``local_jacobians``' own autograd computation gets its
own independent cross-check in this module's test file (central finite
differences on held-out probes) -- an autograd Jacobian that silently
validates itself against nothing is exactly what R2 forbids.

Vocabulary (REGLES section 3, eliminatory). ``psi``/``pi`` are MLPs in a real
run (``psi_hidden``/``pi_hidden`` default to ``(64,)`` and are not config
keys -- phase-C's own gold plants a SINGLE LINEAR layer, which is not D3's
regime), and the head is contextual, so ``J_x``/``J_u`` are genuinely
state-dependent: what this module screens is a FIELD of local Jacobians,
and its spectral content a field of local spectra -- never "the operator" or
"the spectrum". On an LTI system a correctly trained model should collapse
that field to a point; that collapse is a MEASURED outcome, reported here
via the field's dispersion across probes, never assumed. Every number this
module returns is DISPLAY ONLY, never a stability certificate
(``sld.predictors.base.Predictor.local_operator``'s own docstring: frozen-
context, no h-loop, no scheduling path -- the same limitation applies here).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor

from sld.audit._common import require_cpu_float64_module
from sld.data.generate import build_dataset
from sld.data.regimes import OURegime
from sld.encoders.pretrain import (
    build_identity_representation,
    FrozenRepresentationArtifact,
    pretrain_representation,
)
from sld.experiments import protocol as protocol_mod
from sld.experiments.factorial import (
    _build_dataset_for,
    _parse_shard,
    _preflight_resolve_and_check_collisions,
    _resolve_stratum_representations,
    _shard_cells,
    _SYSTEMS,
    _validate_completion_sentinel,
    expand,
)
from sld.experiments.gate import (
    FreeRolloutResult,
    RolloutDivergence,
    _read_run_digests,
    evaluate_run_free_rollout,
    load_run_double,
)
from sld.predictors.base import PredictorState
from sld.systems.coupled_oscillators import CoupledOscillators


def discrete_reference(A_cont: Tensor, B_cont: Tensor, delta: float) -> tuple[Tensor, Tensor]:
    """Exact zero-order-hold discretization via the Van Loan (1978) augmented
    matrix.

    ``expm([[delta*A, delta*I], [0, 0]])`` has top-left block
    ``exp(delta*A)`` and top-right block ``Phi = int_0^delta exp(sA) ds``,
    so ``B_bar = Phi @ B`` falls out of the SAME matrix exponential.
    ``torch.matrix_exp`` is scaling-and-squaring on the full ``2n x 2n``
    matrix -- algorithmically independent of ``rotation_scaling_zoh``'s
    closed-form ``(I, J)`` coefficients and of phase A's RK4 polynomial
    (R2, module docstring).

    Args:
        A_cont: ``(n, n)`` continuous generator.
        B_cont: ``(n, d_u)`` continuous input map.
        delta: the ZOH step. Not validated here (finiteness/positivity is
            ``StructuredPredictor``'s own constructor guard); a caller
            handing this a bad ``delta`` gets whatever ``torch.matrix_exp``
            does with it, same as ``torch.matrix_exp`` on any other input.

    Returns:
        ``(A_d, B_d)``: ``A_d = exp(delta * A_cont)``,
        ``B_d = Phi @ B_cont``.
    """
    n = A_cont.shape[-1]
    aug = torch.zeros(2 * n, 2 * n, dtype=A_cont.dtype)
    aug[:n, :n] = delta * A_cont
    aug[:n, n:] = delta * torch.eye(n, dtype=A_cont.dtype)
    expm = torch.matrix_exp(aug)
    return expm[:n, :n], expm[:n, n:] @ B_cont


def eig_relative_error(A_learned: Tensor, A_reference: Tensor) -> float:
    """Max relative error between two eigenvalue multisets, under the BEST
    pairing over all ``n!`` permutations.

    NEVER sorts. A sort-based pairing (by real part, by magnitude, by
    whatever key) silently mispairs whenever two eigenvalues tie on that key
    under floating-point noise -- exactly what a symmetric coupled-oscillator
    system produces: its two discrete modes share the SAME real part
    (``exp(delta * alpha)``, one ``alpha`` for both) and differ only in
    ``omega``. A general (non-block-aware) eigensolver computes that shared
    real part independently for each mode, with a few ULP of independent
    rounding noise on each side of a comparison -- enough to flip a
    real-part-first sort's tie-break between the learned operator's
    eigenvalues and the reference's, silently pairing each learned
    eigenvalue with the WRONG physical mode while still reporting a
    deceptively small number.
    ``tests/experiments/test_learned_oscillator_gold.py`` demonstrates this
    failure deterministically (not left to incidental rounding), reusing
    the exact rationale ``tests/predictors/test_structured_oscillator_gold.py``
    ``_eig_max_error`` already states.

    ponytail: brute force over ``n!`` permutations -- exact, and cheap at
    ``n = d = 4`` (24 permutations, D3's own state-space dimension).
    Ceiling: stops being free past ``n ~ 8`` (``8! = 40320``); swap in a
    Hungarian assignment (``scipy.optimize.linear_sum_assignment``) then,
    not before.

    Args:
        A_learned: ``(n, n)``, e.g. ``J_x`` from ``local_jacobians``.
        A_reference: ``(n, n)``, the reference discrete generator. Its
            eigenvalues are assumed nonzero -- true for ANY ZOH
            discretization of a finite continuous generator, since
            ``exp(...)`` is never zero -- so the relative error below is
            always well-defined.

    Returns:
        The relative error under the pairing that minimizes the maximum
        per-pair relative error.

    Raises:
        ValueError: the two matrices carry a different number of
            eigenvalues (shape mismatch).
    """
    xs = torch.linalg.eigvals(A_learned).tolist()
    ys = torch.linalg.eigvals(A_reference).tolist()
    if len(xs) != len(ys):
        raise ValueError(
            f"eigenvalue count mismatch: A_learned has {len(xs)}, A_reference has {len(ys)}"
        )
    return min(
        max(abs(x - ys[p[i]]) / abs(ys[p[i]]) for i, x in enumerate(xs))
        for p in itertools.permutations(range(len(ys)))
    )


def action_relative_error(B_learned: Tensor, B_reference: Tensor) -> float:
    """Relative Frobenius-norm error between two input maps of the same shape.

    A direct comparison, no permutation or basis search -- and none is
    needed here: fed ``J_u`` (``local_jacobians``) against ``Phi B``, both
    already live in the SAME (state) coordinates, unlike the raw ``B_bar``
    from ``local_operator`` this bound compared against in an earlier,
    basis-dependent version of this module (see the module docstring).

    Args:
        B_learned: ``(n, d_u)``.
        B_reference: ``(n, d_u)``, same shape, assumed nonzero.

    Returns:
        ``||B_learned - B_reference||_F / ||B_reference||_F``.

    Raises:
        ValueError: ``B_reference`` is exactly zero (relative error is
            undefined).
    """
    ref_norm = torch.linalg.matrix_norm(B_reference, ord="fro")
    if ref_norm.item() == 0.0:
        raise ValueError("B_reference is exactly zero; relative error is undefined")
    return (torch.linalg.matrix_norm(B_learned - B_reference, ord="fro") / ref_norm).item()


def local_jacobians(
    model, x: Tensor, u: Tensor, state: PredictorState | None = None
) -> tuple[Tensor, Tensor]:
    """``(J_x, J_u)``: the Jacobians of the FROZEN-CONTEXT effective one-step
    map ``x_hat = pi(A_bar @ psi(x) + B_bar @ u)`` at a single probe
    ``(x, u)``, by autograd.

    ``A_bar``/``B_bar`` are read from ``model.local_operator`` at
    ``(psi(x), u, state)`` and then DETACHED before differentiating --
    frozen, exactly ``local_operator``'s own contract (base.py: "a
    frozen-context, per-family-space reading misses the h-loop... by
    construction"). ``psi``/``pi`` (``model.lift``/``model.project``) stay
    LIVE, so for a real model (nonlinear MLPs, ``psi_hidden``/``pi_hidden``
    defaulting to ``(64,)``) ``J_x``/``J_u`` are genuine, state-dependent
    Jacobians, not a fixed operator (module docstring, vocabulary note).

    ``state`` defaults to ``PredictorState()`` (fresh, no history): with
    ``u`` structurally excluded from the context
    (``StructuredPredictor``'s own module docstring) this isolates exactly
    what the packet asks for -- does the map vary with ``x`` alone.

    Args:
        model: exposes ``lift``, ``project``, ``local_operator`` (
            ``sld.predictors.structured.StructuredPredictor`` and its three
            arms).
        x: ``(d,)`` -- ONE state-space probe (not lifted).
        u: ``(d_u,)`` -- ONE action.
        state: ``PredictorState``, defaults to fresh.

    Returns:
        ``J_x``: ``(d, d)``. ``J_u``: ``(d, d_u)``.
    """
    if state is None:
        state = PredictorState()
    h = model.lift(x.unsqueeze(0))
    A_bar, B_bar = model.local_operator(h, u.unsqueeze(0), state)
    A_bar = A_bar[0].detach()
    B_bar = B_bar[0].detach()

    def f(x_: Tensor, u_: Tensor) -> Tensor:
        h_ = model.lift(x_.unsqueeze(0))[0]
        h_next = A_bar @ h_ + B_bar @ u_
        return model.project(h_next.unsqueeze(0))[0]

    J_x = torch.autograd.functional.jacobian(lambda x_: f(x_, u), x)
    J_u = torch.autograd.functional.jacobian(lambda u_: f(x, u_), u)
    return J_x, J_u


def oscillator_bounds(
    model,
    system,
    *,
    delta: float,
    probe_states: Tensor,
    probe_actions: Tensor,
) -> dict:
    """PROTOCOL.md section 19.2's spectral and action bounds, screened over a
    batch of probe STATES (not lifted states -- see the module docstring).

    Builds ``(J_x, J_u)`` at each probe via ``local_jacobians`` and compares
    against the SAME physical reference, built once from
    ``system.A_mat``/``system.B_mat`` via ``discrete_reference``. This is a
    FIELD of local Jacobians (module docstring): the error AND its
    dispersion across probes, never a single number standing in for "the
    operator". DISPLAY ONLY -- never a stability certificate.

    Args:
        model: exposes ``lift``/``project``/``local_operator`` (
            ``sld.predictors.structured.StructuredPredictor`` and its three
            arms). Must be CPU float64 (``require_cpu_float64_module``,
            REGLES section 5): the 1e-3 thresholds this feeds are far below
            float32's ~1e-7 headroom.
        system: exposes ``A_mat``/``B_mat`` (numpy, float64) -- e.g.
            ``sld.systems.coupled_oscillators.CoupledOscillators``.
        delta: the ZOH step. Must equal the model's own ``model.delta`` for
            the comparison to be meaningful; not checked here (a mismatch
            is a caller error this function has no way to detect from
            ``system`` alone).
        probe_states: ``(N, d)`` STATE-space probes, float64.
        probe_actions: ``(N, d_u)``, float64, ``N`` matching
            ``probe_states``.

    Returns:
        A dict: ``eig_relative_error``/``action_relative_error`` (the MAX
        over probes -- what a gate would threshold on), plus
        ``_mean``/``_std``/``_values`` for each (the per-probe field, for
        display).

    Raises:
        ValueError: ``probe_states``/``probe_actions`` disagree on their
            leading (``N``) axis, or ``N < 1``.
    """
    require_cpu_float64_module(model, "model")
    if probe_states.shape[0] != probe_actions.shape[0]:
        raise ValueError(
            f"probe_states and probe_actions must share their leading (N) axis, "
            f"got {probe_states.shape[0]} vs {probe_actions.shape[0]}"
        )
    if probe_states.shape[0] < 1:
        raise ValueError("at least one probe state is required")

    A_cont = torch.as_tensor(system.A_mat, dtype=torch.float64)
    B_cont = torch.as_tensor(system.B_mat, dtype=torch.float64)
    A_ref, B_ref = discrete_reference(A_cont, B_cont, delta)

    eig_errs: list[float] = []
    action_errs: list[float] = []
    state = PredictorState()
    for i in range(probe_states.shape[0]):
        J_x, J_u = local_jacobians(model, probe_states[i], probe_actions[i], state)
        eig_errs.append(eig_relative_error(J_x, A_ref))
        action_errs.append(action_relative_error(J_u, B_ref))

    eig_t = torch.tensor(eig_errs, dtype=torch.float64)
    action_t = torch.tensor(action_errs, dtype=torch.float64)
    return {
        "eig_relative_error": eig_t.max().item(),
        "eig_relative_error_mean": eig_t.mean().item(),
        "eig_relative_error_std": eig_t.std(unbiased=False).item(),
        "eig_relative_error_values": eig_errs,
        "action_relative_error": action_t.max().item(),
        "action_relative_error_mean": action_t.mean().item(),
        "action_relative_error_std": action_t.std(unbiased=False).item(),
        "action_relative_error_values": action_errs,
    }


# ==============================================================================
# PERSISTENCE (I/O starts here). Everything ABOVE this point is pure
# computation (no ``json``/``os``/``csv``/``yaml``/``argparse`` touched);
# everything BELOW reads or writes disk -- the same split ``gate.py``'s own
# module docstring declares, deliberately copied.
#
# D3's four bounds (PROTOCOL.md sections 19.1, 19.2, 19.7) are a DIFFERENT
# verdict from D4's positive-control gate (``gate.py``'s own ``gate.json``):
# a state bound at ``H = 100`` (not ``H_GATE = 25``), a spectral bound, an
# action bound, and the gain-against-baselines check ``gate.py``'s own
# ``evaluate_positive_control`` already computes as its ``lower_bound`` --
# reused here as bound 4, never a second computation of the same ratio (R2).
# Persisted to a DIFFERENT artifact, ``d3_gold.json``, because the schema is
# genuinely different: four named bounds plus a probe field's dispersion,
# not one verdict plus a curve.
#
# This section never runs INSIDE training: it reads already-trained runs off
# disk, exactly the way ``gate.evaluate_gate_campaign`` scores
# ``sld-factorial``'s own runs (this module's own docstring point: D3's
# state/spectral/action bounds were derived and committed BEFORE any D3 run
# existed -- section 19.6).
# ==============================================================================

# bound 1's threshold and horizon (PROTOCOL.md section 19.7: "training uses
# h_fr = 25 while bound 1 scores at H = 100" -- declared deliberately, not
# derivable from any other constant in this repo, so it is named here once).
_D3_STATE_H100_THRESHOLD: float = 1e-3
_D3_HORIZON: int = 100
# bounds 2/3's thresholds (PROTOCOL.md section 19.2). No threshold moved by
# this module (19.6): all three numbers below are the SAME 1e-3 already
# fixed for D3's state bound (12.3/19.0), never a separate constant that
# could drift from it.
_D3_SPECTRAL_THRESHOLD: float = 1e-3
_D3_ACTION_THRESHOLD: float = 1e-3

_D3_SCHEMA_VERSION = "d3_gold/1"
_D3_STATUSES = frozenset({"passes", "fails", "divergence"})

# closed schemas, exactly like gate.py's own _TOP_KEYS/_PROTOCOL_KEYS/etc: a
# missing OR an extra key both fail validation, never silently tolerated.
_D3_TOP_KEYS = frozenset(
    {"schema", "identity", "digests", "split", "thresholds", "bounds", "dispersion", "status", "first_step"}
)
_D3_IDENTITY_KEYS = frozenset({"cell_id", "arm", "seed", "stratum_system", "stratum_regime", "run_dir"})
_D3_DIGEST_KEYS = frozenset({"config_hash", "data_hash", "representation_digest"})
# every quantity the verdict was ACTUALLY computed from -- the three literal
# 1e-3 cutoffs, bound 1's horizon and the run's own delta (used to build the
# reference), and the gain check's own inputs (h_gate/gamma/tau_1, plus what
# tau_1 was derived from, PROTOCOL.md section 15.1) -- same "publish every
# threshold used" convention as gate.py's own _PROTOCOL_KEYS, so a later
# change to any one of them is detectable against a d3_gold.json already on
# disk (read_d3_gold_artifact's own ``expected_thresholds``).
_D3_THRESHOLD_KEYS = frozenset(
    {
        "state_h100",
        "spectral",
        "action",
        "horizon",
        "delta",
        "h_gate",
        "gamma",
        "tau_vpt",
        "tau_1",
        "gamma_1",
        "e_1_pers",
        "e_recon",
        "attainable",
    }
)
_D3_BOUND_NAMES = frozenset({"state_h100", "spectral", "action", "gain"})
_D3_DISPERSION_KEYS = frozenset(
    {
        "spectral_mean",
        "spectral_std",
        "spectral_values",
        "action_mean",
        "action_std",
        "action_values",
        "n_probes",
    }
)

# the index into z_true/u a probe is drawn from -- the SAME index (`_T0` in
# gate.py) the free rollout ITSELF treats as its initial condition x_0, so
# bounds 2/3's probes are not a second, independently invented sampling
# scheme: one probe per test episode, at the exact frame bound 1's own
# rollout starts from.
_D3_PROBE_INDEX: int = 1


def _system_for_dataset(ds_cfg: dict):
    """Rebuild the ``system`` object one ``datasets:`` entry declares -- the
    SAME name -> class lookup ``sld.experiments.factorial._build_dataset_for``
    uses (``factorial._SYSTEMS``), reused rather than a second private
    registry of one entry, so a system registered there is available here
    too without a second edit (R2)."""
    spec = ds_cfg["system"]
    return _SYSTEMS[spec["name"]](**spec.get("params", {}))


def _probe_states_from_split(eval_split: dict) -> tuple[Tensor, Tensor]:
    """State-space probes for bounds 2/3 (PROTOCOL.md section 19.2's "field
    of local Jacobians... published... together with the field's
    dispersion"), drawn from the TEST split: one probe per episode, at
    ``_D3_PROBE_INDEX`` of ``z_true``/``u`` -- the SAME frame gate.py's own
    free rollout treats as its initial condition ``x_0``. Genuinely
    STATE-space probes only because D3's own stratum representation is the
    identity (section 19.1: ``d_z == d_x``) -- this function does not itself
    check that; the campaign that calls it only ever does so under an
    identity representation.

    Returns:
        ``(probe_states, probe_actions)``, CPU float64, shape ``(B, d)``.

    Raises:
        ValueError: ``eval_split`` has no episodes, or fewer physical
            steps/actions per episode than ``_D3_PROBE_INDEX`` requires.
    """
    z_true = np.asarray(eval_split["z_true"], dtype=np.float64)
    u = np.asarray(eval_split["u"], dtype=np.float64)
    if z_true.shape[0] < 1:
        raise ValueError("eval_split has no episodes -- at least one probe state is required")
    if z_true.shape[1] <= _D3_PROBE_INDEX:
        raise ValueError(
            f"eval_split has {z_true.shape[1]} physical steps per episode, too few for "
            f"probe index {_D3_PROBE_INDEX}"
        )
    if u.shape[1] <= _D3_PROBE_INDEX:
        raise ValueError(
            f"eval_split has {u.shape[1]} actions per episode, too few for probe index "
            f"{_D3_PROBE_INDEX}"
        )
    probe_states = torch.as_tensor(z_true[:, _D3_PROBE_INDEX, :], dtype=torch.float64)
    probe_actions = torch.as_tensor(u[:, _D3_PROBE_INDEX, :], dtype=torch.float64)
    return probe_states, probe_actions


def write_d3_gold_artifact(
    run_dir: str | Path,
    result: FreeRolloutResult | RolloutDivergence,
    *,
    identity: dict,
    thresholds: dict,
    split: str,
    bounds_field: dict | None = None,
) -> Path:
    """Persist one D3 run's four-bound verdict to ``run_dir/d3_gold.json``.

    ``result`` is either the ``FreeRolloutResult`` bound 1 (and, via its own
    ``positive_control``, bound 4) were computed from, or the
    ``RolloutDivergence`` it raised -- mirrors ``gate.write_gate_artifact``'s
    own contract, including the terminal ``"divergence"`` handling
    (PROTOCOL.md section 11.7: no fabricated number). ``bounds_field`` is
    ``oscillator_bounds``'s own return (bounds 2/3 -- the spectral/action
    field, worst case AND dispersion); REQUIRED when ``result`` is a
    ``FreeRolloutResult`` (there is no other source for it), ignored on
    divergence.

    ``digests`` (``config_hash``/``data_hash``/``representation_digest``) are
    NEVER taken from the caller: re-read from the run's own artifacts on disk
    via ``gate._read_run_digests`` -- the EXACT function
    ``gate.write_gate_artifact`` itself uses (R2), never a second
    implementation of "what digests does this run's own disk state carry".

    Written atomically (temp file + ``os.replace`` in the SAME directory) and
    LAST -- same discipline as ``run_complete.json``/``gate.json``.

    Raises:
        ValueError: ``identity`` or ``thresholds`` does not carry exactly its
            required keys; a ``thresholds`` value is non-finite (except
            ``attainable``); ``run_dir`` has no valid completion sentinel
            (refuses to score an unfinished or unverified run); ``config.yaml``
            is unreadable; ``result`` is a ``FreeRolloutResult`` but
            ``bounds_field`` is not supplied, or its own ``e_h`` is shorter
            than ``thresholds["horizon"]``; ``result`` is neither a
            ``FreeRolloutResult`` nor a ``RolloutDivergence``.
    """
    run_dir = Path(run_dir)

    if set(identity) != _D3_IDENTITY_KEYS:
        raise ValueError(f"identity must carry exactly {sorted(_D3_IDENTITY_KEYS)}, got {sorted(identity)}")
    if set(thresholds) != _D3_THRESHOLD_KEYS:
        raise ValueError(f"thresholds must carry exactly {sorted(_D3_THRESHOLD_KEYS)}, got {sorted(thresholds)}")
    for key, value in thresholds.items():
        if key == "attainable":
            continue
        if not math.isfinite(float(value)):
            raise ValueError(f"thresholds[{key!r}] = {value!r} must be finite")

    # digests reread from the run's OWN artifacts on disk -- never from
    # caller memory (a caller's in-memory bookkeeping could be stale).
    # _read_run_digests ALSO revalidates the completion sentinel, so a run
    # that stopped being complete between the caller's own check and this
    # write is still refused here, by name.
    digests = _read_run_digests(run_dir)

    if isinstance(result, RolloutDivergence):
        status = "divergence"
        first_step = result.first_step
        bounds = None
        dispersion = None
    elif isinstance(result, FreeRolloutResult):
        if bounds_field is None:
            raise ValueError(
                "bounds_field (oscillator_bounds's own return) is required when result is "
                "a FreeRolloutResult -- bounds 2/3 have no other source"
            )
        horizon = int(thresholds["horizon"])
        if len(result.e_h) < horizon:
            raise ValueError(
                f"result.e_h has {len(result.e_h)} entries, fewer than thresholds['horizon']={horizon}"
            )
        e_h_100 = float(result.e_h[horizon - 1])
        eig_max = float(bounds_field["eig_relative_error"])
        action_max = float(bounds_field["action_relative_error"])
        pc = result.positive_control
        bounds = {
            "state_h100": {
                "value": e_h_100,
                "threshold": float(thresholds["state_h100"]),
                "passes": e_h_100 < thresholds["state_h100"],
            },
            "spectral": {
                "value": eig_max,
                "threshold": float(thresholds["spectral"]),
                "passes": eig_max < thresholds["spectral"],
            },
            "action": {
                "value": action_max,
                "threshold": float(thresholds["action"]),
                "passes": action_max < thresholds["action"],
            },
            # bound 4 (PROTOCOL.md section 19.4's "gain against both trivial
            # baselines") is READ from the SAME FreeRolloutResult bound 1 was
            # computed from -- gate.py's own evaluate_positive_control's
            # lower_bound is exactly this ratio (e_h_gate < min(pers, lin) /
            # gamma) -- never a second calculation.
            "gain": {
                "m_low": float(pc.m_low),
                "passes": bool(pc.lower_bound),
            },
        }
        dispersion = {
            "spectral_mean": float(bounds_field["eig_relative_error_mean"]),
            "spectral_std": float(bounds_field["eig_relative_error_std"]),
            "spectral_values": [float(v) for v in bounds_field["eig_relative_error_values"]],
            "action_mean": float(bounds_field["action_relative_error_mean"]),
            "action_std": float(bounds_field["action_relative_error_std"]),
            "action_values": [float(v) for v in bounds_field["action_relative_error_values"]],
            "n_probes": len(bounds_field["eig_relative_error_values"]),
        }
        # ALL FOUR bounds must pass for "passes" -- one bound failing is
        # "fails", never averaged or partially credited (REGLES section 7.1's
        # own negative: "only one of four passing must never read as passes").
        status = "passes" if all(b["passes"] for b in bounds.values()) else "fails"
        first_step = None
    else:
        raise ValueError(
            f"result must be a FreeRolloutResult or a RolloutDivergence, got {type(result)!r}"
        )

    payload = {
        "schema": _D3_SCHEMA_VERSION,
        "identity": identity,
        "digests": digests,
        "split": split,
        "thresholds": thresholds,
        "bounds": bounds,
        "dispersion": dispersion,
        "status": status,
        "first_step": first_step,
    }

    d3_path = run_dir / "d3_gold.json"
    tmp_path = d3_path.with_name(d3_path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload))
    os.replace(tmp_path, d3_path)
    return d3_path


def read_d3_gold_artifact(
    run_dir: str | Path,
    *,
    expected_thresholds: dict | None = None,
    expected_digests: dict | None = None,
) -> dict:
    """Load and VALIDATE ``run_dir/d3_gold.json`` -- trusts nothing about the
    bytes on disk, same doctrine as ``gate.read_gate_artifact``.

    ``expected_thresholds``/``expected_digests`` are the caller's OWN current
    values (e.g. the live module constants, or a freshly recomputed
    representation digest); any key present there that disagrees with what
    is recorded in ``d3_gold.json`` is a refusal NAMING that key -- this is
    what makes "a threshold changed after the run was scored" or "the
    representation moved since this verdict was written" detectable rather
    than silently stale.

    Raises:
        ValueError: ``d3_gold.json`` is missing, corrupt/unreadable, not
            schema ``"d3_gold/1"``, missing a required key at any level,
            carries an unknown ``status``, carries a non-finite value where
            one is required, is internally inconsistent (a ``"divergence"``
            status with non-null ``bounds``/``dispersion``, a non-
            ``"divergence"`` status with a non-null ``first_step``, or a
            ``status`` that disagrees with whether every bound actually
            passed), or disagrees with a caller-supplied
            ``expected_thresholds``/``expected_digests`` entry.
    """
    run_dir = Path(run_dir)
    path = run_dir / "d3_gold.json"
    if not path.exists():
        raise ValueError(f"no d3_gold.json at {path}")
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        raise ValueError(f"d3_gold.json at {path} is corrupt or unreadable ({exc!r})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"d3_gold.json at {path} is not a JSON object")

    if payload.get("schema") != _D3_SCHEMA_VERSION:
        raise ValueError(
            f"d3_gold.json at {path} has schema {payload.get('schema')!r}, expected {_D3_SCHEMA_VERSION!r}"
        )
    missing_top = _D3_TOP_KEYS - set(payload)
    if missing_top:
        raise ValueError(f"d3_gold.json at {path} is missing top-level keys {sorted(missing_top)}")

    identity = payload["identity"]
    if not isinstance(identity, dict) or set(identity) != _D3_IDENTITY_KEYS:
        raise ValueError(f"d3_gold.json at {path}: identity must carry exactly {sorted(_D3_IDENTITY_KEYS)}")

    digests = payload["digests"]
    if not isinstance(digests, dict) or set(digests) != _D3_DIGEST_KEYS:
        raise ValueError(f"d3_gold.json at {path}: digests must carry exactly {sorted(_D3_DIGEST_KEYS)}")

    thresholds = payload["thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != _D3_THRESHOLD_KEYS:
        raise ValueError(f"d3_gold.json at {path}: thresholds must carry exactly {sorted(_D3_THRESHOLD_KEYS)}")
    for key, value in thresholds.items():
        if key == "attainable":
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"d3_gold.json at {path}: thresholds[{key!r}] = {value!r} is not a finite number")

    status = payload["status"]
    if status not in _D3_STATUSES:
        raise ValueError(f"d3_gold.json at {path}: unknown status {status!r}")

    first_step = payload["first_step"]
    bounds = payload["bounds"]
    dispersion = payload["dispersion"]
    if status == "divergence":
        if bounds is not None or dispersion is not None:
            raise ValueError(f"d3_gold.json at {path}: status is 'divergence' but bounds/dispersion is not null")
        if not isinstance(first_step, int) or isinstance(first_step, bool) or first_step < 1:
            raise ValueError(
                f"d3_gold.json at {path}: status is 'divergence', first_step must be a positive "
                f"int, got {first_step!r}"
            )
    else:
        if first_step is not None:
            raise ValueError(
                f"d3_gold.json at {path}: status is {status!r} but first_step is not null "
                f"(got {first_step!r})"
            )
        if not isinstance(bounds, dict) or set(bounds) != _D3_BOUND_NAMES:
            raise ValueError(f"d3_gold.json at {path}: bounds must carry exactly {sorted(_D3_BOUND_NAMES)}")
        for name in ("state_h100", "spectral", "action"):
            entry = bounds[name]
            if not isinstance(entry, dict) or set(entry) != {"value", "threshold", "passes"}:
                raise ValueError(
                    f"d3_gold.json at {path}: bounds[{name!r}] must carry exactly value/threshold/passes"
                )
            for numkey in ("value", "threshold"):
                v = entry[numkey]
                if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                    raise ValueError(
                        f"d3_gold.json at {path}: bounds[{name!r}][{numkey!r}] = {v!r} is not a finite number"
                    )
            if not isinstance(entry["passes"], bool):
                raise ValueError(f"d3_gold.json at {path}: bounds[{name!r}]['passes'] must be a boolean")
        gain = bounds["gain"]
        if not isinstance(gain, dict) or set(gain) != {"m_low", "passes"}:
            raise ValueError(f"d3_gold.json at {path}: bounds['gain'] must carry exactly m_low/passes")
        m_low = gain["m_low"]
        if not isinstance(m_low, (int, float)) or isinstance(m_low, bool) or not math.isfinite(m_low):
            raise ValueError(f"d3_gold.json at {path}: bounds['gain']['m_low'] = {m_low!r} is not a finite number")
        if not isinstance(gain["passes"], bool):
            raise ValueError(f"d3_gold.json at {path}: bounds['gain']['passes'] must be a boolean")

        all_pass = all(bounds[name]["passes"] for name in _D3_BOUND_NAMES)
        if status == "passes" and not all_pass:
            raise ValueError(f"d3_gold.json at {path}: status is 'passes' but not every bound passes")
        if status == "fails" and all_pass:
            raise ValueError(f"d3_gold.json at {path}: status is 'fails' but every bound passes")

        if not isinstance(dispersion, dict) or set(dispersion) != _D3_DISPERSION_KEYS:
            raise ValueError(
                f"d3_gold.json at {path}: dispersion must carry exactly {sorted(_D3_DISPERSION_KEYS)}"
            )
        for key in ("spectral_mean", "spectral_std", "action_mean", "action_std"):
            v = dispersion[key]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                raise ValueError(f"d3_gold.json at {path}: dispersion[{key!r}] = {v!r} is not a finite number")
        for key in ("spectral_values", "action_values"):
            values = dispersion[key]
            if not isinstance(values, list) or not values:
                raise ValueError(f"d3_gold.json at {path}: dispersion[{key!r}] must be a non-empty list")
            if not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values
            ):
                raise ValueError(f"d3_gold.json at {path}: dispersion[{key!r}] contains a non-finite value")
        n_probes = dispersion["n_probes"]
        if not isinstance(n_probes, int) or isinstance(n_probes, bool) or n_probes < 1:
            raise ValueError(f"d3_gold.json at {path}: dispersion['n_probes'] must be a positive int")

    if expected_thresholds is not None:
        for key, expected_value in expected_thresholds.items():
            if key not in _D3_THRESHOLD_KEYS:
                raise ValueError(f"expected_thresholds names unknown threshold {key!r}")
            if thresholds[key] != expected_value:
                raise ValueError(
                    f"d3_gold.json at {path}: threshold {key!r} changed since this run's d3_gold "
                    f"was written -- recorded {thresholds[key]!r}, expected {expected_value!r}"
                )

    if expected_digests is not None:
        for key, expected_value in expected_digests.items():
            if key not in _D3_DIGEST_KEYS:
                raise ValueError(f"expected_digests names unknown digest {key!r}")
            if digests[key] != expected_value:
                raise ValueError(
                    f"d3_gold.json at {path}: digest {key!r} changed since this run's d3_gold was "
                    f"written -- recorded {digests[key]!r}, expected {expected_value!r}"
                )

    return payload


# ------------------------------------------------------------------------
# The campaign: the same experiment YAML sld-factorial trains from, walked
# again -- read-only over the trained runs, never a second training path.
# Mirrors gate.evaluate_gate_campaign's own structure exactly (R2's
# corollary: two campaigns that walk the same grid should look like it).
# ------------------------------------------------------------------------


def _d3_cell_status(run_dir: Path) -> tuple[str, str]:
    """``(status, state_h100)`` for one cell's ``run_dir``, read from
    ``d3_gold.json`` alone -- mirrors ``gate._gate_cell_status``'s
    discipline: read from DISK, never from what this process happened to
    compute, so a shard that never touched this cell still reports another
    shard's already-written verdict correctly."""
    d3_path = Path(run_dir) / "d3_gold.json"
    if not d3_path.exists():
        return "missing", ""
    try:
        payload = read_d3_gold_artifact(run_dir)
    except ValueError:
        # present but untrustworthy -- distinct from both a valid verdict
        # and a cell that was never scored at all (same M9 doctrine
        # factorial._cell_status/gate._gate_cell_status apply to an invalid
        # sentinel/gate.json)
        return "invalid_d3_gold", ""
    status = payload["status"]
    bounds = payload.get("bounds")
    if not bounds:
        return status, ""
    return status, str(bounds["state_h100"]["value"])


def _write_d3_gold_index(results_root: Path, exp_name: str, cells: list[dict]) -> None:
    """Regenerate ``results_root/exp_name/d3_gold_index.csv`` from the full,
    resolved cell list -- ``factorial._write_index``/``gate._write_gate_index``'s
    own contract, copied: ALWAYS the full expected grid (never just this
    shard's slice), atomic write (temp file + ``os.replace`` in the same
    directory). A cell this invocation never touched, or never trained at
    all, appears with status ``"missing"`` rather than being silently absent
    from the file.
    """
    index_dir = results_root / exp_name
    index_dir.mkdir(parents=True, exist_ok=True)
    final_path = index_dir / "d3_gold_index.csv"
    tmp_path = final_path.with_name(final_path.name + ".tmp")
    with open(tmp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_dir", "regime", "family", "structure", "seed", "status", "state_h100"])
        for cell in cells:
            run_dir = cell["run_dir"]
            status, state_h100 = _d3_cell_status(run_dir)
            writer.writerow(
                [run_dir, cell["dataset"], cell["family"], cell["structure"], cell["seed"], status, state_h100]
            )
    os.replace(tmp_path, final_path)


def _score_d3_cell(
    cell: dict,
    datasets: dict[str, dict],
    representations: dict[str, FrozenRepresentationArtifact],
    systems: dict[str, object],
    results_root: Path,
    n: int,
) -> dict:
    """Score one cell: refuse loudly (naming the cell) if its run is not
    complete, evaluate the four D3 bounds on the TEST split (PROTOCOL.md
    section 19.4/15.5/16.4: "the gate scores on test; anything that
    *selects* reads val" -- this function never touches ``val``), write
    ``d3_gold.json``, and return a small summary dict. NEVER raises for a
    merely-incomplete or diverged cell -- both are reported, by name, and the
    campaign loop continues (mirrors ``gate._evaluate_gate_cell``'s own F17d
    discipline: one bad cell must not silently drop every other cell's
    already-computed verdict from the index).
    """
    run_dir = cell["run_dir"]
    label = (
        f"[{cell['index'] + 1}/{n}] {cell['dataset']} {cell['backbone']} "
        f"{cell['structure']} {cell['family']} seed={cell['seed']}"
    )
    sentinel = _validate_completion_sentinel(run_dir)
    if sentinel is None:
        print(f"{label} -> {run_dir} (REFUSED: no valid run_complete.json -- run is not complete)")
        return {"index": cell["index"], "run_dir": str(run_dir), "status": "missing"}

    representation = representations[cell["dataset"]]
    system = systems[cell["dataset"]]
    eval_split = datasets[cell["dataset"]]["test"]

    e_1_pers = representation.manifest["e_1_pers"]
    e_recon = representation.e_recon
    tau_1_value = protocol_mod.tau_1(e_1_pers)
    attainable = protocol_mod.tau1_validity(e_recon, e_1_pers)["attainable"]

    thresholds = {
        "state_h100": _D3_STATE_H100_THRESHOLD,
        "spectral": _D3_SPECTRAL_THRESHOLD,
        "action": _D3_ACTION_THRESHOLD,
        "horizon": _D3_HORIZON,
        "delta": cell["config"]["delta"],
        "h_gate": protocol_mod.H_GATE,
        "gamma": protocol_mod.GAMMA,
        "tau_vpt": protocol_mod.TAU_VPT,
        "tau_1": tau_1_value,
        "gamma_1": protocol_mod.GAMMA_1,
        "e_1_pers": e_1_pers,
        "e_recon": e_recon,
        "attainable": attainable,
    }
    identity = {
        "cell_id": cell["index"],
        "arm": {"backbone": cell["backbone"], "structure": cell["structure"], "family": cell["family"]},
        "seed": cell["seed"],
        "stratum_system": cell["config"]["stratum_system"],
        "stratum_regime": cell["config"]["stratum_regime"],
        "run_dir": str(run_dir.relative_to(results_root)),
    }

    try:
        result = evaluate_run_free_rollout(
            run_dir,
            representation,
            eval_split,
            horizon=_D3_HORIZON,
            tau=protocol_mod.TAU_VPT,
            h_gate=protocol_mod.H_GATE,
            gamma=protocol_mod.GAMMA,
            tau_1=tau_1_value,
        )
    except RolloutDivergence as exc:
        write_d3_gold_artifact(run_dir, exc, identity=identity, thresholds=thresholds, split="test")
        print(f"{label} -> {run_dir} (DIVERGENCE at free-running step {exc.first_step})")
        return {"index": cell["index"], "run_dir": str(run_dir), "status": "divergence"}

    # bounds 2/3 need the raw predictor (never a rollout) -- load_run_double
    # is the SAME digest-checked loader evaluate_run_free_rollout used above
    # (R2), called again here rather than threading a private return value
    # out of that function, so this call re-validates independently.
    _encoder64, predictor64, run_cfg = load_run_double(run_dir, representation)
    thresholds["delta"] = run_cfg["delta"]  # PROTOCOL.md wording: "delta read from the run's config"
    probe_states, probe_actions = _probe_states_from_split(eval_split)
    bounds_field = oscillator_bounds(
        predictor64, system, delta=run_cfg["delta"], probe_states=probe_states, probe_actions=probe_actions
    )

    d3_path = write_d3_gold_artifact(
        run_dir, result, identity=identity, thresholds=thresholds, split="test", bounds_field=bounds_field
    )
    status, _ = _d3_cell_status(run_dir)  # read back from disk (never trust in-memory computation twice)
    print(f"{label} -> {run_dir} (OK, status={status})")
    return {"index": cell["index"], "run_dir": str(run_dir), "status": status, "d3_gold_path": str(d3_path)}


def score_oscillator_campaign(config_path: str | Path, *, shard: tuple[int, int] | None = None) -> list[dict]:
    """Score every cell of ``config_path``'s D3 grid against its
    already-trained run, writing one ``d3_gold.json`` per cell plus one
    campaign-wide ``d3_gold_index.csv`` -- PROTOCOL.md section 19.4's D3
    analogue of ``gate.evaluate_gate_campaign``, built the SAME way: reuses
    ``sld.experiments.factorial.expand`` for the cell list (never a parallel
    expansion that could drift from what ``sld-factorial`` itself trained,
    R2), the SAME global preflight (``_preflight_resolve_and_check_collisions``)
    over the FULL grid before ``shard`` narrows which cells THIS invocation
    scores, and an index that always covers the full grid.

    Args:
        config_path: the experiment YAML -- the SAME file (and schema)
            ``sld-factorial --config`` trained the grid from
            (``learned_oscillator_gold.yaml``).
        shard: ``(i, k)``, 1-indexed shard ``i`` of ``k``, or ``None`` to
            score every cell.

    Returns:
        One summary dict per scored cell (``run_cells`` order): at least
        ``{"index", "run_dir", "status"}``, plus ``"d3_gold_path"`` when a
        ``d3_gold.json`` was written. ``status`` is one of ``"passes"``,
        ``"fails"``, ``"divergence"``, or ``"missing"`` (the run at this cell
        has no valid completion sentinel -- reported here, never silently
        dropped from the list).

    Raises:
        ValueError: the config carries no ``representation:`` block (D3's
            four bounds always score a run against the frozen representation
            it was trained with); anything ``expand``/the preflight/
            ``evaluate_run_free_rollout``/``oscillator_bounds`` itself
            refuses (a digest mismatch, an insufficient split, a config
            schema error) -- these are campaign-wide identity/data problems,
            not a per-seed fluke, so they abort the whole campaign rather
            than being swallowed per-cell.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    cells = expand(config, dry_run=False)
    if config.get("representation") is None:
        raise ValueError(
            f"{config_path}: score_oscillator_campaign requires a `representation:` block -- "
            "D3's four bounds always score a run against the frozen representation it was "
            "trained with (PROTOCOL.md section 19.1)"
        )

    exp_name = config["exp_name"]
    results_root = Path(config.get("base", {}).get("results_root", "results"))
    datasets = {name: _build_dataset_for(cfg) for name, cfg in config["datasets"].items()}
    representations = _resolve_stratum_representations(config, results_root, exp_name, datasets)
    systems = {name: _system_for_dataset(cfg) for name, cfg in config["datasets"].items()}

    # global preflight over the FULL grid (mirrored from
    # gate.evaluate_gate_campaign/factorial.main): every cell's run_dir
    # resolved and collision-checked BEFORE any scoring, and BEFORE sharding
    # narrows the loop below.
    _preflight_resolve_and_check_collisions(cells, datasets, representations)

    run_cells = _shard_cells(cells, shard)
    n = len(cells)
    results = [
        _score_d3_cell(cell, datasets, representations, systems, results_root, n) for cell in run_cells
    ]

    _write_d3_gold_index(results_root, exp_name, cells)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sld-d3-gate")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--shard",
        type=_parse_shard,
        default=None,
        metavar="i/k",
        help=(
            "score only cells i-1::k of the grid (1<=i<=k); expansion and the collision "
            "preflight stay global, same convention as sld-factorial/sld-gate --shard "
            "(PROTOCOL.md section 14.7)."
        ),
    )
    args = parser.parse_args(argv)
    results = score_oscillator_campaign(args.config, shard=args.shard)
    any_failed = any(r["status"] in ("missing", "divergence") for r in results)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


# ==============================================================================
# SECONDARY DIAGNOSTIC (PROTOCOL.md section 19.3), EXPLICITLY NON-GATING.
#
# Section 19.1 removes the frozen shared representation from D3's GATED path
# (the identity representation is the repair). Section 19.3 asks that this
# removal be turned into a NUMBER: what would the frozen chain have cost D3,
# measured on the only stratum where the question is answerable. This
# section computes and persists that number. It is read by nobody upstream
# of a verdict: ``score_oscillator_campaign``/``write_d3_gold_artifact``
# above never call anything below this point, and nothing below this point
# writes into a D3 run directory or a campaign index.
# ==============================================================================

_DIAGNOSTIC_SCHEMA_VERSION = "d3_frozen_chain_cost/1"


def measure_frozen_chain_cost(
    *,
    system_params: dict | None = None,
    theta_ou: float = 1.0,
    n_episodes: int = 256,
    T: int = 120,
    dt: float = 0.05,
    data_seed: int = 0,
    d_z: int = 4,
    encoder_hidden: tuple[int, ...] = (64, 64),
    pretrain_seed: int = 0,
    epochs: int = 50,
    lr: float = 1e-3,
    tau_ema: float = 0.99,
    view_offset: int = 1,
    w_align: float = 1.0,
    w_var: float = 1.0,
    w_cov: float = 1.0,
    w_recon: float = 1.0,
    threshold: float = 1e-3,
) -> dict:
    """PROTOCOL.md section 19.3's non-gating diagnostic.

    A predictor trained in one latent cannot be evaluated in another, so
    "the same cells through the frozen chain" (19.3's literal wording) has
    no executable meaning -- no predictor is retrained here. What DOES have
    a meaning, and carries 19.1's whole argument, is the FLOOR: on the SAME
    oscillator stratum, build a PRETRAINED representation with D4's own
    declared hyperparameters (section 19.7's D4 block) -- no predictor
    trained anywhere, exactly like the identity representation D3 actually
    scores against -- and compare its ``e_recon``/``e_1_pers`` to the
    identity representation's own. Both go through
    ``sld.encoders.pretrain._freeze_and_measure`` (via
    ``build_identity_representation``/``pretrain_representation``, R2): a
    measured comparison, not a repeated assumption. This is what the frozen
    chain would have COST D3, expressed as a number rather than left
    unquantified.

    This function decides NOTHING and gates NOTHING: its result is never
    consumed by ``score_oscillator_campaign`` or any other verdict, and
    ``write_frozen_chain_cost_artifact`` below writes it to a root disjoint
    from every D3/D4 results root.

    Args:
        system_params, theta_ou, n_episodes, T, dt, data_seed: the D3 grid's
            own declared dataset (``learned_oscillator_gold.yaml``:
            ``CoupledOscillators``/``OURegime``) -- the defaults reproduce it
            exactly, so "the same oscillator stratum" is true by
            construction rather than by two configs kept in sync by hand.
        d_z, encoder_hidden, pretrain_seed, epochs, lr, tau_ema, view_offset,
            w_align, w_var, w_cov, w_recon: D4's own declared representation
            hyperparameters (PROTOCOL.md section 19.7's D4 block) -- the
            defaults ARE that declaration, not a stand-in for it.
        threshold: the D3 state bound's own ``1e-3`` (section 19.7) -- the
            number both ``e_recon``s are read against.

    Returns:
        A dict: ``schema``, ``non_gating: True``, ``system``/``regime``,
        ``threshold``, and ``identity``/``pretrained`` sub-dicts, each
        carrying ``e_recon``, ``e_1_pers`` and ``e_recon_over_threshold``.
    """
    system = CoupledOscillators(**(system_params or {}))
    regime = OURegime(theta_ou=theta_ou)
    data = build_dataset(system, regime, None, n_episodes, T, dt, data_seed)

    identity = build_identity_representation(
        data, system="CoupledOscillators", regime="OURegime", d_z=d_z
    )
    pretrained = pretrain_representation(
        data,
        system="CoupledOscillators",
        regime="OURegime",
        d_z=d_z,
        encoder_hidden=tuple(encoder_hidden),
        seed=pretrain_seed,
        epochs=epochs,
        lr=lr,
        tau_ema=tau_ema,
        view_offset=view_offset,
        w_align=w_align,
        w_var=w_var,
        w_cov=w_cov,
        w_recon=w_recon,
    )

    def _summary(art: FrozenRepresentationArtifact) -> dict:
        return {
            "e_recon": art.e_recon,
            "e_1_pers": art.manifest["e_1_pers"],
            "e_recon_over_threshold": art.e_recon / threshold,
        }

    return {
        "schema": _DIAGNOSTIC_SCHEMA_VERSION,
        "non_gating": True,
        "system": "CoupledOscillators",
        "regime": "OURegime",
        "threshold": threshold,
        "identity": _summary(identity),
        "pretrained": _summary(pretrained),
    }


def write_frozen_chain_cost_artifact(result: dict, path: str | Path) -> Path:
    """Atomic write of ``measure_frozen_chain_cost``'s own return dict --
    temp file + ``os.replace`` in the same directory, same discipline as
    every other artifact this module writes. Never under
    ``results_d3_oscillator/`` (a live grid root under concurrent training,
    PROTOCOL.md section 19.7) -- the caller's ``path`` decides the root;
    this function only makes the write atomic.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    os.replace(tmp_path, path)
    return path
