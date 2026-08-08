"""D3's spectral and action bounds (PROTOCOL.md section 19, amendment 10 as
rewritten -- section 19.2).

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

import itertools

import torch
from torch import Tensor

from sld.audit._common import require_cpu_float64_module
from sld.predictors.base import PredictorState


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
