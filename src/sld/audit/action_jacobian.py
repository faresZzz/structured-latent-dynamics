"""The action Jacobian of a free rollout, and its conditioning.

``d x_H / d u``: how the state reached after ``H`` free-running steps responds
to the exogenous action sequence that drove it there. Its condition number is
what figure F reports per arm at ``H = 25``.

Why this is an audit and not a training-time quantity (REGLES section 2): it
measures a trained predictor without modifying it, it depends only on the
``Predictor`` CONTRACT (``rollout``), and it runs offline, on CPU, in float64
like every module under ``sld.audit``. There is deliberately no per-family
branch here -- a structured arm and an `mlp` arm go down the identical path,
which is the only way the number compares across the structure axis at all.

Two independent references, per REGLES 7.1, and the second is the one that
matters:

- **positive** -- on an LTI cell whose rollout is `x_H = sum_t A^(H-1-t) B u_t`,
  the Jacobian is known in closed form and `jacobian_action` reproduces it;
- **negative** -- every Jacobian this module returns is cross-checked against
  CENTRED finite differences of the SAME rollout, to better than `1e-8`
  relative. Autograd and finite differences share no code path, so agreement
  is evidence; and `check_against_finite_differences` is called by
  `conditioning` itself, so a figure cannot be drawn from an unchecked
  Jacobian by forgetting to ask.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from sld.audit._common import require_cpu_float64_module, require_cpu_float64_tensor

# Centred differences on a float64 rollout: the truncation error falls as
# eps**2 while the cancellation error grows as macheps/eps, so the total is
# minimised near macheps**(1/3) ~ 6e-6. Chosen by that argument, and the
# 1e-8 tolerance below is what it actually achieves on these rollouts, not a
# tolerance widened until the test passed.
_FD_EPS = 1e-6
_FD_RTOL = 1e-8


@dataclass(frozen=True)
class ActionJacobian:
    """``d x_H / d u`` for one episode, plus what it is conditioned like.

    Attributes:
        jacobian: shape ``(d_h, T * d_u)``, the derivative of the state after
            ``T`` free-running steps with respect to the whole flattened
            action sequence.
        singular_values: shape ``(min(d_h, T*d_u),)``, descending.
        cond: ``s_max / s_min``, or ``inf`` when ``s_min`` is exactly zero --
            never a clipped or regularised ratio. A rank-deficient action map
            is a real property of the arm and is reported as ``inf``.
        fd_max_rel_error: the largest relative disagreement with centred
            finite differences over all entries. Carried on the result rather
            than discarded, so a figure can print what its own check achieved.
    """

    jacobian: Tensor
    singular_values: Tensor
    cond: float
    fd_max_rel_error: float


def _rollout_endpoint(predictor, h0: Tensor, u_flat: Tensor, horizon: int, d_u: int) -> Tensor:
    """``x_H`` as a function of the FLATTENED action sequence.

    One episode, shaped to the ``(T, B, d_u)`` contract with ``B = 1``. This
    is the single definition of "the rollout" both the autograd path and the
    finite-difference path call -- if it were written twice, the negative
    check below would be comparing two implementations of the derivative of
    two different functions, and would prove nothing.
    """
    u_seq = u_flat.reshape(horizon, 1, d_u)
    return predictor.rollout(h0.reshape(1, -1), u_seq)[-1, 0]


def jacobian_action(predictor, h0: Tensor, u_seq: Tensor) -> Tensor:
    """``d x_H / d u`` by autograd, shape ``(d_h, T * d_u)``.

    Args:
        predictor: any ``Predictor``; only ``rollout`` is used.
        h0: shape ``(d_h,)``, the rollout's initial condition.
        u_seq: shape ``(T, d_u)``, the exogenous action sequence.

    Raises:
        ValueError: a tensor is not CPU float64, or a shape is wrong.
    """
    require_cpu_float64_module(predictor, "predictor")
    require_cpu_float64_tensor(h0, "h0")
    require_cpu_float64_tensor(u_seq, "u_seq")
    if h0.dim() != 1:
        raise ValueError(f"h0 must be (d_h,), got {tuple(h0.shape)}")
    if u_seq.dim() != 2:
        raise ValueError(f"u_seq must be (T, d_u), got {tuple(u_seq.shape)}")

    horizon, d_u = u_seq.shape
    u_flat = u_seq.reshape(-1).clone()
    with torch.enable_grad():
        jac = torch.autograd.functional.jacobian(
            lambda u: _rollout_endpoint(predictor, h0, u, horizon, d_u),
            u_flat,
            vectorize=False,
        )
    return jac.detach()


def finite_difference_action_jacobian(predictor, h0: Tensor, u_seq: Tensor, eps: float = _FD_EPS) -> Tensor:
    """The same Jacobian by CENTRED finite differences -- the independent
    reference REGLES 7.1's corollary requires.

    Shares no code with ``jacobian_action`` beyond the rollout itself: no
    autograd graph, no backward pass, only forward evaluations of the very
    function whose derivative is claimed.
    """
    horizon, d_u = u_seq.shape
    u_flat = u_seq.reshape(-1)
    n = u_flat.numel()
    columns = []
    with torch.no_grad():
        for i in range(n):
            step = torch.zeros_like(u_flat)
            step[i] = eps
            plus = _rollout_endpoint(predictor, h0, u_flat + step, horizon, d_u)
            minus = _rollout_endpoint(predictor, h0, u_flat - step, horizon, d_u)
            columns.append((plus - minus) / (2 * eps))
    return torch.stack(columns, dim=1)


def check_against_finite_differences(
    predictor, h0: Tensor, u_seq: Tensor, jac: Tensor, *, rtol: float = _FD_RTOL
) -> float:
    """Return the max relative disagreement; raise if it exceeds ``rtol``.

    The scale is the larger of the two magnitudes at each entry, floored at 1
    so an entry that is zero in both does not divide by nothing and does not
    count as a relative failure either.
    """
    fd = finite_difference_action_jacobian(predictor, h0, u_seq)
    scale = torch.maximum(jac.abs(), fd.abs()).clamp(min=1.0)
    rel = ((jac - fd).abs() / scale).max().item()
    if not (rel <= rtol):
        raise ValueError(
            f"action Jacobian disagrees with centred finite differences: max relative "
            f"error {rel:.3e} > {rtol:.0e}. The Jacobian is NOT usable for a figure "
            f"(PROTOCOL 23.5(3): the discrepancy is the result)."
        )
    return rel


def conditioning(predictor, h0: Tensor, u_seq: Tensor, *, rtol: float = _FD_RTOL) -> ActionJacobian:
    """``d x_H / d u``, its spectrum and its condition number -- checked.

    The finite-difference cross-check runs HERE, unconditionally, rather than
    in a test that a figure could be written without running. A number this
    module hands out has been checked by construction.
    """
    jac = jacobian_action(predictor, h0, u_seq)
    rel = check_against_finite_differences(predictor, h0, u_seq, jac, rtol=rtol)
    svals = torch.linalg.svdvals(jac)
    s_min = svals[-1].item()
    cond = float("inf") if s_min == 0.0 else svals[0].item() / s_min
    return ActionJacobian(jacobian=jac, singular_values=svals, cond=cond, fd_max_rel_error=rel)
