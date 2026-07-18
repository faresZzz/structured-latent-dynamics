"""The 1-D gates counter-example (REGLES sections 5, 7; PLAN Wave 4a).

This is the doctrine test: it documents IN CODE why a displayed spectrum is
never a stability object, and why the total Jacobian is the only object the
audit trusts. The map is

    F(h) = a(h) * h,   a(h) = 0.5 + 0.4 * tanh(K * (h - 1)),   K = 100.

Read through a frozen gate (the ``local_operator`` convention: the gate
``a(h)`` is context, not state), the multiplier satisfies
``|a(h)| <= 0.9`` everywhere -- every locally displayed "spectrum" looks
comfortably contractive. But the true one-step derivative at ``h = 1`` is

    F'(1) = a(1) + 1 * a'(1) = 0.5 + 0.4 * K = 40.5,

a 45x expansion the frozen-gate reading cannot see, because the expansion
lives entirely in the gate's own dependence on the state. ``total_jacobian``
must see 40.5; ``local_spectra_field`` must display <= 0.9 at the very same
point. Both halves are asserted against each other on the SAME model.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from sld.audit.jacobian import conditional_jacobian, total_jacobian
from sld.audit.spectral import local_spectra_field
from sld.predictors.base import PredictorModule, PredictorState

K = 100.0
GATE_CEILING = 0.9  # sup of |a(h)| = 0.5 + 0.4 * |tanh| < 0.9
TRUE_DERIVATIVE_AT_1 = 40.5  # a(1) + a'(1) = 0.5 + 0.4 * K


def _gate(h: Tensor) -> Tensor:
    return 0.5 + 0.4 * torch.tanh(K * (h - 1.0))


class _GatedScalar(PredictorModule):
    """1-D gated map F(h) = a(h) h with the gate exposed as frozen context.

    ``local_operator`` returns the multiplier ``a(h)`` with the gate frozen
    at the probe point -- exactly the selective-SSM convention (the reading
    never differentiates through the gate). That frozen reading is what
    ``local_spectra_field`` displays, and what this test proves blind.
    """

    def step(self, h: Tensor, u: Tensor, state: PredictorState):
        return _gate(h) * h, state

    def local_operator(self, h: Tensor, u: Tensor, state: PredictorState):
        a = _gate(h)  # (B, 1); gate value at the probe, frozen
        A = a.unsqueeze(-1)  # (B, 1, 1)
        B = torch.zeros(h.shape[0], 1, u.shape[-1], dtype=h.dtype)
        return A, B


def test_gates_counterexample_display_blind_audit_sees():
    model = _GatedScalar()

    # -- the display half: roll from h0 = 1.0 (the cliff point is sampled at
    # t = 0) and read the field of local spectra along the trajectory.
    h0 = torch.tensor([[1.0]], dtype=torch.float64)
    u_seq = torch.zeros(8, 1, 1, dtype=torch.float64)
    times, eigvals = local_spectra_field(model, h0, u_seq)

    # Pin the "same point" claim: the FIRST displayed reading must be taken
    # at t = 0, i.e. AT h = 1 where the gate is a(1) = 0.5 exactly. Without
    # this, a regression that samples after the step (or returns zeros)
    # would keep the ceiling assertions below vacuously green.
    assert times[0] == 0
    lam_at_cliff = eigvals[0, 0, 0]
    assert abs(lam_at_cliff.imag) < 1e-12
    assert abs(lam_at_cliff.real - 0.5) < 1e-9, (
        f"displayed value at the cliff point must be a(1) = 0.5, got {lam_at_cliff}"
    )

    displayed_max = float(np.abs(eigvals).max())
    print(f"[gates counterexample] displayed max |lambda| = {displayed_max:.6f}")
    assert displayed_max <= GATE_CEILING + 1e-12, (
        "the frozen-gate display must look contractive everywhere -- "
        "otherwise this counter-example no longer demonstrates anything"
    )

    # -- the audit half: the total Jacobian at the very point the display
    # sampled first (h = 1, where the displayed value is a(1) = 0.5).
    def step_fn(q: Tensor, u: Tensor) -> Tensor:
        h, new_state = model.step(q.unsqueeze(0), u.unsqueeze(0), PredictorState())
        assert new_state.m is None  # 1-D map: the whole augmented state is h
        return h.squeeze(0)

    q = torch.tensor([1.0], dtype=torch.float64)
    u = torch.zeros(1, dtype=torch.float64)
    J = total_jacobian(step_fn, q, u)
    print(f"[gates counterexample] total Jacobian at h=1 = {J[0, 0]:.9f}")
    assert abs(J[0, 0] - TRUE_DERIVATIVE_AT_1) < 1e-9

    # -- the doctrine, in one inequality: the display and the audit disagree
    # by a factor > 45 at the same point of the same model (and by exactly
    # 40.5 / 0.5 = 81 at the pinned cliff reading).
    assert J[0, 0] > 45.0 * displayed_max
    assert J[0, 0] / lam_at_cliff.real > 80.0


def test_hidden_coupling_contractive_conditional_expanding_total():
    """REGLES section 7's second counter-example, in augmented-state form.

    The conditional (1,1) block -- ``h`` responding to ``h`` with the
    auxiliary state ``m`` frozen -- is contractive (0.5), yet the TOTAL
    augmented Jacobian is expanding (spectral radius > 2) because the
    expansion runs entirely through the ``h <-> m`` coupling (the
    exploding-cache-next-to-a-contractive-block failure mode). A caller
    that audits only the conditional block passes a stability read that the
    total Jacobian refutes -- which is exactly why ``conditional_jacobian``
    never concludes alone.
    """
    M = torch.tensor([[0.5, 2.0], [2.0, 0.1]], dtype=torch.float64)

    def step_fn(q: Tensor, u: Tensor) -> Tensor:
        return M @ q + u

    q = torch.zeros(2, dtype=torch.float64)
    u = torch.zeros(2, dtype=torch.float64)
    J = total_jacobian(step_fn, q, u)

    cond_block = conditional_jacobian(J, d_h=1)
    rho_cond = float(np.abs(np.linalg.eigvals(cond_block)).max())
    rho_total = float(np.abs(np.linalg.eigvals(J)).max())
    print(f"[hidden coupling] rho(conditional)={rho_cond:.3f}, rho(total)={rho_total:.3f}")
    assert rho_cond < 1.0 < rho_total

    # independent oracle: the total spectral radius is that of M itself
    rho_expected = float(np.abs(np.linalg.eigvals(M.numpy())).max())
    assert abs(rho_total - rho_expected) < 1e-12
