"""Two-check suite (REGLES section 7.1) for the shared structured cell.

Positive: ONE hand-derived step, exercising psi, the head, A_bar, B_bar, u and
pi in a single number. Negative: four local bypass mutants (lift, projection,
action, context), each of which must FAIL that same oracle, plus the guards and
the numerical seams of ``rotation_scaling_zoh`` itself.

All references here are INDEPENDENT of the implementation under test:
``torch.matrix_exp`` for ``A_bar`` and the Van Loan augmented-matrix
construction for ``Phi = int_0^delta exp(sA) ds``. Neither is used in
production; both are computed by a different algorithm than the (I, J)
coefficient-basis closed form in ``blocks2x2``.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

from sld.predictors.base import PredictorState
from sld.predictors.blocks2x2 import block_diagonal, rotation_scaling_zoh
from sld.predictors.structured import ContextBackbone, StructuredPredictor

DTYPE = torch.float64


# ---------------------------------------------------------------------------
# independent references (tests only -- never imported by production code)
# ---------------------------------------------------------------------------
def _generator(alpha: float, omega: float) -> Tensor:
    return torch.tensor([[alpha, -omega], [omega, alpha]], dtype=DTYPE)


def _phi_van_loan(A: Tensor, delta: float) -> Tensor:
    """``int_0^delta exp(sA) ds`` via ``expm([[A, I], [0, 0]])``.

    For ``M = [[A, I], [0, 0]]``, ``exp(delta*M)`` has top-right block exactly
    ``int_0^delta exp(sA) ds`` (Van Loan 1978). A completely different
    algorithm from the (I, J) closed form / series being tested.
    """
    n = A.shape[-1]
    M = torch.zeros(2 * n, 2 * n, dtype=A.dtype)
    M[:n, :n] = A
    M[:n, n:] = torch.eye(n, dtype=A.dtype)
    return torch.matrix_exp(delta * M)[:n, n:]


# ---------------------------------------------------------------------------
# a minimal backbone: c_t = h_t + m_t, with m carrying the PAST only
# ---------------------------------------------------------------------------
class _SumBackbone(nn.Module):
    """``kappa`` stand-in for the three real arms.

    ``m_{t+1} = m_t + h_t + W u_t``, so past actions DO reach later contexts
    (which is what makes the u-exclusion test non-vacuous) while ``u_t`` cannot
    reach ``c_t``: ``context_step`` is not given one.
    """

    def __init__(self, d_h: int, d_u: int, dtype: torch.dtype = DTYPE) -> None:
        super().__init__()
        self.d_context = d_h
        self.W = nn.Linear(d_u, d_h, dtype=dtype)

    def context_step(self, h: Tensor, state: PredictorState) -> Tensor:
        m = torch.zeros_like(h) if state.m is None else state.m
        return h + m

    def absorb(self, h: Tensor, u: Tensor, state: PredictorState) -> PredictorState:
        m = torch.zeros_like(h) if state.m is None else state.m
        return PredictorState(m=m + h + self.W(u))


def test_sum_backbone_satisfies_the_protocol():
    assert isinstance(_SumBackbone(4, 1), ContextBackbone)


# ---------------------------------------------------------------------------
# THE POSITIVE -- derived by hand (REGLES section 7.1)
# ---------------------------------------------------------------------------
# r = 1, d_u = 1, d_z = 2, Delta = 1.
#   A_bar = exp(Delta*alpha) * R(Delta*omega) must equal 0.5 * R(pi/2)
#     => exp(alpha) = 1/2  => alpha = -ln 2 ;  omega = pi/2.
#   B_bar = Phi B_cont must equal [1, 2]^T. With Phi = A^{-1}(A_bar - I),
#     B_cont = (A_bar - I)^{-1} A [1, 2]^T. In the (I, J) basis,
#     A_bar - I = -I + (1/2)J, whose inverse is (-I - (1/2)J)/(5/4), and
#     A = -ln2 * I + (pi/2) J, giving
#     (A_bar - I)^{-1} A = ((4 ln2 + pi)/5) I + ((2 ln2 - 2 pi)/5) J.
#     Applied to [1, 2]^T:  first  = c - 2d = 5 pi / 5     = pi
#                           second = d + 2c = 10 ln2 / 5   = 2 ln 2
#     => B_cont = [pi, 2 ln 2]^T  (EXACT closed form).
#   psi swaps the two coordinates, so z = [1, 2] lifts to h0 = [2, 1].
#   h1 = A_bar h0 + B_bar u = [0*2 - 0.5*1, 0.5*2 + 0*1] + 3*[1, 2]
#      = [-0.5, 1] + [3, 6] = [2.5, 7]
#   z_next = pi(h1) = swap([2.5, 7]) = [7, 2.5].
GOLD_ALPHA = math.log(0.5)
GOLD_OMEGA = math.pi / 2.0
GOLD_B_CONT = (math.pi, 2.0 * math.log(2.0))
GOLD_Z = (1.0, 2.0)
GOLD_U = 3.0
GOLD_Z_NEXT = (7.0, 2.5)


def _gold_model(cls: type[StructuredPredictor] = StructuredPredictor) -> StructuredPredictor:
    """The hand-built rig. ``cls`` lets a bypass mutant reuse it verbatim."""
    torch.manual_seed(0)
    backbone = _SumBackbone(d_h=2, d_u=1)
    model = cls(
        backbone,
        d_z=2,
        r=1,
        d_u=1,
        delta=1.0,
        psi_hidden=(),  # single Linear: psi/pi must be exactly the swap
        pi_hidden=(),
        dtype=DTYPE,
    )
    swap = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=DTYPE)
    with torch.no_grad():
        model.psi[0].weight.copy_(swap)
        model.psi[0].bias.zero_()
        model.pi[0].weight.copy_(swap)  # the swap is its own inverse
        model.pi[0].bias.zero_()
        # NEUTRAL CONTEXT: W_theta = 0 exactly, so the head output is theta_0
        # for every c -- the property the LTI gold of wave C3 depends on.
        model.head.weight.zero_()
        model.head.bias.copy_(
            torch.tensor([GOLD_ALPHA, GOLD_OMEGA, *GOLD_B_CONT], dtype=DTYPE)
        )
    return model


def _gold_forward(model: StructuredPredictor) -> Tensor:
    z = torch.tensor([GOLD_Z], dtype=DTYPE)
    u = torch.tensor([[GOLD_U]], dtype=DTYPE)
    h0 = model.lift(z)
    h1, _ = model.step(h0, u, PredictorState())
    return model.project(h1)


def test_gold_single_step_hand_derived():
    """The positive: z=[1,2], u=3 -> z_next=[7,2.5], derived by hand above."""
    z_next = _gold_forward(_gold_model())
    expected = torch.tensor([GOLD_Z_NEXT], dtype=DTYPE)
    err = (z_next - expected).abs().max().item()
    print(f"[gold] z_next={z_next.tolist()} expected={GOLD_Z_NEXT} err={err:.3e}")
    torch.testing.assert_close(z_next, expected, rtol=0.0, atol=1e-12)


def test_gold_operator_matches_the_hand_derived_matrices():
    """The same rig, one level down: A_bar and B_bar themselves."""
    model = _gold_model()
    h0 = model.lift(torch.tensor([GOLD_Z], dtype=DTYPE))
    u = torch.tensor([[GOLD_U]], dtype=DTYPE)
    A_bar, B_bar = model.local_operator(h0, u, PredictorState())
    torch.testing.assert_close(
        A_bar, torch.tensor([[[0.0, -0.5], [0.5, 0.0]]], dtype=DTYPE), rtol=0.0, atol=1e-14
    )
    torch.testing.assert_close(
        B_bar, torch.tensor([[[1.0], [2.0]]], dtype=DTYPE), rtol=0.0, atol=1e-14
    )


# ---------------------------------------------------------------------------
# THE NEGATIVES -- four bypass mutants, LOCAL to this file (production code
# carries no bypass flag). Each must fail the oracle above.
# ---------------------------------------------------------------------------
class _NoLift(StructuredPredictor):
    """Bypass 1: the transition starts from z instead of psi(z)."""

    def lift(self, z: Tensor) -> Tensor:
        return z


class _NoProject(StructuredPredictor):
    """Bypass 2: the read-out is the raw h_next instead of pi(h_next)."""

    def project(self, h: Tensor) -> Tensor:
        return h


class _NoAction(StructuredPredictor):
    """Bypass 3: the B_bar u term is dropped."""

    def step(self, h, u, state):
        A_bar, _ = self._operator_from_context(self.backbone.context_step(h, state))
        return (A_bar @ h.unsqueeze(-1)).squeeze(-1), self.backbone.absorb(h, u, state)


class _FrozenOperator(StructuredPredictor):
    """Bypass 4: the head's alpha/omega are ignored and pinned to zero."""

    def _operator_from_context(self, c):
        theta = self.head(c)
        pinned = torch.zeros_like(theta[..., : self.r])
        B_cont = theta[..., 2 * self.r :].reshape(*theta.shape[:-1], self.d_h, self.d_u)
        blocks, B_bar = rotation_scaling_zoh(pinned, pinned, self.delta, B_cont)
        return block_diagonal(blocks), B_bar


@pytest.mark.parametrize(
    "mutant", [_NoLift, _NoProject, _NoAction, _FrozenOperator], ids=lambda c: c.__name__
)
def test_bypass_mutants_all_fail_the_gold_oracle(mutant):
    """A test that survives a bypass proves nothing. All four must go red."""
    expected = torch.tensor([GOLD_Z_NEXT], dtype=DTYPE)
    z_next = _gold_forward(_gold_model(mutant))
    dev = (z_next - expected).abs().max().item()
    print(f"[mutant {mutant.__name__}] z_next={z_next.tolist()} deviation={dev:.6g}")
    assert dev > 1e-6, f"{mutant.__name__} still passes the oracle -- the oracle is blind"
    with pytest.raises(AssertionError):
        torch.testing.assert_close(z_next, expected, rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# rotation_scaling_zoh: guards, independent reference, seams, gradients
# ---------------------------------------------------------------------------
def test_delta_must_be_finite_and_strictly_positive():
    """NaN fails every comparison, so a `<= 0`-only guard would admit it."""
    alpha = torch.tensor([0.3, -0.2], dtype=DTYPE)
    omega = torch.tensor([1.0, 0.5], dtype=DTYPE)
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and strictly positive"):
            rotation_scaling_zoh(alpha, omega, bad)
    # a per-block delta with ONE bad entry must also be refused
    with pytest.raises(ValueError, match="finite and strictly positive"):
        rotation_scaling_zoh(alpha, omega, torch.tensor([0.1, float("nan")], dtype=DTYPE))
    # positive control: a good delta goes through
    A_bar, _ = rotation_scaling_zoh(alpha, omega, 0.1)
    assert torch.isfinite(A_bar).all()


def test_a_bar_matches_matrix_exp_on_a_generic_block():
    """Independent reference (R2): torch.matrix_exp on the hand-built G.

    Generic (alpha, omega) -- neither zero nor a special angle -- so a sign or
    transpose error in the closed form cannot hide behind a symmetry.
    """
    alpha_v, omega_v, delta = 0.37, -1.234, 0.83
    A_bar, _ = rotation_scaling_zoh(
        torch.tensor([alpha_v], dtype=DTYPE), torch.tensor([omega_v], dtype=DTYPE), delta
    )
    ref = torch.matrix_exp(delta * _generator(alpha_v, omega_v))
    err = (A_bar[0] - ref).abs().max().item()
    print(f"[A_bar vs matrix_exp] err={err:.3e}")
    assert err < 1e-14


def test_batched_alpha_omega_and_b_cont_against_the_references():
    """The batched path (closed in this wave) checked block by block.

    The reference is per-(row, pair) ``matrix_exp`` / Van Loan, never a
    row-by-row rerun of the function under test.
    """
    torch.manual_seed(3)
    n_batch, n_pairs, m, delta = 4, 3, 2, 0.37
    alpha = torch.randn(n_batch, n_pairs, dtype=DTYPE)
    omega = torch.randn(n_batch, n_pairs, dtype=DTYPE)
    B_cont = torch.randn(n_batch, 2 * n_pairs, m, dtype=DTYPE)

    A_bar, B_bar = rotation_scaling_zoh(alpha, omega, delta, B_cont)
    assert A_bar.shape == (n_batch, n_pairs, 2, 2)
    assert B_bar.shape == (n_batch, 2 * n_pairs, m)

    worst_a = worst_b = 0.0
    for b in range(n_batch):
        for i in range(n_pairs):
            G = _generator(alpha[b, i].item(), omega[b, i].item())
            worst_a = max(worst_a, (A_bar[b, i] - torch.matrix_exp(delta * G)).abs().max().item())
            ref_b = _phi_van_loan(G, delta) @ B_cont[b, 2 * i : 2 * i + 2]
            worst_b = max(worst_b, (B_bar[b, 2 * i : 2 * i + 2] - ref_b).abs().max().item())
    print(f"[batched] max err A_bar={worst_a:.3e} B_bar={worst_b:.3e}")
    assert worst_a < 1e-13 and worst_b < 1e-13

    # batched delta on the same batched inputs (the old B_bar restriction)
    delta_batched = torch.rand(n_batch, n_pairs, dtype=DTYPE) * 0.5 + 1e-3
    A_b, B_b = rotation_scaling_zoh(alpha, omega, delta_batched, B_cont)
    assert A_b.shape == (n_batch, n_pairs, 2, 2) and B_b.shape == (n_batch, 2 * n_pairs, m)
    G = _generator(alpha[1, 2].item(), omega[1, 2].item())
    d12 = delta_batched[1, 2].item()
    ref = _phi_van_loan(G, d12) @ B_cont[1, 4:6]
    assert (B_b[1, 4:6] - ref).abs().max().item() < 1e-13


def test_phi_is_entire_through_zero_and_across_the_branch_seam():
    """At (0, 0): B_bar == delta*B_cont EXACTLY, no raise, no NaN.

    Then both sides of the ``delta^2 (alpha^2+omega^2) <= 1e-6`` switch are
    checked against Van Loan: the two branches must join, not merely each be
    self-consistent.
    """
    delta = 1.0  # puts the switch exactly at alpha^2 + omega^2 == 1e-6
    B_cont = torch.tensor([[0.7], [-0.3]], dtype=DTYPE)
    zero = torch.zeros(1, dtype=DTYPE)

    A_bar, B_bar = rotation_scaling_zoh(zero, zero, delta, B_cont)
    assert torch.isfinite(A_bar).all() and torch.isfinite(B_bar).all()
    torch.testing.assert_close(A_bar[0], torch.eye(2, dtype=DTYPE), rtol=0.0, atol=1e-15)
    torch.testing.assert_close(B_bar, delta * B_cont, rtol=0.0, atol=0.0)  # EXACT
    print(f"[Phi at (0,0)] B_bar={B_bar.flatten().tolist()} delta*B_cont ok")

    # Tolerance is the SERIES branch's declared guarantee (blocks2x2 docstring):
    # relative truncation error < 1e-10, dominated by the dropped delta^4 A^4/24
    # term, worst at the switch itself (~1.5e-11 measured at mag=5e-4). Still
    # sharp: dropping the delta^2 A^2/6 term instead would give ~1e-7 here.
    worst = 0.0
    # 5e-4 -> denom 5e-7 (series side); 1e-3 -> denom 2e-6 (closed-form side);
    # 7.07e-4 / 7.08e-4 straddle the switch (2*mag^2 == 1e-6 at mag=7.071e-4)
    for mag in (0.0, 1e-9, 5e-4, 7.07e-4, 7.08e-4, 1e-3, 1e-2, 0.5):
        a = torch.tensor([mag], dtype=DTYPE)
        _, Bb = rotation_scaling_zoh(a, a, delta, B_cont)
        ref = _phi_van_loan(_generator(mag, mag), delta) @ B_cont
        rel = (Bb - ref).abs().max().item() / ref.abs().max().item()
        worst = max(worst, rel)
        assert rel < 1e-10, f"mag={mag} rel err={rel:.3e}"
    print(f"[branch seam] max REL err vs Van Loan across the switch = {worst:.3e}")


def test_gradients_are_finite_at_the_removable_singularity():
    """At (0, 0) the value is fine; the trap is the UNTAKEN torch.where branch
    poisoning the gradient (0 * inf = nan). Checked with isfinite, because a
    NaN is invisible to a comparison."""
    alpha = torch.zeros(1, dtype=DTYPE, requires_grad=True)
    omega = torch.zeros(1, dtype=DTYPE, requires_grad=True)
    delta = torch.tensor(0.7, dtype=DTYPE, requires_grad=True)
    B_cont = torch.tensor([[0.7], [-0.3]], dtype=DTYPE, requires_grad=True)

    A_bar, B_bar = rotation_scaling_zoh(alpha, omega, delta, B_cont)
    loss = A_bar.sum() + B_bar.sum()
    grads = torch.autograd.grad(loss, (alpha, omega, delta, B_cont))
    for name, g in zip(("alpha", "omega", "delta", "B_cont"), grads, strict=True):
        print(f"[grad @ (0,0)] d/d{name} = {g.flatten().tolist()}")
        assert torch.isfinite(g).all(), f"non-finite gradient wrt {name}"


@pytest.mark.parametrize("alpha0,omega0", [(0.0, 0.0), (5e-4, 5e-4), (1e-3, 1e-3)])
def test_gradients_match_central_differences_across_the_seam(alpha0, omega0):
    """Independent reference: centred finite differences, forward passes only.

    ``h = 1e-5`` is the float64 sweet spot here (truncation ~h^2, rounding
    ~eps/h, both ~1e-10) and is 100x smaller than the distance from either
    test point to the branch switch, so the perturbed evaluations stay on the
    same branch as the point being differentiated.
    """
    delta = 1.0
    B_cont = torch.tensor([[0.7], [-0.3]], dtype=DTYPE)

    def loss_at(a: float, o: float) -> float:
        with torch.no_grad():
            A_bar, B_bar = rotation_scaling_zoh(
                torch.tensor([a], dtype=DTYPE), torch.tensor([o], dtype=DTYPE), delta, B_cont
            )
            return (A_bar.sum() + B_bar.sum()).item()

    alpha = torch.tensor([alpha0], dtype=DTYPE, requires_grad=True)
    omega = torch.tensor([omega0], dtype=DTYPE, requires_grad=True)
    A_bar, B_bar = rotation_scaling_zoh(alpha, omega, delta, B_cont)
    g_alpha, g_omega = torch.autograd.grad(A_bar.sum() + B_bar.sum(), (alpha, omega))

    h = 1e-5
    fd_alpha = (loss_at(alpha0 + h, omega0) - loss_at(alpha0 - h, omega0)) / (2 * h)
    fd_omega = (loss_at(alpha0, omega0 + h) - loss_at(alpha0, omega0 - h)) / (2 * h)
    err_alpha = abs(g_alpha.item() - fd_alpha)
    err_omega = abs(g_omega.item() - fd_omega)
    print(
        f"[grad seam @ ({alpha0},{omega0})] autograd=({g_alpha.item():.9f},"
        f"{g_omega.item():.9f}) fd=({fd_alpha:.9f},{fd_omega:.9f}) "
        f"abs err=({err_alpha:.3e},{err_omega:.3e})"
    )
    assert err_alpha < 1e-8 and err_omega < 1e-8


# ---------------------------------------------------------------------------
# StructuredPredictor: the contract decisions, each with a way to fail
# ---------------------------------------------------------------------------
def test_head_slice_layout_is_alpha_omega_then_vec_b():
    """A silent permutation of the three head slices is invisible to a suite
    that only checks shapes. Here each slice is isolated by a one-hot theta_0:
    the resulting operator pins WHICH slice was read as what."""
    r, d_u = 2, 1
    model = StructuredPredictor(
        _SumBackbone(2 * r, d_u), d_z=2, r=r, d_u=d_u, delta=1.0, dtype=DTYPE
    )
    with torch.no_grad():
        model.head.weight.zero_()
        # alpha = (0.3, 0), omega = (0, 0.5), vec B = (1, 2, 3, 4)
        model.head.bias.copy_(torch.tensor([0.3, 0.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0], dtype=DTYPE))
    h = torch.zeros(1, 2 * r, dtype=DTYPE)
    A_bar, B_bar = model.local_operator(h, torch.zeros(1, d_u, dtype=DTYPE), PredictorState())

    # block 0: alpha=0.3, omega=0 -> pure scaling exp(0.3)*I, no rotation
    torch.testing.assert_close(
        A_bar[0, :2, :2], math.exp(0.3) * torch.eye(2, dtype=DTYPE), rtol=0.0, atol=1e-14
    )
    # block 1: alpha=0, omega=0.5 -> pure rotation R(0.5), unit modulus
    torch.testing.assert_close(
        A_bar[0, 2:, 2:], torch.matrix_exp(_generator(0.0, 0.5)), rtol=0.0, atol=1e-14
    )
    assert torch.all(A_bar[0, :2, 2:] == 0.0) and torch.all(A_bar[0, 2:, :2] == 0.0)
    # vec B is row-major into (2r, d_u): rows (1,2) feed block 0, (3,4) block 1
    B_cont = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=DTYPE)
    for i, (a, o) in enumerate([(0.3, 0.0), (0.0, 0.5)]):
        ref = _phi_van_loan(_generator(a, o), 1.0) @ B_cont[2 * i : 2 * i + 2]
        torch.testing.assert_close(B_bar[0, 2 * i : 2 * i + 2], ref, rtol=0.0, atol=1e-13)


def test_neutral_context_is_exactly_theta_zero():
    """W_theta = 0 must make the operator identical for every context -- the
    property wave C3's LTI gold rests on."""
    model = _gold_model()  # head.weight already zeroed
    u = torch.tensor([[GOLD_U]], dtype=DTYPE)
    ops = [
        model.local_operator(h, u, PredictorState(m=m))
        for h, m in (
            (torch.tensor([[1.0, 2.0]], dtype=DTYPE), None),
            (torch.tensor([[-5.0, 9.0]], dtype=DTYPE), None),
            (torch.tensor([[0.0, 0.0]], dtype=DTYPE), torch.tensor([[3.0, -1.0]], dtype=DTYPE)),
        )
    ]
    for A_bar, B_bar in ops[1:]:
        torch.testing.assert_close(A_bar, ops[0][0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(B_bar, ops[0][1], rtol=0.0, atol=0.0)


def test_head_init_follows_protocol_13_6():
    """PROTOCOL 13.6 positive, unpatched: ``W_theta`` is EXACTLY zero at
    construction (atol=0), and ``theta_0``'s three slices land in the ranges
    this repo already draws for the same three quantities elsewhere --
    ``nu``/``omega`` of ``RotationScalingBlocks`` (blocks2x2.py:186-187) and
    ``B_cont`` of ``KoopmanSSMPredictor`` (koopman_ssm.py:154). Also: alpha
    must NOT depend on ``c`` at init -- proven by feeding two very different
    contexts and checking the operator comes out bit-identical, not just
    close."""
    r, d_u = 3, 2
    model = StructuredPredictor(
        _SumBackbone(2 * r, d_u), d_z=2, r=r, d_u=d_u, delta=0.3, dtype=DTYPE
    )

    torch.testing.assert_close(
        model.head.weight, torch.zeros_like(model.head.weight), rtol=0.0, atol=0.0
    )

    bias = model.head.bias.detach()
    alpha0, omega0, b0 = bias[:r], bias[r : 2 * r], bias[2 * r :]
    assert torch.all((alpha0 >= -1.5) & (alpha0 <= -0.5))
    assert torch.all((omega0 >= 0.1) & (omega0 <= 2.0))
    # N(0, 0.01**2): 6-sigma is 0.06, generous enough not to flake
    assert b0.abs().max().item() < 0.06

    u = torch.zeros(1, d_u, dtype=DTYPE)
    A1, B1 = model.local_operator(
        torch.full((1, 2 * r), 1e3, dtype=DTYPE), u, PredictorState()
    )
    A2, B2 = model.local_operator(
        torch.full((1, 2 * r), -1e3, dtype=DTYPE), u, PredictorState()
    )
    torch.testing.assert_close(A1, A2, rtol=0.0, atol=0.0)
    torch.testing.assert_close(B1, B2, rtol=0.0, atol=0.0)


def test_context_actually_reaches_the_operator():
    """The mirror of the test above: with W_theta != 0 the operator MUST move
    with the context, otherwise the head is wired to nothing.

    PROTOCOL 13.6 zeroes W_theta at construction, so this bench must pose its
    own non-zero W_theta rather than lean on a default random init."""
    torch.manual_seed(7)
    model = StructuredPredictor(
        _SumBackbone(4, 1), d_z=2, r=2, d_u=1, delta=0.5, dtype=DTYPE
    )
    with torch.no_grad():
        model.head.weight.copy_(torch.randn_like(model.head.weight))
    u = torch.zeros(1, 1, dtype=DTYPE)
    A1, B1 = model.local_operator(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=DTYPE), u, PredictorState())
    A2, B2 = model.local_operator(torch.tensor([[0.0, 1.0, -2.0, 3.0]], dtype=DTYPE), u, PredictorState())
    assert (A1 - A2).abs().max().item() > 1e-6
    assert (B1 - B2).abs().max().item() > 1e-6


def test_u_never_enters_the_context_of_its_own_step():
    """Structural by the two-call protocol; measured here anyway: changing u_t
    must leave (A_bar_t, B_bar_t) bit-identical.

    The tail check needs the operator to actually move with the context (a
    PAST u must reach a LATER context); PROTOCOL 13.6 zeroes W_theta at
    construction, so this bench poses its own non-zero W_theta."""
    torch.manual_seed(11)
    model = StructuredPredictor(_SumBackbone(4, 3), d_z=2, r=2, d_u=3, delta=0.5, dtype=DTYPE)
    with torch.no_grad():
        model.head.weight.copy_(torch.randn_like(model.head.weight))
    h = torch.randn(2, 4, dtype=DTYPE)
    A1, B1 = model.local_operator(h, torch.randn(2, 3, dtype=DTYPE), PredictorState())
    A2, B2 = model.local_operator(h, torch.randn(2, 3, dtype=DTYPE) * 100.0, PredictorState())
    torch.testing.assert_close(A1, A2, rtol=0.0, atol=0.0)
    torch.testing.assert_close(B1, B2, rtol=0.0, atol=0.0)

    # ... but a PAST u must reach a LATER context (else the check is vacuous)
    u0_a, u0_b = torch.zeros(2, 3, dtype=DTYPE), torch.ones(2, 3, dtype=DTYPE)
    _, s_a = model.step(h, u0_a, PredictorState())
    _, s_b = model.step(h, u0_b, PredictorState())
    A_a, _ = model.local_operator(h, u0_a, s_a)
    A_b, _ = model.local_operator(h, u0_a, s_b)
    assert (A_a - A_b).abs().max().item() > 1e-6


def test_step_is_exactly_affine_in_h_and_u_via_local_operator():
    """local_operator is the EXACT pair, not a linearization: reconstructing
    the step from it must reproduce step() to machine precision."""
    torch.manual_seed(13)
    model = StructuredPredictor(_SumBackbone(6, 2), d_z=3, r=3, d_u=2, delta=0.4, dtype=DTYPE)
    h = torch.randn(5, 6, dtype=DTYPE)
    u = torch.randn(5, 2, dtype=DTYPE)
    state = PredictorState(m=torch.randn(5, 6, dtype=DTYPE))
    h_next, _ = model.step(h, u, state)
    A_bar, B_bar = model.local_operator(h, u, state)
    assert A_bar.shape == (5, 6, 6) and B_bar.shape == (5, 6, 2)
    rebuilt = (A_bar @ h.unsqueeze(-1) + B_bar @ u.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(rebuilt, h_next, rtol=0.0, atol=1e-14)


def test_recurrence_stays_in_h_and_rollout_starts_fresh():
    """pi is read-only: perturbing it must not move a rollout at all. And the
    inherited rollout must equal a manual step loop started from a FRESH state
    (no context crossing an episode boundary)."""
    torch.manual_seed(17)
    model = StructuredPredictor(_SumBackbone(4, 2), d_z=3, r=2, d_u=2, delta=0.3, dtype=DTYPE)
    h0 = torch.randn(3, 4, dtype=DTYPE)
    u_seq = torch.randn(6, 3, 2, dtype=DTYPE)

    out = model.rollout(h0, u_seq)
    assert out.shape == (6, 3, 4)

    h, state = h0, PredictorState()
    manual = []
    for t in range(6):
        h, state = model.step(h, u_seq[t], state)
        manual.append(h)
    torch.testing.assert_close(out, torch.stack(manual), rtol=0.0, atol=0.0)

    with torch.no_grad():
        for p in model.pi.parameters():
            p.add_(torch.randn_like(p) * 10.0)
    torch.testing.assert_close(model.rollout(h0, u_seq), out, rtol=0.0, atol=0.0)

    # calling rollout twice must give the same answer: state is not carried over
    torch.testing.assert_close(model.rollout(h0, u_seq), out, rtol=0.0, atol=0.0)


def test_alpha_is_free_so_unstable_modes_are_representable():
    """PROTOCOL 13.2: no -exp / -softplus reparametrization may hide here.
    A positive alpha from the head must produce |eig(A_bar)| > 1."""
    model = StructuredPredictor(_SumBackbone(2, 1), d_z=2, r=1, d_u=1, delta=1.0, dtype=DTYPE)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.copy_(torch.tensor([0.8, 0.0, 1.0, 0.0], dtype=DTYPE))
    A_bar, _ = model.local_operator(
        torch.zeros(1, 2, dtype=DTYPE), torch.zeros(1, 1, dtype=DTYPE), PredictorState()
    )
    modulus = torch.linalg.eigvals(A_bar[0]).abs().max().item()
    print(f"[alpha free] alpha=0.8 -> |eig(A_bar)|={modulus:.6f}")
    assert modulus == pytest.approx(math.exp(0.8), abs=1e-12)
    assert modulus > 1.0


def test_constructor_rejects_bad_delta_and_r():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and strictly positive"):
            StructuredPredictor(_SumBackbone(2, 1), d_z=2, r=1, d_u=1, delta=bad, dtype=DTYPE)
    with pytest.raises(ValueError, match="r must be"):
        StructuredPredictor(_SumBackbone(2, 1), d_z=2, r=0, d_u=1, delta=1.0, dtype=DTYPE)


# ===========================================================================
# WAVE C2 -- the three real context producers behind the wrapper above.
#
# Every positive below is checked against arithmetic written OUT here in plain
# Python (math.erf / math.log1p / math.exp) plus the two independent matrix
# references already used above (torch.matrix_exp, Van Loan). The production
# formulas are never called to produce an expected value.
# ===========================================================================
from sld.predictors.min_ssm import (  # noqa: E402
    SelectiveSSMContextBackbone,
    SelectiveSSMPredictor,
)
from sld.predictors.mlp import MLPContextBackbone  # noqa: E402
from sld.predictors.structured import (  # noqa: E402
    MLPKPredictor,
    SSMKPredictor,
    TransformerKPredictor,
)
from sld.predictors.transformer import CausalTransformerContextBackbone  # noqa: E402

FACADES = (MLPKPredictor, TransformerKPredictor, SSMKPredictor)


def _facade(cls, seed: int = 0, **kw):
    """A facade at a declared, small, shared configuration."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        params = dict(d_z=3, d_u=2, delta=0.05, r=2, width=8, dtype=DTYPE)
        params.update(kw)
        return cls(**params)


def _gelu(x: float) -> float:
    """Exact (erf) GELU, from python's math -- not torch's kernel."""
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def _softplus(x: float) -> float:
    return math.log1p(math.exp(x)) if x < 30.0 else x


def _hand_operator(alpha: float, omega: float, delta: float, b_cont: list[float]):
    """(A_bar, B_bar) from scalars, via matrix_exp + Van Loan only."""
    G = _generator(alpha, omega)
    A_bar = torch.matrix_exp(delta * G)
    B_bar = _phi_van_loan(G, delta) @ torch.tensor(b_cont, dtype=DTYPE).reshape(-1, 1)
    return A_bar, B_bar


# ---------------------------------------------------------------------------
# wiring: the canonical signature really is wired to the backbones (R7 -- read
# the call site, not the docstring)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", FACADES)
def test_facade_builds_a_conforming_backbone_with_d_context_equal_to_width(cls):
    """R7 -- read the CALL SITE, not the docstring. Every dimension is given a
    DIFFERENT value (d_z=5, d_u=2, r=3, width=12 => d_h=6), so a facade that
    forwarded its arguments in the wrong order, or dropped `width`, cannot hide
    behind two slots that happen to hold the same number."""
    model = _facade(cls, d_z=5, d_u=2, r=3, width=12)
    assert isinstance(model.backbone, ContextBackbone)
    assert model.backbone.d_context == 12
    assert (model.d_z, model.d_u, model.r, model.d_h) == (5, 2, 3, 6)
    assert model.psi[0].in_features == 5
    assert model.head.in_features == 12
    assert model.head.out_features == 2 * 3 + 6 * 2

    # and the backbone itself must have been built on d_h / d_u, not on
    # whatever else was in scope
    bb = model.backbone
    if cls is MLPKPredictor:
        assert (bb.net[0].in_features, bb.net[0].out_features) == (6, 12)
        assert bb.net[2].out_features == 12
    elif cls is TransformerKPredictor:
        assert bb.q_in.in_features == 6  # the query sees h ONLY
        assert bb.pair_in.in_features == 8  # the cache sees the pair (h, u)
        assert bb.d_model == 12 and bb.d_head * bb.n_heads == 12
    else:
        assert bb.W_c.in_features == 6 + 12  # [h ; m]
        assert bb.W_in.in_features == 8  # [h ; u]
        assert bb.d_state == 12


@pytest.mark.parametrize("cls", FACADES)
def test_facades_are_thin_and_never_reimplement_the_transition(cls):
    """The three arms must be ONE treatment: a facade that defined its own
    step/local_operator/_operator_from_context would silently make them three."""
    for name in ("step", "local_operator", "_operator_from_context", "lift", "project"):
        assert getattr(cls, name) is getattr(StructuredPredictor, name), name


# ---------------------------------------------------------------------------
# POSITIVE 1 -- MLP backbone: two imposed contexts, SAME h and u
# ---------------------------------------------------------------------------
# r=1, d_u=1, Delta=1, width=2. Backbone = Linear(2,2) -> GELU -> Linear(2,2):
#   c(beta) = W2 @ gelu(h + beta) + b2,  with W2 = diag(2, 3), b2 = [0.1, -0.2]
# Head (2 -> 4): theta = W_head c + b_head, sliced [alpha | omega | vec B].
# With h = [1, 2] and the two biases below, by hand:
#   beta1 = [ 0.5, -1.5] -> pre = [1.5, 0.5]
#   beta2 = [-1.0,  0.25] -> pre = [0.0, 2.25]
# gelu(0) = 0 exactly, which is why beta2 is worth writing down: it puts one
# coordinate of the context on the ONE point where a wrong-but-plausible GELU
# (tanh approximation) still agrees, so the discrimination has to come from
# the other coordinate.
MLP_H = [1.0, 2.0]
MLP_U = 0.7
MLP_BETA1 = [0.5, -1.5]
MLP_BETA2 = [-1.0, 0.25]
MLP_W2 = [[2.0, 0.0], [0.0, 3.0]]
MLP_B2 = [0.1, -0.2]
HEAD_W = [[0.3, -0.2], [0.5, 0.1], [0.7, 0.0], [0.0, -0.4]]
HEAD_B = [-0.4, 0.2, 0.1, 0.3]


def _mlp_rig(beta: list[float]) -> MLPKPredictor:
    model = _facade(MLPKPredictor, d_z=2, d_u=1, delta=1.0, r=1, width=2)
    net = model.backbone.net
    with torch.no_grad():
        net[0].weight.copy_(torch.eye(2, dtype=DTYPE))
        net[0].bias.copy_(torch.tensor(beta, dtype=DTYPE))
        net[2].weight.copy_(torch.tensor(MLP_W2, dtype=DTYPE))
        net[2].bias.copy_(torch.tensor(MLP_B2, dtype=DTYPE))
        model.head.weight.copy_(torch.tensor(HEAD_W, dtype=DTYPE))
        model.head.bias.copy_(torch.tensor(HEAD_B, dtype=DTYPE))
    return model


def _mlp_by_hand(beta: list[float]):
    """c, A_bar, B_bar, h_next -- written out, torch only for expm/Van Loan."""
    c = [
        MLP_W2[i][i] * _gelu(MLP_H[i] + beta[i]) + MLP_B2[i] for i in range(2)
    ]
    theta = [HEAD_W[j][0] * c[0] + HEAD_W[j][1] * c[1] + HEAD_B[j] for j in range(4)]
    A_bar, B_bar = _hand_operator(theta[0], theta[1], 1.0, [theta[2], theta[3]])
    h = torch.tensor(MLP_H, dtype=DTYPE).reshape(2, 1)
    h_next = (A_bar @ h + B_bar * MLP_U).reshape(-1)
    return torch.tensor(c, dtype=DTYPE), A_bar, B_bar.reshape(-1), h_next


def test_mlp_backbone_two_contexts_match_the_hand_computation():
    h = torch.tensor([MLP_H], dtype=DTYPE)
    u = torch.tensor([[MLP_U]], dtype=DTYPE)
    got, want = [], []
    for beta in (MLP_BETA1, MLP_BETA2):
        model = _mlp_rig(beta)
        c = model.backbone.context_step(h, PredictorState())
        A_bar, B_bar = model.local_operator(h, u, PredictorState())
        h_next, _ = model.step(h, u, PredictorState())
        got.append((c[0], A_bar[0], B_bar[0, :, 0], h_next[0]))
        want.append(_mlp_by_hand(beta))

    for k, (g, w) in enumerate(zip(got, want)):
        errs = [(a - b).abs().max().item() for a, b in zip(g, w)]
        print(f"[mlp-hand] beta{k + 1} c={g[0].tolist()} err(c,A,B,h_next)={errs}")
        for a, b in zip(g, w):
            torch.testing.assert_close(a, b, rtol=0.0, atol=1e-12)

    # THE difference itself, which is what "the context reaches the operator"
    # means -- same h, same u, two contexts
    d_A = (got[0][1] - got[1][1]).abs().max().item()
    d_h = (got[0][3] - got[1][3]).abs().max().item()
    print(f"[mlp-hand] operator spread={d_A:.6e} output spread={d_h:.6e}")
    torch.testing.assert_close(
        got[0][1] - got[1][1], want[0][1] - want[1][1], rtol=0.0, atol=1e-12
    )
    torch.testing.assert_close(
        got[0][3] - got[1][3], want[0][3] - want[1][3], rtol=0.0, atol=1e-12
    )
    assert d_A > 0.1 and d_h > 0.1  # non-vacuous: the two contexts really differ


def test_mlp_backbone_is_memoryless_at_the_call_site():
    """c_t = MLP(h_t) and nothing else: garbage in state.m must not move it,
    and absorb must hand back the SAME state object."""
    model = _mlp_rig(MLP_BETA1)
    h = torch.tensor([MLP_H], dtype=DTYPE)
    u = torch.tensor([[MLP_U]], dtype=DTYPE)
    clean = model.backbone.context_step(h, PredictorState())
    dirty = model.backbone.context_step(h, PredictorState(m=torch.full((1, 2), 9.0, dtype=DTYPE)))
    torch.testing.assert_close(clean, dirty, rtol=0.0, atol=0.0)
    state = PredictorState()
    assert model.backbone.absorb(h, u, state) is state
    assert model.rollout(h, u.unsqueeze(0).repeat(4, 1, 1)).shape == (4, 1, 2)


# ---------------------------------------------------------------------------
# POSITIVE 2 -- SSM backbone: SAME h and u, two KNOWN recurrent states m
# ---------------------------------------------------------------------------
# d_h = 2, width = d_state = 2, d_u = 1, Delta = 1.
#   c   = W_c [h; m] + b_c
#   d_t = clamp(softplus(W_delta c + b_delta), 1e-4, 10)
#   m'  = exp(d_t * (-exp(a_log))) * m + d_t * (W_b c + b_b) * (W_in [h; u] + b_in)
SSM_H = [1.0, 2.0]
SSM_U = 0.7
SSM_M_A = [0.4, -0.6]
SSM_M_B = [-1.25, 2.0]
# a third state chosen to drive softplus(W_delta c) PAST delta_max: with
# c = [51.1, 51.8] the raw step sizes are ~25.3 and ~13.2, so both coordinates
# land on the clamp. Added after a mutation that deleted the clamp survived the
# two states above, which keep delta_t inside the interval (wave report).
SSM_M_CLAMP = [100.0, 100.0]
SSM_WC = [[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5]]
SSM_BC = [0.1, -0.2]
SSM_WD = [[0.5, 0.0], [0.0, 0.25]]
SSM_BD = [-0.3, 0.2]
SSM_WB = [[1.0, 0.0], [0.0, -1.0]]
SSM_BB = [0.05, 0.05]
SSM_WIN = [[0.2, 0.1, 0.4], [-0.3, 0.5, 0.6]]
SSM_BIN = [0.0, 0.1]
SSM_A_CONT = [-0.5, -0.8]  # a_log = log(0.5), log(0.8)


def _ssm_rig() -> SSMKPredictor:
    model = _facade(SSMKPredictor, d_z=2, d_u=1, delta=1.0, r=1, width=2)
    bb = model.backbone
    with torch.no_grad():
        bb.W_c.weight.copy_(torch.tensor(SSM_WC, dtype=DTYPE))
        bb.W_c.bias.copy_(torch.tensor(SSM_BC, dtype=DTYPE))
        bb.W_delta.weight.copy_(torch.tensor(SSM_WD, dtype=DTYPE))
        bb.W_delta.bias.copy_(torch.tensor(SSM_BD, dtype=DTYPE))
        bb.W_b.weight.copy_(torch.tensor(SSM_WB, dtype=DTYPE))
        bb.W_b.bias.copy_(torch.tensor(SSM_BB, dtype=DTYPE))
        bb.W_in.weight.copy_(torch.tensor(SSM_WIN, dtype=DTYPE))
        bb.W_in.bias.copy_(torch.tensor(SSM_BIN, dtype=DTYPE))
        bb.a_log.copy_(torch.tensor([math.log(-a) for a in SSM_A_CONT], dtype=DTYPE))
        model.head.weight.copy_(torch.tensor(HEAD_W, dtype=DTYPE))
        model.head.bias.copy_(torch.tensor(HEAD_B, dtype=DTYPE))
    return model


def _ssm_by_hand(m: list[float]):
    x = SSM_H + m
    c = [sum(SSM_WC[i][j] * x[j] for j in range(4)) + SSM_BC[i] for i in range(2)]
    theta = [HEAD_W[j][0] * c[0] + HEAD_W[j][1] * c[1] + HEAD_B[j] for j in range(4)]
    A_bar, B_bar = _hand_operator(theta[0], theta[1], 1.0, [theta[2], theta[3]])

    d_t = [
        min(max(_softplus(sum(SSM_WD[i][j] * c[j] for j in range(2)) + SSM_BD[i]), 1e-4), 10.0)
        for i in range(2)
    ]
    b_t = [sum(SSM_WB[i][j] * c[j] for j in range(2)) + SSM_BB[i] for i in range(2)]
    hu = SSM_H + [SSM_U]
    v_t = [sum(SSM_WIN[i][j] * hu[j] for j in range(3)) + SSM_BIN[i] for i in range(2)]
    m_next = [
        math.exp(d_t[i] * SSM_A_CONT[i]) * m[i] + d_t[i] * b_t[i] * v_t[i] for i in range(2)
    ]
    return (
        torch.tensor(c, dtype=DTYPE),
        A_bar,
        B_bar.reshape(-1),
        torch.tensor(m_next, dtype=DTYPE),
    )


def test_ssm_backbone_two_known_states_match_the_hand_computation():
    model = _ssm_rig()
    h = torch.tensor([SSM_H], dtype=DTYPE)
    u = torch.tensor([[SSM_U]], dtype=DTYPE)
    got, want = [], []
    for m in (SSM_M_A, SSM_M_B, SSM_M_CLAMP):
        state = PredictorState(m=torch.tensor([m], dtype=DTYPE))
        c = model.backbone.context_step(h, state)
        A_bar, B_bar = model.local_operator(h, u, state)
        m_next = model.backbone.absorb(h, u, state).m
        got.append((c[0], A_bar[0], B_bar[0, :, 0], m_next[0]))
        want.append(_ssm_by_hand(m))

    for k, (g, w) in enumerate(zip(got, want)):
        errs = [(a - b).abs().max().item() for a, b in zip(g, w)]
        print(f"[ssm-hand] m{k} c={g[0].tolist()} err(c,A,B,m_next)={errs}")
        # the two O(1) states get the strict absolute tolerance; the clamp
        # state drives |A_bar| to ~1e2, where atol=1e-12 would be measuring
        # float64 rounding (2.2e-16 * 1e2 * a ~10-operation chain) rather than
        # correctness -- its measured relative error is 2.7e-14
        rtol = 1e-13 if k == 2 else 0.0
        for a, b in zip(g, w):
            torch.testing.assert_close(a, b, rtol=rtol, atol=1e-12)

    d_c = (got[0][0] - got[1][0]).abs().max().item()
    d_A = (got[0][1] - got[1][1]).abs().max().item()
    print(f"[ssm-hand] context spread={d_c:.6e} operator spread={d_A:.6e}")
    assert d_c > 0.1 and d_A > 0.1

    # the third state must really have been on the clamp, otherwise the case
    # silently degrades into a third copy of the first two
    c_clamp = got[2][0]
    raw = [
        _softplus(sum(SSM_WD[i][j] * c_clamp[j].item() for j in range(2)) + SSM_BD[i])
        for i in range(2)
    ]
    print(f"[ssm-hand] clamp case: raw softplus={[f'{r:.3f}' for r in raw]} vs delta_max=10.0")
    assert min(raw) > 10.0


def test_ssm_backbone_shares_the_one_recurrence_with_the_standalone_family():
    """Same parameters in both objects => bit-identical m_next. This is what
    "one context producer, two transitions" has to mean; it also catches a
    future edit that touches only one of the two call sites."""
    torch.manual_seed(3)
    d_h, d_u, width = 4, 2, 6
    bb = SelectiveSSMContextBackbone(d_h, d_u, width, dtype=DTYPE)
    ref = SelectiveSSMPredictor(d_h=d_h, d_u=d_u, d_state=width).to(DTYPE)
    with torch.no_grad():
        for name in ("W_c", "W_delta", "W_b", "W_in"):
            getattr(ref, name).weight.copy_(getattr(bb, name).weight)
            getattr(ref, name).bias.copy_(getattr(bb, name).bias)
        ref.a_log.copy_(bb.a_log)
    h = torch.randn(3, d_h, dtype=DTYPE)
    u = torch.randn(3, d_u, dtype=DTYPE)
    state = PredictorState(m=torch.randn(3, width, dtype=DTYPE))
    _, ref_state = ref.step(h, u, state)
    torch.testing.assert_close(bb.absorb(h, u, state).m, ref_state.m, rtol=0.0, atol=0.0)


def test_selective_ssm_state_dict_keys_survive_the_extraction():
    """NON-REGRESSION for wave C2's refactor: the recurrence moved to a free
    function precisely so no key of the historical family would move. Pinned
    by NAME here (a set literal, not a re-read of the model), so any future
    'let's put the parameters in a sub-module' shows up as a red test rather
    than as a checkpoint that no longer loads."""
    keys = set(SelectiveSSMPredictor(d_h=4, d_u=2, d_state=8).state_dict())
    assert keys == {
        "a_log",
        "W_c.weight", "W_c.bias",
        "W_delta.weight", "W_delta.bias",
        "W_b.weight", "W_b.bias",
        "W_in.weight", "W_in.bias",
        "W_out.weight", "W_out.bias",
    }


# ---------------------------------------------------------------------------
# THE SENTINELS, and the mutants that must fail them
# ---------------------------------------------------------------------------
def _context_spread_over_h(model, h_a: Tensor, h_b: Tensor) -> float:
    """max |c(h_a) - c(h_b)| at the same (empty) past."""
    ca = model.backbone.context_step(h_a, PredictorState())
    cb = model.backbone.context_step(h_b, PredictorState())
    return (ca - cb).abs().max().item()


def _context_spread_over_prefix(model, h: Tensor, pre_a, pre_b) -> float:
    """max |c(h | prefix_a) - c(h | prefix_b)| at the SAME current h."""
    outs = []
    for prefix in (pre_a, pre_b):
        state = PredictorState()
        for h_s, u_s in prefix:
            state = model.backbone.absorb(h_s, u_s, state)
        outs.append(model.backbone.context_step(h, state))
    return (outs[0] - outs[1]).abs().max().item()


def _probe_inputs(d_h: int = 4, d_u: int = 2, seed: int = 5):
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, dtype=DTYPE)

    h = rnd(2, d_h)
    h_b = rnd(2, d_h)
    pre_a = [(rnd(2, d_h), rnd(2, d_u)) for _ in range(3)]
    pre_b = [(rnd(2, d_h), rnd(2, d_u)) for _ in range(3)]
    return h, h_b, pre_a, pre_b


class _DeafBackbone(nn.Module):
    """MUTANT: keeps the real backbone's memory but reports a CONSTANT context.

    This is the failure the whole wave is about -- a wrapper that carries a
    backbone it never listens to. Its rollout still runs, its shapes are still
    right, and it is still a perfectly valid LPV cell with a constant operator.
    Only the sentinels below tell it apart from the real thing.
    """

    def __init__(self, inner) -> None:
        super().__init__()
        self.inner = inner
        self.d_context = inner.d_context
        self.register_buffer("c0", torch.full((inner.d_context,), 0.3, dtype=DTYPE))

    def context_step(self, h: Tensor, state: PredictorState) -> Tensor:
        return self.c0.expand(h.shape[0], -1)

    def absorb(self, h: Tensor, u: Tensor, state: PredictorState) -> PredictorState:
        return self.inner.absorb(h, u, state)


@pytest.mark.parametrize("cls", FACADES)
def test_every_backbone_context_responds_to_h_and_the_deaf_mutant_does_not(cls):
    """NEGATIVE 1 (half a): 3/3 contexts alive in h."""
    model = _facade(cls)
    h_a, h_b, _, _ = _probe_inputs()
    real = _context_spread_over_h(model, h_a, h_b)

    deaf = StructuredPredictor(
        _DeafBackbone(model.backbone), d_z=3, r=2, d_u=2, delta=0.05, dtype=DTYPE
    )
    mutant = _context_spread_over_h(deaf, h_a, h_b)
    print(f"[sentinel-h] {cls.__name__}: real={real:.6e} deaf={mutant:.6e}")
    assert real > 1e-3
    assert mutant == 0.0


@pytest.mark.parametrize("cls", FACADES)
def test_prefix_sentinel_separates_the_memoryful_arms_from_mlp_and_from_the_deaf_mutant(cls):
    """NEGATIVE 1 (half b): the past reaches the context -- except for the MLP
    arm, whose memorylessness is the DEFINING property of that cell and is
    asserted here as an exact zero, not waved through."""
    model = _facade(cls)
    h, _, pre_a, pre_b = _probe_inputs()
    real = _context_spread_over_prefix(model, h, pre_a, pre_b)

    deaf = StructuredPredictor(
        _DeafBackbone(model.backbone), d_z=3, r=2, d_u=2, delta=0.05, dtype=DTYPE
    )
    mutant = _context_spread_over_prefix(deaf, h, pre_a, pre_b)
    print(f"[sentinel-prefix] {cls.__name__}: real={real:.6e} deaf={mutant:.6e}")
    if cls is MLPKPredictor:
        assert real == 0.0  # by construction, and it must stay exactly 0
    else:
        assert real > 1e-3
    assert mutant == 0.0


class _LeakyRollout(StructuredPredictor):
    """MUTANT: keeps the context state on the module between rollouts -- the
    one-line 'optimization' that silently makes episode k+1 depend on k."""

    def rollout(self, h0: Tensor, u_seq: Tensor) -> Tensor:
        state = getattr(self, "_leak", PredictorState())
        h, outs = h0, []
        for t in range(u_seq.shape[0]):
            h, state = self.step(h, u_seq[t], state)
            outs.append(h)
        self._leak = state
        return torch.stack(outs, dim=0)


_MEMORYFUL_FACADES = (TransformerKPredictor, SSMKPredictor)


@pytest.mark.parametrize("cls", _MEMORYFUL_FACADES)
def test_no_state_crosses_an_episode_boundary_at_the_rollout_call_site(cls):
    """NEGATIVE 3, at the REAL call site: rollout, twice, same inputs. Bitwise
    equal for the real arms; the leaky mutant is caught by the same two lines.

    MLP is deliberately NOT parametrized here: its backbone has no `m` at all
    (``MLPContextBackbone.absorb`` is a no-op, see ``mlp.py``), so
    ``_LeakyRollout`` on that arm has nothing to leak and would drift by
    0.000000e+00 even under a mutation that deletes context-dependence
    entirely -- a case that cannot fail. See
    ``test_mlp_backbone_has_no_state_to_leak_across_an_episode_boundary``
    below for the non-vacuous version of "no state crosses a boundary" for
    that arm.

    The leaky-mutant half needs the operator to depend on context (else a
    leaked state can never move the output); PROTOCOL 13.6 zeroes W_theta at
    construction, so this bench poses its own non-zero W_theta before the
    leaky copy."""
    model = _facade(cls)
    if model.head.weight.numel() > 0:
        with torch.no_grad():
            model.head.weight.copy_(
                torch.randn(*model.head.weight.shape, dtype=DTYPE, generator=torch.Generator().manual_seed(9))
            )
    h0 = torch.randn(2, model.d_h, dtype=DTYPE, generator=torch.Generator().manual_seed(7))
    u_seq = torch.randn(6, 2, 2, dtype=DTYPE, generator=torch.Generator().manual_seed(8))

    first = model.rollout(h0, u_seq)
    second = model.rollout(h0, u_seq)
    print(f"[episode] {cls.__name__}: rollout drift={(first - second).abs().max().item():.3e}")
    torch.testing.assert_close(second, first, rtol=0.0, atol=0.0)

    leaky = _LeakyRollout(model.backbone, d_z=3, r=2, d_u=2, delta=0.05, dtype=DTYPE)
    with torch.no_grad():  # same head as `model`, so only the state differs
        leaky.head.weight.copy_(model.head.weight)
        leaky.head.bias.copy_(model.head.bias)
    leak_first = leaky.rollout(h0, u_seq)
    leak_second = leaky.rollout(h0, u_seq)
    drift = (leak_first - leak_second).abs().max().item()
    print(f"[episode] {cls.__name__}: leaky mutant drift={drift:.6e}")
    assert drift > 1e-6


def test_mlp_backbone_has_no_state_to_leak_across_an_episode_boundary():
    """NEGATIVE 3 for the MLP arm, non-vacuous version. The parametrized
    leaky-rollout mutant above is meaningless here: with no `m` to carry,
    `_LeakyRollout` on MLP is bitwise identical to the real rollout, so that
    bench's drift is 0.000000e+00 even for a correct implementation -- it
    cannot distinguish real code from broken code. The actual claim to test
    for a memoryless backbone is narrower and falsifiable: (1) `absorb` never
    writes anything into `state.m` -- there is truly nothing to carry -- and
    (2) `context_step` is blind to whatever `state` it is handed, so a future
    state leaked in by a caller could not reach the context even if one
    existed. If MLPContextBackbone were ever given real memory without this
    reasoning being updated, this test would catch it."""
    model = _facade(MLPKPredictor)
    bb = model.backbone
    h = torch.randn(2, model.d_h, dtype=DTYPE, generator=torch.Generator().manual_seed(21))
    u = torch.randn(2, model.d_u, dtype=DTYPE, generator=torch.Generator().manual_seed(22))

    absorbed = bb.absorb(h, u, PredictorState())
    assert absorbed.m is None  # nothing was ever written into the state

    real_context = bb.context_step(h, PredictorState())
    junk_state = PredictorState(
        m=torch.randn(2, 7, dtype=DTYPE, generator=torch.Generator().manual_seed(23))
    )
    junk_context = bb.context_step(h, junk_state)
    print(
        f"[episode] MLPKPredictor: context spread over junk state="
        f"{(junk_context - real_context).abs().max().item():.6e}"
    )
    torch.testing.assert_close(junk_context, real_context, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("cls", FACADES)
def test_perturbing_the_future_leaves_every_earlier_output_bit_identical(cls):
    """NEGATIVE 4. atol = rtol = 0 on the past, with the POSITIVE CONTROL that
    the same bench does see a difference when the PAST is perturbed instead --
    otherwise 'no leak' would just mean 'nothing measured'."""
    model = _facade(cls)
    K = 4
    T = 8
    gen = torch.Generator().manual_seed(11)
    h0 = torch.randn(2, model.d_h, dtype=DTYPE, generator=gen)
    u_seq = torch.randn(T, 2, 2, dtype=DTYPE, generator=gen)
    base = model.rollout(h0, u_seq)

    future = u_seq.clone()
    future[K:] += 5.0
    out_future = model.rollout(h0, future)
    past_gap = (out_future[:K] - base[:K]).abs().max().item()
    at_k = (out_future[K] - base[K]).abs().max().item()
    print(f"[causality] {cls.__name__}: gap on t<{K} = {past_gap:.3e}, at t={K} = {at_k:.6e}")
    torch.testing.assert_close(out_future[:K], base[:K], rtol=0.0, atol=0.0)
    assert at_k > 1e-6  # the perturbation is real, it just cannot travel back

    # POSITIVE CONTROL: perturb the PAST and the same window must move
    past = u_seq.clone()
    past[0] += 5.0
    out_past = model.rollout(h0, past)
    control = (out_past[:K] - base[:K]).abs().max().item()
    print(f"[causality] {cls.__name__}: positive control on the same window = {control:.6e}")
    assert control > 1e-6


@pytest.mark.parametrize("cls", FACADES)
def test_u_t_cannot_reach_c_t_but_does_reach_c_next(cls):
    """NEGATIVE 5, three halves (the third was added after a mutation survived
    the first two).

    Half one alone would pass on a backbone that ignores u entirely; half two
    alone would pass on one that leaks it. Both read the operator, so BOTH are
    blind to a `step` that absorbs before it reads the context: `local_operator`
    would still be innocent while `step` itself leaked. Half three closes that
    by testing `step` at the REAL call site (R7) through the property the
    exclusion exists to protect (PROTOCOL 13.1): with the state frozen, `step`
    is EXACTLY affine in u, so the operator read at one u must reproduce `step`
    at every other u."""
    model = _facade(cls)
    gen = torch.Generator().manual_seed(13)
    h = torch.randn(2, model.d_h, dtype=DTYPE, generator=gen)
    u_a = torch.randn(2, 2, dtype=DTYPE, generator=gen)
    u_b = u_a + 3.0
    # a non-trivial past, so the state m is not the degenerate None
    state = PredictorState()
    for _ in range(2):
        state = model.backbone.absorb(
            torch.randn(2, model.d_h, dtype=DTYPE, generator=gen),
            torch.randn(2, 2, dtype=DTYPE, generator=gen),
            state,
        )

    # HALF ONE: same step -- c_t, A_bar_t, B_bar_t must not move at all
    c = model.backbone.context_step(h, state)
    A_a, B_a = model.local_operator(h, u_a, state)
    A_b, B_b = model.local_operator(h, u_b, state)
    torch.testing.assert_close(A_a, A_b, rtol=0.0, atol=0.0)
    torch.testing.assert_close(B_a, B_b, rtol=0.0, atol=0.0)

    # HALF TWO: the NEXT step must see it (except for the memoryless arm)
    next_a = model.backbone.context_step(h, model.backbone.absorb(h, u_a, state))
    next_b = model.backbone.context_step(h, model.backbone.absorb(h, u_b, state))
    forward = (next_a - next_b).abs().max().item()
    same_step = (A_a - A_b).abs().max().item()
    print(
        f"[u-exclusion] {cls.__name__}: |dA_t/du_t|={same_step:.1e} "
        f"|dc_(t+1)/du_t|={forward:.6e} (|c_t|={c.abs().max().item():.3e})"
    )
    if cls is MLPKPredictor:
        assert forward == 0.0  # memoryless: u never reaches ANY context, declared
    else:
        assert forward > 1e-6

    # HALF THREE: `step` itself, at the real call site. The operator read at
    # u = 0 must reproduce step at u_a and at u_b -- which is false the moment
    # the context of step t is built from a state that has already absorbed u_t.
    A0, B0 = model.local_operator(h, torch.zeros_like(u_a), state)
    for tag, u in (("u_a", u_a), ("u_b", u_b)):
        h_next, _ = model.step(h, u, state)
        affine = (A0 @ h.unsqueeze(-1) + B0 @ u.unsqueeze(-1)).squeeze(-1)
        err = (h_next - affine).abs().max().item()
        print(f"[u-exclusion] {cls.__name__}: step-vs-affine at {tag} err={err:.3e}")
        torch.testing.assert_close(h_next, affine, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("cls", FACADES)
def test_fifty_step_rollout_is_finite_from_the_real_entry_point(cls):
    """H = 50 over 20 INIT SEEDS, started where every real caller starts:
    ``h0 = lift(z_0)``, at the declared config (width 8, r = 2), swept over
    Delta = 0.02, 0.10, 0.50. This is a WIRING check and NOT a stability
    claim; ``alpha`` is FREE by PROTOCOL 13.2, so nothing bounds
    ``exp(Delta*alpha)``, and ``c_t`` is an unnormalized read of ``h_t``,
    which can close a runaway loop: ``|h|`` up -> ``|c|`` up -> ``alpha`` up
    -> ``|h|`` up faster.

    PROTOCOL 13.6 measured this BEFORE the fix, with the default random
    ``W_theta`` (20 seeds, ``d_z=4``, ``r=2``, ``width=16``), finite count:

        Delta   mlp_k    transformer_k    ssm_k
        0.02    20/20    20/20            20/20
        0.10    20/20    19/20            20/20
        0.50    15/20    8/20             13/20

    Zeroing ``W_theta`` at construction (this module) removes the confound at
    the starting point: 20/20 finite is now expected on every arm at every
    Delta below. See ``test_fifty_step_rollout_finiteness_bench_can_detect_a_
    runaway`` for the positive control proving this bench can still fail.
    """
    for delta in (0.02, 0.10, 0.50):
        finite = 0
        worst = 0.0
        for seed in range(20):
            model = _facade(cls, seed=seed, delta=delta)
            gen = torch.Generator().manual_seed(21)
            h0 = model.lift(torch.randn(2, model.d_z, dtype=DTYPE, generator=gen))
            out = model.rollout(h0, torch.randn(50, 2, 2, dtype=DTYPE, generator=gen))
            assert out.shape == (50, 2, model.d_h)
            if torch.isfinite(out).all():
                finite += 1
                worst = max(worst, out.abs().max().item())
        print(
            f"[H=50] {cls.__name__} delta={delta}: finite on {finite}/20 seeds, "
            f"worst max|h| = {worst:.6e}"
        )
        assert finite == 20


@pytest.mark.parametrize("cls", FACADES)
def test_fifty_step_rollout_finiteness_bench_can_detect_a_runaway(cls):
    """Positive control for the bench above: a finiteness test that can never
    fail proves nothing. Force ``alpha_0`` strongly positive by hand (so
    ``exp(Delta*alpha)`` grows every step instead of shrinking) and confirm
    the SAME rollout bench, same entry point, same Delta = 0.50, counts
    non-finite seeds -- i.e. it is capable of catching the runaway PROTOCOL
    13.6 measured, not just reporting a number that is always 20/20."""
    non_finite = 0
    for seed in range(20):
        model = _facade(cls, seed=seed, delta=0.50)
        with torch.no_grad():
            model.head.bias[: model.r].fill_(50.0)  # force a strongly unstable alpha
        gen = torch.Generator().manual_seed(21)
        h0 = model.lift(torch.randn(2, model.d_z, dtype=DTYPE, generator=gen))
        out = model.rollout(h0, torch.randn(50, 2, 2, dtype=DTYPE, generator=gen))
        if not torch.isfinite(out).all():
            non_finite += 1
    print(f"[H=50 positive control] {cls.__name__}: non-finite on {non_finite}/20 seeds")
    assert non_finite > 0


def test_transformer_backbone_refuses_to_run_past_max_len():
    model = _facade(TransformerKPredictor, max_len=3)
    h = torch.randn(1, model.d_h, dtype=DTYPE)
    u = torch.randn(1, 2, dtype=DTYPE)
    state = PredictorState()
    for _ in range(3):
        h, state = model.step(h, u, state)
    assert state.m.shape[4] == 3
    with pytest.raises(ValueError, match="max_len"):
        model.step(h, u, state)


def test_transformer_and_ssm_backbones_reject_impossible_configurations():
    with pytest.raises(ValueError, match="divisible"):
        CausalTransformerContextBackbone(4, 2, width=9, n_heads=2)
    with pytest.raises(ValueError, match="finite"):
        SelectiveSSMContextBackbone(4, 2, 8, delta_min=float("nan"))
    with pytest.raises(ValueError, match="delta_min < delta_max"):
        SelectiveSSMContextBackbone(4, 2, 8, delta_min=1.0, delta_max=0.5)
    assert MLPContextBackbone(4, 8).d_context == 8


# ---------------------------------------------------------------------------
# r has two different defaults depending on the entry point (budget.py vs
# these classes' own signature) -- pinned so the divergence stays visible
# instead of silent.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", FACADES)
def test_r_default_differs_between_get_predictor_and_direct_construction(cls):
    """``get_predictor`` routes through ``budget._kwargs_for``, which sets
    ``r=d_z`` (2x lifting ratio, matching koopman_ssm's ``n_pairs=d_z``); the
    facade classes themselves default to ``r=2``. Same knob, two different
    values depending on the door you walked in -- both are correct for their
    own path, but a reader who only saw one of the two would be surprised by
    the other."""
    from sld.predictors.registry import get_predictor

    family = {MLPKPredictor: "mlp_k", TransformerKPredictor: "transformer_k", SSMKPredictor: "ssm_k"}[cls]
    d_z = 4
    via_budget = get_predictor(family, d_z, 1, delta=0.05)
    assert via_budget.r == d_z

    direct = cls(d_z=d_z, d_u=1, delta=0.05, dtype=DTYPE)
    assert direct.r == 2
