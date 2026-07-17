"""Tests for KoopmanSSMPredictor.

Discipline (REGLES sections 5, 6):
 - float64 for every exactness/gold test; references built INDEPENDENTLY of the
   path under test (Van Loan matrix_exp, numpy analytic eigenvalues, explicit
   python loops), never through the model's own closed form.
 - local generators only (torch.Generator / np.random.default_rng) for data and
   random tensors; ``torch.manual_seed`` is used solely to make nn.Module
   parameter init reproducible.
 - tolerances tied to the claim: exact steps at 1e-12/1e-14, autograd Jacobians
   at 1e-10, the gold eigenvalue gate at 1e-2 relative (the wave gate).
"""

import numpy as np
import pytest
import torch
import torch.func as tfunc
import torch.nn.functional as F

from sld.predictors.base import PredictorState
from sld.predictors.blocks2x2 import block_diagonal
from sld.predictors.koopman_ssm import KoopmanSSMPredictor
from sld.systems.base import DynamicalSystem
from sld.systems.damped_pendulum import DampedPendulum

DTYPE = torch.float64


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.linalg.norm(a - b) / torch.linalg.norm(b)).item()


# ---------------------------------------------------------------------------
# (a) exactness of the "constant" regime step vs an independent Van Loan ZOH
# ---------------------------------------------------------------------------
def test_step_exact_constant():
    torch.manual_seed(0)
    n_pairs, d_u, delta = 3, 2, 0.1
    model = KoopmanSSMPredictor(
        d_z=4, n_pairs=n_pairs, d_u=d_u, delta=delta, regime="constant", dtype=DTYPE
    )
    d_h = 2 * n_pairs
    # independent reference: Van Loan augmented matrix M = [[A_cont, B_cont],[0,0]];
    # expm(delta*M) has the exact ZOH (A_bar, B_bar_0) as its top-left / top-right
    # blocks -- matrix_exp, NOT the closed-form path step uses internally.
    A_cont = block_diagonal(model.blocks.continuous_blocks()).detach()
    B_cont = model.B_cont.detach()
    M = torch.zeros(d_h + d_u, d_h + d_u, dtype=DTYPE)
    M[:d_h, :d_h] = A_cont
    M[:d_h, d_h:] = B_cont
    E = torch.matrix_exp(delta * M)
    A_ref, B_ref = E[:d_h, :d_h], E[:d_h, d_h:]

    g = torch.Generator().manual_seed(4)
    h = torch.randn(6, d_h, dtype=DTYPE, generator=g)
    u = torch.randn(6, d_u, dtype=DTYPE, generator=g)
    manual = h @ A_ref.T + u @ B_ref.T
    out, _ = model.step(h, u, PredictorState())
    err = (out - manual).abs().max().item()
    print(f"[step_exact_constant] max abs err vs Van Loan = {err:.3e}")
    assert err < 1e-12


# ---------------------------------------------------------------------------
# (b) bilinear interaction einsum vs explicit loop; and the affine point
# ---------------------------------------------------------------------------
def test_bilinear_interaction_and_affine_point():
    torch.manual_seed(0)
    n_pairs, d_u, delta = 3, 2, 0.1
    d_h = 2 * n_pairs
    model = KoopmanSSMPredictor(
        d_z=4, n_pairs=n_pairs, d_u=d_u, delta=delta, regime="bilinear", dtype=DTYPE
    )
    gj = torch.Generator().manual_seed(7)
    with torch.no_grad():
        model.B_j.copy_(0.5 * torch.randn(model.B_j.shape, dtype=DTYPE, generator=gj))

    g = torch.Generator().manual_seed(8)
    h = torch.randn(5, d_h, dtype=DTYPE, generator=g)
    u = torch.randn(5, d_u, dtype=DTYPE, generator=g)
    out, _ = model.step(h, u, PredictorState())
    A_bar, B_bar_0 = model._discrete_affine()
    interaction = (out - (h @ A_bar.T + u @ B_bar_0.T)).detach()

    # independent path: explicit sum_j u_j (B_bar_j h) over batch and channel
    manual = torch.zeros_like(interaction)
    Bj = model.B_j.detach()
    for b in range(h.shape[0]):
        for j in range(d_u):
            manual[b] += u[b, j] * (Bj[j] @ h[b])
    err = (interaction - manual).abs().max().item()
    print(f"[bilinear interaction] einsum vs loop max abs err = {err:.3e}")
    assert err < 1e-14

    # affine point: with B_j = 0, bilinear == constant on shared params, exactly
    torch.manual_seed(0)
    m_const = KoopmanSSMPredictor(
        d_z=4, n_pairs=n_pairs, d_u=d_u, delta=delta, regime="constant", dtype=DTYPE
    )
    torch.manual_seed(0)
    m_bilin = KoopmanSSMPredictor(
        d_z=4, n_pairs=n_pairs, d_u=d_u, delta=delta, regime="bilinear", dtype=DTYPE
    )
    m_bilin.blocks.load_state_dict(m_const.blocks.state_dict())
    with torch.no_grad():
        m_bilin.B_cont.copy_(m_const.B_cont)  # B_j stays at its zeros init
    oc, _ = m_const.step(h, u, PredictorState())
    ob, _ = m_bilin.step(h, u, PredictorState())
    err2 = (oc - ob).abs().max().item()
    print(f"[affine point] bilinear(B_j=0) vs constant max abs err = {err2:.3e}")
    assert err2 == 0.0


# ---------------------------------------------------------------------------
# (c) local_operator exactness (constant/bilinear) & selective coherence
# ---------------------------------------------------------------------------
def test_local_operator_exact_constant_bilinear():
    for regime in ("constant", "bilinear"):
        torch.manual_seed(0)
        model = KoopmanSSMPredictor(
            d_z=3, n_pairs=2, d_u=2, delta=0.1, regime=regime, dtype=DTYPE
        )
        if regime == "bilinear":
            gj = torch.Generator().manual_seed(5)
            with torch.no_grad():
                model.B_j.copy_(
                    0.3 * torch.randn(model.B_j.shape, dtype=DTYPE, generator=gj)
                )
        g = torch.Generator().manual_seed(1)
        h = torch.randn(4, model.d_h, dtype=DTYPE, generator=g)
        u = torch.randn(4, model.d_u, dtype=DTYPE, generator=g)
        A, B = model.local_operator(h, u, PredictorState())

        def single_step(hh, uu):
            out, _ = model.step(hh.unsqueeze(0), uu.unsqueeze(0), PredictorState())
            return out.squeeze(0)

        worst = 0.0
        for b in range(h.shape[0]):
            Jh = tfunc.jacrev(single_step, argnums=0)(h[b], u[b])
            Ju = tfunc.jacrev(single_step, argnums=1)(h[b], u[b])
            worst = max(worst, _rel(Jh, A[b]), _rel(Ju, B[b]))
        print(f"[local_op {regime}] worst rel err vs jacrev = {worst:.3e}")
        assert worst < 1e-10


def test_local_operator_selective_coherence():
    torch.manual_seed(0)
    model = KoopmanSSMPredictor(
        d_z=4, n_pairs=3, d_u=2, delta=0.05, regime="selective", dtype=DTYPE
    )
    # keep delta_t = clamp(softplus(c), 1e-4, 0.5) strictly inside its bounds so
    # the code c genuinely modulates the block -- otherwise a saturated clamp
    # would zero d(delta_t)/dh and the "freezing is real" check would be vacuous
    with torch.no_grad():
        model.W_c.weight.mul_(0.3)
        model.W_c.bias.fill_(-1.5)
    g = torch.Generator().manual_seed(2)
    h = torch.randn(4, model.d_h, dtype=DTYPE, generator=g)
    u = torch.randn(4, model.d_u, dtype=DTYPE, generator=g)
    A, _ = model.local_operator(h, u, PredictorState())
    c = model.W_c(torch.cat([h, u], dim=-1)).detach()

    def frozen(hh, uu, cc):
        return model._frozen_step(
            hh.unsqueeze(0), uu.unsqueeze(0), cc.unsqueeze(0)
        ).squeeze(0)

    worst = 0.0
    for b in range(h.shape[0]):
        Jh = tfunc.jacrev(frozen, argnums=0)(h[b], u[b], c[b])
        worst = max(worst, _rel(Jh, A[b]))
    print(f"[local_op selective] frozen-c coherence rel err = {worst:.3e}")
    assert worst < 1e-5

    # freezing is real: the raw step (c depends on h) has a different Jacobian
    def raw(hh, uu):
        out, _ = model.step(hh.unsqueeze(0), uu.unsqueeze(0), PredictorState())
        return out.squeeze(0)

    Jraw = tfunc.jacrev(raw, argnums=0)(h[0], u[0])
    diff = (Jraw - A[0]).abs().max().item()
    print(f"[local_op selective] raw vs frozen-A max diff = {diff:.3e}")
    assert diff > 1e-6


# ---------------------------------------------------------------------------
# (d) pi (and psi) are read-only w.r.t. the rollout recurrence
# ---------------------------------------------------------------------------
def test_pi_read_only_in_rollout():
    torch.manual_seed(0)
    model = KoopmanSSMPredictor(
        d_z=3, n_pairs=2, d_u=2, delta=0.1, regime="bilinear", dtype=DTYPE
    )
    g = torch.Generator().manual_seed(3)
    h0 = torch.randn(5, model.d_h, dtype=DTYPE, generator=g)
    u_seq = torch.randn(8, 5, model.d_u, dtype=DTYPE, generator=g)
    model.rollout(h0, u_seq).sum().backward()

    for name, p in model.pi.named_parameters():
        assert p.grad is None or p.grad.abs().max().item() == 0.0, f"pi.{name} got grad"
    # psi is also outside rollout (rollout starts from a given h0)
    for name, p in model.psi.named_parameters():
        assert p.grad is None or p.grad.abs().max().item() == 0.0, f"psi.{name} got grad"
    # the recurrence params DID receive gradient
    assert model.B_cont.grad is not None and model.B_cont.grad.abs().max().item() > 0
    assert model.blocks.nu.grad is not None and model.blocks.nu.grad.abs().max().item() > 0
    assert model.blocks.omega.grad is not None and model.blocks.omega.grad.abs().max() > 0
    print("[pi_read_only] pi/psi no grad; blocks + B_cont have grad -- ok")


# ---------------------------------------------------------------------------
# (e) GOLD: bilinear Koopman recovers the linearized-pendulum discrete spectrum
# ---------------------------------------------------------------------------
class LinearizedPendulum(DynamicalSystem):
    """Linear surrogate dx/dt = A_lin x + B_lin u from the pendulum's Jacobian.

    A linear system so the true ZOH discretization lies exactly in the model
    class -- the gold's identifiability object. RK4 at dt=0.05 is exact on a
    linear field far beyond the 1e-2 gate.
    """

    state_dim = 2
    action_dim = 1
    u_max = 2.0
    partial_indices = (0,)

    def __init__(self) -> None:
        self.A_lin, self.B_lin = DampedPendulum().linearize(np.zeros(2), np.zeros(1))

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.A_lin @ x + self.B_lin @ u

    def sample_ic(self, rng: np.random.Generator, regime: str = "default") -> np.ndarray:
        if regime == "default":
            return rng.uniform(-0.5, 0.5, size=2)
        raise ValueError(f"unknown IC regime: {regime!r}")


def _make_dataset(system, dt, n_ep, T, rng):
    states, actions = [], []
    for _ in range(n_ep):
        x0 = system.sample_ic(rng)
        u_seq = np.clip(rng.normal(0.0, 0.6, size=(T, system.action_dim)), -2.0, 2.0)
        states.append(system.integrate(x0, u_seq, dt))  # (T+1, d_x)
        actions.append(u_seq)
    x = torch.from_numpy(np.stack(states, axis=1))  # (T+1, n_ep, d_x)
    u = torch.from_numpy(np.stack(actions, axis=1))  # (T, n_ep, d_u)
    return x, u


def _free_running_windows(x, u, H):
    """Cut trajectories into non-overlapping length-H free-running windows.

    Returns (x0_win (K*n_ep, d_x), u_win (H, K*n_ep, d_u), tgt_win (H, K*n_ep, d_x)):
    each window is a genuine multi-step rollout from its true start state to its
    next-H targets.
    """
    T = u.shape[0]
    K = T // H
    x0s, us, tgts = [], [], []
    for k in range(K):
        s = k * H
        x0s.append(x[s])              # (n_ep, d_x) window start state
        us.append(u[s : s + H])       # (H, n_ep, d_u)
        tgts.append(x[s + 1 : s + H + 1])  # (H, n_ep, d_x) next states
    return torch.cat(x0s, dim=0), torch.cat(us, dim=1), torch.cat(tgts, dim=1)


def test_gold_bilinear_koopman_spectrum():
    dt = 0.05
    system = LinearizedPendulum()
    rng = np.random.default_rng(0)
    states, actions = _make_dataset(system, dt, n_ep=64, T=100, rng=rng)
    # DEVIATION FROM SPEC (documented, honest): a one-step teacher-forced loss
    # ALONE does NOT identify the block spectrum. With linear equal-dim psi/pi
    # there is a gauge freedom -- pi_W A_bar psi_W = M is solvable for ANY
    # (alpha, omega) -- so one-step training reaches MSE ~1e-33 with a wrong
    # spectrum. A free-running term is what breaks the gauge: matching
    # pi_W A_bar^k psi_W = M^k for k >= 2 forces psi_W pi_W = I, i.e. pi = psi^-1,
    # which makes A_bar SIMILAR to M and pins its eigenvalues. The spec flagged
    # this free-running term as "optional if it helps"; here it is necessary.
    x0_win, u_win, tgt_win = _free_running_windows(states, actions, H=10)

    torch.manual_seed(0)
    model = KoopmanSSMPredictor(
        d_z=2, n_pairs=1, d_u=1, delta=dt, regime="bilinear",
        constrained=True, psi_hidden=(), pi_hidden=(), dtype=DTYPE,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1.5e-2)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[800, 1500, 1900], gamma=0.2
    )
    for _ in range(2200):
        opt.zero_grad()
        preds = model.rollout(model.lift(x0_win), u_win)  # (H, B, d_h)
        loss = F.mse_loss(model.project(preds), tgt_win)
        loss.backward()
        opt.step()
        sched.step()
    print(f"[gold] final free-running MSE = {loss.item():.3e}")

    # learned discrete pair: eigenvalues of the assembled 2x2 A_bar
    A_bar_blocks, _ = model.blocks.discretize_zoh(dt)
    learned = torch.linalg.eigvals(block_diagonal(A_bar_blocks)).detach().numpy()
    # independent reference: exp(dt * continuous analytic eigenvalues), numpy only
    expected = np.exp(dt * DampedPendulum().analytic_eigenvalues())
    learned = learned[np.argsort(learned.imag)]
    expected = expected[np.argsort(expected.imag)]
    rel = np.abs(learned - expected) / np.abs(expected)
    print(f"[gold] expected pair = {expected}")
    print(f"[gold] learned  pair = {learned}")
    print(f"[gold] per-eigenvalue relative error = {rel}")
    assert np.all(rel < 1e-2), f"gold spectrum gate: rel={rel}"

    # on linear data the bilinear coupling must vanish relative to B_bar_0
    _, B_bar_0 = model._discrete_affine()
    ratio = (torch.linalg.norm(model.B_j) / torch.linalg.norm(B_bar_0)).item()
    print(f"[gold] ||B_j|| / ||B_bar_0|| = {ratio:.3e}")
    assert ratio < 0.1


# ---------------------------------------------------------------------------
# (f) overfit gate: one nonlinear pendulum trajectory, one-step MSE < 1e-3
# ---------------------------------------------------------------------------
def test_overfit_single_trajectory():
    dt = 0.05
    system = DampedPendulum()
    rng = np.random.default_rng(1)
    x0 = system.sample_ic(rng)
    u_seq = np.clip(rng.normal(0.0, 0.6, size=(100, 1)), -2.0, 2.0)
    traj = system.integrate(x0, u_seq, dt)  # (101, 2)
    x_t = torch.from_numpy(traj[:-1])  # (100, 2), time as batch
    x_next = torch.from_numpy(traj[1:])
    u_t = torch.from_numpy(u_seq)  # (100, 1)

    torch.manual_seed(0)
    model = KoopmanSSMPredictor(
        d_z=2, n_pairs=2, d_u=1, delta=dt, regime="bilinear",
        psi_hidden=(32,), pi_hidden=(32,), dtype=DTYPE,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[2000, 3500], gamma=0.2)
    for _ in range(5000):
        opt.zero_grad()
        h_next, _ = model.step(model.lift(x_t), u_t, PredictorState())
        loss = F.mse_loss(model.project(h_next), x_next)
        loss.backward()
        opt.step()
        sched.step()
    print(f"[overfit] final one-step MSE = {loss.item():.3e}")
    assert loss.item() < 1e-3


# ---------------------------------------------------------------------------
# (g) guards
# ---------------------------------------------------------------------------
def test_guards():
    with pytest.raises(ValueError):
        KoopmanSSMPredictor(d_z=2, n_pairs=1, d_u=1, delta=0.1, regime="nope", dtype=DTYPE)
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            KoopmanSSMPredictor(
                d_z=2, n_pairs=1, d_u=1, delta=bad, regime="constant", dtype=DTYPE
            )
