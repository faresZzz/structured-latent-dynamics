import math

import numpy as np
import pytest
import torch
from torch import nn

from sld.objectives.jac_reg import TotalJacobianPenalty


def _linear_step(A):
    # rows independent: the Jacobian of every row is A
    return lambda q: q @ A.T


def test_unbiased_on_linear_map_gold():
    # gold test mirroring spike Experiment 1: on a linear step the penalty
    # estimates ||A||_F^2; mean over 200 rollout draws within 5% at k=8
    torch.manual_seed(0)
    d = 16
    A = torch.randn(d, d, dtype=torch.float64)
    truth = (A * A).sum().item()
    g = torch.Generator().manual_seed(1)
    pen = TotalJacobianPenalty(k_probes=8, step_subsample=1.0, generator=g)
    q_seq = torch.randn(4, 8, d, dtype=torch.float64)
    draws = [pen(_linear_step(A), q_seq).item() for _ in range(200)]
    mean_est = sum(draws) / len(draws)
    rel_err = abs(mean_est - truth) / truth
    print(f"[jac_reg gold] mean={mean_est:.3f} truth={truth:.3f} rel_err={rel_err:.3%}")
    assert rel_err < 0.05


def test_u_is_frozen_the_object_is_J_wrt_q():
    # step(q, u) = q A^T + u B^T: J^q = A regardless of B — the penalty must
    # not see B (u is frozen per step, by definition of the augmented-state
    # Jacobian)
    torch.manual_seed(0)
    d, m = 8, 3
    A = torch.randn(d, d, dtype=torch.float64)
    B = 100.0 * torch.randn(d, m, dtype=torch.float64)  # huge B: leakage would scream
    truth = (A * A).sum().item()

    def step(q, u):
        return q @ A.T + u @ B.T

    g = torch.Generator().manual_seed(2)
    pen = TotalJacobianPenalty(k_probes=8, step_subsample=1.0, generator=g)
    q_seq = torch.randn(3, 16, d, dtype=torch.float64)
    u_seq = torch.randn(3, 16, m, dtype=torch.float64)
    draws = [pen(step, q_seq, u_seq).item() for _ in range(200)]
    mean_est = sum(draws) / len(draws)
    assert abs(mean_est - truth) / truth < 0.05


def test_u_row_alignment_with_jointly_dependent_jacobian():
    # adversarial-review lock: the earlier u-test had J independent of u, so
    # a q/u row misalignment was invisible (and any u-only Jacobian is
    # permutation-invariant in the mean). Here step(q, u) = u_0*q_0*q gives
    # J = u_0*(q_0*I + e_0 q^T)... with ||J||_F^2 = u_0^2*(d*q_0^2 + ||q||^2
    # + 2*q_0^2), jointly and non-separably dependent on the (q, u) PAIR,
    # and u_0 := 1 + q_0^2 couples them: the aligned analytic mean (~200)
    # and the decoupled one (~72) are far apart, so a pairing bug cannot
    # hide inside the 5% tolerance. (J = u_0*(q e_0^T + q_0 I): the q
    # column sits at j=0, i.e. d(q_0*q_i)/dq_j has the q_i's in column 0.)
    torch.manual_seed(0)
    d = 5
    T, B = 3, 16

    def step(q, u):
        return u[:, :1] * q[:, :1] * q

    q_seq = torch.randn(T, B, d, dtype=torch.float64)
    u_seq = 1.0 + q_seq[..., :1] ** 2  # (T, B, 1), coupled to q's row
    q0 = q_seq[..., 0]
    bracket = d * q0**2 + (q_seq**2).sum(-1) + 2.0 * q0**2
    truth = (u_seq[..., 0] ** 2 * bracket).mean().item()

    g = torch.Generator().manual_seed(11)
    pen = TotalJacobianPenalty(k_probes=8, step_subsample=1.0, generator=g)
    draws = [pen(step, q_seq, u_seq).item() for _ in range(300)]
    mean_est = sum(draws) / len(draws)
    print(f"[jac_reg u-align] mean={mean_est:.2f} aligned-truth={truth:.2f}")
    assert abs(mean_est - truth) / truth < 0.05


def test_gradients_reach_module_params():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(6, 32), nn.Tanh(), nn.Linear(32, 6))
    g = torch.Generator().manual_seed(3)
    pen = TotalJacobianPenalty(k_probes=2, step_subsample=1.0, generator=g)
    q_seq = torch.randn(5, 4, 6)
    value = pen(lambda q: net(q), q_seq)
    value.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(gr).all() for gr in grads)
    assert any(gr.abs().sum() > 0 for gr in grads)


def test_subsample_unbiased_including_empty_draws():
    # adversarial-review lock: a plain mean over kept steps would be biased
    # low by 1 - (1-p)^T; the Horvitz-Thompson normalization must make the
    # estimator unbiased over the Bernoulli subsampling, with the zero of an
    # empty draw participating in the average
    torch.manual_seed(0)
    d = 6
    A = torch.randn(d, d, dtype=torch.float64)
    truth = (A * A).sum().item()
    g = torch.Generator().manual_seed(7)
    pen = TotalJacobianPenalty(k_probes=2, step_subsample=0.5, generator=g)
    q_seq = torch.randn(4, 8, d, dtype=torch.float64)  # T=4: P(empty) ~ 6%
    draws = [pen(_linear_step(A), q_seq).item() for _ in range(400)]
    mean_est = sum(draws) / len(draws)
    n_empty = sum(1 for v in draws if v == 0.0)
    print(f"[jac_reg HT] mean={mean_est:.3f} truth={truth:.3f} empty_draws={n_empty}")
    assert n_empty > 0  # the zero branch genuinely participates
    assert abs(mean_est - truth) / truth < 0.10


def test_zero_sampled_steps_returns_zero():
    # force an empty Bernoulli draw via a tiny keep probability and a seed
    # known to keep nothing on the first draw for T=2
    g = torch.Generator().manual_seed(0)
    pen = TotalJacobianPenalty(k_probes=2, step_subsample=1e-9, generator=g)
    q_seq = torch.randn(2, 3, 4)
    value = pen(lambda q: q, q_seq)
    assert value.item() == 0.0


def test_deterministic_given_generator_seed():
    torch.manual_seed(0)
    A = torch.randn(5, 5, dtype=torch.float64)
    q_seq = torch.randn(6, 7, 5, dtype=torch.float64)
    vals = []
    for _ in range(2):
        g = torch.Generator().manual_seed(42)
        pen = TotalJacobianPenalty(k_probes=2, step_subsample=0.5, generator=g)
        vals.append(pen(_linear_step(A), q_seq).item())
    assert vals[0] == vals[1]


def test_training_with_penalty_lowers_spectral_norm():
    # the plan's behavioral test: fit a 1-step map whose target has spectral
    # norm 2.0; with the penalty (strength mu) the trained Jacobian's
    # spectral norm must come out strictly lower than without
    torch.manual_seed(0)
    d = 8
    A_target = torch.randn(d, d)
    A_target = 2.0 * A_target / torch.linalg.matrix_norm(A_target, ord=2)
    q_train = torch.randn(256, d)
    y_train = q_train @ A_target.T

    def train(mu: float) -> float:
        torch.manual_seed(1)  # same init for both arms
        net = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, d))
        g = torch.Generator().manual_seed(2)
        pen = TotalJacobianPenalty(k_probes=2, step_subsample=1.0, generator=g)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        for _ in range(300):
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(q_train), y_train)
            if mu > 0:
                loss = loss + mu * pen(lambda q: net(q), q_train.unsqueeze(0))
            loss.backward()
            opt.step()
        # spectral norm of the trained Jacobian, median over test points
        q_test = torch.randn(16, d)
        norms = []
        for q in q_test:
            J = torch.autograd.functional.jacobian(lambda x: net(x), q)
            norms.append(torch.linalg.matrix_norm(J, ord=2).item())
        norms.sort()
        return norms[len(norms) // 2]

    norm_without = train(mu=0.0)
    norm_with = train(mu=0.01)
    print(f"[jac_reg training] spectral norm without={norm_without:.3f} with={norm_with:.3f}")
    assert norm_with < norm_without


def test_rectangular_step_uses_input_dimension():
    # the multiplier is the INPUT dimension d (E[vv^T] = I_d/d lives in
    # input space): verified on a genuinely rectangular map J = A (o != d)
    torch.manual_seed(0)
    d_in, d_out = 6, 11
    A = torch.randn(d_out, d_in, dtype=torch.float64)
    truth = (A * A).sum().item()
    g = torch.Generator().manual_seed(21)
    pen = TotalJacobianPenalty(k_probes=8, step_subsample=1.0, generator=g)
    q_seq = torch.randn(3, 8, d_in, dtype=torch.float64)
    draws = [pen(_linear_step(A), q_seq).item() for _ in range(300)]
    mean_est = sum(draws) / len(draws)
    assert abs(mean_est - truth) / truth < 0.05


# -- on_state_steps: the pytree (h, m) companion for state-threading families ---


def _linear_block_step(Ah, Am, Bh, Bm):
    # h_next = h @ Ah^T + m @ Am^T ; m_next = h @ Bh^T + m @ Bm^T -- rows
    # independent, so the per-sample block Jacobian is the SAME fixed matrix
    # [[Ah, Am], [Bh, Bm]] for every row, at every step
    def step_fn(h, m, u):
        return h @ Ah.T + m @ Am.T, h @ Bh.T + m @ Bm.T

    return step_fn


def test_on_state_steps_unbiased_on_linear_augmented_map_gold():
    # gold test mirroring the tensor-path gold: a linear augmented (h, m)
    # step has a KNOWN block Jacobian; on_state_steps must recover its
    # analytic ||.||_F^2 (per-sample mean) within 5% over >=200 draws
    torch.manual_seed(0)
    d_h, d_m, B, T = 4, 3, 5, 3
    Ah = torch.randn(d_h, d_h, dtype=torch.float64)
    Am = torch.randn(d_h, d_m, dtype=torch.float64)
    Bh = torch.randn(d_m, d_h, dtype=torch.float64)
    Bm = torch.randn(d_m, d_m, dtype=torch.float64)
    J_block = torch.zeros(d_h + d_m, d_h + d_m, dtype=torch.float64)
    J_block[:d_h, :d_h], J_block[:d_h, d_h:] = Ah, Am
    J_block[d_h:, :d_h], J_block[d_h:, d_h:] = Bh, Bm
    truth = (J_block * J_block).sum().item()  # ||J_i||_F^2, identical every sample/step

    step_fn = _linear_block_step(Ah, Am, Bh, Bm)
    steps = [
        (
            torch.randn(B, d_h, dtype=torch.float64),
            torch.randn(B, d_m, dtype=torch.float64),
            torch.randn(B, 2, dtype=torch.float64),
        )
        for _ in range(T)
    ]
    g = torch.Generator().manual_seed(31)
    pen = TotalJacobianPenalty(k_probes=8, step_subsample=1.0, generator=g)
    draws = [pen.on_state_steps(step_fn, steps).item() for _ in range(200)]
    mean_est = sum(draws) / len(draws)
    rel_err = abs(mean_est - truth) / truth
    print(f"[on_state_steps gold] mean={mean_est:.3f} truth={truth:.3f} rel_err={rel_err:.3%}")
    assert rel_err < 0.05


def test_on_state_steps_subsample_unbiased_including_empty_draws():
    # same Horvitz-Thompson lock as test_subsample_unbiased_including_empty_draws,
    # through the pytree path: small T, subsample < 1, empty draws must
    # genuinely participate as a zero and the mean must stay unbiased
    torch.manual_seed(0)
    d_h, d_m, B, T = 3, 2, 6, 4  # T=4: P(empty) ~ 6% at p=0.5
    Ah = torch.randn(d_h, d_h, dtype=torch.float64)
    Am = torch.randn(d_h, d_m, dtype=torch.float64)
    Bh = torch.randn(d_m, d_h, dtype=torch.float64)
    Bm = torch.randn(d_m, d_m, dtype=torch.float64)
    J_block = torch.zeros(d_h + d_m, d_h + d_m, dtype=torch.float64)
    J_block[:d_h, :d_h], J_block[:d_h, d_h:] = Ah, Am
    J_block[d_h:, :d_h], J_block[d_h:, d_h:] = Bh, Bm
    truth = (J_block * J_block).sum().item()

    step_fn = _linear_block_step(Ah, Am, Bh, Bm)
    steps = [
        (
            torch.randn(B, d_h, dtype=torch.float64),
            torch.randn(B, d_m, dtype=torch.float64),
            torch.randn(B, 2, dtype=torch.float64),
        )
        for _ in range(T)
    ]
    g = torch.Generator().manual_seed(17)
    pen = TotalJacobianPenalty(k_probes=4, step_subsample=0.5, generator=g)
    draws = [pen.on_state_steps(step_fn, steps).item() for _ in range(400)]
    mean_est = sum(draws) / len(draws)
    n_empty = sum(1 for v in draws if v == 0.0)
    print(
        f"[on_state_steps HT] mean={mean_est:.3f} truth={truth:.3f} empty_draws={n_empty}"
    )
    assert n_empty > 0  # the zero branch genuinely participates
    assert abs(mean_est - truth) / truth < 0.10


def test_on_state_steps_gradients_reach_module_params():
    # mirrors test_gradients_reach_module_params for the pytree path, with a
    # MIXED thread: the first step has m=None (fresh state), later steps
    # carry the tensor m -- exercising both branches of the None-drop logic
    torch.manual_seed(0)

    class ToyStateNet(nn.Module):
        def __init__(self, d_h, d_m, d_u):
            super().__init__()
            self.w_h = nn.Linear(d_h + d_u, d_h)
            self.w_m0 = nn.Linear(d_h, d_m)
            self.w_m = nn.Linear(d_h + d_m, d_m)

        def step(self, h, m, u):
            h_next = torch.tanh(self.w_h(torch.cat([h, u], dim=-1)))
            if m is None:
                m_next = torch.tanh(self.w_m0(h))
            else:
                m_next = torch.tanh(self.w_m(torch.cat([h, m], dim=-1)))
            return h_next, m_next

    d_h, d_m, d_u, B, T = 4, 3, 2, 5, 4
    net = ToyStateNet(d_h, d_m, d_u)

    def step_fn(h, m, u):
        return net.step(h, m, u)

    m, steps = None, []
    for _ in range(T):
        h_t = torch.randn(B, d_h)
        u_t = torch.randn(B, d_u)
        steps.append((h_t, m, u_t))
        _, m = net.step(h_t, m, u_t)

    g = torch.Generator().manual_seed(4)
    pen = TotalJacobianPenalty(k_probes=2, step_subsample=1.0, generator=g)
    value = pen.on_state_steps(step_fn, steps)
    value.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(gr).all() for gr in grads)
    assert any(gr.abs().sum() > 0 for gr in grads)


# -- identity-passthrough exclusion for growing-memory families (Wave 5) -------
#
# The transformer's KV cache copies every old entry verbatim into m_next, so
# d(m_next)/d(m) contains an exact identity block: a structural floor of
# m.numel() on ||J^q||_F^2 and sigma_max(J^q) >= 1 for ANY parameters. The
# penalized (and calibrated) object excludes that block. These tests are the
# proof that the subtracted term is EXACTLY that floor and nothing else.


def _append_family(A, C, W):
    """Toy pure-append family with a hand-computable Jacobian.

    ``h_next = h A^T + m[-1] C^T`` and ``m_next = cat([m, (h W^T)[None]])``.
    ``m`` has layout ``(L, B, d_m)`` -- the batch axis sits in the MIDDLE on
    purpose, mimicking the transformer cache: nothing in the estimator may
    depend on knowing that.

    Per-sample block Jacobian w.r.t. ``(h_i, m[:, i, :])``:
      d h_next / d h        = A                       (d_h x d_h)
      d h_next / d m        = C on the LAST slot only (d_h x d_m)
      d m_next / d h        = W on the NEW slot only  (d_m x d_h)
      d m_next / d m        = I on the L old slots    (identity passthrough)
    hence  ||J_excl||_F^2 = ||A||^2 + ||C||^2 + ||W||^2   (independent of L)
    and    ||J_full||_F^2 = ||J_excl||_F^2 + L * d_m.
    """

    def step_fn(h, m, u):
        h_next = h @ A.T + m[-1] @ C.T
        m_next = torch.cat([m, (h @ W.T).unsqueeze(0)], dim=0)
        return h_next, m_next

    return step_fn


def _flattened_append_family(A, C, W, d_h, d_m):
    """The SAME map, presented as a single tensor state (no m/m_next split).

    With ``m=None`` the estimator has no growing leaf to exclude, so it
    returns the FULL ``||J||_F^2`` -- identical map, identical probe space,
    identical normalization. This is how the tests read "what the estimator
    would have returned without the exclusion" without touching the code
    under test.
    """
    inner = _append_family(A, C, W)

    def step_fn(q, m, u):
        h = q[:, :d_h]
        L = (q.shape[1] - d_h) // d_m
        m_t = q[:, d_h:].reshape(-1, L, d_m).permute(1, 0, 2)
        h_next, m_next = inner(h, m_t, u)
        flat_next = m_next.permute(1, 0, 2).reshape(q.shape[0], -1)
        return torch.cat([h_next, flat_next], dim=-1), None

    return step_fn


def test_on_state_steps_excludes_exactly_the_identity_floor_gold():
    # analytic gold on the toy append family: the excluded estimator must
    # converge to ||A||^2 + ||C||^2 + ||W||^2, and the SAME map presented
    # without the m split must converge to that plus L * d_m -- proving the
    # subtracted term is exactly the identity passthrough
    torch.manual_seed(0)
    d_h, d_m, d_u, B = 4, 2, 2, 5
    A = torch.randn(d_h, d_h, dtype=torch.float64)
    C = torch.randn(d_h, d_m, dtype=torch.float64)
    W = torch.randn(d_m, d_h, dtype=torch.float64)
    truth_excl = float((A**2).sum() + (C**2).sum() + (W**2).sum())

    lengths = [1, 2, 3]
    steps = [
        (
            torch.randn(B, d_h, dtype=torch.float64),
            torch.randn(L, B, d_m, dtype=torch.float64),
            torch.randn(B, d_u, dtype=torch.float64),
        )
        for L in lengths
    ]
    truth_full = truth_excl + sum(L * d_m for L in lengths) / len(lengths)

    g = torch.Generator().manual_seed(11)
    pen = TotalJacobianPenalty(k_probes=8, step_subsample=1.0, generator=g)
    excl = [pen.on_state_steps(_append_family(A, C, W), steps).item() for _ in range(300)]
    mean_excl = sum(excl) / len(excl)

    flat_steps = [
        (torch.cat([h, m.permute(1, 0, 2).reshape(B, -1)], dim=-1), None, u)
        for h, m, u in steps
    ]
    g_full = torch.Generator().manual_seed(11)
    pen_full = TotalJacobianPenalty(k_probes=8, step_subsample=1.0, generator=g_full)
    flat_fn = _flattened_append_family(A, C, W, d_h, d_m)
    full = [pen_full.on_state_steps(flat_fn, flat_steps).item() for _ in range(300)]
    mean_full = sum(full) / len(full)

    err_excl = abs(mean_excl - truth_excl) / truth_excl
    err_full = abs(mean_full - truth_full) / truth_full
    print(
        f"[append exclusion gold] excluded mean={mean_excl:.4f} truth={truth_excl:.4f} "
        f"rel_err={err_excl:.3%} | NOT excluded mean={mean_full:.4f} "
        f"truth={truth_full:.4f} rel_err={err_full:.3%} | "
        f"identity floor = mean(L)*d_m = {truth_full - truth_excl:.1f}"
    )
    assert err_excl < 0.05
    assert err_full < 0.05
    # the gap between the two objects IS the identity floor, analytically exact
    assert abs((truth_full - truth_excl) - sum(L * d_m for L in lengths) / len(lengths)) < 1e-12


def test_on_state_steps_fixed_size_state_is_never_excluded():
    # non-regression: a family whose m keeps a CONSTANT numel gets NO
    # subtraction, even when d(m_next)/d(m) is literally the identity --
    # m_next is a transformation of m there (min_ssm), not an appended copy,
    # and the whole block is a legitimate learned target of the penalty
    torch.manual_seed(0)
    d_h, d_m, d_u, B, T = 3, 4, 2, 6, 3
    A = torch.randn(d_h, d_h, dtype=torch.float64)
    truth = float((A**2).sum()) + d_m  # ||A||^2 + ||I_{d_m}||_F^2, NOT ||A||^2

    def step_fn(h, m, u):
        return h @ A.T, m  # same numel in and out; identity recurrence on m

    steps = [
        (
            torch.randn(B, d_h, dtype=torch.float64),
            torch.randn(B, d_m, dtype=torch.float64),
            torch.randn(B, d_u, dtype=torch.float64),
        )
        for _ in range(T)
    ]
    g = torch.Generator().manual_seed(5)
    pen = TotalJacobianPenalty(k_probes=8, step_subsample=1.0, generator=g)
    draws = [pen.on_state_steps(step_fn, steps).item() for _ in range(300)]
    mean_est = sum(draws) / len(draws)
    rel_err = abs(mean_est - truth) / truth
    print(f"[fixed-size no exclusion] mean={mean_est:.4f} truth={truth:.4f} rel_err={rel_err:.3%}")
    assert rel_err < 0.05


def test_on_state_steps_rejects_growth_that_does_not_copy_m():
    # append hypothesis violated in the DEFICIT direction: m grows but the old
    # entries are scaled down instead of copied, so ||J v||^2 - ||v_m||^2 goes
    # negative. That must raise, never be clamped to zero (REGLES section 5).
    torch.manual_seed(0)
    d_h, d_m, d_u, B, L = 2, 3, 2, 4, 2

    def step_fn(h, m, u):
        m_next = torch.cat([0.1 * m, torch.zeros(1, m.shape[1], d_m, dtype=m.dtype)], dim=0)
        return torch.zeros_like(h), m_next

    steps = [
        (
            torch.randn(B, d_h, dtype=torch.float64),
            torch.randn(L, B, d_m, dtype=torch.float64),
            torch.randn(B, d_u, dtype=torch.float64),
        )
    ]
    g = torch.Generator().manual_seed(2)
    pen = TotalJacobianPenalty(k_probes=2, step_subsample=1.0, generator=g)
    with pytest.raises(ValueError, match="does NOT copy its old memory entries"):
        pen.on_state_steps(step_fn, steps)


def test_on_state_steps_rejects_shrinking_memory():
    # outside the declared taxonomy (fixed size or pure append): refused
    # loudly rather than handled by a guess
    torch.manual_seed(0)
    d_h, d_m, d_u, B, L = 2, 3, 2, 4, 3

    def step_fn(h, m, u):
        return h, m[1:]

    steps = [
        (
            torch.randn(B, d_h, dtype=torch.float64),
            torch.randn(L, B, d_m, dtype=torch.float64),
            torch.randn(B, d_u, dtype=torch.float64),
        )
    ]
    g = torch.Generator().manual_seed(3)
    pen = TotalJacobianPenalty(k_probes=2, step_subsample=1.0, generator=g)
    with pytest.raises(ValueError, match="shrinking recurrent memory"):
        pen.on_state_steps(step_fn, steps)


def test_on_state_steps_and_dense_audit_reject_disappearing_memory_identically():
    # m10 (adversarial review round 2): m_t present at input but step_fn
    # returning m_next=None used to be silently treated as "nothing to
    # exclude" here, while the dense audit already rejected the SAME toy
    # family as a shrinking-to-zero memory. Both modules must reject this
    # identically now.
    from sld.audit.jacobian import augmented_jacobian_excluding_identity

    torch.manual_seed(0)
    d_h, d_m, d_u, B = 2, 3, 2, 4

    def step_fn(h, m, u):
        return h, None  # m present in, vanishes out

    h_t = torch.randn(B, d_h, dtype=torch.float64)
    m_t = torch.randn(B, d_m, dtype=torch.float64)
    u_t = torch.randn(B, d_u, dtype=torch.float64)

    g = torch.Generator().manual_seed(9)
    pen = TotalJacobianPenalty(k_probes=2, step_subsample=1.0, generator=g)
    with pytest.raises(ValueError, match="shrinking"):
        pen.on_state_steps(step_fn, [(h_t, m_t, u_t)])

    # single-sample form for the dense audit (batch axis is the caller's
    # business there) -- same map, same disappearing-memory shape
    with pytest.raises(ValueError, match="shrinking"):
        augmented_jacobian_excluding_identity(step_fn, h_t[0], m_t[0], u_t[0])


def test_on_state_steps_exclusion_leaves_parameter_gradients_untouched():
    # the justification for excluding the block is that its derivative w.r.t.
    # the parameters is identically zero -- so with THE SAME probes, the
    # exclusion must (a) lower the penalty VALUE by exactly the subtracted
    # sum(||v_m||^2) term and (b) leave every parameter gradient identical.
    # A cross-presentation comparison (pytree vs single tensor) is NOT valid
    # here: different leaf shapes consume the RNG stream in different orders,
    # so the probes -- and hence the stochastic estimates -- differ. Instead
    # this test replicates on_state_steps' exact probe stream (white-box, on
    # purpose: 1x rand(T) keep mask, then per probe randn per leaf normalized
    # by the concatenated norm) and hand-builds the NON-excluded estimate
    # with identical probes as the oracle.
    torch.manual_seed(0)
    d_h, d_m, d_u, B, L = 3, 2, 2, 4, 2
    k_probes, seed = 3, 7
    net = nn.Linear(d_h, d_m, dtype=torch.float64)

    def grown(h, m, u):
        return h * 0.5, torch.cat([m, net(h).unsqueeze(0)], dim=0)

    h_t = torch.randn(B, d_h, dtype=torch.float64)
    m_t = torch.randn(L, B, d_m, dtype=torch.float64)
    u_t = torch.randn(B, d_u, dtype=torch.float64)

    # -- excluded path: the real implementation --
    net.zero_grad(set_to_none=True)
    g = torch.Generator().manual_seed(seed)
    pen = TotalJacobianPenalty(k_probes=k_probes, step_subsample=1.0, generator=g)
    value_excl = pen.on_state_steps(grown, [(h_t, m_t, u_t)])
    value_excl.backward()
    grads_excl = {
        n: (p.grad.clone() if p.grad is not None else None)
        for n, p in net.named_parameters()
    }

    # -- oracle: same probes, NO exclusion --
    net.zero_grad(set_to_none=True)
    g = torch.Generator().manual_seed(seed)
    torch.rand(1, generator=g)  # the keep-mask draw (T=1, p=1.0)
    leaves = (h_t.detach(), m_t.detach())
    d_total = sum(leaf.numel() for leaf in leaves)

    def fn(h, m):
        return grown(h, m, u_t)

    probe_sq = torch.zeros((), dtype=torch.float64)
    subtracted = torch.zeros((), dtype=torch.float64)
    for _ in range(k_probes):
        vh = torch.randn(h_t.shape, generator=g, dtype=torch.float64)
        vm = torch.randn(m_t.shape, generator=g, dtype=torch.float64)
        norm = torch.cat([vh.reshape(-1), vm.reshape(-1)]).norm()
        vh, vm = vh / norm, vm / norm
        _, jv = torch.func.jvp(fn, leaves, (vh, vm))
        probe_sq = probe_sq + sum(leaf.pow(2).sum() for leaf in jv)
        subtracted = subtracted + (vm * vm).sum()
    scale = d_total / (k_probes * B * 1 * 1.0)
    value_full = scale * probe_sq
    expected_gap = scale * subtracted
    value_full.backward()
    grads_full = {
        n: (p.grad.clone() if p.grad is not None else None)
        for n, p in net.named_parameters()
    }

    print(
        f"[exclusion gradient invariance] excl={value_excl.item():.6f} "
        f"full={value_full.item():.6f} expected_gap={expected_gap.item():.6f}"
    )
    # (a) the value gap is EXACTLY the subtracted identity-passthrough term
    gap = value_full.item() - value_excl.item()
    assert gap > 0
    assert abs(gap - expected_gap.item()) <= 1e-12 * max(1.0, abs(gap))
    # (b) gradients identical: same no-grad set on both paths -- the bias is
    # legitimately gradient-free (the JVP differentiates w.r.t. the INPUT and
    # d(W h + b)/dh = W never sees b; Wave 2 doctrine, not a bug) -- and the
    # weight gradients agree to float64 tightness
    none_excl = {n for n, g_ in grads_excl.items() if g_ is None}
    none_full = {n for n, g_ in grads_full.items() if g_ is None}
    assert none_excl == none_full == {"bias"}
    assert torch.allclose(
        grads_excl["weight"], grads_full["weight"], rtol=0, atol=1e-12
    )


def test_on_state_steps_matches_the_dense_audit_of_the_same_object():
    # coherence: what training PENALIZES and what the calibration MEASURES
    # must be the same object, or the calibrated margin means nothing
    from sld.audit.jacobian import augmented_jacobian_excluding_identity

    torch.manual_seed(0)
    d_h, d_m, d_u, B, L = 4, 3, 2, 1, 3  # B=1: per-sample object, comparable to the audit
    A = torch.randn(d_h, d_h, dtype=torch.float64)
    C = torch.randn(d_h, d_m, dtype=torch.float64)
    W = torch.randn(d_m, d_h, dtype=torch.float64)
    step_fn = _append_family(A, C, W)

    h_t = torch.randn(B, d_h, dtype=torch.float64)
    m_t = torch.randn(L, B, d_m, dtype=torch.float64)
    u_t = torch.randn(B, d_u, dtype=torch.float64)

    dense = augmented_jacobian_excluding_identity(step_fn, h_t, m_t, u_t)
    g = torch.Generator().manual_seed(23)
    pen = TotalJacobianPenalty(k_probes=8, step_subsample=1.0, generator=g)
    draws = [pen.on_state_steps(step_fn, [(h_t, m_t, u_t)]).item() for _ in range(400)]
    mean_est = sum(draws) / len(draws)
    rel_err = abs(mean_est - dense.frobenius_sq) / dense.frobenius_sq
    print(
        f"[estimator vs audit] stochastic={mean_est:.4f} dense={dense.frobenius_sq:.4f} "
        f"rel_err={rel_err:.3%} identity_rows={dense.n_identity_rows}"
    )
    assert dense.n_identity_rows == m_t.numel()
    assert rel_err < 0.05


def _dense_full_frobenius_sq(step_fn, h, m, u) -> float:
    """The NOT-excluded ``||J||_F^2`` of ``(h, m, u) -> (h_next, m_next)``.

    Mirrors ``audit.jacobian.augmented_jacobian_excluding_identity``'s own
    internal flatten-then-``jacrev`` construction, WITHOUT the identity-row
    deletion, so the m13 test below can pin the identity floor against an
    exact dense quantity rather than a second stochastic estimate.
    """
    h_shape, m_shape = h.shape, m.shape
    n_h = h.numel()

    def flat_map(q_flat):
        h_arg = q_flat[:n_h].reshape(h_shape)
        m_arg = q_flat[n_h:].reshape(m_shape)
        out_h, out_m = step_fn(h_arg, m_arg, u)
        return torch.cat([out_h.reshape(-1), out_m.reshape(-1)])

    q_flat = torch.cat([h.reshape(-1), m.reshape(-1)])
    J = torch.func.jacrev(flat_map)(q_flat)
    return float((J**2).sum())


def test_on_state_steps_matches_the_dense_audit_on_the_real_transformer():
    # m13 (adversarial review round 2): the claimed real-transformer
    # regression lock was absent -- only toy append families exercised the
    # exclusion machinery. Build the ACTUAL production CausalTransformerPredictor,
    # thread it a few steps to reach a LATE cache state (t >= 2 going in),
    # then check (a) the dense audit excludes exactly m.numel() rows, (b) the
    # stochastic exclusion is finite, positive, and close to the dense one,
    # and (c) the identity floor -- the gap between the NOT-excluded and the
    # excluded dense Frobenius norms -- is (near-)exactly m.numel().
    from sld.audit.jacobian import augmented_jacobian_excluding_identity
    from sld.predictors.base import PredictorState
    from sld.predictors.transformer import CausalTransformerPredictor

    torch.manual_seed(0)
    d_z, d_u = 4, 1
    # constructed directly at a small width (same pattern as
    # tests/training/test_budget.py): match_budget cannot hit 2000 +/-5%
    # for this family -- its smallest d_model already overshoots
    predictor = CausalTransformerPredictor(
        d_h=d_z, d_u=d_u, d_model=8, n_heads=2, n_layers=2
    ).double()
    predictor.eval()

    def step_fn(h, m, u):
        h_next, state = predictor.step(h, u, PredictorState(m=m))
        return h_next, state.m

    B = 1
    h = torch.randn(B, d_z, dtype=torch.float64)
    m = None
    with torch.no_grad():
        for _ in range(3):  # warm up the cache to a LATE state (t >= 2 going in)
            u = torch.randn(B, d_u, dtype=torch.float64)
            h, m = step_fn(h, m, u)
    u_t = torch.randn(B, d_u, dtype=torch.float64)

    dense = augmented_jacobian_excluding_identity(step_fn, h, m, u_t)
    print(
        f"[m13 real transformer] n_identity_rows={dense.n_identity_rows} "
        f"m.numel()={m.numel()} d_in={dense.d_in} sigma_max={dense.sigma_max:.6g} "
        f"frobenius_sq(dense excluded)={dense.frobenius_sq:.6g}"
    )
    # (a) the dense audit's identity-row count is exactly the KV cache size
    assert dense.n_identity_rows == m.numel()
    assert math.isfinite(dense.sigma_max) and dense.sigma_max > 0.0
    assert math.isfinite(dense.frobenius_sq) and dense.frobenius_sq > 0.0

    # (b) the stochastic estimator (what training actually penalizes) is
    # finite, positive, and agrees with the dense measurement
    g = torch.Generator().manual_seed(99)
    pen = TotalJacobianPenalty(k_probes=16, step_subsample=1.0, generator=g)
    draws = [pen.on_state_steps(step_fn, [(h, m, u_t)]).item() for _ in range(200)]
    mean_est = sum(draws) / len(draws)
    assert math.isfinite(mean_est) and mean_est > 0.0
    rel_err = abs(mean_est - dense.frobenius_sq) / dense.frobenius_sq
    print(
        f"[m13 real transformer] stochastic_excluded_mean={mean_est:.6g} rel_err={rel_err:.3%}"
    )
    assert rel_err < 0.10

    # (c) the identity floor: the NOT-excluded dense Frobenius norm exceeds
    # the excluded one by exactly m.numel() -- compared to a second dense
    # quantity, not a stochastic one, for a clean lock
    full_frobenius_sq = _dense_full_frobenius_sq(step_fn, h, m, u_t)
    floor = full_frobenius_sq - dense.frobenius_sq
    print(
        f"[m13 real transformer] floor={floor:.6f} vs m.numel()={m.numel()} "
        f"(full={full_frobenius_sq:.6g}, excluded={dense.frobenius_sq:.6g})"
    )
    assert abs(floor - m.numel()) / m.numel() < 1e-6


# ---------------------------------------------------------------------------
# PROTOCOL.md section 10.3 amendment: the declared, gauge-normalized metric
# ---------------------------------------------------------------------------


def test_normalize_jacobian_basic_and_shape_errors():
    from sld.objectives.jac_reg import normalize_jacobian

    J = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    s_in = torch.tensor([1.0, 10.0, 100.0]).numpy()
    s_out = torch.tensor([2.0, 5.0]).numpy()
    M = normalize_jacobian(J, s_in, s_out)
    expected = (J * s_in[None, :]) / s_out[:, None]
    assert (M == expected).all()

    with pytest.raises(ValueError, match="s_in must have shape"):
        normalize_jacobian(J, s_in[:-1], s_out)
    with pytest.raises(ValueError, match="s_out must have shape"):
        normalize_jacobian(J, s_in, s_out[:-1])
    with pytest.raises(ValueError, match="finite"):
        normalize_jacobian(J, s_in, [float("nan"), 1.0])
    with pytest.raises(ValueError, match="positive"):
        normalize_jacobian(J, s_in, [0.0, 1.0])


def test_gauge_indeterminate_on_zero_or_nonfinite_or_missing_scale():
    from sld.objectives.jac_reg import Gauge, MetricIndeterminate

    good = Gauge(s_h=torch.ones(1, 3, dtype=torch.float64))
    assert torch.equal(good.h_scale(), torch.ones(1, 3, dtype=torch.float64))

    zero = Gauge(s_h=torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64))
    with pytest.raises(MetricIndeterminate, match="zero"):
        zero.h_scale()

    nonfinite = Gauge(s_h=torch.tensor([[1.0, float("nan"), 1.0]], dtype=torch.float64))
    with pytest.raises(MetricIndeterminate, match="non-finite"):
        nonfinite.h_scale()

    negative = Gauge(s_h=torch.tensor([[1.0, -2.0, 1.0]], dtype=torch.float64))
    with pytest.raises(MetricIndeterminate):
        negative.h_scale()

    # m present but the gauge was built with no m scale at all: a
    # construction bug, not a data-dependent indeterminacy -> ValueError
    no_m = Gauge(s_h=torch.ones(1, 2, dtype=torch.float64))
    with pytest.raises(ValueError, match="no m scale"):
        no_m.m_scale(torch.zeros(1, 3, dtype=torch.float64))

    # s_m and s_m_positions are mutually exclusive
    with pytest.raises(ValueError, match="never both"):
        Gauge(
            s_h=torch.ones(1, 2, dtype=torch.float64),
            s_m=torch.ones(1, 2, dtype=torch.float64),
            s_m_positions=torch.ones(2, 1, 2, dtype=torch.float64),
            position_axis=0,
        )
    # position_axis and s_m_positions must be set together
    with pytest.raises(ValueError, match="together"):
        Gauge(s_h=torch.ones(1, 2, dtype=torch.float64), s_m_positions=torch.ones(2, 1, 2, dtype=torch.float64))


def test_gauge_growing_position_coverage():
    from sld.objectives.jac_reg import Gauge, MetricIndeterminate

    # position axis 0 (mirrors _append_family's m layout (L, B, d_m)),
    # reference rollout covers 5 positions, 2 features each
    s_positions = torch.arange(1, 11, dtype=torch.float64).reshape(5, 1, 2)
    gauge = Gauge(s_h=torch.ones(1, 3, dtype=torch.float64), s_m_positions=s_positions, position_axis=0)

    m3 = torch.zeros(3, 1, 2, dtype=torch.float64)
    scale3 = gauge.m_scale(m3)
    assert torch.equal(scale3, s_positions[:3])

    m5 = torch.zeros(5, 1, 2, dtype=torch.float64)
    assert torch.equal(gauge.m_scale(m5), s_positions)

    m6 = torch.zeros(6, 1, 2, dtype=torch.float64)  # beyond the 5-position reference coverage
    with pytest.raises(MetricIndeterminate, match="beyond"):
        gauge.m_scale(m6)


def test_call_gauge_matches_normalized_dense_frobenius_gold():
    # cross-check on the TENSOR path (mlp-like, q == h, no m): the
    # gauge-normalized stochastic estimator must target
    # ||normalize_jacobian(A, s, s)||_F^2 exactly, not ||A||_F^2 -- the
    # declared metric applied identically to what training penalizes.
    torch.manual_seed(0)
    d = 5
    A = torch.randn(d, d, dtype=torch.float64)
    # deliberately span several orders of magnitude, like h vs m's variance
    # gap this amendment exists to fix
    s = torch.tensor([1e-3, 1.0, 1e2, 3.0, 1e-1], dtype=torch.float64).reshape(1, d)
    from sld.objectives.jac_reg import Gauge, normalize_jacobian

    truth = float((normalize_jacobian(A.numpy(), s.numpy().reshape(-1), s.numpy().reshape(-1)) ** 2).sum())
    raw = float((A * A).sum())
    assert abs(truth - raw) / raw > 0.5  # the gauge must actually CHANGE the quantity

    gauge = Gauge(s_h=s)
    g = torch.Generator().manual_seed(1)
    pen = TotalJacobianPenalty(k_probes=8, step_subsample=1.0, generator=g, gauge=gauge)
    q_seq = torch.randn(4, 8, d, dtype=torch.float64)
    draws = [pen(_linear_step(A), q_seq).item() for _ in range(200)]
    mean_est = sum(draws) / len(draws)
    rel_err = abs(mean_est - truth) / truth
    print(f"[gauge gold, tensor path] mean={mean_est:.4f} truth={truth:.4f} rel_err={rel_err:.3%}")
    assert rel_err < 0.05


def test_on_state_steps_gauge_matches_dense_normalized_audit_fixed_size_state():
    # THE training/audit consistency lock, pytree path (min_ssm-like: fixed-
    # size m, nothing excluded). What TotalJacobianPenalty penalizes and
    # what audit.jacobian.augmented_sigma_max_normalized measures must be
    # built from the SAME normalize_jacobian call on the SAME dense J --
    # the exact incoherence an adversarial review already found one layer
    # down (the un-normalized object) must not reappear here.
    from sld.audit.jacobian import augmented_jacobian_excluding_identity
    from sld.objectives.jac_reg import Gauge, normalize_jacobian

    torch.manual_seed(0)
    d_h, d_m, d_u, B = 4, 3, 2, 1
    A = torch.randn(d_h, d_h, dtype=torch.float64)
    C = torch.randn(d_h, d_m, dtype=torch.float64)
    Wm = torch.randn(d_m, d_m, dtype=torch.float64)
    Wh = torch.randn(d_m, d_h, dtype=torch.float64)

    def step_fn(h, m, u):
        h_next = h @ A.T + m @ C.T
        m_next = m @ Wm.T + h @ Wh.T  # FIXED size: no growth, nothing excluded
        return h_next, m_next

    h_t = torch.randn(B, d_h, dtype=torch.float64)
    m_t = torch.randn(B, d_m, dtype=torch.float64)
    u_t = torch.randn(B, d_u, dtype=torch.float64)

    s_h = torch.tensor([1e-2, 1.0, 5.0, 1e2], dtype=torch.float64).reshape(1, d_h)
    s_m = torch.tensor([1e3, 1e-3, 2.0], dtype=torch.float64).reshape(1, d_m)
    gauge = Gauge(s_h=s_h, s_m=s_m)

    dense = augmented_jacobian_excluding_identity(step_fn, h_t, m_t, u_t)
    assert dense.n_identity_rows == 0  # fixed-size state: nothing excluded
    s_in = torch.cat([s_h.reshape(-1), s_m.reshape(-1)]).numpy()
    s_out = s_in  # h and m are both fixed-size/non-growing here: S_out == S_in
    M = normalize_jacobian(dense.jacobian, s_in, s_out)
    truth_frobenius_sq = float((M**2).sum())

    g = torch.Generator().manual_seed(23)
    pen = TotalJacobianPenalty(k_probes=8, step_subsample=1.0, generator=g, gauge=gauge)
    draws = [pen.on_state_steps(step_fn, [(h_t, m_t, u_t)]).item() for _ in range(400)]
    mean_est = sum(draws) / len(draws)
    rel_err = abs(mean_est - truth_frobenius_sq) / truth_frobenius_sq
    print(
        f"[consistency lock, pytree/fixed-size] stochastic={mean_est:.4f} "
        f"dense_normalized={truth_frobenius_sq:.4f} rel_err={rel_err:.3%}"
    )
    assert rel_err < 0.05

    # and the audit's own entry point agrees with the same truth's sigma_max
    from sld.audit.jacobian import augmented_sigma_max_normalized
    from sld.audit.spectral import operator_norm

    audited = augmented_sigma_max_normalized(step_fn, h_t, m_t, u_t, gauge)
    assert abs(audited - operator_norm(M)) <= 1e-10 * max(1.0, operator_norm(M))


def test_on_state_steps_gauge_growing_memory_matches_dense_normalized_audit():
    # THE training/audit consistency lock, GROWING pytree path
    # (transformer-like): the identity-passthrough exclusion must survive
    # the gauge exactly, because a persisting coordinate's S_out equals its
    # S_in (Gauge.s_m_positions's declared property) -- verified here
    # end-to-end rather than merely asserted in a docstring.
    from sld.audit.jacobian import augmented_jacobian_excluding_identity, augmented_sigma_max_normalized
    from sld.audit.spectral import operator_norm
    from sld.objectives.jac_reg import Gauge, normalize_jacobian

    torch.manual_seed(0)
    d_h, d_m, d_u, B, L = 3, 2, 2, 1, 2  # L=2 old positions already in the cache
    A = torch.randn(d_h, d_h, dtype=torch.float64)
    C = torch.randn(d_h, d_m, dtype=torch.float64)
    W = torch.randn(d_m, d_h, dtype=torch.float64)
    step_fn = _append_family(A, C, W)

    h_t = torch.randn(B, d_h, dtype=torch.float64)
    m_t = torch.randn(L, B, d_m, dtype=torch.float64)
    u_t = torch.randn(B, d_u, dtype=torch.float64)

    s_h = torch.tensor([1e-2, 1.0, 1e2], dtype=torch.float64).reshape(1, d_h)
    # one frozen (position, feature) scale per absolute position, reference
    # rollout covers up to 4 positions (>= L + 1 needed here)
    s_positions = (10.0 ** torch.arange(-2, 2, dtype=torch.float64)).reshape(4, 1, 1) * torch.tensor(
        [1.0, 3.0], dtype=torch.float64
    ).reshape(1, 1, 2)
    gauge = Gauge(s_h=s_h, s_m_positions=s_positions, position_axis=0)

    dense = augmented_jacobian_excluding_identity(step_fn, h_t, m_t, u_t)
    assert dense.n_identity_rows == m_t.numel()  # growing memory: old L positions excluded

    s_in = torch.cat([s_h.reshape(-1), s_positions[:L].reshape(-1)]).numpy()
    s_out_full = torch.cat([s_h.reshape(-1), s_positions[: L + 1].reshape(-1)]).numpy()
    s_out = np.delete(s_out_full, dense.excluded_row_indices)
    M = normalize_jacobian(dense.jacobian, s_in, s_out)
    truth_frobenius_sq = float((M**2).sum())

    g = torch.Generator().manual_seed(23)
    pen = TotalJacobianPenalty(k_probes=8, step_subsample=1.0, generator=g, gauge=gauge)
    draws = [pen.on_state_steps(step_fn, [(h_t, m_t, u_t)]).item() for _ in range(400)]
    mean_est = sum(draws) / len(draws)
    rel_err = abs(mean_est - truth_frobenius_sq) / truth_frobenius_sq
    print(
        f"[consistency lock, pytree/growing] stochastic={mean_est:.4f} "
        f"dense_normalized={truth_frobenius_sq:.4f} rel_err={rel_err:.3%}"
    )
    assert rel_err < 0.05

    audited = augmented_sigma_max_normalized(step_fn, h_t, m_t, u_t, gauge)
    assert abs(audited - operator_norm(M)) <= 1e-8 * max(1.0, operator_norm(M))
