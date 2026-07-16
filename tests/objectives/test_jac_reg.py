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
