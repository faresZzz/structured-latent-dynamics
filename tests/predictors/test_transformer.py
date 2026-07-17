import numpy as np
import pytest
import torch

from sld.data.regimes import OURegime
from sld.predictors.base import PredictorState
from sld.predictors.transformer import CausalTransformerPredictor
from sld.systems.damped_pendulum import DampedPendulum

D_H, D_U = 2, 1
D_MODEL, N_HEADS, N_LAYERS = 16, 2, 2
D_HEAD = D_MODEL // N_HEADS


def _make(seed: int, **kwargs) -> CausalTransformerPredictor:
    # nn.Module init reads torch's global RNG internally with no generator
    # hook; fork_rng() seeds it locally for this construction only and
    # restores the prior global state on exit, so tests stay independent.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return CausalTransformerPredictor(
            d_h=D_H, d_u=D_U, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, **kwargs
        )


def test_step_shapes_and_cache_growth():
    model = _make(0)
    B = 4
    gen = torch.Generator().manual_seed(1)
    h = torch.randn(B, D_H, generator=gen)
    u = torch.randn(B, D_U, generator=gen)

    h1, state1 = model.step(h, u, PredictorState())
    assert h1.shape == (B, D_H)
    assert state1.m.shape == (N_LAYERS, 2, B, N_HEADS, 1, D_HEAD)

    h2, state2 = model.step(h1, u, state1)
    assert h2.shape == (B, D_H)
    assert state2.m.shape == (N_LAYERS, 2, B, N_HEADS, 2, D_HEAD)


def test_rollout_shape():
    model = _make(1)
    B, T = 3, 9
    gen = torch.Generator().manual_seed(2)
    h0 = torch.randn(B, D_H, generator=gen)
    u_seq = torch.randn(T, B, D_U, generator=gen)
    out = model.rollout(h0, u_seq)
    assert out.shape == (T, B, D_H)
    assert torch.all(torch.isfinite(out))


def test_step_rollout_parity():
    # driving step manually from a fresh state must be bit-identical to
    # rollout (which does exactly this loop internally) -- no randomness
    # anywhere in the forward pass, so atol=0/rtol=0 is the right tolerance
    model = _make(2)
    B, T = 2, 6
    gen = torch.Generator().manual_seed(3)
    h0 = torch.randn(B, D_H, generator=gen)
    u_seq = torch.randn(T, B, D_U, generator=gen)

    out_rollout = model.rollout(h0, u_seq)

    h, state = h0, PredictorState()
    manual_outs = []
    for t in range(T):
        h, state = model.step(h, u_seq[t], state)
        manual_outs.append(h)
    out_manual = torch.stack(manual_outs, dim=0)

    torch.testing.assert_close(out_manual, out_rollout, atol=0.0, rtol=0.0)


def test_cache_reset_between_rollout_calls():
    # rollout always starts from a fresh PredictorState(), so two identical
    # calls back to back must give identical outputs -- no leakage
    model = _make(3)
    B, T = 2, 5
    gen = torch.Generator().manual_seed(4)
    h0 = torch.randn(B, D_H, generator=gen)
    u_seq = torch.randn(T, B, D_U, generator=gen)

    out_a = model.rollout(h0, u_seq)
    out_b = model.rollout(h0, u_seq)
    torch.testing.assert_close(out_a, out_b, atol=0.0, rtol=0.0)


def test_causality_perturbation_does_not_leak_backward_in_time():
    model = _make(4)
    B, T = 2, 8
    gen = torch.Generator().manual_seed(5)
    h0 = torch.randn(B, D_H, generator=gen)
    u_seq = torch.randn(T, B, D_U, generator=gen)

    out_base = model.rollout(h0, u_seq)

    perturb_t = 3
    u_seq_perturbed = u_seq.clone()
    u_seq_perturbed[perturb_t] += 10.0
    out_perturbed = model.rollout(h0, u_seq_perturbed)

    # strictly earlier outputs are untouched
    torch.testing.assert_close(
        out_perturbed[:perturb_t], out_base[:perturb_t], atol=0.0, rtol=0.0
    )
    # the perturbed step itself (and onward) must actually change --
    # otherwise the test would pass vacuously with a broken/no-op model
    diff = (out_perturbed[perturb_t] - out_base[perturb_t]).abs().max().item()
    assert diff > 0.0


def test_local_operator_is_generic_none():
    model = _make(5)
    gen = torch.Generator().manual_seed(6)
    h = torch.randn(2, D_H, generator=gen)
    u = torch.randn(2, D_U, generator=gen)
    assert model.local_operator(h, u, PredictorState()) is None


def test_max_len_guard_raises():
    model = _make(6, max_len=3)
    gen = torch.Generator().manual_seed(7)
    h = torch.randn(1, D_H, generator=gen)
    u = torch.randn(1, D_U, generator=gen)

    state = PredictorState()
    for _ in range(3):
        h, state = model.step(h, u, state)
    with pytest.raises(ValueError):
        model.step(h, u, state)


def _one_pendulum_trajectory(T: int = 40, dt: float = 0.05, seed: int = 0):
    system = DampedPendulum()
    rng = np.random.default_rng(seed)  # local Generator, no global numpy RNG touched
    x0, u_seq_np = OURegime().sample_episode(system, T=T, dt=dt, rng=rng)
    traj = system.integrate(x0, u_seq_np, dt=dt)  # (T+1, state_dim), float64
    h = torch.as_tensor(traj, dtype=torch.float32)
    u = torch.as_tensor(u_seq_np, dtype=torch.float32)
    return h, u


def test_overfit_one_trajectory():
    # T=40 (not the spec's T~=100): teacher forcing here backprops through
    # the ENTIRE unrolled T-step KV-cache sequentially (no batching over
    # time is possible, unlike the MLP family), so wall-clock is O(T) per
    # training step; T=40 keeps the gate well under the ~30s budget while
    # still exercising cache growth well past a handful of steps.
    h, u = _one_pendulum_trajectory()
    T = h.shape[0] - 1
    h_b = h.unsqueeze(1)  # (T+1, B=1, d_h), single-trajectory batch
    u_b = u.unsqueeze(1)  # (T, B=1, d_u)

    model = _make(seed=0, max_len=T + 1)
    opt = torch.optim.Adam(model.parameters(), lr=6e-3)

    loss = torch.tensor(float("inf"))
    for _ in range(55):
        opt.zero_grad()
        state = PredictorState()
        preds = []
        for t in range(T):
            h_pred, state = model.step(h_b[t], u_b[t], state)  # teacher-forced one-step
            preds.append(h_pred)
        preds = torch.stack(preds, dim=0)
        loss = torch.mean((preds - h_b[1:]) ** 2)
        loss.backward()
        opt.step()

    print(f"[overfit-transformer] final teacher-forced one-step MSE = {loss.item():.3e}")
    assert loss.item() < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-q"])


def test_cache_content_is_used():
    # adversarial review round 1: length-based tests pass even if cached K/V
    # were appended but ignored. Same current token, same position, two
    # DIFFERENT prefixes -> the outputs must differ, proving attention reads
    # the cache CONTENT.
    with torch.random.fork_rng():
        torch.manual_seed(11)
        model = CausalTransformerPredictor(d_h=3, d_u=2, d_model=16, n_heads=2)
    gen = torch.Generator().manual_seed(12)

    def run_prefix(seed_offset: int) -> PredictorState:
        g = torch.Generator().manual_seed(100 + seed_offset)
        state = PredictorState()
        for _ in range(2):
            h_p = torch.randn(4, 3, generator=g)
            u_p = torch.randn(4, 2, generator=g)
            _, state = model.step(h_p, u_p, state)
        return state

    state_a, state_b = run_prefix(0), run_prefix(1)
    h_cur = torch.randn(4, 3, generator=gen)
    u_cur = torch.randn(4, 2, generator=gen)
    out_a, _ = model.step(h_cur, u_cur, state_a)
    out_b, _ = model.step(h_cur, u_cur, state_b)
    diff = (out_a - out_b).abs().max().item()
    print(f"[cache-content] max |out_a - out_b| = {diff:.3e}")
    assert diff > 1e-6, "outputs identical across different prefixes: cache content unused"
