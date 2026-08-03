import numpy as np
import pytest
import torch
from torch import Tensor, nn

from sld.data.regimes import OURegime
from sld.predictors.base import PredictorState
from sld.predictors.transformer import CausalTransformerPredictor
from sld.systems.damped_pendulum import DampedPendulum

GAMMA_OVERFIT = 10.0  # declared before measuring any model, never revised (REGLES 7.2)

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

    # --- persistence-relative gate (REGLES 7.1, added 2026-08-03) ------------
    # The absolute ``loss < 1e-3`` above is unmoored: it says nothing about
    # whether the model beat doing NOTHING on this trajectory. Measured on the
    # four families' own trajectories, persistence (h_hat_{t+1} = h_t) scores
    # mlp 1.8692e-02, transformer 3.8355e-02, min_ssm 1.7737e-03 and
    # koopman_ssm 9.0675e-04 -- so on koopman_ssm the 1e-3 gate is satisfied by
    # persistence itself (the test was literally vacuous there), and on min_ssm
    # it clears it by only 1.77x. GAMMA_OVERFIT is declared here, BEFORE any
    # model number was looked at, and is not revised afterwards.
    persistence = torch.mean((h_b[:-1] - h_b[1:]) ** 2).item()
    factor = persistence / loss.item()
    print(f"[overfit-transformer] persistence MSE = {persistence:.4e} | factor = {factor:.1f}x")
    assert loss.item() < persistence / GAMMA_OVERFIT

    with torch.no_grad():
        fr = model.rollout(h_b[0], u_b[:5])
    fr_mse = torch.mean((fr - h_b[1:6]) ** 2).item()
    pers_fr = torch.mean((h_b[0].unsqueeze(0) - h_b[1:6]) ** 2).item()
    print(f"[overfit-transformer] free-rollout H=5 MSE = {fr_mse:.4e} (persistence {pers_fr:.4e})")
    assert torch.isfinite(fr).all()
    assert fr_mse < pers_fr

    assert not (persistence < persistence / GAMMA_OVERFIT)


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


# ---------------------------------------------------------------------------
# REGLES 7.1 -- the attention math itself. The tests above prove a prefix
# INFLUENCES the output, and nothing more: none of them independently verify
# Q/K/V, the 1/sqrt(d_k) scale, the softmax axis, or the positional index.
# An implementation normalizing on the wrong axis, dropping the scale, or
# off-by-one on positions would pass every test above. The oracle below is a
# SECOND, independent line-by-line computation of the same math (never
# calling model.step/_CausalBlock.forward) on a 1-head, 1-layer, hand-set
# tiny model.
# ---------------------------------------------------------------------------

_ATTN_TOL = 1e-6  # R3: float32 forward pass, <=4-step recursive chain -- no
# long accumulation here, so the tight float32 bound applies as-is


def _make_hand_set_attention_model() -> CausalTransformerPredictor:
    """Minimal (1 head, 1 layer, ``d_model=4``) transformer, every weight
    assigned BY HAND under ``torch.no_grad()`` via deterministic arithmetic
    ramps -- never ``torch.randn`` -- so ``_manual_attention_step`` below is
    a genuine second computation of the same numbers, not a comparison
    against the model's own random init.

    Neutralized, documented (REGLES 7.1 explicitly allows isolating
    attention when the full layer is too rich to reproduce honestly --
    LayerNorm and the MLP here):
      - ``blocks[0].ln1`` is replaced by ``nn.Identity()``: the oracle skips
        pre-attention LayerNorm entirely, so it must be a real no-op in the
        model too, not just assumed away in the reference.
      - ``blocks[0].mlp``'s two ``nn.Linear``s both have weight AND bias
        zeroed, forcing the whole MLP branch to contribute exactly 0 to the
        residual stream regardless of what ``ln2``/GELU/the first Linear
        produce upstream. ``ln2`` itself is left as its real, default-init
        LayerNorm -- it is simply multiplied away to nothing downstream, so
        its output never reaches the model's output.
    Everything the task asks to verify is real and hand-set, NOT
    neutralized: ``in_proj`` (token read-in), ``qkv`` (Q/K/V), ``attn_out``
    (attention's own output projection), and ``out_proj`` (final read-out).
    """
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = CausalTransformerPredictor(
            d_h=2, d_u=1, d_model=4, n_heads=1, n_layers=1, d_ff=4, max_len=16
        )

    def ramp(*shape: int, scale: float, offset: float = 0.0) -> Tensor:
        n = 1
        for s in shape:
            n *= s
        return torch.arange(n, dtype=torch.float32).reshape(shape) * scale + offset

    with torch.no_grad():
        block = model.blocks[0]
        model.in_proj.weight.copy_(ramp(*model.in_proj.weight.shape, scale=0.03, offset=-0.05))
        model.in_proj.bias.copy_(ramp(*model.in_proj.bias.shape, scale=0.01))
        block.qkv.weight.copy_(ramp(*block.qkv.weight.shape, scale=0.02, offset=-0.15))
        block.qkv.bias.copy_(ramp(*block.qkv.bias.shape, scale=0.01, offset=-0.02))
        block.attn_out.weight.copy_(ramp(*block.attn_out.weight.shape, scale=0.04, offset=-0.08))
        block.attn_out.bias.zero_()
        block.mlp[0].weight.zero_()
        block.mlp[0].bias.zero_()
        block.mlp[2].weight.zero_()
        block.mlp[2].bias.zero_()
        model.out_proj.weight.copy_(ramp(*model.out_proj.weight.shape, scale=0.05, offset=-0.1))
        model.out_proj.bias.zero_()
        block.ln1 = nn.Identity()
    return model


def _manual_attention_step(
    model: CausalTransformerPredictor,
    h: Tensor,
    u: Tensor,
    k_cache: Tensor | None,
    v_cache: Tensor | None,
    t: int,
    *,
    no_scale: bool = False,
    wrong_softmax_axis: bool = False,
    shift_pos: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Independent, full-prefix rewrite of one ``step`` call (single head,
    single layer, ``ln1``/MLP neutralized -- see
    ``_make_hand_set_attention_model``): ``Q=xW_q``, ``K=xW_k``, ``V=xW_v``,
    scores ``QK^T/sqrt(d_k)``, softmax over the KEY axis, ``@V``, the
    attention output projection, then the residual read-out -- written from
    scratch here, never calling ``model.step``/``_CausalBlock.forward``.

    ``no_scale``/``wrong_softmax_axis``/``shift_pos`` each replace exactly
    one line of the otherwise-correct computation with one of the three
    classic attention bugs the task asks to rule out.
    """
    block = model.blocks[0]
    d_head = model.d_head  # == d_model since n_heads=1
    B = h.shape[0]

    x_in = torch.cat([h, u], dim=-1)
    token = x_in @ model.in_proj.weight.T + model.in_proj.bias
    token = token + model.pe[t + shift_pos]  # volet 3: positional index
    x = token  # ln1 neutralized to Identity, so this IS the attention input

    qkv = x @ block.qkv.weight.T + block.qkv.bias
    q, k_new, v_new = qkv.chunk(3, dim=-1)
    q4 = q.view(B, 1, 1, d_head)
    k_new4 = k_new.view(B, 1, 1, d_head)
    v_new4 = v_new.view(B, 1, 1, d_head)
    if k_cache is None:
        k_cache4, v_cache4 = k_new4, v_new4
    else:
        k_cache4 = torch.cat([k_cache, k_new4], dim=2)
        v_cache4 = torch.cat([v_cache, v_new4], dim=2)

    scale = 1.0 if no_scale else d_head**-0.5  # volet 1: the 1/sqrt(d_k) factor
    scores = torch.einsum("bhqd,bhkd->bhqk", q4, k_cache4) * scale
    axis = -2 if wrong_softmax_axis else -1  # volet 2: query axis vs key axis
    attn = torch.softmax(scores, dim=axis)
    out = torch.einsum("bhqk,bhkd->bhqd", attn, v_cache4)
    out = out.reshape(B, d_head)

    x = x + (out @ block.attn_out.weight.T + block.attn_out.bias)
    y = x  # MLP branch neutralized to exactly 0 -- see builder docstring
    h_next = h + (y @ model.out_proj.weight.T + model.out_proj.bias)
    return h_next, k_cache4, v_cache4


def _run_actual(model: CausalTransformerPredictor, h0: Tensor, u_seq: Tensor) -> list[Tensor]:
    h, state = h0, PredictorState()
    outs = []
    for t in range(u_seq.shape[0]):
        h, state = model.step(h, u_seq[t], state)
        outs.append(h)
    return outs


def _run_oracle(
    model: CausalTransformerPredictor, h0: Tensor, u_seq: Tensor, **flags: bool | int
) -> list[Tensor]:
    h, k_cache, v_cache = h0, None, None
    outs = []
    for t in range(u_seq.shape[0]):
        h, k_cache, v_cache = _manual_attention_step(
            model, h, u_seq[t], k_cache, v_cache, t, **flags
        )
        outs.append(h)
    return outs


def _attn_test_inputs() -> tuple[Tensor, Tensor]:
    gen = torch.Generator().manual_seed(42)
    B, T = 2, 4
    h0 = torch.randn(B, 2, generator=gen)
    u_seq = torch.randn(T, B, 1, generator=gen)
    return h0, u_seq


def test_attention_matches_hand_written_full_prefix_reference():
    # Positive (REGLES 7.1): a SECOND, independent computation of Q/K/V, the
    # 1/sqrt(d_k) scale, the causal softmax, and the output projection --
    # see _make_hand_set_attention_model for exactly what was neutralized
    # (ln1, MLP) and why.
    model = _make_hand_set_attention_model()
    h0, u_seq = _attn_test_inputs()

    actual_outs = _run_actual(model, h0, u_seq)
    oracle_outs = _run_oracle(model, h0, u_seq)

    diffs = [(a - o).abs().max().item() for a, o in zip(actual_outs, oracle_outs)]
    print(f"[attn-oracle] positive per-step max|diff| = {[f'{d:.3e}' for d in diffs]}")
    assert max(diffs) < _ATTN_TOL


def test_attention_negative_missing_inv_sqrt_dk_scale():
    # Volet 1: drop the 1/sqrt(d_k) factor. Degenerate at t=0 -- a single
    # cached key makes softmax scale-invariant (softmax of ONE scalar is
    # always 1.0 regardless of its value) -- but diverges from t=1 on, once
    # the cache holds >= 2 keys and relative score magnitudes start to
    # matter.
    model = _make_hand_set_attention_model()
    h0, u_seq = _attn_test_inputs()

    actual_outs = _run_actual(model, h0, u_seq)
    false_outs = _run_oracle(model, h0, u_seq, no_scale=True)

    diffs = [(a - o).abs().max().item() for a, o in zip(actual_outs, false_outs)]
    print(f"[attn-oracle] no-scale per-step max|diff| = {[f'{d:.3e}' for d in diffs]}")
    assert diffs[0] < _ATTN_TOL  # documented degeneracy at t=0, not a test bug
    assert diffs[-1] > _ATTN_TOL  # but the final step must clearly diverge


def test_attention_negative_softmax_wrong_axis():
    # Volet 2: softmax over the query axis (size 1) instead of the key axis.
    # Also degenerate at t=0 -- both axes have size 1 on the very first step,
    # so "the other axis" coincides with the right one there -- but from
    # t=1 on the key axis has >= 2 entries and the bug turns the weighted
    # average into an unnormalized SUM over the whole cache (softmax over a
    # size-1 axis is identically 1), which grows with cache length.
    model = _make_hand_set_attention_model()
    h0, u_seq = _attn_test_inputs()

    actual_outs = _run_actual(model, h0, u_seq)
    false_outs = _run_oracle(model, h0, u_seq, wrong_softmax_axis=True)

    diffs = [(a - o).abs().max().item() for a, o in zip(actual_outs, false_outs)]
    print(f"[attn-oracle] wrong-axis per-step max|diff| = {[f'{d:.3e}' for d in diffs]}")
    assert diffs[0] < _ATTN_TOL  # documented degeneracy at t=0, not a test bug
    assert diffs[-1] > _ATTN_TOL


def test_attention_negative_shifted_positional_index():
    # Volet 3: position t+1 instead of t. Unlike the other two volets, this
    # bug is visible immediately at t=0 -- the positional table enters the
    # residual stream directly, with no cache-size degeneracy to hide
    # behind.
    model = _make_hand_set_attention_model()
    h0, u_seq = _attn_test_inputs()

    actual_outs = _run_actual(model, h0, u_seq)
    false_outs = _run_oracle(model, h0, u_seq, shift_pos=1)

    diffs = [(a - o).abs().max().item() for a, o in zip(actual_outs, false_outs)]
    print(f"[attn-oracle] shifted-position per-step max|diff| = {[f'{d:.3e}' for d in diffs]}")
    assert all(d > _ATTN_TOL for d in diffs)
