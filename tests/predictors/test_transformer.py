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


# ===========================================================================
# WAVE C2 -- CausalTransformerContextBackbone: the query is h_t alone, the
# keys/values are the PAST pairs (h_s, u_s) plus a learned BOS entry.
#
# Same method as the attention oracle above: every weight is assigned by hand
# from arithmetic ramps, and `_manual_context` recomputes the whole thing from
# scratch in full-prefix form -- it never calls the backbone.
# ===========================================================================
from sld.predictors.transformer import CausalTransformerContextBackbone  # noqa: E402

CTX_DTYPE = torch.float64
CTX_TOL = 1e-10  # float64 forward, single layer -- see the wave report


def _ramp(*shape: int, scale: float, offset: float = 0.0) -> Tensor:
    n = 1
    for s in shape:
        n *= s
    return torch.arange(n, dtype=CTX_DTYPE).reshape(shape) * scale + offset


def _hand_set_context_backbone() -> CausalTransformerContextBackbone:
    """1 head, 1 layer, width 4, float64, every weight a deterministic ramp.

    Neutralized and declared (REGLES 7.1 allows isolating attention when the
    full layer is too rich to reproduce honestly): ``ln1`` becomes
    ``nn.Identity`` so the oracle may skip it -- it must be a real no-op in the
    module, not merely assumed away -- and both MLP ``Linear``s are zeroed
    weight AND bias so that branch contributes exactly 0 whatever ``ln2`` and
    GELU do upstream. Everything the wave is about stays real and hand-set:
    ``q_in`` (query read-in, h ONLY), ``pair_in`` (cache read-in, the pair),
    ``kv_proj``, ``q_proj``, ``attn_out`` and the learned BOS entry.
    """
    with torch.random.fork_rng():
        torch.manual_seed(0)
        bb = CausalTransformerContextBackbone(
            d_h=2, d_u=1, width=4, n_heads=1, n_layers=1, d_ff=4,
            max_len=16, dtype=CTX_DTYPE,
        )
    block = bb.blocks[0]
    with torch.no_grad():
        bb.q_in.weight.copy_(_ramp(*bb.q_in.weight.shape, scale=0.03, offset=-0.05))
        bb.q_in.bias.copy_(_ramp(*bb.q_in.bias.shape, scale=0.01))
        bb.pair_in.weight.copy_(_ramp(*bb.pair_in.weight.shape, scale=0.04, offset=-0.07))
        bb.pair_in.bias.copy_(_ramp(*bb.pair_in.bias.shape, scale=0.02, offset=-0.01))
        block.q_proj.weight.copy_(_ramp(*block.q_proj.weight.shape, scale=0.02, offset=-0.15))
        block.q_proj.bias.copy_(_ramp(*block.q_proj.bias.shape, scale=0.01, offset=-0.02))
        block.kv_proj.weight.copy_(_ramp(*block.kv_proj.weight.shape, scale=0.015, offset=-0.2))
        block.kv_proj.bias.copy_(_ramp(*block.kv_proj.bias.shape, scale=0.005, offset=-0.03))
        block.attn_out.weight.copy_(_ramp(*block.attn_out.weight.shape, scale=0.04, offset=-0.08))
        block.attn_out.bias.zero_()
        block.bos_kv.copy_(_ramp(*block.bos_kv.shape, scale=0.06, offset=-0.1))
        block.mlp[0].weight.zero_()
        block.mlp[0].bias.zero_()
        block.mlp[2].weight.zero_()
        block.mlp[2].bias.zero_()
        block.ln1 = nn.Identity()
    return bb


def _manual_context(
    bb: CausalTransformerContextBackbone,
    h: Tensor,
    pairs: list[tuple[Tensor, Tensor]],
    t: int,
    *,
    drop_bos: bool = False,
    shift_pos: int = 0,
    query_attends_to_itself: bool = False,
) -> Tensor:
    """Independent full-prefix rewrite of ``context_step`` after ``len(pairs)``
    ``absorb`` calls -- written from scratch, never calling the backbone.

    Each keyword replaces exactly one line of the otherwise-correct
    computation with one specific way of getting the contract wrong.
    """
    block = bb.blocks[0]
    d_head = bb.d_head  # == width, n_heads == 1
    B = h.shape[0]

    keys, values = [], []
    if not drop_bos:
        keys.append(block.bos_kv[0].reshape(1, d_head).expand(B, d_head))
        values.append(block.bos_kv[1].reshape(1, d_head).expand(B, d_head))
    for s, (h_s, u_s) in enumerate(pairs):
        tok = torch.cat([h_s, u_s], dim=-1) @ bb.pair_in.weight.T + bb.pair_in.bias
        tok = tok + bb.pe[s].to(CTX_DTYPE)
        kv = tok @ block.kv_proj.weight.T + block.kv_proj.bias
        k_s, v_s = kv.chunk(2, dim=-1)
        keys.append(k_s)
        values.append(v_s)

    x = h @ bb.q_in.weight.T + bb.q_in.bias + bb.pe[t + shift_pos].to(CTX_DTYPE)
    if query_attends_to_itself:
        # what `_CausalBlock` does, and what this backbone must NOT do: build
        # k/v from the CURRENT token (which carries no u) and let it into the
        # cache it is about to read
        kv_self = x @ block.kv_proj.weight.T + block.kv_proj.bias
        k_self, v_self = kv_self.chunk(2, dim=-1)
        keys.append(k_self)
        values.append(v_self)

    q = x @ block.q_proj.weight.T + block.q_proj.bias
    K = torch.stack(keys, dim=1)  # (B, S, d_head)
    V = torch.stack(values, dim=1)
    scores = torch.einsum("bd,bsd->bs", q, K) * (d_head**-0.5)
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bs,bsd->bd", attn, V)
    x = x + (out @ block.attn_out.weight.T + block.attn_out.bias)
    return x  # MLP branch neutralized to exactly 0 -- see the builder docstring


def _ctx_inputs(n_pairs: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    pairs = [
        (
            torch.randn(2, 2, generator=gen, dtype=CTX_DTYPE),
            torch.randn(2, 1, generator=gen, dtype=CTX_DTYPE),
        )
        for _ in range(n_pairs)
    ]
    return pairs


def _absorb_all(bb, pairs) -> PredictorState:
    state = PredictorState()
    for h_s, u_s in pairs:
        state = bb.absorb(h_s, u_s, state)
    return state


def test_context_backbone_two_prefixes_match_the_full_prefix_reference():
    """THE POSITIVE: same current token, same position t=2, two known prefixes.
    Both contexts must equal an independently written full-prefix computation,
    and they must differ from each other -- otherwise the prefix is decorative.
    """
    bb = _hand_set_context_backbone()
    h = torch.tensor([[0.3, -0.8], [1.1, 0.2]], dtype=CTX_DTYPE)
    pre_a = _ctx_inputs(2, seed=101)
    pre_b = _ctx_inputs(2, seed=202)

    got, want = [], []
    for prefix in (pre_a, pre_b):
        got.append(bb.context_step(h, _absorb_all(bb, prefix)))
        want.append(_manual_context(bb, h, prefix, t=2))

    errs = [(g - w).abs().max().item() for g, w in zip(got, want)]
    spread = (got[0] - got[1]).abs().max().item()
    print(f"[ctx-oracle] positive err={[f'{e:.3e}' for e in errs]} prefix spread={spread:.6e}")
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=0.0, atol=CTX_TOL)
    assert spread > 1e-3


def test_context_backbone_is_defined_at_t0_through_the_learned_bos():
    """The empty cache is not a special case: the BOS entry is always there,
    so c_0 exists AND depends on h_0. The negative half is the same oracle with
    the BOS entry dropped, which must diverge once a real entry is present."""
    bb = _hand_set_context_backbone()
    h_a = torch.tensor([[0.3, -0.8], [1.1, 0.2]], dtype=CTX_DTYPE)
    h_b = torch.tensor([[-1.4, 0.9], [0.5, -0.3]], dtype=CTX_DTYPE)

    c_a = bb.context_step(h_a, PredictorState())
    c_b = bb.context_step(h_b, PredictorState())
    err = (c_a - _manual_context(bb, h_a, [], t=0)).abs().max().item()
    h_dep = (c_a - c_b).abs().max().item()
    print(f"[ctx-oracle] t=0 err={err:.3e} dependence on h_0={h_dep:.6e}")
    torch.testing.assert_close(c_a, _manual_context(bb, h_a, [], t=0), rtol=0.0, atol=CTX_TOL)
    assert torch.isfinite(c_a).all()
    assert h_dep > 1e-3

    # NEGATIVE: an implementation whose attention forgets the BOS entry
    prefix = _ctx_inputs(1, seed=303)
    real = bb.context_step(h_a, _absorb_all(bb, prefix))
    no_bos = _manual_context(bb, h_a, prefix, t=1, drop_bos=True)
    gap = (real - no_bos).abs().max().item()
    print(f"[ctx-oracle] NEGATIVE drop-bos gap={gap:.6e}")
    assert gap > CTX_TOL


def test_context_backbone_query_must_not_attend_to_itself():
    """NEGATIVE: the `_CausalBlock` behaviour (q, k, v from the same token) is
    exactly what this backbone must not do -- the current token carries no
    ``u``, so letting it into its own cache would treat it differently from
    every past entry and open the door to a u_t leak."""
    bb = _hand_set_context_backbone()
    h = torch.tensor([[0.3, -0.8], [1.1, 0.2]], dtype=CTX_DTYPE)
    prefix = _ctx_inputs(2, seed=404)
    real = bb.context_step(h, _absorb_all(bb, prefix))
    selfish = _manual_context(bb, h, prefix, t=2, query_attends_to_itself=True)
    gap = (real - selfish).abs().max().item()
    print(f"[ctx-oracle] NEGATIVE self-attention gap={gap:.6e}")
    assert gap > CTX_TOL


def test_context_backbone_positions_must_follow_the_prefix():
    """NEGATIVE: the query's positional index must be the number of absorbed
    pairs. Shifting it by one is invisible to shapes and to every finiteness
    check, and it is caught here."""
    bb = _hand_set_context_backbone()
    h = torch.tensor([[0.3, -0.8], [1.1, 0.2]], dtype=CTX_DTYPE)
    prefix = _ctx_inputs(2, seed=505)
    real = bb.context_step(h, _absorb_all(bb, prefix))
    shifted = _manual_context(bb, h, prefix, t=2, shift_pos=1)
    gap = (real - shifted).abs().max().item()
    print(f"[ctx-oracle] NEGATIVE shifted-position gap={gap:.6e}")
    assert gap > CTX_TOL


class _DeadCacheBackbone(CausalTransformerContextBackbone):
    """MUTANT: absorb accumulates nothing. Shapes stay right, rollouts stay
    finite, and every context silently becomes the t=0 context."""

    def absorb(self, h: Tensor, u: Tensor, state: PredictorState) -> PredictorState:
        return state


def test_context_backbone_negative_dead_cache_kills_the_prefix_sentinel():
    """NEGATIVE: two different prefixes must give two different contexts. With
    a dead cache the spread collapses to exactly 0 -- the measured gap between
    the real backbone and the mutant IS the sentinel's discriminating power."""
    h = torch.tensor([[0.3, -0.8], [1.1, 0.2]], dtype=CTX_DTYPE)
    pre_a = _ctx_inputs(3, seed=606)
    pre_b = _ctx_inputs(3, seed=707)

    real_bb = _hand_set_context_backbone()
    spread_real = (
        real_bb.context_step(h, _absorb_all(real_bb, pre_a))
        - real_bb.context_step(h, _absorb_all(real_bb, pre_b))
    ).abs().max().item()

    with torch.random.fork_rng():
        torch.manual_seed(0)
        dead = _DeadCacheBackbone(d_h=2, d_u=1, width=4, n_heads=1, n_layers=1, dtype=CTX_DTYPE)
    spread_dead = (
        dead.context_step(h, _absorb_all(dead, pre_a))
        - dead.context_step(h, _absorb_all(dead, pre_b))
    ).abs().max().item()
    print(f"[ctx-oracle] NEGATIVE dead-cache: real spread={spread_real:.6e} mutant={spread_dead:.6e}")
    assert spread_real > 1e-3
    assert spread_dead == 0.0


def test_context_backbone_cache_layout_grows_one_entry_per_absorb():
    bb = CausalTransformerContextBackbone(d_h=2, d_u=1, width=8, n_heads=2, n_layers=3)
    state = PredictorState()
    assert state.m is None
    for s in range(4):
        assert bb.context_step(torch.randn(5, 2), state).shape == (5, 8)
        state = bb.absorb(torch.randn(5, 2), torch.randn(5, 1), state)
        # (n_layers, 2, B, n_heads, S, d_head) -- same layout as the family
        assert state.m.shape == (3, 2, 5, 2, s + 1, 4)
