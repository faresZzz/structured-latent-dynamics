"""Minimal causal transformer predictor: the generic O(H^2)-per-trajectory baseline.

Naming discipline (non-negotiable, REGLES vocabulary rules): this is a
*minimal causal transformer* over the (h, u) token stream, nothing more. It
has no claim to structured dynamics or to any operator-theoretic reading --
``local_operator`` stays the inherited ``None`` and the audit falls back to
autograd. Attention cost grows quadratically with the number of steps
already seen in the current trajectory (each new token attends to the full
cache), so a rollout of length T costs O(T^2) in total -- the family is a
generic capacity baseline, not a scalable architecture, and must never be
oversold as one.

Token construction: at step t, ``Linear(concat([h_t, u_t])) -> d_model``,
plus a FIXED sinusoidal positional encoding (no learned position
parameters, so the parameter-count comparison against the other families
stays fair) added at position ``t = len(cache)``. Two pre-LN causal blocks
(multi-head self-attention + MLP) follow. Output head is residual, matching
the ``mlp`` family: ``h_next = h + Linear(d_model -> d_h)(y_t)``.

KV cache: ``PredictorState.m`` packs keys and values for every layer into
ONE tensor of shape ``(n_layers, 2, B, n_heads, t, d_head)`` (dim 1 indexes
K=0/V=1). ``m=None`` is the empty cache. ``step`` appends the new token's
K/V and returns the grown cache; ``rollout`` (inherited from
``PredictorModule``) always starts from a fresh ``PredictorState()``, which
is what resets the cache between episodes. Callers driving ``step``
directly are responsible for passing a fresh ``PredictorState()`` at the
start of every episode -- reusing a state across episode boundaries would
leak one trajectory's context into another's predictions.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from sld.predictors.base import PredictorModule, PredictorState


def _build_sinusoidal_pe(max_len: int, d_model: int) -> Tensor:
    # standard fixed sin/cos table (Vaswani et al.), even/odd columns split
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)[:, : d_model // 2]
    return pe


class _CausalBlock(nn.Module):
    """One pre-LN block: causal multi-head self-attention + MLP, both residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: Tensor, cache: Tensor | None) -> tuple[Tensor, Tensor]:
        # x: (B, 1, d_model), the single new token. cache: (2, B, n_heads, t, d_head) or None.
        B = x.shape[0]
        q, k_new, v_new = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        k_new = k_new.view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        v_new = v_new.view(B, 1, self.n_heads, self.d_head).transpose(1, 2)

        if cache is None:
            k_cache, v_cache = k_new, v_new
        else:
            k_cache = torch.cat([cache[0], k_new], dim=2)
            v_cache = torch.cat([cache[1], v_new], dim=2)

        # new token attends to the full cache (itself included) -- causal by
        # construction, since the cache never holds future positions
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k_cache) * (self.d_head**-0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v_cache)
        out = out.transpose(1, 2).reshape(B, 1, self.n_heads * self.d_head)
        x = x + self.attn_out(out)
        x = x + self.mlp(self.ln2(x))

        new_cache = torch.stack([k_cache, v_cache], dim=0)
        return x, new_cache


class CausalTransformerPredictor(PredictorModule):
    """Minimal causal transformer over the (h, u) token stream.

    Args:
        d_h: latent state dimension.
        d_u: action dimension.
        d_model: token/hidden dimension.
        n_heads: attention heads (must divide ``d_model``).
        n_layers: number of pre-LN causal blocks.
        d_ff: MLP hidden width per block; defaults to ``4 * d_model``.
        max_len: maximum trajectory length the fixed positional table (and
            therefore the KV cache) supports; ``step`` raises ``ValueError``
            past this length.
    """

    def __init__(
        self,
        d_h: int,
        d_u: int,
        d_model: int = 32,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int | None = None,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        d_ff = 4 * d_model if d_ff is None else d_ff

        self.d_h = d_h
        self.d_u = d_u
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_head = d_model // n_heads
        self.max_len = max_len

        self.in_proj = nn.Linear(d_h + d_u, d_model)
        self.blocks = nn.ModuleList(
            _CausalBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        )
        self.out_proj = nn.Linear(d_model, d_h)
        # fixed (non-learned) table: a buffer, not a parameter
        self.register_buffer("pe", _build_sinusoidal_pe(max_len, d_model), persistent=False)

    def step(
        self, h: Tensor, u: Tensor, state: PredictorState
    ) -> tuple[Tensor, PredictorState]:
        m = state.m
        t = 0 if m is None else m.shape[4]
        if t >= self.max_len:
            raise ValueError(
                f"cache length {t} would exceed max_len={self.max_len} on this step"
            )

        token = self.in_proj(torch.cat([h, u], dim=-1))  # (B, d_model)
        token = token + self.pe[t].to(dtype=token.dtype, device=token.device)
        x = token.unsqueeze(1)  # (B, 1, d_model), the single query token

        new_cache_layers = []
        for layer_idx, block in enumerate(self.blocks):
            layer_cache = None if m is None else m[layer_idx]
            x, layer_kv = block(x, layer_cache)
            new_cache_layers.append(layer_kv)
        new_m = torch.stack(new_cache_layers, dim=0)  # (n_layers, 2, B, n_heads, t+1, d_head)

        y = x.squeeze(1)  # (B, d_model)
        h_next = h + self.out_proj(y)
        return h_next, PredictorState(m=new_m)
