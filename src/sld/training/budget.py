"""Parameter-budget matching and analytic FLOP accounting for the four
predictor families (fairness contract: comparing families means comparing
them at the SAME parameter budget, OBJECTIFS' cross-family comparison).

``match_budget`` searches each family's single width knob by build-and-count
(construct, ``count_params``, compare) rather than a closed-form inverse --
exact by construction, and fast at this scale (widths in the tens to low
hundreds for the 10k-30k targets used in this study).

``count_flops_per_step``/``count_flops_per_trajectory`` are a SEPARATE axis
from the parameter budget: two families matched on the same parameter count
can still cost very different FLOPs per step -- most visibly the
transformer, whose attention cost grows with trajectory position.
"""

from __future__ import annotations

import torch
from torch import nn

from sld.predictors.koopman_ssm import KoopmanSSMPredictor
from sld.predictors.min_ssm import SelectiveSSMPredictor
from sld.predictors.mlp import MLPPredictor
from sld.predictors.transformer import CausalTransformerPredictor

# family name -> constructor, the single source of truth reused by registry.py
_FAMILY_CONSTRUCTORS: dict[str, type] = {
    "mlp": MLPPredictor,
    "transformer": CausalTransformerPredictor,
    "min_ssm": SelectiveSSMPredictor,
    "koopman_ssm": KoopmanSSMPredictor,
}


def count_params(model: nn.Module) -> int:
    """Sum of ``p.numel()`` over trainable (``requires_grad``) parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _linear_macs(module: nn.Module) -> int:
    """Sum of ``in_features * out_features`` over every ``nn.Linear`` in ``module``.

    A dense-matmul MAC count for whatever ``nn.Linear`` layers actually exist
    in the given submodule -- the constant (trajectory-position-independent)
    core of every family's step cost.
    """
    return sum(
        m.in_features * m.out_features for m in module.modules() if isinstance(m, nn.Linear)
    )


def _transformer_flops(model: CausalTransformerPredictor, t: int) -> int:
    """``f0 + c*t`` per-step cost of the causal transformer's ``step``.

    ``nn.Linear`` MACs (``in_proj``, each block's ``qkv``/``attn_out``/MLP,
    ``out_proj``) are ``t``-INDEPENDENT: they only ever see the current
    single token, never the cache. The two per-block attention einsums
    (scores against the cache, weighted sum over the cache) each contract
    against a ``(t + 1)``-long K/V cache -- ``2 * d_model * (t + 1)`` MACs
    per block, LINEAR in ``t``, which is the ``c * t`` term. Softmax and
    LayerNorm costs are declared out of scope (normalization/activation,
    same declared exemption as GELU elsewhere).
    """
    n_layers, d_model, d_h = model.n_layers, model.d_model, model.d_h
    linear_macs = _linear_macs(model)
    # elementwise residual adds (2 per block) + the fixed positional-encoding
    # add + the final residual readout add, all t-independent
    elementwise_const = (2 * n_layers + 1) * d_model + d_h
    c = 4 * n_layers * d_model  # 2 einsums/block * 2 FLOPs/MAC * d_model MACs/block
    f0 = 2 * linear_macs + elementwise_const + c
    return f0 + c * t


def _koopman_flops(model: KoopmanSSMPredictor) -> int:
    """Per-step cost of ``KoopmanSSMPredictor.step`` (``psi``/``pi`` excluded:
    they are read-only, never called inside ``step``/``rollout``).

    ``constant``/``bilinear`` share the affine core ``h @ A_bar.T + u @
    B_bar_0.T``: dense matmuls AS EXECUTED (the block-diagonal ``A_bar`` is
    realized densely by ``block_diagonal`` before the matmul -- fairness
    counts what runs, not the operator's sparsity), plus one elementwise
    combine add. ``bilinear`` additionally pays the interaction term
    ``sum_j u_j (B_bar_j h)``, a trilinear contraction counted at its
    dominant matmul-equivalent cost (``d_u`` per-block matrix-vector
    products of size ``d_h x d_h``). ``selective`` bypasses the affine core
    entirely (a different code path in ``step``): its cost is ``W_c``'s
    Linear plus a per-sample dense ``(B, d_h, d_h)``/``(B, d_h, d_u)``
    matvec pair, with the ``B_bar_t`` Hadamard product counted linearly.
    ``discretize_zoh``'s O(``n_pairs``) trig/exp block assembly is declared
    negligible, alongside the other families' activation costs.
    """
    d_h, d_u, n_pairs = model.d_h, model.d_u, model.n_pairs

    if model.regime == "selective":
        wc_macs = (d_h + d_u) * n_pairs  # W_c: Linear(d_h + d_u, n_pairs)
        av_macs = d_h * d_h  # einsum "bkl,bl->bk": dense A_bar_t @ h
        bu_macs = d_h * d_u  # einsum "bkj,bj->bk": dense B_bar_t @ u
        elementwise = d_h * d_u + d_h  # B_bar_t Hadamard product + combine add
        return 2 * (wc_macs + av_macs + bu_macs) + elementwise

    affine_macs = d_h * d_h + d_u * d_h
    flops = 2 * affine_macs + d_h  # + elementwise combine add
    if model.regime == "bilinear":
        interaction_macs = d_u * d_h * d_h
        flops += 2 * interaction_macs + d_h  # + elementwise combine add
    return flops


def count_flops_per_step(model: nn.Module, t: int = 1) -> int:
    """Analytic per-step FLOPs of ``model.step``, counted AS IMPLEMENTED.

    Convention (declared, not measured): 1 MAC = 2 FLOPs; every Linear /
    matmul / einsum actually executed by the step is counted at its dense
    shape (a block-diagonal operator realized as a dense matmul -- as three
    of the four families do -- counts dense: fairness counts what runs, not
    the operator's sparsity); elementwise combine/residual ops are counted
    linearly (1 op = 1 FLOP); activation costs (GELU, softplus, the
    exp/cos/sin inside ``blocks2x2``'s ZOH assembly) are ignored (declared).

    ``t`` is the transformer's current KV-cache length (tokens already seen
    before this step): its attention cost grows linearly in ``t``, so its
    per-step cost is ``f0 + c*t``. The other three families are markovian
    (``koopman_ssm``) or recurrent-with-constant-cost (``min_ssm``), so
    their step cost does not depend on trajectory position -- ``t`` is
    accepted for a uniform signature but ignored.

    Raises:
        TypeError: ``model`` is not one of the four predictor families.
    """
    if isinstance(model, MLPPredictor):
        return 2 * _linear_macs(model.net) + model.d_h  # + residual combine add
    if isinstance(model, CausalTransformerPredictor):
        return _transformer_flops(model, t)
    if isinstance(model, SelectiveSSMPredictor):
        # W_c, W_delta, W_b, W_in, W_out; elementwise: A_bar_t*m, B_bar_t*v_t,
        # their sum (3*d_state), plus the residual combine add (d_h)
        elementwise = 3 * model.d_state + model.d_h
        return 2 * _linear_macs(model) + elementwise
    if isinstance(model, KoopmanSSMPredictor):
        return _koopman_flops(model)
    raise TypeError(f"count_flops_per_step: unsupported predictor type {type(model).__name__}")


def count_flops_per_trajectory(model: nn.Module, H: int) -> int:
    """Sum of ``count_flops_per_step(model, t)`` for ``t = 0 .. H - 1``.

    Quadratic in ``H`` for the transformer (``sum_{t=0}^{H-1} (f0 + c*t) =
    H*f0 + c*H*(H-1)/2`` -- the O(H^2) compute bill behind the O(H^2)-per-
    trajectory baseline framing in ``transformer.py``'s module docstring);
    exactly linear in ``H`` for the other three families, whose per-step
    cost does not depend on ``t``.
    """
    return sum(count_flops_per_step(model, t) for t in range(H))


def _kwargs_for(family: str, w: int, d_z: int, d_u: int, fixed: dict) -> dict:
    """Full constructor kwargs for ``family`` at width knob ``w`` (``fixed`` applied last)."""
    if family == "mlp":
        kwargs = dict(d_h=d_z, d_u=d_u, hidden=(w, w))
    elif family == "transformer":
        n_heads = fixed.get("n_heads", 2)
        kwargs = dict(
            d_h=d_z,
            d_u=d_u,
            d_model=w,
            n_heads=n_heads,
            n_layers=2,
            d_ff=4 * w,
            max_len=fixed.get("max_len", 512),
        )
    elif family == "min_ssm":
        kwargs = dict(d_h=d_z, d_u=d_u, d_state=w)
    else:  # family == "koopman_ssm"; "delta" presence already checked by the caller
        n_pairs = fixed.get("n_pairs", d_z)  # default lifting ratio 2x: d_h = 2*d_z
        kwargs = dict(
            d_z=d_z,
            n_pairs=n_pairs,
            d_u=d_u,
            delta=fixed["delta"],
            regime=fixed.get("regime", "bilinear"),
            constrained=fixed.get("constrained", True),
            psi_hidden=(w,),
            pi_hidden=(w,),
        )
    kwargs.update(fixed)
    return kwargs


def _search_knob(build_fn, target: int, lo: int, step: int) -> int:
    """Integer search over a monotonic-non-decreasing knob ``w`` in
    ``{lo, lo + step, lo + 2*step, ...}`` for the value whose built model's
    parameter count is closest to ``target``.

    ``build_fn(w)`` constructs and returns the model at knob value ``w``.
    Parameter count is monotonic non-decreasing in ``w`` for every family's
    single width knob (verified by construction of each family's kwargs:
    wider layers strictly add parameters), so build-and-count with a
    doubling bracket followed by bisection is exact (not approximate) and
    fast at this scale.
    """

    def params_at(w: int) -> int:
        return count_params(build_fn(w))

    lo_w = hi_w = lo
    while params_at(hi_w) < target:
        lo_w = hi_w
        hi_w = hi_w * 2 if hi_w > lo else lo + step
        if hi_w > 1_000_000:
            # a fixed= override pinning the width knob makes params_at constant:
            # fail loudly instead of doubling forever
            raise ValueError(
                "match_budget: knob search diverged -- is the family's width knob "
                "actually free (not pinned by a fixed= override)?"
            )

    while hi_w - lo_w > step:
        mid = lo_w + ((hi_w - lo_w) // (2 * step)) * step
        if params_at(mid) < target:
            lo_w = mid
        else:
            hi_w = mid

    candidates = {w for w in (lo_w, hi_w) if w >= lo}
    return min(candidates, key=lambda w: abs(params_at(w) - target))


def match_budget(family: str, d_z: int, d_u: int, target_params: int = 10_000, **fixed) -> dict:
    """Full constructor kwargs for ``family`` whose built model's parameter
    count lands within +/-5% of ``target_params``.

    Searches the family's single width knob by build-and-count (see
    ``_search_knob``):
      - ``"mlp"``: knob = hidden width ``w``, ``hidden=(w, w)``.
      - ``"transformer"``: knob = ``d_model`` (kept divisible by
        ``n_heads``, default 2), ``d_ff = 4 * d_model``, ``n_layers=2``.
      - ``"min_ssm"``: knob = ``d_state``.
      - ``"koopman_ssm"``: knob = ``w`` with ``psi_hidden=(w,)``,
        ``pi_hidden=(w,)``; ``n_pairs`` defaults to ``d_z`` (2x lifting
        ratio: ``d_h = 2 * d_z``); ``delta`` is REQUIRED in ``fixed`` (no
        default -- it is a physical ZOH step, not a capacity knob).

    ``fixed`` keys are passed straight through to the constructor kwargs
    (applied after the searched width, so they can override it too).

    Raises:
        ValueError: unknown ``family``; ``family="koopman_ssm"`` without
            ``delta`` in ``fixed``; or the closest achievable parameter
            count is still more than 5% off ``target_params``.
    """
    if family not in _FAMILY_CONSTRUCTORS:
        raise ValueError(
            f"unknown family: {family!r}, expected one of {tuple(_FAMILY_CONSTRUCTORS)}"
        )
    if family == "koopman_ssm" and "delta" not in fixed:
        raise ValueError(
            "match_budget('koopman_ssm', ...) requires delta=... (the fixed ZOH step) -- "
            "it is a physical constant with no sane default, unlike the other families' "
            "width knob"
        )

    lo = step = fixed.get("n_heads", 2) if family == "transformer" else 1

    def build(w: int):
        return _FAMILY_CONSTRUCTORS[family](**_kwargs_for(family, w, d_z, d_u, fixed))

    # every construction below is throwaway (shape/param-count driven only);
    # fork the global RNG so the search leaves no trace on it -- the caller's
    # real, seeded construction happens after match_budget returns
    with torch.random.fork_rng():
        w = _search_knob(build, target_params, lo=lo, step=step)
        kwargs = _kwargs_for(family, w, d_z, d_u, fixed)
        achieved = count_params(build(w))

    rel_err = abs(achieved - target_params) / target_params
    if rel_err > 0.05:
        raise ValueError(
            f"match_budget: could not hit target_params={target_params} for family={family!r} "
            f"within +/-5% -- closest achieved is {achieved} ({rel_err:.1%} off) at "
            f"kwargs={kwargs}"
        )
    return kwargs
