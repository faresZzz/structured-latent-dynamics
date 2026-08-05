"""Parameter-budget matching and analytic FLOP accounting for the seven
predictor families (fairness contract: comparing families means comparing
them at the SAME parameter budget, OBJECTIFS' cross-family comparison).

``match_budget`` searches each family's single width knob by build-and-count
(construct, ``count_params``, compare) rather than a closed-form inverse --
exact by construction, and fast at this scale (widths in the tens to low
hundreds for the 10k-30k targets used in this study).

Scope of the matching (declared): the PREDICTOR's parameters only. The
encoder is outside this budget -- any factorial arm that varies the encoder
(frame vs window) is NOT capacity-matched by this module, and the
pre-registration (Wave 5) must either hold the encoder fixed across
compared arms or declare encoder capacity as a separate, unmatched factor.

``count_flops_per_step``/``count_flops_per_trajectory`` are a SEPARATE axis
from the parameter budget: two families matched on the same parameter count
can still cost very different FLOPs per step -- most visibly the
transformer, whose attention cost grows with trajectory position.
"""

from __future__ import annotations

import torch
from torch import nn

from sld.predictors.koopman_ssm import KoopmanSSMPredictor
from sld.predictors.min_ssm import SelectiveSSMContextBackbone, SelectiveSSMPredictor
from sld.predictors.mlp import MLPContextBackbone, MLPPredictor
from sld.predictors.registry import STRUCTURED_FAMILIES, _FAMILY_CONSTRUCTORS
from sld.predictors.structured import StructuredPredictor
from sld.predictors.transformer import (
    CausalTransformerContextBackbone,
    CausalTransformerPredictor,
)

# ``delta`` is a physical ZOH step, not a capacity knob: no family that has
# one gets a default for it (the historical koopman_ssm and every structured
# arm alike -- PROTOCOL.md section 13.1 gives them the same fixed Delta).
_DELTA_REQUIRED = frozenset({"koopman_ssm", *STRUCTURED_FAMILIES})

# families whose width knob feeds multi-head attention and must therefore
# stay divisible by ``n_heads`` (the structured transformer arm has one too:
# ``CausalTransformerContextBackbone`` rejects width % n_heads != 0)
_HEAD_DIVISIBLE = frozenset({"transformer", "transformer_k"})


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
    # the ZOH input integral Phi @ B_cont IS executed by every step (the
    # affine operator is re-discretized at use time): per pair a (2x2)@(2xd_u)
    # block matmul, 4*d_u MACs -- "as executed" counts it (adversarial
    # review round 1); the O(n_pairs) trig/exp scalar assembly stays declared
    # negligible
    zoh_b_macs = 4 * d_u * n_pairs
    flops = 2 * (affine_macs + zoh_b_macs) + d_h  # + elementwise combine add
    if model.regime == "bilinear":
        interaction_macs = d_u * d_h * d_h
        flops += 2 * interaction_macs + d_h  # + elementwise combine add
    return flops


def _structured_backbone_flops(backbone: nn.Module, t: int) -> int:
    """Per-step cost of ``kappa_b``: ``context_step`` AND ``absorb``, as executed.

    ``StructuredPredictor.step`` calls both exactly once, so both are on the
    step's critical path. This is the ONLY place the three structured arms
    differ (PROTOCOL.md section 13.1); everything downstream of ``c_t`` is
    shared and counted by ``_structured_core_flops``.

    ``t`` is the number of pairs already absorbed. It matters only for the
    transformer backbone, whose two attention einsums contract against a
    ``(t + 1)``-long K/V cache (the learned BOS entry plus ``t`` cached
    pairs) -- the ``c * t`` term, LINEAR in ``t``, exactly as in
    ``_transformer_flops``. The other two backbones are memoryless
    (``mlp_k``) or constant-cost recurrent (``ssm_k``).

    Raises:
        TypeError: ``backbone`` is not one of the three context producers.
    """
    if isinstance(backbone, MLPContextBackbone):
        # absorb is a no-op (memoryless): only context_step's MLP runs
        return 2 * _linear_macs(backbone)
    if isinstance(backbone, SelectiveSSMContextBackbone):
        w = backbone.d_state
        # W_c runs TWICE per step -- once in context_step, once again inside
        # absorb (the ponytail duplicate declared in min_ssm.py). _linear_macs
        # counts each Linear once, so its second execution is added here.
        wc_macs = backbone.W_c.in_features * backbone.W_c.out_features
        # delta_t*a_cont, A_bar_t*m, delta_t*b_t (B_bar_t), B_bar_t*v_t, and
        # their sum; softplus/clamp/exp are activation-exempt (declared)
        elementwise = 5 * w
        return 2 * (_linear_macs(backbone) + wc_macs) + elementwise
    if isinstance(backbone, CausalTransformerContextBackbone):
        d_model, n_layers = backbone.d_model, backbone.n_layers
        # every Linear (q_in, pair_in, and each block's q_proj/kv_proj/
        # attn_out/mlp) runs exactly once per step: the query path in
        # context_step, kv_proj in absorb
        c = 4 * n_layers * d_model  # 2 einsums/block * 2 FLOPs/MAC * d_model
        # 2 positional-encoding adds (query token + pair token) + 2 residual
        # adds per block; LayerNorm and softmax are declared out of scope
        elementwise = 2 * d_model + 2 * n_layers * d_model
        return 2 * _linear_macs(backbone) + elementwise + c * (t + 1)
    raise TypeError(
        f"_structured_backbone_flops: unsupported context backbone "
        f"{type(backbone).__name__}"
    )


def _structured_core_flops(model: StructuredPredictor) -> int:
    """The shared part of a structured step: head, ZOH input integral, transition.

    Identical for the three arms by construction -- that is what makes
    "identical treatment, three backbones" a true sentence, and it is why
    ``psi``/``pi`` are NOT here: like ``koopman_ssm``'s, they are read-only
    and never called inside ``step`` (they enter the trajectory bill in
    ``count_flops_per_trajectory``).

    Counted as executed: the affine head ``theta_0 + W_theta c`` (one
    Linear); ``B_bar = Phi @ B_cont``, ``r`` block matmuls of ``(2, 2) @ (2,
    d_u)``, i.e. ``4 * d_u * r`` MACs -- the same term ``_koopman_flops``
    counts for the same integral; and the transition ``A_bar h + B_bar u``
    at its DENSE shape, since ``block_diagonal`` materializes ``A_bar``
    densely before the matmul. The ``exp``/``cos``/``sin`` block assembly is
    declared negligible, as everywhere else in this module.
    """
    d_h, d_u, r = model.d_h, model.d_u, model.r
    macs = _linear_macs(model.head) + 4 * d_u * r + d_h * d_h + d_h * d_u
    return 2 * macs + d_h  # + the elementwise combine add


def count_flops_per_step(model: nn.Module, t: int = 1) -> int:
    """Analytic per-step FLOPs of ``model.step``, counted AS IMPLEMENTED.

    Convention (declared, not measured): 1 MAC = 2 FLOPs; every Linear /
    matmul / einsum actually executed by the step is counted at its dense
    shape (a block-diagonal operator realized as a dense matmul -- as every
    family carrying 2x2 blocks does, ``koopman_ssm`` and the three
    structured arms alike -- counts dense: fairness counts what runs, not
    the operator's sparsity); elementwise combine/residual ops are counted
    linearly (1 op = 1 FLOP); activation costs (GELU, softplus, the
    exp/cos/sin inside ``blocks2x2``'s ZOH assembly) are ignored (declared).

    ``t`` is the current KV-cache length (tokens already seen before this
    step) of the two attention-based arms, ``transformer`` and
    ``transformer_k``: their attention cost grows linearly in ``t``, so
    their per-step cost is ``f0 + c*t``. The other five families are
    markovian (``mlp``, ``koopman_ssm``, ``mlp_k``) or
    recurrent-with-constant-cost (``min_ssm``, ``ssm_k``), so their step
    cost does not depend on trajectory position -- ``t`` is accepted for a
    uniform signature but ignored.

    Raises:
        TypeError: ``model`` is not one of the seven predictor families.
    """
    if isinstance(model, MLPPredictor):
        return 2 * _linear_macs(model.net) + model.d_h  # + residual combine add
    if isinstance(model, CausalTransformerPredictor):
        return _transformer_flops(model, t)
    if isinstance(model, SelectiveSSMPredictor):
        # W_c, W_delta, W_b, W_in, W_out; elementwise: delta_t*a_cont (the
        # exp itself is activation-exempt), delta_t*b_t (B_bar), A_bar_t*m,
        # B_bar_t*v_t, their sum (5*d_state), plus the residual combine
        # add (d_h)
        elementwise = 5 * model.d_state + model.d_h
        return 2 * _linear_macs(model) + elementwise
    if isinstance(model, KoopmanSSMPredictor):
        return _koopman_flops(model)
    if isinstance(model, StructuredPredictor):
        # ONE branch for the three arms: the transition is shared, only
        # kappa_b differs (PROTOCOL.md section 13.1)
        return _structured_core_flops(model) + _structured_backbone_flops(model.backbone, t)
    raise TypeError(f"count_flops_per_step: unsupported predictor type {type(model).__name__}")


def count_flops_per_trajectory(model: nn.Module, H: int) -> int:
    """Full read-out bill of an H-step trajectory (not just ``H * step``).

    Quadratic in ``H`` for the two attention arms, ``transformer`` and
    ``transformer_k`` (``sum_{t=0}^{H-1} (f0 + c*t) = H*f0 + c*H*(H-1)/2``
    -- the O(H^2) compute bill behind the O(H^2)-per-trajectory baseline
    framing in ``transformer.py``'s module docstring); AFFINE in ``H`` for
    the other five families (constant per-step cost, plus for the lifted
    families a constant lift).

    For every LIFTED family -- ``koopman_ssm`` and the three structured arms
    -- the bill includes what a read-out rollout actually executes beyond
    ``step``: ONE lift ``psi`` at the start and ONE projection ``pi`` per
    step (their psi/pi hold a large share of the matched parameters --
    excluding them would systematically undercount those families,
    adversarial review round 1). The other families' read-out is their step
    output itself; their per-step Linears are already fully counted.
    """
    total = sum(count_flops_per_step(model, t) for t in range(H))
    if isinstance(model, (KoopmanSSMPredictor, StructuredPredictor)):
        # 2 FLOPs/MAC over psi once + pi at every step
        total += 2 * _linear_macs(model.psi) + H * 2 * _linear_macs(model.pi)
    return total


def _kwargs_for(family: str, w: int, d_z: int, d_u: int, fixed: dict) -> dict:
    """Full constructor kwargs for ``family`` at width knob ``w`` (``fixed`` applied last).

    The chain below is CLOSED: an unknown family raises. It used to end in a
    bare ``else`` that WAS the ``koopman_ssm`` branch, so a family added to
    the registry and forgotten here would silently have been built with
    Koopman's kwargs instead of erroring.

    The three structured arms share ONE branch, not three: their constructor
    signature is identical by design (``structured.py``), with ``width`` the
    single budget knob of all three.
    """
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
    elif family == "koopman_ssm":  # "delta" presence already checked by the caller
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
    elif family in STRUCTURED_FAMILIES:
        # ONE branch for mlp_k/transformer_k/ssm_k: same signature, same
        # single knob. ``r`` defaults to d_z for the same 2x lifting ratio
        # koopman_ssm gets from n_pairs=d_z (d_h = 2*d_z), so the lifted
        # width is identical across all four lifted families at fixed d_z.
        kwargs = dict(
            d_z=d_z,
            d_u=d_u,
            delta=fixed["delta"],
            # r defaults to d_z HERE, not to the r=2 default the three
            # constructors carry in structured.py -- deliberately: this is
            # the only path get_predictor uses, so every run sees r=d_z,
            # and a hand-built StructuredPredictor arm (bypassing
            # get_predictor) is the only place r=2 is ever actually seen.
            r=fixed.get("r", d_z),
            width=w,
            psi_hidden=(w,),
            pi_hidden=(w,),
        )
    else:
        raise ValueError(
            f"_kwargs_for: no kwargs recipe for family {family!r} -- a family in the "
            f"registry must have a branch here, expected one of "
            f"{tuple(_FAMILY_CONSTRUCTORS)}"
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
      - ``"mlp_k"``/``"transformer_k"``/``"ssm_k"``: TWO stages, identical
        for the three arms -- coarse ``width`` (the context producer, last
        rung at or below target), then fine ``psi_hidden``/``pi_hidden``.
        Head, psi, pi and context are all inside the budget. ``r`` defaults
        to ``d_z``; ``delta`` is REQUIRED, same reason as ``koopman_ssm``.
        ``transformer_k``'s width stays divisible by ``n_heads``. Why two
        stages and not one: see the measurement in the body.

    ``fixed`` keys are passed straight through to the constructor kwargs
    (applied after the searched width, so they can override it too).

    Raises:
        ValueError: unknown ``family``; a family in ``_DELTA_REQUIRED``
            without ``delta`` in ``fixed``; or the closest achievable
            parameter count is still more than 5% off ``target_params``.
    """
    if family not in _FAMILY_CONSTRUCTORS:
        raise ValueError(
            f"unknown family: {family!r}, expected one of {tuple(_FAMILY_CONSTRUCTORS)}"
        )
    if family in _DELTA_REQUIRED and "delta" not in fixed:
        raise ValueError(
            f"match_budget({family!r}, ...) requires delta=... (the fixed ZOH step) -- "
            "it is a physical constant with no sane default, unlike the other families' "
            "width knob"
        )

    lo = step = fixed.get("n_heads", 2) if family in _HEAD_DIVISIBLE else 1

    def build_at(w: int, extra: dict):
        return _FAMILY_CONSTRUCTORS[family](**_kwargs_for(family, w, d_z, d_u, {**fixed, **extra}))

    # every construction below is throwaway (shape/param-count driven only);
    # fork the global RNG so the search leaves no trace on it -- the caller's
    # real, seeded construction happens after match_budget returns
    with torch.random.fork_rng():
        w = _search_knob(lambda w: build_at(w, {}), target_params, lo=lo, step=step)
        trim: dict = {}
        if family in STRUCTURED_FAMILIES:
            # MEASURED, and the reason this second stage exists: the width
            # knob alone cannot hold +/-5% for transformer_k. Its context
            # producer costs ~24*width^2 (2 blocks x (q_proj + kv_proj +
            # attn_out + a 4x MLP)) and width must stay a multiple of
            # n_heads, so at target_params=10_000, d_z=4, d_u=1, r=4 the
            # ladder steps 9442 (width 18) -> 11448 (width 20): -5.6% then
            # +14.5%, no rung inside the contract. That is a property of a
            # quadratic knob at this scale, not of a particular target.
            #
            # So: COARSE stage = width (the context producer), stepped DOWN
            # to the last rung at or below target so the fine stage only
            # ever adds; FINE stage = psi/pi hidden width, ~26 params per
            # unit here, i.e. ~0.3% resolution. Applied identically to the
            # three structured arms -- one procedure, one branch, and the
            # budget still covers psi, pi, head AND context.
            while w > lo and count_params(build_at(w, {})) > target_params:
                w -= step
            p = _search_knob(
                lambda p: build_at(w, {"psi_hidden": (p,), "pi_hidden": (p,)}),
                target_params,
                lo=1,
                step=1,
            )
            trim = {"psi_hidden": (p,), "pi_hidden": (p,)}
        kwargs = _kwargs_for(family, w, d_z, d_u, {**fixed, **trim})
        achieved = count_params(build_at(w, trim))

    rel_err = abs(achieved - target_params) / target_params
    if rel_err > 0.05:
        raise ValueError(
            f"match_budget: could not hit target_params={target_params} for family={family!r} "
            f"within +/-5% -- closest achieved is {achieved} ({rel_err:.1%} off) at "
            f"kwargs={kwargs}"
        )
    return kwargs
