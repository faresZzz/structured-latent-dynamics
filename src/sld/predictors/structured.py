"""The shared structured transition: one cell, three interchangeable backbones.

PROTOCOL section 13.1 (decision C0) fixes ONE transition for every structured
arm of the 3x2 factorial; only the context producer ``kappa_b`` differs between
the MLP, transformer and SSM backbones. This module holds that transition, so
the three arms are literally the same treatment:

    h_0                          = psi(z_0)
    c_t                          = kappa_b(h_t, m_t)          # past only
    m_{t+1}                      = absorb_b(h_t, u_t, m_t)
    (alpha_t, omega_t, vec B_t)  = theta_0 + W_theta c_t
    G_{t,i}                      = [[alpha_{t,i}, -omega_{t,i}],
                                    [omega_{t,i},  alpha_{t,i}]]
    A_bar_t                      = blkdiag_i exp(Delta G_{t,i})
    B_bar_t                      = blkdiag_i (int_0^Delta exp(s G_{t,i}) ds) B_t
    h_{t+1}                      = A_bar_t h_t + B_bar_t u_t
    z_hat_{t+1}                  = pi(h_{t+1})

Four decisions are part of that contract and are enforced structurally here,
not by convention:

- ``u_t`` is EXCLUDED from ``c_t``. That is why the backbone protocol is split
  into TWO calls: ``context_step`` cannot see ``u_t`` because it is not passed
  one, and ``absorb`` is the only place the pair ``(h_t, u_t)`` enters the
  context state. The invariant is structural, not a discipline a test has to
  police. It is what keeps the cell linear in the action, hence what makes
  "identical treatment, three backbones" a true sentence.
- ``alpha`` is FREE — no ``-exp(nu)``, no ``-softplus``. The structured axis
  must not smuggle a contraction treatment in with it (PROTOCOL section 13.2);
  stable AND unstable modes stay representable and stability is measured, not
  imposed.
- The head is a single affine map ``theta_0 + W_theta c`` (one ``nn.Linear``,
  its bias IS ``theta_0``). No hidden second MLP — that would be a fourth
  backbone in disguise, and it would destroy the exact meaning of a neutral
  context (``W_theta = 0`` gives ``theta_0`` for every ``c``), which the LTI
  gold test depends on.
- ``B`` is CONTINUOUS and goes through the exact ZOH input integral. A directly
  parametrized discrete ``B`` would put the physical gold system outside the
  model class.

Vocabulary (REGLES section 3, eliminatory): ``A_bar_t`` depends on the context,
so this is an LPV / cocycle transition. It is NOT a Koopman operator, which
names either a FIXED linear operator on observables or the bilinear form with
an explicit ``sum_j u_j B_bar_j h`` interaction — neither of which is what is
built here.

Contract (``base.py``): ``step``/``rollout`` iterate purely in H-space; ``pi``
is read-only and never feeds the recurrence. ``rollout`` is inherited unchanged,
so state never crosses an episode boundary.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from sld.predictors.base import PredictorModule, PredictorState
from sld.predictors.blocks2x2 import block_diagonal, rotation_scaling_zoh
from sld.predictors.koopman_ssm import _make_mlp
from sld.predictors.min_ssm import SelectiveSSMContextBackbone
from sld.predictors.mlp import MLPContextBackbone
from sld.predictors.transformer import CausalTransformerContextBackbone


@runtime_checkable
class ContextBackbone(Protocol):
    """``kappa_b``: the only thing that differs between the structured arms.

    Deliberately TWO calls rather than one. A single
    ``kappa(h, u, m) -> (c, m_next)`` would make it a code-review question
    whether ``u_t`` leaked into ``c_t``; splitting it makes the answer a
    signature. ``context_step`` is not handed ``u_t``, so ``alpha_t``,
    ``omega_t`` and ``B_t`` cannot depend on it.
    """

    d_context: int

    def context_step(self, h: Tensor, state: PredictorState) -> Tensor:
        """``c_t`` from ``h_t`` and the past encoded in ``state.m``.

        ``u_t`` DOES NOT ENTER HERE. Shapes: ``h`` is ``(B, d_h)``, the
        returned context is ``(B, d_context)``.
        """
        ...

    def absorb(self, h: Tensor, u: Tensor, state: PredictorState) -> PredictorState:
        """``m_{t+1}``: fold the current pair ``(h_t, u_t)`` into the state.

        This is the only place ``u_t`` reaches the context, and it reaches it
        only for steps ``s > t``.
        """
        ...


class StructuredPredictor(PredictorModule):
    """Lift, context-scheduled 2x2 rotation-scaling blocks, exact ZOH, project.

    Args:
        backbone: the ``ContextBackbone`` producing ``c_t``. Registered as a
            submodule when it is an ``nn.Module`` (the three real arms are).
        d_z: dimension of the encoder latent fed to ``lift``.
        r: number of 2x2 blocks; the lifted state has ``d_h = 2*r``.
        d_u: action dimension.
        delta: the fixed ZOH step ``Delta``. Finite and strictly positive.
        psi_hidden: hidden widths of ``psi``; ``()`` makes it a single Linear.
        pi_hidden: hidden widths of ``pi``; ``()`` makes it a single Linear.
        dtype: parameter dtype at construction.
    """

    def __init__(
        self,
        backbone: ContextBackbone,
        d_z: int,
        r: int,
        d_u: int,
        delta: float,
        psi_hidden: tuple[int, ...] = (64,),
        pi_hidden: tuple[int, ...] = (64,),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if r < 1:
            raise ValueError(f"r must be >= 1, got {r}")
        delta_f = float(delta)
        # NaN fails every comparison, so `<= 0` alone would admit NaN and +inf
        if not math.isfinite(delta_f) or delta_f <= 0.0:
            raise ValueError(f"delta must be finite and strictly positive, got {delta!r}")
        self.backbone = backbone
        self.d_z = d_z
        self.r = r
        self.d_u = d_u
        self.d_h = 2 * r
        self.delta = delta_f
        self.psi = _make_mlp(d_z, self.d_h, psi_hidden, dtype)
        self.pi = _make_mlp(self.d_h, d_z, pi_hidden, dtype)
        # ONE affine head. bias IS theta_0; weight IS W_theta, so zeroing the
        # weight makes the context exactly neutral. Output layout, in order:
        #   [0 : r)                 alpha
        #   [r : 2r)                omega
        #   [2r : 2r + 2r*d_u)      vec B, row-major into (2r, d_u)
        self.head = nn.Linear(backbone.d_context, 2 * r + self.d_h * d_u, dtype=dtype)
        # PROTOCOL 13.6: W_theta starts at EXACTLY zero so the three arms
        # start from the same LTI system regardless of their context norm
        # (measured 5x apart at init between mlp_k/ssm_k and transformer_k,
        # which otherwise confounds the factorial). theta_0 (the bias) is
        # drawn from the ranges this repo already uses for the same three
        # quantities elsewhere: alpha <- nu of RotationScalingBlocks
        # (blocks2x2.py), omega <- omega of the same, vec B <- B_cont of
        # KoopmanSSMPredictor (koopman_ssm.py). This is an INIT choice only:
        # alpha stays FREE (no -exp(nu), no softplus, see module docstring)
        # and nothing later bounds it during training. It is not the
        # contraction constraint returning under another name.
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias[: r].uniform_(-1.5, -0.5)  # alpha_0
            self.head.bias[r : 2 * r].uniform_(0.1, 2.0)  # omega_0
            self.head.bias[2 * r :].normal_(0.0, 0.01)  # vec B_0

    # -- read-only maps -------------------------------------------------------

    def lift(self, z: Tensor) -> Tensor:
        """``psi``: encoder latent ``(B, d_z)`` -> lifted state ``(B, 2r)``."""
        return self.psi(z)

    def project(self, h: Tensor) -> Tensor:
        """``pi``: lifted state ``(B, 2r)`` -> latent ``(B, d_z)``.

        READ-ONLY: used for the loss and for reporting, never inside
        ``step``/``rollout`` to produce the next ``h`` (base.py contract).
        """
        return self.pi(h)

    # -- the structured transition -------------------------------------------

    def _operator_from_context(self, c: Tensor) -> tuple[Tensor, Tensor]:
        """``(A_bar, B_bar)`` from a context ``c``: ``(B, 2r, 2r)``, ``(B, 2r, d_u)``.

        The single place the head output is sliced, so ``step`` and
        ``local_operator`` cannot drift apart.
        """
        theta = self.head(c)
        alpha = theta[..., : self.r]
        omega = theta[..., self.r : 2 * self.r]
        B_cont = theta[..., 2 * self.r :].reshape(*theta.shape[:-1], self.d_h, self.d_u)
        A_bar_blocks, B_bar = rotation_scaling_zoh(alpha, omega, self.delta, B_cont)
        assert B_bar is not None  # B_cont was given
        return block_diagonal(A_bar_blocks), B_bar

    def step(
        self, h: Tensor, u: Tensor, state: PredictorState
    ) -> tuple[Tensor, PredictorState]:
        c = self.backbone.context_step(h, state)
        A_bar, B_bar = self._operator_from_context(c)
        h_next = (A_bar @ h.unsqueeze(-1) + B_bar @ u.unsqueeze(-1)).squeeze(-1)
        # u_t is folded into the context only AFTER it has acted, so it can
        # influence c_s for s > t and never c_t itself
        return h_next, self.backbone.absorb(h, u, state)

    def local_operator(
        self, h: Tensor, u: Tensor, state: PredictorState
    ) -> tuple[Tensor, Tensor]:
        """The EXACT ``(A_bar_t, B_bar_t)`` pair at the frozen context.

        Not a linearization: the transition is genuinely affine in ``(h, u)``
        once ``c_t`` is fixed, so this pair reproduces ``step`` exactly.
        ``u`` is accepted for contract compatibility and unused by construction
        — the context does not depend on it.
        """
        return self._operator_from_context(self.backbone.context_step(h, state))


# ---------------------------------------------------------------------------
# The three arms. Thin facades: they build a backbone and delegate EVERYTHING
# else to StructuredPredictor, so "identical treatment, three backbones" is
# true by construction rather than by review. None of them overrides `step`.
#
# One canonical signature, deliberately: (d_z, d_u, delta, r, width,
# psi_hidden, pi_hidden, dtype) is shared verbatim by all three, with `width`
# as the ONE budget knob, and every backbone-specific extra keyword-only with
# a default. One branch in `budget._kwargs_for` / `trainer._FAMILY_CONFIG_KEYS`
# then covers the three families instead of three.
#
# NOTE the r default below (r=2) is NOT the default `get_predictor` gives you:
# `budget._kwargs_for` passes `r=fixed.get("r", d_z)` for the structured
# families, so every real run built through `get_predictor` gets r=d_z (the
# same 2x lifting ratio koopman_ssm gets from n_pairs=d_z), never r=2. r=2
# only shows up when one of these three classes is constructed BY HAND,
# bypassing get_predictor/budget entirely -- as the gold tests here do.
# Aligning the two defaults is deliberately NOT done: it would change the
# lifted width of every hand-built arm and any test pinned to it.
# ---------------------------------------------------------------------------
class MLPKPredictor(StructuredPredictor):
    """Structured cell, MEMORYLESS context: ``c_t = MLP(h_t)``.

    ``width`` is the hidden AND output width of that MLP, so
    ``d_context == width``.
    """

    def __init__(
        self,
        d_z: int,
        d_u: int,
        delta: float,
        r: int = 2,
        width: int = 64,
        psi_hidden: tuple[int, ...] = (64,),
        pi_hidden: tuple[int, ...] = (64,),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            MLPContextBackbone(2 * r, width, dtype=dtype),
            d_z, r, d_u, delta, psi_hidden, pi_hidden, dtype,
        )


class TransformerKPredictor(StructuredPredictor):
    """Structured cell, attention over the past pairs as context.

    ``width`` is ``d_model``, so ``d_context == width``; it must be divisible
    by ``n_heads``.
    """

    def __init__(
        self,
        d_z: int,
        d_u: int,
        delta: float,
        r: int = 2,
        width: int = 64,
        psi_hidden: tuple[int, ...] = (64,),
        pi_hidden: tuple[int, ...] = (64,),
        dtype: torch.dtype = torch.float32,
        *,
        n_heads: int = 2,
        n_layers: int = 2,
        max_len: int = 512,
    ) -> None:
        super().__init__(
            CausalTransformerContextBackbone(
                2 * r, d_u, width,
                n_heads=n_heads, n_layers=n_layers, max_len=max_len, dtype=dtype,
            ),
            d_z, r, d_u, delta, psi_hidden, pi_hidden, dtype,
        )


class SSMKPredictor(StructuredPredictor):
    """Structured cell, selective-SSM recurrent state as context.

    ``width`` is the recurrent state dimension ``d_state``, so
    ``d_context == width``.
    """

    def __init__(
        self,
        d_z: int,
        d_u: int,
        delta: float,
        r: int = 2,
        width: int = 64,
        psi_hidden: tuple[int, ...] = (64,),
        pi_hidden: tuple[int, ...] = (64,),
        dtype: torch.dtype = torch.float32,
        *,
        delta_min: float = 1e-4,
        delta_max: float = 10.0,
    ) -> None:
        super().__init__(
            SelectiveSSMContextBackbone(
                2 * r, d_u, width, delta_min=delta_min, delta_max=delta_max, dtype=dtype
            ),
            d_z, r, d_u, delta, psi_hidden, pi_hidden, dtype,
        )
