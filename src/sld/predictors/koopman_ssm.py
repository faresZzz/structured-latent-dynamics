"""Koopman / structured-SSM predictor: a lifted linear-in-H recurrence.

The state is lifted by ``psi`` into an H-space of dimension ``d_h = 2*n_pairs``
carried by ``RotationScalingBlocks`` (one conjugate eigenvalue pair per block),
and read out by ``pi``. Three transition regimes span the Koopman axis of the
study (REGLES section 3 vocabulary is eliminatory and enforced here):

- ``"constant"`` — a fixed discrete operator ``A_bar`` with inputs entering
  additively through ``B_bar_0``. The autonomous part is the fixed-linear-operator
  reading (Koopman regime (i), EDMD-style: Williams, Kevrekidis & Rowley 2015,
  https://arxiv.org/abs/1408.4408). With inputs added this is the AFFINE actuated
  form ``A_bar h + B_bar_0 u`` — affine, never "bilinear".

- ``"bilinear"`` — regime (ii), the core cell: the affine form PLUS the explicit
  interaction ``sum_j u_j (B_bar_j h)``. This is the discrete bilinear Koopman
  realization (Bruder, Fu & Vasudevan 2021, https://arxiv.org/abs/2010.09961),
  motivated by control-affine lifting whose generator obeys
  ``dh/dt = A h + sum_j u_j N_j h + B u`` (Goswami & Paley 2017, IEEE CDC,
  https://doi.org/10.1109/CDC.2017.8264584).

- ``"selective"`` — the LPV ablation. A per-pair, state/input-dependent step
  ``delta_t = f(h, u)`` modulates the block field. This regime steps OUTSIDE the
  Koopman frame (the transition is no longer a fixed nor bilinear operator on
  observables): it is a linear-parameter-varying (LPV) A-field ablation, named as
  such and kept for contrast, not as a Koopman model.

Contract (predictors/base.py): ``step`` and ``rollout`` iterate PURELY in H-space;
``pi`` is read-only — it is never applied inside ``step`` or ``rollout``. The
recurrence is markovian in ``h`` for every regime, so the augmented-state slot
``m`` stays ``None`` throughout (no context threaded across steps).

``local_operator`` is the audit READING of the local linear(ized) transition
(Fig. 2 instrument), never the object of a stability claim (REGLES sections 3, 5):
stability is audited offline on the total Jacobian of the augmented state, not on
any operator displayed here.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sld.predictors.base import PredictorModule, PredictorState
from sld.predictors.blocks2x2 import RotationScalingBlocks, block_diagonal

_VALID_REGIMES = ("constant", "bilinear", "selective")


def _make_mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...], dtype: torch.dtype) -> nn.Module:
    """MLP ``in_dim -> out_dim`` with GELU between hidden layers.

    ``hidden=()`` collapses to a single ``Linear`` (NO nonlinearity) — this
    matters for identifiability: the gold test needs ``psi``/``pi`` linear so the
    true ZOH-discretized linear system lies exactly in the model class.
    """
    sizes = [in_dim, *hidden, out_dim]
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], dtype=dtype))
        if i < len(sizes) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class KoopmanSSMPredictor(PredictorModule):
    """Lifted rotation-scaling recurrence with three transition regimes.

    Args:
        d_z: dimension of the model's own latent (the ``z`` fed to ``lift``).
        n_pairs: number of 2x2 rotation-scaling blocks; ``d_h = 2 * n_pairs``.
        d_u: action dimension.
        delta: fixed ZOH step for the ``constant``/``bilinear`` regimes and the
            base scale for the ``selective`` clamp. Must be finite and > 0.
        regime: one of ``{"constant", "bilinear", "selective"}``.
        constrained: ``True`` is the ON cell (alpha = -exp(nu), contractive
            stability axis); ``False`` is the free OFF cell.
        psi_hidden, pi_hidden: hidden sizes of the lifting / projection MLPs;
            ``()`` means a single linear map.
        dtype: parameter dtype at construction.
        delta_min, delta_max: clamp bounds for the ``selective`` per-pair step;
            ``delta_max=None`` defaults to ``10 * delta``.
    """

    def __init__(
        self,
        d_z: int,
        n_pairs: int,
        d_u: int,
        delta: float,
        regime: str = "bilinear",
        constrained: bool = True,
        psi_hidden: tuple[int, ...] = (64,),
        pi_hidden: tuple[int, ...] = (64,),
        dtype: torch.dtype = torch.float32,
        delta_min: float = 1e-4,
        delta_max: float | None = None,
    ) -> None:
        super().__init__()
        if regime not in _VALID_REGIMES:
            raise ValueError(f"unknown regime: {regime!r}, expected one of {_VALID_REGIMES}")
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError("delta must be finite and strictly positive")

        self.d_z = d_z
        self.n_pairs = n_pairs
        self.d_h = 2 * n_pairs
        self.d_u = d_u
        self.delta = float(delta)
        self.regime = regime
        self.delta_min = float(delta_min)
        self.delta_max = float(10.0 * delta if delta_max is None else delta_max)
        # the clamp interval must be a real interval containing the base step,
        # else the selective gate is dead (saturated clamp = zero gradient)
        # or silently broken (NaN/inverted bounds) from the first call
        if not (math.isfinite(self.delta_min) and math.isfinite(self.delta_max)):
            raise ValueError("delta_min and delta_max must be finite")
        if not (0.0 < self.delta_min < self.delta_max):
            raise ValueError("need 0 < delta_min < delta_max")
        if regime == "selective" and not (self.delta_min <= self.delta <= self.delta_max):
            raise ValueError("selective regime needs delta_min <= delta <= delta_max")

        # read-out maps: pi is read-only w.r.t. the recurrence (base.py contract)
        self.psi = _make_mlp(d_z, self.d_h, psi_hidden, dtype)
        self.pi = _make_mlp(self.d_h, d_z, pi_hidden, dtype)

        # transition core: continuous-time rotation-scaling generators + a
        # continuous input matrix B_cont (d_h, d_u); ZOH-discretized at use time
        self.blocks = RotationScalingBlocks(
            n_pairs, constrained=constrained, mode="continuous", dtype=dtype
        )
        self.B_cont = nn.Parameter(0.01 * torch.randn(self.d_h, d_u, dtype=dtype))

        # regime-specific parameters are created LAST so the shared params
        # (psi, pi, blocks, B_cont) keep a regime-independent construction order
        if regime == "bilinear":
            # discrete-time bilinear coupling B_bar_j, init ZEROS: starts at the
            # affine point (identifiability -- symmetric with the constant
            # regime -- and the gold's vanishing-coupling gate on linear data)
            self.B_j = nn.Parameter(torch.zeros(d_u, self.d_h, self.d_h, dtype=dtype))
        elif regime == "selective":
            # LPV code net: c_t = W_c(concat([h, u])) -> per-pair delta_t
            self.W_c = nn.Linear(self.d_h + d_u, n_pairs, dtype=dtype)
            # bias so softplus(bias) == delta: at init the gate sits at the
            # base step, INSIDE the clamp interval -- a zero bias would give
            # softplus(0) ~ 0.693, saturated against delta_max = 10*delta for
            # any delta < 0.07, zeroing every W_c gradient (dead gate,
            # adversarial review round 1)
            with torch.no_grad():
                self.W_c.bias.fill_(math.log(math.expm1(self.delta)))

    # -- lifting / projection -------------------------------------------------

    def lift(self, z: Tensor) -> Tensor:
        """Lift a latent ``z`` (B, d_z) into H-space ``h`` (B, d_h)."""
        return self.psi(z)

    def project(self, h: Tensor) -> Tensor:
        """Read out ``h`` (B, d_h) to latent space ``z`` (B, d_z). READ-ONLY:
        never called inside ``step``/``rollout`` (base.py contract)."""
        return self.pi(h)

    # -- shared discrete affine operator (constant / bilinear) ---------------

    def _discrete_affine(self) -> tuple[Tensor, Tensor]:
        """Assembled fixed operator ``(A_bar (d_h, d_h), B_bar_0 (d_h, d_u))``."""
        A_bar_blocks, B_bar_0 = self.blocks.discretize_zoh(self.delta, self.B_cont)
        return block_diagonal(A_bar_blocks), B_bar_0

    # -- selective (LPV) per-sample operators --------------------------------

    def _selective_operators(self, delta_t: Tensor) -> tuple[Tensor, Tensor]:
        """Per-sample ``(A_bar_t, B_bar_t)`` from a per-pair step ``delta_t``.

        ``A_bar_t`` uses the exact batched-delta ZOH A-path; the input path uses
        the Euler simplification ``B_bar_t = delta_expanded * B_cont`` (each
        pair's delta repeated over its two rows). This is a deliberate
        Mamba-style simplification: exact ZOH ``B_bar`` with batched delta is not
        supported by blocks2x2, and the declared object of study in this regime
        is the LPV ``A`` field, not the input integral.
        """
        A_bar_blocks, _ = self.blocks.discretize_zoh(delta_t)  # (B, n_pairs, 2, 2)
        A_bar_t = block_diagonal(A_bar_blocks)  # (B, d_h, d_h)
        delta_rows = delta_t.repeat_interleave(2, dim=-1)  # (B, d_h)
        B_bar_t = delta_rows.unsqueeze(-1) * self.B_cont  # (B, d_h, d_u)
        return A_bar_t, B_bar_t

    def _frozen_step(self, h: Tensor, u: Tensor, c: Tensor) -> Tensor:
        """Selective step with the input-dependent code ``c`` held CONSTANT.

        Freezing ``c`` makes the transition linear in ``(h, u)`` with operator
        ``(A_bar_t, B_bar_t)`` — the coherence reference the tests differentiate
        against, and the pair returned by ``local_operator`` in this regime.
        """
        delta_t = torch.clamp(F.softplus(c), self.delta_min, self.delta_max)
        A_bar_t, B_bar_t = self._selective_operators(delta_t)
        return torch.einsum("bkl,bl->bk", A_bar_t, h) + torch.einsum(
            "bkj,bj->bk", B_bar_t, u
        )

    # -- step (H-space recurrence; pi never applied here) --------------------

    def step(
        self, h: Tensor, u: Tensor, state: PredictorState
    ) -> tuple[Tensor, PredictorState]:
        if self.regime == "selective":
            # LPV: the code c_t depends on (h, u); step differentiates through it
            c = self.W_c(torch.cat([h, u], dim=-1))
            return self._frozen_step(h, u, c), state

        A_bar, B_bar_0 = self._discrete_affine()
        # affine actuated update A_bar h + B_bar_0 u
        h_next = h @ A_bar.T + u @ B_bar_0.T
        if self.regime == "bilinear":
            # interaction term sum_j u_j (B_bar_j h), written as one named einsum
            interaction = torch.einsum("bj,jkl,bl->bk", u, self.B_j, h)
            h_next = h_next + interaction
        # markovian in h: no context threaded across steps, m stays None
        return h_next, state

    # -- local_operator (audit READING, not a stability object) --------------

    def local_operator(
        self, h: Tensor, u: Tensor, state: PredictorState
    ) -> tuple[Tensor, Tensor]:
        batch = h.shape[0]
        if self.regime == "constant":
            # exact fixed operator, broadcast to the batch
            A_bar, B_bar_0 = self._discrete_affine()
            A = A_bar.unsqueeze(0).expand(batch, -1, -1)
            B = B_bar_0.unsqueeze(0).expand(batch, -1, -1)
            return A, B

        if self.regime == "bilinear":
            # EXACT first-order Jacobians of the bilinear step:
            #   dstep/dh = A_bar + sum_j u_j B_bar_j
            #   dstep/du[:, j] = B_bar_0[:, j] + B_bar_j h
            A_bar, B_bar_0 = self._discrete_affine()
            A = A_bar.unsqueeze(0) + torch.einsum("bj,jkl->bkl", u, self.B_j)
            B = B_bar_0.unsqueeze(0) + torch.einsum("jkl,bl->bkj", self.B_j, h)
            return A, B

        # selective: frozen-c pair (A_bar_t, B_bar_t) at c(h, u) held fixed
        c = self.W_c(torch.cat([h, u], dim=-1))
        delta_t = torch.clamp(F.softplus(c), self.delta_min, self.delta_max)
        return self._selective_operators(delta_t)
