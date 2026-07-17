"""Mamba-style minimal selective SSM predictor: a recurrent hidden state
``m`` whose discrete transition operator is chosen by the current state
itself (Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective
State Spaces", 2023, https://arxiv.org/abs/2312.00752).

Vocabulary discipline (REGLES section 3, eliminatory): the transition
operator ``A_bar(c_t)`` is state-dependent, so this family is a linear
parameter-varying (LPV) system read as a cocycle along the trajectory --
never a Koopman operator (see the one disclaimer sentence in the class
docstring, and nowhere else in this file).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sld.predictors.base import PredictorModule, PredictorState


class SelectiveSSMPredictor(PredictorModule):
    """Mamba-style minimal selective SSM in a recurrent state ``m`` (B, d_state).

    Disclaimer (REGLES section 3): the transition operator ``A_bar_t`` below
    is chosen by a function of the current state, so this is an LPV/cocycle
    system, not a Koopman operator -- a Koopman operator is fixed on the
    observables, and this one is not.

    Per step, with ``x_t = concat([h_t, u_t])`` (d_in = d_h + d_u):
      - selectivity context ``c_t = W_c(concat([h_t, m_t]))``, so the gate
        depends on both the readout state and the recurrent state (the
        plan's ``c_t = omega(h_t, m_t)``);
      - step size ``delta_t = clamp(softplus(W_delta(c_t)), delta_min,
        delta_max)``, strictly positive by construction (softplus + clamp,
        REGLES section 5);
      - continuous diagonal generator ``a_cont = -exp(a_log)`` (learned,
        negative real, no oscillatory modes in this minimal family) gives
        the exact zero-order hold (ZOH) of a diagonal system,
        ``A_bar_t = exp(delta_t * a_cont)``, with every entry in (0, 1);
      - input-dependent gate ``b_t = W_b(c_t)`` and input projection
        ``v_t = W_in(x_t)``, combined through Mamba's declared Euler-style
        simplification ``B_bar_t = delta_t * b_t`` (an approximation to the
        exact ZOH input integral, not the exact integral itself);
      - recurrence ``m_next = A_bar_t * m_t + B_bar_t * v_t`` (elementwise,
        batch-vectorized -- the sequential scan over time lives only in
        ``rollout``'s step loop, per ``PredictorModule``);
      - residual readout ``h_next = h + W_out(m_next)``.

    ``local_operator`` exposes the frozen-context first-order pair of the
    RECURRENT update (this family's recurrent space is ``m``, so
    ``d_r = d_state``): with the selectivity context ``c`` frozen, it is a
    READING for the audit instrument (the local spectra field of the LPV
    system) -- never the object of stability claims (base.py, REGLES
    section 5).
    """

    def __init__(
        self,
        d_h: int,
        d_u: int,
        d_state: int = 16,
        delta_min: float = 1e-4,
        delta_max: float = 10.0,
    ) -> None:
        super().__init__()
        # step-size bounds must define a real interval: NaN/inverted/negative
        # bounds would silently break the softplus+clamp guard downstream
        if not (math.isfinite(delta_min) and math.isfinite(delta_max)):
            raise ValueError("delta_min and delta_max must be finite")
        if not (0.0 < delta_min < delta_max):
            raise ValueError("need 0 < delta_min < delta_max")
        self.d_h = d_h
        self.d_u = d_u
        self.d_state = d_state
        self.delta_min = delta_min
        self.delta_max = delta_max

        self.W_c = nn.Linear(d_h + d_state, d_state)
        self.W_delta = nn.Linear(d_state, d_state)
        self.W_b = nn.Linear(d_state, d_state)
        self.W_in = nn.Linear(d_h + d_u, d_state)
        self.W_out = nn.Linear(d_state, d_h)
        # zero-init the residual readout (standard residual-branch practice):
        # at construction h_next == h exactly, which removes the expansive
        # h-residual loop the default Linear init produced (adversarial
        # review round 1). This is NOT a full-state boundedness guarantee:
        # the internal recurrence on m stays unconstrained (b_t grows with m
        # through the selectivity path), so free-running stability remains an
        # AUDIT object (Wave 4), never a construction claim; the test locks
        # (h, m) finiteness over 200 steps at default init, no more.
        nn.init.zeros_(self.W_out.weight)
        nn.init.zeros_(self.W_out.bias)

        # continuous diagonal generator a_cont = -exp(a_log) < 0, initialized
        # so a_cont in [-1, -0.1]: negative real diagonal (no oscillatory
        # modes), keeps A_bar_t = exp(delta_t * a_cont) well inside (0, 1)
        # at construction time
        a_log_init = torch.empty(d_state).uniform_(math.log(0.1), math.log(1.0))
        self.a_log = nn.Parameter(a_log_init)

    def _gate_params(self, c: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """delta_t, A_bar_t, b_t from a selectivity context ``c`` (B, d_state).

        Shared by ``_frozen_recurrence`` and ``local_operator`` so both read
        the same formula off ``c`` -- treated here as a given tensor, never
        recomputed from ``h``/``m``.
        """
        # step size: softplus + clamp guard (REGLES section 5), strictly positive
        delta_t = torch.clamp(F.softplus(self.W_delta(c)), self.delta_min, self.delta_max)
        a_cont = -torch.exp(self.a_log)
        # exact ZOH of a diagonal system: A_bar_t = exp(delta_t * a_cont) in (0, 1)
        A_bar_t = torch.exp(delta_t * a_cont)
        b_t = self.W_b(c)
        return delta_t, A_bar_t, b_t

    def _frozen_recurrence(self, m: Tensor, h: Tensor, u: Tensor, c: Tensor) -> Tensor:
        """m_next with the selectivity context ``c`` treated as a CONSTANT.

        ``delta_t``/``b_t`` are derived from the given ``c`` only (no
        dependence on ``m``/``h`` through ``c`` here); ``h`` still feeds the
        direct input path ``v_t = W_in(concat([h, u]))``, which is unrelated
        to the selectivity gate. Stable helper used by ``step`` and by the
        audit-coherence tests (frozen-context first-order pair).
        """
        delta_t, A_bar_t, b_t = self._gate_params(c)
        # input-dependent gate, Mamba's declared Euler-style simplification
        x_t = torch.cat([h, u], dim=-1)
        v_t = self.W_in(x_t)
        B_bar_t = delta_t * b_t
        return A_bar_t * m + B_bar_t * v_t

    def step(
        self, h: Tensor, u: Tensor, state: PredictorState
    ) -> tuple[Tensor, PredictorState]:
        m = state.m if state.m is not None else h.new_zeros(h.shape[0], self.d_state)
        # selectivity context depends on both the readout state and the
        # recurrent state: c_t = omega(h_t, m_t)
        c = self.W_c(torch.cat([h, m], dim=-1))
        m_next = self._frozen_recurrence(m, h, u, c)
        # residual readout
        h_next = h + self.W_out(m_next)
        return h_next, PredictorState(m=m_next)

    def local_operator(
        self, h: Tensor, u: Tensor, state: PredictorState
    ) -> tuple[Tensor, Tensor]:
        """Frozen-context first-order pair (A, B) of the RECURRENT update.

        A READING for the audit (local spectra field of the LPV system,
        REGLES section 3), never the object of stability claims (base.py).
        With ``c = c(h, m)`` frozen: ``A`` is the diag-embedded ``A_bar_t``,
        shape (B, d_state, d_state); ``B`` is
        ``diag(B_bar_t) @ W_in[:, d_h:]`` (the input-block columns of
        ``W_in``, since ``v_t = W_in(concat([h, u]))``), shape
        (B, d_state, d_u).
        """
        m = state.m if state.m is not None else h.new_zeros(h.shape[0], self.d_state)
        c = self.W_c(torch.cat([h, m], dim=-1))
        delta_t, A_bar_t, b_t = self._gate_params(c)
        B_bar_t = delta_t * b_t
        A = torch.diag_embed(A_bar_t)
        # dv_t/du at frozen c: the u-block columns of W_in, constant in (h, u)
        W_in_u = self.W_in.weight[:, self.d_h :]
        B = B_bar_t.unsqueeze(-1) * W_in_u.unsqueeze(0)
        return A, B
