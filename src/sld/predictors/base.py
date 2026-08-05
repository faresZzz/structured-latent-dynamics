"""Shared contract for the seven predictor families.

Semantics locked by OBJECTIFS section 4.4 -- subclasses must not weaken them:

- ``step`` and ``rollout`` iterate **in the predictor's latent space H**.
  For lifted families (``koopman_ssm`` and the three structured arms
  ``mlp_k``/``transformer_k``/``ssm_k``) the read-out projection ``pi`` is
  read-only: it never feeds back into the recurrence.
- ``local_operator`` is a READING of the local linear(ized) transition,
  exposed for the audit instrument (Fig. 2 of the plan) -- it feeds the
  DISPLAYED local spectra field and nothing else. It is NEVER the object of
  stability claims (REGLES sections 3 and 5), and the audit's quantitative
  objects (total Jacobians, Jacobian products, LTV Gramians) are built by
  autograd on the FULL augmented state ``(h, m)`` offline (Wave 4), never
  from this pair: a frozen-context, per-family-space reading misses the
  ``h``-loop and the control-through-scheduling paths by construction.
- ``PredictorState`` makes the augmented state explicit. ``m`` carries
  whatever context a family threads across steps (selective-SSM hidden
  state, transformer KV cache); ``m=None`` is the universal fresh state.
  State must never be carried across episode boundaries: ``rollout``
  always starts fresh, and callers driving ``step`` directly are
  responsible for resetting the state between episodes.

Shape conventions (time-major, matching ``objectives.jac_reg``):
``h``: ``(B, d_h)``; ``u``: ``(B, d_u)``; ``u_seq``: ``(T, B, d_u)``;
``rollout`` returns the predictions ``h_1 .. h_T`` as ``(T, B, d_h)``
(the initial ``h0`` is not repeated in the output).
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

import torch
from torch import Tensor, nn


class PredictorState(NamedTuple):
    """Explicit augmented-state container threaded through ``step``.

    ``m=None`` is the universal fresh state: every family interprets it as
    "no context yet" (zero SSM state, empty KV cache). Families that need
    no context (``mlp``) keep ``m=None`` forever.
    """

    m: Tensor | None = None


@runtime_checkable
class Predictor(Protocol):
    """The contract shared by the seven families (OBJECTIFS section 4.4).

    ``local_operator`` returns the frozen-context first-order pair
    ``(A, B)`` of the family's *recurrent-state* update -- ``A`` is
    ``(B, d_r, d_r)`` and ``B`` is ``(B, d_r, d_u)`` where ``d_r`` is the
    dimension of the space the recurrence actually lives in (``d_h`` for
    ``koopman_ssm`` and the structured arms, the hidden-state dimension for
    ``min_ssm``). Families
    with no structured operator (``mlp``, ``transformer``) return ``None``
    and the audit falls back to autograd.
    """

    def step(
        self, h: Tensor, u: Tensor, state: PredictorState
    ) -> tuple[Tensor, PredictorState]: ...

    def rollout(self, h0: Tensor, u_seq: Tensor) -> Tensor: ...

    def local_operator(
        self, h: Tensor, u: Tensor, state: PredictorState
    ) -> tuple[Tensor, Tensor] | None: ...


class PredictorModule(nn.Module):
    """Concrete base: shared free-running ``rollout`` over ``step``.

    Subclasses implement ``step`` (and optionally override
    ``local_operator``). ``rollout`` starts from the universal fresh state,
    so calling it once per episode is what guarantees the "no state across
    episodes" rule (e.g. the transformer KV cache is rebuilt per call).
    """

    def step(
        self, h: Tensor, u: Tensor, state: PredictorState
    ) -> tuple[Tensor, PredictorState]:
        raise NotImplementedError

    def rollout(self, h0: Tensor, u_seq: Tensor) -> Tensor:
        if u_seq.dim() != 3:
            raise ValueError(f"u_seq must be (T, B, d_u), got {tuple(u_seq.shape)}")
        if h0.dim() != 2 or h0.shape[0] != u_seq.shape[1]:
            raise ValueError(
                f"h0 must be (B, d_h) with B == u_seq.shape[1], "
                f"got {tuple(h0.shape)} vs {tuple(u_seq.shape)}"
            )
        # Sequential over time by contract; vectorized over the batch inside step.
        h, state = h0, PredictorState()
        outs = []
        for t in range(u_seq.shape[0]):
            h, state = self.step(h, u_seq[t], state)
            outs.append(h)
        return torch.stack(outs, dim=0)

    def local_operator(
        self, h: Tensor, u: Tensor, state: PredictorState
    ) -> tuple[Tensor, Tensor] | None:
        # Generic families expose no structured operator; audit uses autograd.
        return None
