"""Total-Jacobian penalty: the common constraint factor mu_J.

Hutchinson-style estimator of ||J^q||_F^2 along training rollouts, where
J^q is the Jacobian of the predictor step with respect to the AUGMENTED
state q = (h, m) — never the conditional block alone. This penalty is the
factor shared by every predictor family in the factorial (constraint axis
ON/OFF); the spectral separation regularizer is a family-specific ablation
and must never be folded in here.

Estimator. For probes v ~ Uniform(S^{d-1}), E[v v^T] = I/d, hence
E||J v||^2 = ||J||_F^2 / d and the unbiased estimate at a state q is

    (d / k) * sum_{j=1..k} || JVP(step_fn, q, v_j) ||_2^2.

The factor d is load-bearing: without it every margin in the project would
be silently wrong by the state dimension. Verified against an analytic
||A||_F^2 in the gold test and in `reports/spike_jacreg.md`.

Two declared implementation choices (spike decision, see the report):
  1. Detached linearization point: the penalty evaluates J^q at
     q_t.detach() — a semi-gradient that shapes the Jacobian at the visited
     states while treating the visitation distribution as fixed (the
     dropped term contracts SECOND derivatives of the step with the rollout
     sensitivity dq_t/dtheta). Achieved margins are measured post-hoc by
     the audit, so this affects optimization dynamics only, never a
     reported diagnostic.
  2. Deferred single-JVP evaluation: all sampled states and probes are
     evaluated in ONE batched `torch.func.jvp` call per rollout (measured
     overhead 1.36x vs 4-6x for the naive per-step, per-probe loop).
"""

from __future__ import annotations

from typing import Callable

import torch


class TotalJacobianPenalty:
    """Unbiased ||J^q||_F^2 penalty over Bernoulli-sampled rollout steps.

    Args:
        k_probes: fresh sphere probes per sampled step (training config: 2).
        step_subsample: Bernoulli keep-probability per rollout step
            (training config: 0.25).
        generator: explicit torch.Generator for probe/subsampling draws —
            fresh probes every call (a fixed probe set would bias training
            toward shrinking J only along those directions).
    """

    def __init__(
        self,
        k_probes: int = 2,
        step_subsample: float = 0.25,
        generator: torch.Generator | None = None,
    ) -> None:
        if k_probes < 1:
            raise ValueError("k_probes must be >= 1")
        if not 0.0 < step_subsample <= 1.0:
            raise ValueError("step_subsample must be in (0, 1]")
        self.k_probes = k_probes
        self.step_subsample = step_subsample
        self.generator = generator

    def _sphere(self, n: int, d: int, dtype: torch.dtype, device) -> torch.Tensor:
        # uniform on S^{d-1}: normalized isotropic Gaussian, one row per probe
        v = torch.randn(n, d, generator=self.generator, dtype=dtype, device=device)
        return v / v.norm(dim=-1, keepdim=True)

    def __call__(
        self,
        step_fn: Callable[..., torch.Tensor],
        q_seq: torch.Tensor,
        u_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Penalty for one rollout.

        Args:
            step_fn: the predictor step as a row-batched map. Called as
                step_fn(q) if u_seq is None, else step_fn(q, u) with q and u
                row-aligned; must process rows independently so the JVP of
                row i is J_i v_i. The caller flattens (h, m) into q — the
                penalty is defined on the augmented state, by doctrine.
            q_seq: rollout states, shape (T, B, d). Detached internally
                (declared semi-gradient, see module docstring).
            u_seq: optional actions, shape (T, B, m). J^q differentiates
                with respect to q only; u is frozen per step (that is what
                makes the object J^q and not a mixed Jacobian). Detached.

        Returns:
            Scalar penalty, Horvitz-Thompson weighted: the per-step
            (d/k)-estimates of kept steps are summed and divided by
            T * step_subsample (the expected number of kept steps), so the
            estimator is unbiased for the mean over ALL T steps — including
            the zero returned when the Bernoulli draw keeps no step (a plain
            mean over kept steps would be biased low by the factor
            1 - (1-p)^T; caught in adversarial review, locked by test).
            The multiplier d is the INPUT dimension of step_fn (correct
            even for non-square maps: E[vv^T] = I_d/d lives in input space).

            Declared variance (adversarial-review quantification): the
            Bernoulli mask is shared across the batch, so its noise does
            NOT average down with B. For per-step values mu_t the mask-only
            CV is sqrt((1-p)/(p*T_eff)) with T_eff = (sum mu_t)^2 /
            sum(mu_t^2) <= T — at the training config T=20, p=0.25 this is
            ~39 % per rollout in the equal-step LOWER-BOUND case, and grows
            toward sqrt((1-p)/p) when one step dominates. Acceptable for a
            REGULARIZER averaged over optimizer steps; never acceptable for
            a reported number (the audit uses exact dense Jacobians).
        """
        T, B, d = q_seq.shape
        keep = (
            torch.rand(T, generator=self.generator, device=q_seq.device)
            < self.step_subsample
        )
        idx = torch.nonzero(keep, as_tuple=False).flatten()
        if idx.numel() == 0:
            return torch.zeros((), dtype=q_seq.dtype, device=q_seq.device)

        # deferred single-JVP evaluation: stack sampled steps, then repeat
        # k times along the row axis so every probe rides one batched call
        q_sel = q_seq[idx].detach().reshape(-1, d)
        q_rep = q_sel.repeat(self.k_probes, 1)
        v = self._sphere(q_rep.shape[0], d, q_seq.dtype, q_seq.device)

        if u_seq is None:
            fn = step_fn
        else:
            u_rep = u_seq[idx].detach().reshape(len(idx) * B, -1).repeat(self.k_probes, 1)
            fn = lambda q: step_fn(q, u_rep)  # noqa: E731  u frozen: the object is J^q

        _, jv = torch.func.jvp(fn, (q_rep,), (v,))
        # Horvitz-Thompson normalization: sum over the k*S*B rows divided by
        # k*B*T*p == sum over kept steps of the (d/k) batch-mean estimate,
        # divided by the EXPECTED kept count T*p — unbiased over the
        # Bernoulli subsampling, empty draws included
        total = jv.pow(2).sum(dim=-1).sum()
        return d * total / (self.k_probes * B * T * self.step_subsample)
