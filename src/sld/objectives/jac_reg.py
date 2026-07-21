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

``__call__`` covers the tensor state q (B, d) case (``m`` always None, e.g.
``mlp``/``koopman_ssm``). ``on_state_steps`` is the companion for families
that thread a recurrent ``m`` (``min_ssm``/``transformer``): a distinct
estimator on the pytree state (h, m), documented on the method itself --
including the Wave 5 declaration that, for growing-memory families, the
penalized object EXCLUDES the identity embedding of ``m`` into ``m_next``
(the same excluded object the calibration measures through
``audit.jacobian.augmented_jacobian_excluding_identity``).
"""

from __future__ import annotations

from typing import Callable, Sequence

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

    def _sphere_pytree(self, leaves: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        # one probe spanning the WHOLE flattened pytree (every leaf, every
        # batch row together): draw independent Gaussians per leaf, then
        # normalize by the norm of the CONCATENATION so the combined vector
        # is uniform on the sphere of the full input space (assumes all
        # leaves share dtype/device -- true under this repo's float32 policy)
        parts = [
            torch.randn(
                leaf.shape, generator=self.generator, dtype=leaf.dtype, device=leaf.device
            )
            for leaf in leaves
        ]
        norm = torch.cat([p.reshape(-1) for p in parts]).norm()
        return tuple(p / norm for p in parts)

    @staticmethod
    def _exclude_identity_passthrough(
        jv_sq: torch.Tensor,
        v: tuple[torch.Tensor, ...],
        m_t: torch.Tensor | None,
        out: tuple[torch.Tensor, ...],
        t: int,
    ) -> torch.Tensor:
        """Per-probe ``||J v||^2`` with the identity passthrough removed.

        Growth gate (layout-agnostic, decided on NUMEL only -- never on
        ``m``'s axes, which are family-private by ``predictors.base``
        doctrine):

        - ``m_t is None`` (fresh state): nothing to exclude, ``jv_sq``
          returned unchanged.
        - ``m_next.numel() == m_t.numel()``: fixed-size recurrent state
          (``min_ssm``). ``m_next`` is a TRANSFORMATION of ``m``, not a copy
          of it, so there is no identity block and NO subtraction happens.
        - ``m_next.numel() > m_t.numel()``: growing-memory family
          (``transformer`` KV cache). Under pure append the old entries of
          ``m`` are reproduced verbatim in ``m_next``, so
          ``||J_full v||^2 = ||J_excl v||^2 + ||v_m||^2`` exactly (see the
          derivation in ``on_state_steps``) and the exclusion is the
          subtraction of ``||v_m||^2``.
        - ``m_next.numel() < m_t.numel()``: a shrinking memory is outside
          the declared taxonomy; refused loudly rather than handled by a
          guess.

        The subtracted term is a CONSTANT with respect to the parameters
        (``v`` is a fixed draw, ``m_t`` is detached), so the exclusion
        changes the penalty VALUE but leaves its gradient w.r.t. theta
        bit-identical -- which is exactly why excluding a block the penalty
        could never move is legitimate rather than a weakening.

        Verification, not assumption (REGLES section 5, loi du fini): under
        pure append the returned quantity is ``||J_excl v||^2 >= 0``
        EXACTLY, so a significantly negative value proves the append
        hypothesis false for this family and raises. The tolerance is
        ``sqrt(eps)`` relative to the two cancelling terms: the rounding
        floor of each (a pairwise-reduced sum of non-negative squares) is
        ``O(eps)`` relative, while a genuine append violation shows up as an
        ``O(1)`` relative deficit -- ``sqrt(eps)`` sits ~3 decades above the
        noise and ~3 decades below any real violation in float32.

        DECLARED LIMIT (one-sided): this guard sees violations that
        UNDER-shoot the identity floor (e.g. ``m_next = cat([0.5 * m,
        new])``), not ones that OVER-shoot it (``cat([2 * m, new])`` leaves
        the difference positive but wrong). Detecting the second kind
        requires identifying which output slots are copies, which needs a
        dense Jacobian -- too expensive per training step. The exact,
        two-sided verifier is
        ``audit.jacobian.augmented_jacobian_excluding_identity``, which
        checks the identity rows explicitly and is what the calibration
        (PROTOCOL section 5) actually measures.

        DISAPPEARING MEMORY (m10, adversarial review round 2): ``m_t``
        present at input but ``step_fn`` returning ``m_next=None`` is a
        memory shrinking to ZERO elements -- the same shrinking-memory case
        rejected below, not a "nothing to exclude" fresh-state case. ``fn``
        (the caller) filters ``None`` leaves out of its returned tuple, so
        this shows up here as ``len(out) < 2``; it must raise identically to
        ``audit.jacobian.augmented_jacobian_excluding_identity`` (which sees
        ``n_m_next = 0 < n_m`` and rejects it as shrinking), never silently
        fall through as if ``m_t`` had been ``None`` all along.
        """
        if m_t is None:
            return jv_sq
        if len(out) < 2:
            raise ValueError(
                f"step {t}: m_t is present ({m_t.numel()} elements) but step_fn "
                "returned no m_next (None); a threaded memory that DISAPPEARS "
                "is a shrinking-to-zero recurrent memory, outside the declared "
                "taxonomy (fixed-size state or pure append) and the "
                "identity-passthrough exclusion is undefined for it -- same "
                "rejection as audit.jacobian.augmented_jacobian_excluding_identity "
                "(m10: training and the audit must reject this identically)"
            )
        n_m, n_m_next = m_t.numel(), out[1].numel()
        if n_m_next == n_m:
            return jv_sq  # fixed-size recurrent state: no passthrough to exclude
        if n_m_next < n_m:
            raise ValueError(
                f"step {t}: m_next has {n_m_next} elements, fewer than m's {n_m}; "
                "a shrinking recurrent memory is outside the declared taxonomy "
                "(fixed-size state or pure append) and the identity-passthrough "
                "exclusion is undefined for it"
            )
        vm_sq = v[1].pow(2).sum()
        excluded = jv_sq - vm_sq
        tol = (torch.finfo(jv_sq.dtype).eps ** 0.5) * (jv_sq + vm_sq).detach()
        if excluded < -tol:
            raise ValueError(
                f"step {t}: the identity-passthrough exclusion produced "
                f"{float(excluded):.6g} < -{float(tol):.3g} for a growing state "
                f"(m: {n_m} -> {n_m_next} elements). ||J v||^2 - ||v_m||^2 is "
                "||J_excl v||^2 >= 0 under pure append, so this family does NOT "
                "copy its old memory entries verbatim into m_next and the "
                "declared exclusion does not apply to it (REGLES section 5: this "
                "is a bug to chase upstream, never a value to clamp)"
            )
        return excluded

    def on_state_steps(
        self,
        step_fn: Callable[
            [torch.Tensor, torch.Tensor | None, torch.Tensor],
            tuple[torch.Tensor | None, ...],
        ],
        steps: Sequence[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]],
    ) -> torch.Tensor:
        """Penalty over a TF thread whose recurrent state ``m`` is a tensor.

        Companion to ``__call__`` for families that thread
        ``PredictorState.m`` across steps (``min_ssm``, ``transformer``): the
        augmented state at step ``t`` is the pytree ``(h_t, m_t)``, and
        ``m``'s batch axis is family-specific (dim 0 for ``min_ssm``, dim 2
        inside the packed KV cache for ``transformer``) — unknown to this
        module by design (``predictors.base`` keeps ``m`` opaque). The
        row-stratified probes ``__call__`` uses need a known batch axis, so
        they cannot be reused here.

        Declared estimator (design decision): each probe is drawn GLOBALLY
        over the ENTIRE flattened input pytree at step ``t`` — every leaf,
        every batch row together, one standard-normal draw per leaf
        normalized by the norm of the concatenation (``_sphere_pytree``) —
        uniform on the sphere of the full input space R^D_total, with
        ``D_total`` the total numel of the INPUT leaves ``(h_t[, m_t])``.
        Because ``step_fn`` processes batch rows independently, ``J^q_t`` is
        exactly block-diagonal over the batch regardless of ``m``'s (unknown)
        batch axis, so Hutchinson on the WHOLE space still gives an unbiased
        estimate of ``sum_i ||J_i||_F^2`` (the sum of per-sample blocks) —
        ``E[v v^T] = I_D/D`` holds on the full space irrespective of the
        block structure. Dividing by ``B`` then yields the per-sample mean,
        matching ``__call__``'s convention (which averages over the batch
        through its own denominator).

        EXCLUDED OBJECT (declared, Wave 5). For families whose memory GROWS
        by append -- the transformer's KV cache copies every old entry
        verbatim into ``m_next`` and adds one token's worth of new ones --
        ``d(m_next)/d(m)`` contains an exact IDENTITY block. That block puts
        a structural floor of ``m.numel()`` on ``||J^q||_F^2`` and forces
        ``sigma_max(J^q) >= 1``, which would make the declared margin band
        (PROTOCOL section 5) unreachable for that family by construction.
        The penalized object here therefore EXCLUDES the identity embedding
        of ``m`` into ``m_next``. This is legitimate rather than convenient:
        the block is known in advance (it is a copy, not a learned map) and
        its derivative with respect to the parameters is identically zero,
        so no penalty weight could ever move it -- excluding it removes a
        constant, not a constraint. The same excluded object is what the
        audit measures for the calibration
        (``audit.jacobian.augmented_jacobian_excluding_identity``), so the
        quantity penalized during training and the quantity calibrated
        post-hoc are the SAME object.

        How the exclusion stays layout-agnostic. Let ``v = (v_h, v_m)`` be a
        probe. Under pure append ``m_next`` reproduces each entry of ``m`` in
        exactly one slot, so the corresponding rows of ``J^q`` are canonical
        basis vectors and the JVP equals ``v_m`` (up to that re-indexing) on
        those slots. The re-indexing is a partial permutation, hence
        norm-preserving, and therefore

            ||J_full . v||^2 = ||J_excl . v||^2 + ||v_m||^2

        holds for EVERY probe. Excluding the block is thus the subtraction of
        ``||v_m||^2`` -- which never asks WHERE the copied entries live.
        ``m``'s axis layout stays family-private knowledge, as
        ``predictors.base`` requires (Liskov: the objective must not learn a
        family's cache layout to do its job).

        DECLARED LIMIT: that identity holds only under PURE APPEND, and it is
        VERIFIED numerically rather than assumed -- a significantly negative
        excluded estimate raises instead of being clamped. See
        ``_exclude_identity_passthrough`` for the growth gate, the tolerance,
        and the one-sidedness of the training-time check.

        Args:
            step_fn: raw per-step map ``(h, m, u) -> (h_next, m_next)`` with
                ``m``/``m_next`` a tensor or ``None`` independently of each
                other (e.g. ``lambda h, m, u: (lambda hn, st: (hn,
                st.m))(*predictor.step(h, u, PredictorState(m=m)))``). Called
                with ``m`` exactly as given per step (``None`` on a fresh
                state) and ``u`` FROZEN (the differentiated object is J^q,
                not a mixed Jacobian, same doctrine as ``__call__``).
            steps: ``T`` tuples ``(h_t, m_t, u_t)`` — the INPUT state
                captured BEFORE each step of one TF thread; ``m_t`` is
                ``None`` on a fresh state (typically the first step) and a
                tensor afterwards, mixed within one call.

        Returns:
            Scalar penalty, Horvitz-Thompson weighted exactly like
            ``__call__`` (kept-step per-sample estimates summed and divided
            by ``T * step_subsample``; an empty Bernoulli draw returns zero).

        Cost (declared, NOT the ``__call__`` 1.36x): no cross-step or
        cross-probe batching — ``m``'s shape can differ step to step (the
        transformer cache grows by one token per step), so this issues
        ``k_probes`` separate ``torch.func.jvp`` calls per kept step: roughly
        ``k_probes * step_subsample * T`` extra forward passes, unmeasured
        here (`reports/spike_jacreg.md` only measured the tensor path).
        """
        T = len(steps)
        if T == 0:
            raise ValueError("steps must be non-empty")
        h0 = steps[0][0]
        keep = torch.rand(T, generator=self.generator, device=h0.device) < self.step_subsample
        idx = torch.nonzero(keep, as_tuple=False).flatten().tolist()
        if not idx:
            return torch.zeros((), dtype=h0.dtype, device=h0.device)

        total = torch.zeros((), dtype=h0.dtype, device=h0.device)
        for t in idx:
            h_t, m_t, u_t = steps[t]
            h_t = h_t.detach()
            m_t = m_t.detach() if m_t is not None else None
            u_t = u_t.detach()
            B = h_t.shape[0]
            leaves = (h_t,) if m_t is None else (h_t, m_t)
            d_total = sum(leaf.numel() for leaf in leaves)

            def fn(*primals: torch.Tensor, _m_is_none: bool = m_t is None, _u=u_t):
                h_arg = primals[0]
                m_arg = None if _m_is_none else primals[1]
                out = step_fn(h_arg, m_arg, _u)
                return tuple(x for x in out if x is not None)

            probe_sq = torch.zeros((), dtype=h0.dtype, device=h0.device)
            for _ in range(self.k_probes):
                v = self._sphere_pytree(leaves)
                out, jv = torch.func.jvp(fn, leaves, v)
                jv_sq = sum(leaf.pow(2).sum() for leaf in jv)
                probe_sq = probe_sq + self._exclude_identity_passthrough(
                    jv_sq, v, m_t, out, t
                )
            step_estimate = d_total * probe_sq / self.k_probes
            total = total + step_estimate / B

        return total / (T * self.step_subsample)
