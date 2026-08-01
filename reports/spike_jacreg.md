# Spike report — `TotalJacobianPenalty` (Wave 2, S1)

De-risks the single most technically risky component of the repo: the
common-constraint-factor penalty on the total Jacobian, estimated by JVP
probes during training rollouts. Evidence from `scripts/spike_jacreg.py`
(CPU, float64 for unbiasedness, float32 MLPs for timing) plus a supervisor
variant benchmark. Decision at the bottom.

## Experiment 1 — Unbiasedness (d-scaling verified)

For probes $v ~ Uniform(S^{d-1}), E[v v^T] = I/d, so E||Jv||^2 = ||J||_F^2/d$:
the estimator MUST carry the factor d $(d/k) sum_v ||JVP(step, q, v)||^2$.
Verified numerically on a known linear map (J = A exactly, d = 16):

| quantity | value |
|---|---|
| analytic \|\|A\|\|_F² | 257.288 |
| k=8, mean over 200 draws | 256.477 (rel err **0.315 %**, accept < 5 %) |
| k=8, single-draw rel std | 10.9 % |
| k=2, mean over 500 draws | 256.862 (rel err **0.165 %**) |
| k=2, single-draw rel std | 21.8 % (training-time per-step noise) |

Without the factor d the estimate would be low by 16× here — the silent
error this spike existed to catch. The per-step noise at k=2 is averaged
over sampled steps and batch during training; the audit measures achieved
margins post-hoc, so estimator noise never enters a reported number.

## Experiment 2 — Differentiability through `torch.func.jvp`

Penalty on an MLP step (16→64→16, tanh), functional closure over the module:
- `penalty.backward()` reaches every parameter that can influence the
  input-Jacobian (the output bias correctly gets no grad — J does not
  depend on it); all grads finite, nonzero norm 2.94.
- 30 Adam steps minimizing only the penalty: 1.149 → 0.027 (**decreases**).

The regularizer can actually shape the Jacobian during training. No
`functional_call` gymnastics needed: closing over the module and calling it
inside the jvp'd function keeps params connected for reverse-mode.

## Experiment 3 — Overhead at the training config

Toy training step: MLP 32→128→32, batch 64, H=20 rollout, MSE + backward,
median over ≥ 50 reps. Spike numbers, then supervisor variant bench (same
machine, second run — ratios are the comparable quantity):

| variant | ratio vs baseline |
|---|---|
| naive: penalty on **live** rollout state, k jvp calls per sampled step | 4.19× / 6.22× (two runs) |
| detached q, per-step loop over k probes | 4.33× |
| detached q, k probes fused in one jvp per sampled step | **2.28×** |
| detached q, all sampled steps deferred into ONE jvp per rollout | **1.36×** |

Target < 2.5×. The naive implementation fails it; the fused/deferred
implementations clear it with margin. Fused-estimator unbiasedness
re-verified independently: rel err 0.124 %.

## Decision (supervisor)

**Primary design ADOPTED at k_probes=2, step_subsample=0.25** — no fallback
needed — with two declared implementation choices:

1. **Detached linearization point**: the penalty evaluates ||J^q||_F^2 at
   `q_t.detach()` — a semi-gradient, standard in Jacobian-regularization
   practice: the penalty shapes J at the visited states, treating the
   visitation distribution as fixed (dropping the term (d||J||²/dq)
   (dq_t/dθ), which contracts SECOND derivatives of the step with the
   rollout sensitivity). The achieved margins are measured post-hoc by
   the audit, so this choice affects optimization dynamics only, never a
   reported diagnostic.
2. **Deferred single-JVP evaluation**: sampled (detached) states are
   collected during the rollout and all probes evaluated in one batched
   `torch.func.jvp` call (measured 1.36×; the per-step fused variant at
   2.28× remains the fallback if a later predictor's step cannot be
   row-batched).

Fresh probes are drawn per call from an explicit `torch.Generator`
(no fixed-probe bias). The package implementation is
`src/sld/objectives/jac_reg.py`; its gold test reproduces Experiment 1's
unbiasedness bound.

## Post-review amendments (two-model adversarial pass, Codex)

0. **Second high-effort round (fresh thread).** Additional accepted
   findings: (a) the per-step noise quoted in Experiment 1 measures sphere
   noise at p=1 only — the Bernoulli mask is shared across the batch, so
   its contribution does NOT average down with B: mask-only CV =
   sqrt((1-p)/(p*T_eff)) with T_eff = (Σ_t mu_t)² / Σ_t mu_t² ≤ T, which is
   ≈ 39 % per rollout at T=20, p=0.25 in the equal-step LOWER-BOUND case
   (T_eff = T) and approaches sqrt((1-p)/p) ≈ 173 % when one step
   dominates. Declared and
   accepted for a regularizer averaged over optimizer steps; never
   acceptable for a reported number. (b) The dropped semi-gradient term
   contracts SECOND derivatives of the step with rollout sensitivities
   (earlier wording said "third-order"). (c) In `blocks2x2`, the ZOH input
   integral is an entire function of A — B_bar exists at (alpha, omega) =
   (0, 0) (= delta*B); the implementation now branch-selects a series
   evaluation there instead of wrongly raising, and nonpositive delta is
   rejected (the ON stability claim needs delta > 0).

1. **Bernoulli subsampling bias fixed.** A plain mean over kept steps is
   biased low by `1 - (1-p)^T` because an empty draw returns zero
   (0.32 % at T=20, p=0.25 — small, but the docstring claimed exact
   unbiasedness). The implementation now uses Horvitz-Thompson weighting
   (sum over kept steps divided by the expected kept count `T*p`), which is
   unbiased with the empty-draw zero included; locked by
   `test_subsample_unbiased_including_empty_draws`.
2. **Audit independence made explicit.** Reported diagnostics never reuse
   this stochastic training estimator: the audit (`audit/jacobian.py`,
   Wave 4a of the plan) computes exact dense Jacobians by autograd in
   float64 on checkpoints. The penalty influences training only.
