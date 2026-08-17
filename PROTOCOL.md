# PROTOCOL.md

This is the opposable fairness contract of the factorial: everything that
makes cells comparable is declared here, before the runs. Three categories
of run exist in this repo, defined once, here, and reused identically in
every section below and in `README.md` — no section restates this
taxonomy differently:

- **Core grid** — exactly the factorial cells declared by sections 1, 3
  and 4 (budget-matched families × constraint × seeds), trained at the
  LR and μ_J values already pinned by the selection/calibration runs
  below. D1 is 80 runs (`README.md`'s D1 grid arithmetic). Official run
  counting for the core grid starts at the git tag `pre-registration`.
- **Pre-registered selection & calibration runs** — the LR sweeps (OFF
  and ON cells, section 2) and the μ_J calibration (section 5): committed
  and pre-registered before the tag, executed after it, published
  *alongside* the core grid's results, but never counted as part of the
  core grid's run total. Their arithmetic is declared in advance (section
  2, section 5), not tallied after the fact. Run counting for these also
  starts at the tag.
- **Exploratory runs** — everything else, whether it ran *before* the tag
  (spikes, smoke tests, dev iteration) or *after* it (calibration sweeps
  beyond the declared grids, debug runs, anything tried and abandoned):
  logged in the journal (section 9), never silently folded into either
  count above.

Section 9 restates this taxonomy once more, at the point of use, to fix
where exploratory runs get logged — it is not a second, different rule.

---

---

## Reading map

**This file is an audit trail, not a paper.** It is meant to be *queried*, not read
front to back. For the study itself — question, design, results, scope, limits — read
[`FINDINGS.md`](FINDINGS.md); for the result in ninety seconds, [`README.md`](README.md).
Everything below exists so that every number in those two documents can be traced to the
declaration that preceded it.

### If you want to check one thing

| you want | go to |
|---|---|
| whether structure helps, and by how much | **27.8.3** (paired contrast), qualified by **27.9** (what it covers, where it stops) |
| whether the prior is even representable | **20** (planted `1.9e-15`, OLS `2.11e-08`, gradient ~200× away) |
| why gradient descent does not reach it | **20.1** (four candidates eliminated; the 25-step closed rollout is the one that survives) |
| the diagnosis — basis versus coordinates | **26.0** (identity basis `1.0654` = 36 % of `‖A‖`; real Jordan basis `1.97e-15`) |
| how far the results generalize, and where they stop | **25.6** (scope rule), **21.4** (the pendulum boundary, never crossed), **27.8.5** and **28.5** (per-root scope) |
| the confounds, named by the author | **27.2** (rate ≠ backbone), **23.2** (why budget matching is suspended, not applied) |
| what was withdrawn after measurement | **22** (withdrawal of 21.1), **26.0** (25.3's branch 1 refuted structurally), **11** (withdrawal of P1(b)) |
| a defect found in a rule *after* scoring, and left standing | **27.8.4** and **27.8.5** |
| what has been declared but never run | **28** (the oracle-basis root) |
| how a run's identity is fixed | **12** (endpoint conventions), **19.4** (the gate artifact) |

### The four eras of this file

Sections are chronological and never rewritten, so the file has strata. Knowing which
stratum you are in saves most of the confusion.

1. **Sections 1–9 — the original contract.** Budgets, seeds, TF/FR schedule, thresholds.
   Written before anything ran. Sections 6, 7 and 8 were reserved for waves that the
   project never reached and are *open placeholders*, not omissions.
2. **Sections 10–12 — the μ_J era, and its failure.** Structure was first encoded as a
   calibrated Jacobian penalty. The calibration failed its own cross-family
   comparability check; **11** withdraws the prediction that depended on it. This era is
   retired and its material is secondary/regression only — but it is kept, because the
   failure is the most instructive thing in it.
3. **Sections 13–22 — the pendulum era.** The structure axis becomes architectural
   (**13**), the phase-D gate is built and repaired (**14–17**), the ssm family is
   documented as a negative and then rescued by a physically bounded latent step
   (**18**, **24**), D3 proves the path (**20**) and D4's reading is withdrawn (**21**,
   **22**). **21.4** marks the boundary — seven of eight positive-control cells fail on
   `DampedPendulum` — and no later section crosses it.
4. **Sections 23–28 — the oscillator era, where the result lives.** Six self-sufficient
   roots (**23**, **25**), their two dated precisions (**26**), the primary factorial
   across backbones (**27**), and the basis root, declared and unrun (**28**).

### The eighteen amendments, one line each

| § | amendment | what it changed |
|---|---|---|
| 10 | 1 | repaired the μ_J calibration after its first version proved non-comparable across families |
| 11 | 2 | **withdrew** prediction P1(b); the calibrated margin was never validated as physically comparable |
| 12 | 3 | fixed endpoint conventions and introduced the two-sided positive-control gate |
| 13 | 4 | made the structure axis **architectural** — a predictor family, never a numeric penalty |
| 14 | 5 | fixed the phase-D gate budget and its execution contract |
| 15 | 6 | repaired the phase-D gate and declared the second probe |
| 16 | 7 | retained the probe's entry values; declared probe round 3 |
| 17 | 8 | added the gradient guard and re-ran round 3's ssm cells clipped |
| 18 | 9 | narrowed the grid to two backbones; documented the ssm family as a negative |
| 19 | 10 | fixed D3's measurement coordinates and made the spectral and action bounds well-posed |
| 20 | 11 | D3's verdict: the path is proven, exact identification is not reached |
| 21 | 12 | D4's reading: an upper bound that is not arm-neutral, and one declared re-selection |
| 22 | 13 | **withdrew** 21.1's conclusion |
| 23 | 14 | opened the oscillator campaigns: three-level structure factor, sample-size axis, closed-form misspecification axis |
| 24 | 15 | the ssm family trains under a physically bounded latent step |
| 25 | 16 | declared two roots together: the training-horizon regime, then the decomposition |
| 26 | — | two dated precisions: why `nolift` is worse than both arms, and campaign B's hidden mechanism |
| 27 | 17 | lifted the ceiling for ONE root — the backbone axis, never measured |
| 28 | 18 | lifted it a second time, for the one experiment 26.0 named and declined |

### Three conventions that explain the shape of this file

- **The contract precedes the run.** A section is committed before the cells it
  authorises execute. This is why amendments accumulate instead of edits.
- **A withdrawn result is not a falsified one.** Withdrawals are dated and placed beside
  the measurement they revise; nothing is deleted, and no history is rewritten.
- **A defect visible before the data is repaired; a defect visible only after scoring is
  recorded.** That distinction is why **27.7** is an amendment and **27.8.4** is an
  observation, and it is the reason a rule that turned out to be the wrong instrument was
  left exactly where it was written.

---

## 1. Budgets (fixed at Wave 3)

Every predictor family in a comparison cell is matched to the same parameter
target, not the same width. `sld.training.budget.match_budget` searches each
family's single capacity knob by build-and-count (construct, count trainable
parameters, compare) until the achieved count is within **±5 %** of
`target_params` — exact by construction, not a closed-form approximation.

- **Capacity knobs are not config-settable.** `d_model`, `d_ff`, `n_layers`,
  `hidden`, `d_state`, `psi_hidden`, `pi_hidden` are owned by `match_budget`
  and rejected outright by `Trainer` if named directly in a run config
  (`_BUDGET_OWNED_KEYS`, `src/sld/training/trainer.py`). The only capacity
  lever a config declares is `target_params` (default 10 000).
- **FLOPs are counted analytically, as executed** — never as measured wall
  clock. Convention: 1 MAC = 2 FLOPs; every Linear/matmul/einsum the step
  actually runs is counted at its dense shape. A block-diagonal operator
  realized as a dense matmul (three of the four families do this) counts
  dense: fairness counts what runs, not the operator's sparsity.
  Elementwise combine/residual ops count linearly; activation costs
  (GELU, softplus, the exp/cos/sin inside the block-2x2 ZOH assembly) are
  declared out of scope.
- **Transformer per-step cost is `f0 + c*t`**, `t` = current KV-cache length
  — O(H²) per trajectory. This is declared, not oversold: the other three
  families are markovian or recurrent-with-constant-cost, so their
  per-step cost does not depend on trajectory position.

Measured at Wave 3, `target_params=10_000`, `d_z=4, d_u=2`:

| family | achieved params |
|---|---|
| mlp | 9 968 |
| transformer | 10 344 (d_model=20) |
| min_ssm | 10 124 |
| koopman_ssm | 9 992 |

All four inside the ±5 % band (locked by `tests/training/test_budget.py`,
which asserts the relative-error bound per family, not these exact counts —
the counts above are the Wave 3 measurement at this target and this
`(d_z, d_u)`, reproducible by rerunning `match_budget`).

---

## 2. Tuning (fixed at Wave 3, selection rule fully pinned at Wave 5,
LR sequencing revised at Wave 5 round 3, phase 4 consistency check bounded
at Wave 5 round 4)

The **LR sweep** is identical per arm (arm = family × constraint) — the
per-arm fairness law, quoted exactly (`REGLES.md` section 4): *« Équité :
params ±5 %, FLOPs comptés, mêmes updates, même sweep de tuning par bras,
sweeps publiés. »* Both the OFF and the ON cell of every family receive the
same LR sweep — same candidates, same seed, same budget, same score —
rather than ON inheriting its family's OFF-selected value. The ON arm also
receives the bounded phase-4 consistency re-validations below. That is a
declared structural asymmetry, required to keep the retained `(LR, mu_j)`
pair measured; it is not a hidden extra tuning budget, and every such run is
published in the pre-registered selection & calibration category.
`pendulum_core.yaml` carries a single `lr` override per arm (8 arms, not 4
families), shared by *both* regimes (`ou`, `action_conditioned`) — there is
no per-regime learning rate anywhere in the grid.

**Run namespaces.** Core cells use `exp_name: pendulum_core`; every
selection or calibration config uses `exp_name: pendulum_core_selection`.
Because `exp_name` is part of a resolved run identity path, the seed-0,
core-budget selection checkpoint cannot occupy or silently resume the core
seed-0 cell. The categories and their arithmetic remain disjoint exactly as
declared above; the namespaces now are too.

**Sequencing with μ_J calibration (fixed at Wave 5 round 3, revised from
round 2).** The LR sweep and the μ_J calibration (section 5) still cannot
both be resolved in one pass — a sweep on the ON cell needs a `mu_j`, and
calibrating `mu_j` needs an LR — but round 2's fix (OFF sweeps, ON
inherits) violated the per-arm law quoted above: the ON cell's penalized
objective could prefer a different LR than the OFF cell's unpenalized one,
so letting OFF's sweep silently decide ON's LR gave the two arms an
asymmetric tuning budget disguised as a shared one. The break is
pre-registered here, in four ordered phases:

1. **Phase OFF.** The sweep procedure below runs on each family's OFF
   cell only (`mu_j = 0`, no dependency on section 5) and retains one LR
   **per family** — `LR_off(family)` — for that family's OFF arm.
2. **Initial calibration.** Section 5's μ_J calibration (steps 1-6,
   unchanged) runs at `LR_off(family)` and retains `mu_j(family)`. This
   unblocks phase 3; it is not necessarily the calibration that ends up in
   the achieved-margins table (see phase 4).
3. **Phase ON.** The *full* sweep procedure below — same 3 candidates,
   same selection seed, same selection budget, same selection metric as
   phase 1 — now runs on each family's ON cell, with `mu_j(family)` from
   phase 2 held fixed, and retains one LR **per arm** — `LR_on(family)` —
   for that family's ON arm. This sweep is not circular: `mu_j(family)`
   already exists from phase 2, so the ON cell trains end to end for every
   candidate LR. Both arms of every family now receive an identical sweep
   (per-arm law satisfied): phase OFF for the OFF arm, phase ON for the ON
   arm.
4. **Consistency of the selected pair (bounded iteration, revised at
   Wave 5 round 4).** If `LR_on(family) == LR_off(family)`, the margin
   measured by phase 2's 10-epoch calibration is the selection proxy for
   the pair the ON arm actually trains at — selection is done. If `LR_on(family) !=
   LR_off(family)`, the `(LR_off, mu_j)` pair calibrated in phase 2 is no
   longer the pair the ON arm uses, so its 10-epoch selection proxy is
   re-measured — the same post-hoc measurement, section 5 step 3 — at
   `(LR_on(family), mu_j(family))`:
   - If the re-measured proxy median falls in the declared band (section 5,
     `[0.90, 1.00]`), `(LR_on(family), mu_j(family))` is the
     selection-retained pair — selection is done.
   - If it falls outside the band, the pair enters the bounded iteration
     below: **at most two full cycles**, each cycle the same four-step
     unit named in the round 4 decision — *sweep* (the LR fixed entering
     the cycle, already produced by a prior sweep) -> *calibration*
     (recalibrate `mu_j` at that LR) -> *re-sweep* (re-run the full phase
     ON LR sweep at the recalibrated `mu_j`) -> *re-measure* (re-audit the
     10-epoch selection proxy of the resulting pair, only if the LR moved).
     Concretely, for
     cycle `k` (`k = 1, 2`), let `LR^(k-1)` be the LR fixed entering the
     cycle — `LR_on(family)` for `k = 1`, cycle `k-1`'s re-swept LR for
     `k = 2`:
     1. **Recalibrate.** The entire section 5 calibration procedure
        (steps 1-6, including its one allowed refinement point) is rerun
        at `LR^(k-1)` — this is the LR the recalibration measures its
        candidate `mu_j` values at. If no swept or refined value's median
        reaches the band even at this fixed LR, the family's ON cell is
        reported **"band not reached"** immediately (section 5's own
        terminal state; no further cycle is attempted). Otherwise this
        yields `mu_j^(k)`, replacing the previous selection-retained
        `mu_j`.
     2. **Re-sweep.** The *full* phase ON sweep procedure (same 3 LR
        candidates, same selection seed, same selection budget, same
        selection metric as phase 1) reruns with `mu_j^(k)` held fixed,
        yielding `LR^(k)`.
     3. **Re-measure the selection proxy, conditionally.** If `LR^(k) == LR^(k-1)` — the LR
        the recalibration in step 1 just measured `mu_j^(k)` at — the
        selection-retained pair is `(LR^(k), mu_j^(k))` and its proxy is
        already known from that recalibration's own in-band median: no
        extra training run, selection is done. If `LR^(k) != LR^(k-1)`,
        the pair moved again, so the proxy is re-measured (section 5 step
        3) at
        `(LR^(k), mu_j^(k))`:
        - In the band: `(LR^(k), mu_j^(k))` is the selection-retained pair
          — selection is done.
        - Outside the band: if `k = 1`, proceed to cycle `k = 2` with
          `LR^(1)` as the LR fixed entering it; if `k = 2`, the two-cycle
          budget is exhausted and the family's ON cell is reported
          **"selection not converged"**, verbatim — no third cycle, no
          nearest-value fallback. Resolving a "selection not converged"
          family is a protocol amendment, not a choice made inside this
          procedure — the same policy as "band not reached" (section 5):
          it requires a new, dated, committed decision before that
          family's ON cell can enter the core grid.

   Every run this consistency check and its bounded iteration trigger —
   both cycles' recalibrations, re-sweeps, and conditional
   re-measurements — stays in the **pre-registered selection &
   calibration runs** category (header taxonomy): published alongside
   the core grid with its full trace, never logged in section 9, which
   is reserved for runs outside this declared procedure (this corrects a
   round 3 mis-citation that had sent these runs to section 9).

   **Core-budget achieved-margin audit.** Once phase 4 has settled a
   selection-retained `(LR_on, mu_j)` pair, the declared achieved margin is
   audited on that ON arm's retained **OU-regime 50-epoch core checkpoint**,
   seed 0. The action-conditioned regime's retained 50-epoch core checkpoint,
   also seed 0, is audited and reported alongside it as an informational
   margin, but cannot affect the band verdict: the OU result alone defines
   that verdict. This pins the criterion to the same OU distributional slice
   used by the calibration proxy.
   The 10-epoch calibration/re-measurement medians above are explicitly
   unverified **selection proxies**: they choose `mu_j` candidates only and
   are never entered as achieved margins. If the offline audit of the core
   **OU checkpoint** falls outside `[0.90, 1.00]`, report **"band not
   reached at core budget"** verbatim in section 5's table and journal the
   outcome as a dated protocol amendment; no further selection or calibration
   loop is run. The procedure is bounded.

**Sweep procedure** (applies identically to phase OFF and phase ON above —
the same six rules, run once per phase per family):

1. **Sweep**: learning rate in `{1e-4, 3e-4, 1e-3}`, trained on the cell
   being tuned in that phase (the OFF cell in phase OFF; the ON cell, with
   `mu_j(family)` fixed, in phase ON).
2. **Selection seed = 0.**
3. **Selection budget = 50 epochs** — the same `epochs` value the core
   grid itself trains under. This is not invented: it is
   `src/sld/training/trainer.py::_DEFAULTS["epochs"] = 50`, and
   `pendulum_core.yaml` declares no `epochs` key, so 50 is the effective
   budget for both sweep phases and the core grid (`batch_episodes`
   likewise stays at the Trainer default, 8).
4. **Selection metric, common across arms**: free-running validation loss
   at the fixed horizon `h_fr = 5` (section 4) — never teacher-forced
   loss, never a family-specific proxy — **averaged over both regimes**:
   `score(lr) = 0.5 * (val_fr_loss_ou(lr) + val_fr_loss_action_conditioned(lr))`.
   `pendulum_core.yaml` declares one `lr` override slot per arm (no
   per-regime field); this is one selection decision per arm per phase,
   not one per regime, and the two regimes' scores are combined by
   unweighted mean rather than either one silently standing in for both.
5. **Tie-break**: if two or more swept LR values are within
   floating-point-exact equality of the minimum averaged score, the
   **smallest** tied LR is retained.
6. **Confirmation**: each arm's retained learning rate is rerun at 5 seeds
   (section 3) as part of the core grid.

**Retention granularity**: one retained LR **per arm** — 4 families × 2
arms = 8 retentions (`LR_off(family)` and `LR_on(family)` for each of the
4 families), replacing Wave 5 round 2's single per-family retention.
`pendulum_core.yaml`'s comment on the `lr` field (*"filled by PROTOCOL.md
section 2's sweep ... selects 1 LR from {1e-4, 3e-4, 1e-3} ..., then runs
5 seeds at the retained LR"*) stays accurate under this revision without
any change to the yaml itself: the config already carries one `lr` slot
per arm (8 slots, `overrides: {..., lr: null}` on every arm), and each
slot is now filled by its own arm's sweep instead of 6 of the 8 slots
inheriting a sibling arm's value.

Sweep results (both phases) are published alongside the core grid, not
summarized away — the header's taxonomy classifies these as pre-registered
selection & calibration runs: published with the core grid, never counted
in it. Any run outside this sweep (a hunch, a debug run, a wider grid
tried and abandoned) is exploratory and is journaled in section 9, never
silently folded into the reported numbers.

**Declared run arithmetic for this section** (worst case; the realized
selection-run count is reported alongside the achieved-margins table in
section 5, not assumed here). Counted in `Trainer.fit` / run-directory
units. Two
corrections versus the arithmetic below's prior version (both fixed at
Wave 5 round 4): every swept LR candidate trains **both** regimes
independently (section 2 point 4: `score(lr)` averages two separately
trained runs, `val_fr_loss_ou` and `val_fr_loss_action_conditioned`), so a
"3 LR candidates" sweep is 6 `Trainer.fit` runs per family, not 3; and the
initial calibration (section 2 phase 2 / section 5) is counted here too —
it is part of the declared procedure and was wrongly left out of the
prior worst-case total:

- Phase OFF: 4 families × 3 LR candidates × 2 regimes (`ou`,
  `action_conditioned`) = 24 runs.
- Initial calibration (section 2 phase 2, i.e. section 5 steps 1-6 at
  `LR_off(family)`): 4 swept `mu_j` values × 1 seed × 1 regime (OU only —
  section 5 step 1 names no `action_conditioned` variant) = 4 runs per
  family, plus at most 1 refinement point (section 5 step 6) = at most 5
  runs per family; 4 families × 5 = at most 20 runs.
- Phase ON (initial sweep, `mu_j(family)` fixed): same structure as phase
  OFF — 4 families × 3 LR candidates × 2 regimes = 24 runs.
- Consistency check and its bounded iteration (phase 4, revised at Wave 5
  round 4):
  - Initial re-measurement: 1 run (OU regime, seed 0, section 5's
    10-epoch calibration budget) per family whose `LR_on(family) !=
    LR_off(family)` — at most 4 families × 1 = 4 runs.
  - At most two full cycles, each entered only if the selection proxy
    re-measurement above (or the previous cycle's) falls outside the band.
    Each cycle is
    at most: a full section 5 recalibration (≤ 5 runs, OU only) + a full
    phase ON re-sweep (6 runs, both regimes) + one conditional
    re-measurement (≤ 1 run, OU only) = at most 12 runs per cycle per
    family; 2 cycles × 12 = at most 24 runs per family; 4 families × 24 =
    at most 96 runs.
  - Consistency/iteration subtotal: 4 + 96 = at most 100 runs.
- Worst-case total: 24 + 20 + 24 + 100 = **168 `Trainer.fit` runs**. This
  is the ceiling only if every family's `LR_on` differs from its
  `LR_off`, every selection-proxy re-measurement lands outside the declared
  band, and
  every family exhausts both cycles without converging (a "selection not
  converged" verdict on all four families); a single in-band
  re-measurement, or a re-swept LR that stops moving, ends the iteration
  for that family immediately, so the realized number is almost
  certainly far smaller.

The four seed-0 OU achieved-margin **band audits** plus four seed-0
action-conditioned **informational audits** are eight offline measurements on
already retained 50-epoch core checkpoints, not additional `Trainer.fit` runs
or selection runs. They therefore add zero to the 168-run ceiling; both
regimes' results, including any OU "band not reached at core budget" outcome,
are reported in section 5's table.

---

## 3. Seeds (fixed at Wave 0)

5 seeds: `0, 1, 2, 3, 4`. The seed is part of the run's identity — it is a
config field, and the run directory is `<config_hash>-<data_hash>` where
`config_hash` is a hash of the resolved config (seed included). Two runs
that differ only in seed are, correctly, two different run directories.

---

## 4. TF/FR schedule (fixed at Wave 3)

Trainer defaults, declared here as the protocol v0 choice (not silently
inherited from `src/sld/training/trainer.py::_DEFAULTS`):

- `w_tf = 1.0`, `w_fr = 1.0`, `h_fr = 5`.
- **TF** (teacher-forced): `predictor.step` threaded over the true prefix —
  state accumulates across the ground-truth prefix, the model never feeds its
  own prediction back in.
- **FR** (free-running): `predictor.rollout` from `h0` for
  `min(h_fr, T)` steps.

Every family in the factorial trains under the same TF/FR mix; no family
gets a silently different schedule.

---

## 5. μ_J margins (fixed at Wave 5 — this section)

### The constraint axis

- **Cell OFF**: `mu_j = 0` (no penalty weight). For `koopman_ssm`
  specifically, OFF also means `alpha_i` free in `constrained: false` mode —
  hard-wiring stability into the parametrization would deny the family a
  real "off". OFF is "the penalty is absent", not "the family happens to be
  contractive".
- **Cell ON**: `mu_j > 0`. The penalty acts on norms of the **total**
  Jacobian `J^q` on the augmented state `q = (h, m)` along training
  trajectories — never on a per-block or per-family surrogate. Estimated by
  JVP/Hutchinson probes, `k_probes = 2`, `step_subsample = 0.25` —
  pre-registered constants from the Wave 2 spike. **The measured overhead
  (1.36×) is scoped to the tensor-state path only**: families that never
  populate a threaded recurrent state (`mlp`, `koopman_ssm` — every step
  has `state.m is None`, dispatched to `TotalJacobianPenalty.__call__`),
  where all sampled states and probes are evaluated in one deferred,
  batched `torch.func.jvp` call per rollout (`reports/spike_jacreg.md`).
  **Families that thread a recurrent state (`min_ssm`, `transformer`,
  dispatched to `TotalJacobianPenalty.on_state_steps`) are NOT covered by
  that measurement.** `on_state_steps` cannot batch sampled steps together
  because the state's shape can change step to step (the transformer's
  KV cache grows by one token per step), so it issues `k_probes` separate
  `torch.func.jvp` calls per kept step — roughly
  `k_probes * step_subsample * T` extra forward passes per loss evaluation
  (`src/sld/objectives/jac_reg.py::on_state_steps`, declared in its
  docstring, unmeasured). This per-evaluation cost **grows with KV-cache
  length for the transformer** (each of those forward passes gets more
  expensive as `t` increases across the rollout), unlike `min_ssm`'s
  fixed-shape state. The training loop additionally **repeats this
  evaluation on the whole validation split every epoch**
  (`src/sld/training/trainer.py`: `val_total, _ = self._composite_loss(...)`
  under `torch.no_grad()`, same TF branch, whenever `mu_j > 0`), not just on
  the sampled training batch. No overhead multiplier is claimed for the
  threaded-state path — none has been measured. For `koopman_ssm`, ON
  additionally requires `alpha_i <= 0` via the constrained parametrization
  `alpha = -exp(nu)` (`constrained: true`).
- The penalty weight lives in the run config as `mu_j` (the Trainer's
  config key for the training-time penalty weight; 0.0 by default, i.e. the
  OFF cell).

### Identity-block exclusion for growing-memory families (fixed at Wave 5)

**Motivation.** A predictor with growing memory carries its full past
forward verbatim: the transformer's KV cache appends one token per step and
never overwrites earlier entries, so the block of `J^q` mapping "cache
entries already in `m`" to "the same entries copied unchanged into
`m_next`" is structurally an identity embedding. That block forces
`sigma_max(J^q) >= 1` by construction, with a Frobenius-norm floor that
grows with cache length — the adversarial review's own reproduction found
`80 * t` on that block at this repo's dimensions, i.e. 7 920 at `t = 99`.
Neither the training-time penalty nor the post-hoc dense audit can move
this block: its entries do not depend on the predictor's parameters, so the
gradient of any loss with respect to it is exactly zero, and a dense audit
that assumes a fixed-size square Jacobian map cannot even be formed once
the cache has grown (its shape depends on `t`).

**Decision (adopted).** For families whose augmented state `m` grows across
the rollout, the object that is both **penalized** (`mu_j`, training-time)
and **calibrated** (the post-hoc `sigma_max` audit below) is `J^q`
restricted to the complement of the identity embedding of the old `m` into
`m_next` — the new content each step actually writes into the state, never
the verbatim carry-forward of what was already there. Which rows/columns of
the dense Jacobian this excludes is an implementation detail owned by the
code (currently being implemented alongside this document), not a protocol
choice, and this document does not restate implementation-level specifics
or measured numbers it does not have.

**Scope.** In this factorial's four families, only `transformer` has memory
that grows across the rollout (its KV cache appends one token per step).
`min_ssm`'s recurrent state `m` has fixed shape `(B, d_state)` regardless
of `t`; `mlp` and `koopman_ssm` never populate `m` at all (`state.m is
None` throughout every rollout). None of these three families receive this
exclusion — it applies to `transformer` alone.

**Honesty declaration.** Because of this exclusion, the transformer's
penalized object and calibrated margin are measured on a **slightly
different object** than the other three families' complete augmented-state
Jacobian: a Jacobian with the identity carry-forward block removed, rather
than the full `J^q`. This is a declared asymmetry in the constraint axis,
not a hidden one. The exclusion is legitimate only because the excluded
block is known in advance and immovable by training (zero gradient) — never
because it makes the transformer's achieved margin look better after the
fact.

### Calibration procedure (pre-registered, run after the tag, short)

Per family, independently, **at the arm's retained LR** — the *initial*
calibration (section 2 phase 2) runs at `LR_off(family)`, before the
family's ON arm has its own retained LR. Section 2 phase 4's bounded
iteration (revised at Wave 5 round 4) can trigger this same procedure
again, **at most twice more** — once per cycle, each time at the LR fixed
entering that cycle (`LR_on(family)` for the first cycle; the previous
cycle's re-swept LR for the second, if a second cycle is entered). Either
way the LR is fixed by its sweep before this calibration runs, never the
reverse — this is what closes the LR/μ_J circularity. Every median in this
procedure is a 10-epoch **selection proxy**, not a declared achieved margin:
the latter's band verdict is audited only on the retained ON arm's
OU-regime 50-epoch core checkpoint as specified in section 2 phase 4; the
action-conditioned seed-0 checkpoint is measured and reported alongside it
for information only.

1. **Sweep**: `mu_j in {1e-3, 1e-2, 1e-1, 1}`, **selection seed = 0**, the
   pendulum OU regime (`OURegime` on `DampedPendulum`), trained at the
   arm's retained LR (`LR_off(family)` for the initial calibration; the
   LR fixed entering the current cycle, per section 2 phase 4's bounded
   iteration, for a recalibration).
2. **Calibration budget = 10 epochs, `batch_episodes = 8`** (the Trainer
   default, unchanged). An explicit, short budget — one fifth of the core
   grid's 50-epoch budget (section 2) — chosen to separate the four swept
   `mu_j` values without paying the full training cost per calibration
   point; not "short" left for later interpretation.
3. **Post-hoc measurement only** — never the training-time estimator: run
   the exact audit (dense Jacobian by autograd, float64, offline on the
   checkpoint) and take the median, over **states visited in
   teacher-forcing on the val split** (detached; the same TF-only
   distribution the training-time penalty itself targets — see the
   state-distribution honesty declaration below), of `sigma_max(J^q_t)` —
   the operator norm of the dense total Jacobian. This audit uses the
   identity-block-excluded object above for `transformer`, the full `J^q`
   for the other three families.
4. **Declared margin band: median in [0.90, 1.00]** for the ON cell.
5. **Selection-retention rule**: selection-retained `mu_j` = the smallest
   swept grid value whose proxy median falls in the band.
6. **Refinement rule (deterministic, at most one refinement per family)**.
   If no swept value's median falls in the band, read the four medians
   together. The four cases below are exhaustive and read in order — every
   possible set of medians matches exactly one, with no gap and no
   ambiguous overlap:
   - **All four above the band** (`> 1.00`, i.e. even `mu_j = 1` does not
     contract enough): add exactly one grid point at `mu_j = 10` — one
     geometric step above the largest swept value, same decade spacing as
     the sweep — and re-measure only that point.
   - **All four below the band** (`< 0.90`, i.e. even `mu_j = 1e-3`
     over-contracts): add exactly one grid point at `mu_j = 1e-4` — one
     geometric step below the smallest swept value — and re-measure only
     that point.
   - **Monotone, straddling the band** (not all four on the same side of
     the band, and the medians decrease monotonically as `mu_j` increases —
     so there is exactly one adjacent pair of swept values `mu_j_lo <
     mu_j_hi` with `median(mu_j_lo) > 1.00` and `median(mu_j_hi) < 0.90`:
     the sweep steps over the band between two grid points instead of
     landing in it): add exactly one grid point at the **geometric mean**
     `sqrt(mu_j_lo * mu_j_hi)` of that bracketing pair and re-measure only
     that point.
   - **Non-monotone** (not all four on the same side of the band, and the
     medians do **not** decrease monotonically as `mu_j` increases): no
     refinement is attempted, even when one or more adjacent pairs happen
     to bracket the band locally — a non-monotone sweep can have more than
     one such bracketing pair (e.g. medians `[1.2, 0.8, 1.1, 0.7]` bracket
     the band twice), and picking one pair over another would itself be a
     result-dependent degree of freedom. Abstention is the deterministic
     rule here, not evidence that no bracketing pair exists.
   If, after this single refinement (or immediately, in the non-monotone
   case), no value's median falls in the band, the family's ON cell is
   reported as **"band not reached"**, verbatim, in the achieved-margins
   table below — no nearest-value fallback, no post-hoc relaxation of the
   band, no cherry-picking of a value outside it, and no second
   geometric-mean re-bracketing attempt in the monotone-straddle case (the
   "at most one refinement per family" cap above already fixes the limit
   at one). Resolving a "band not reached" family is a protocol amendment,
   not a choice made inside this procedure: it requires a new, dated,
   committed decision before that family's ON cell can enter the core grid.

Achieved margins (to be filled by the **50-epoch core-budget audits**, not by
the 10-epoch calibration). The OU-regime seed-0 margin is the sole band
criterion; the action-conditioned seed-0 margin is always measured and
reported alongside it, informational only. A family may instead read
**"band not reached"** (section 5 step 6's selection terminal state),
**"selection not converged"** (section 2 phase 4's terminal state), or
**"band not reached at core budget"** (the retained ON **OU-regime** seed-0
checkpoint's offline audit) in place of its numeric `mu_j`/OU-median pair —
each is reported verbatim, never papered over with a nearest value:

| family | mu_j selection-retained | median achieved, OU core seed-0 (50 epochs; band criterion) | median achieved, action-conditioned core seed-0 (50 epochs; informational) |
|---|---|---|---|
| mlp | to be filled by selection | to be filled by core-budget audit | to be filled by informational audit |
| transformer | to be filled by selection | to be filled by core-budget audit | to be filled by informational audit |
| min_ssm | to be filled by selection | to be filled by core-budget audit | to be filled by informational audit |
| koopman_ssm | to be filled by selection | to be filled by core-budget audit | to be filled by informational audit |

If section 2 phase 4's bounded iteration triggers one or two
recalibrations for a family, the `mu_j` reported above is the value from the
**last** selection cycle; its OU median is always the subsequent core-budget
band audit, never an intermediate proxy, and its action-conditioned median is
reported alongside it informationally. Superseded intermediate calibrations
(at `LR_off(family)`, or at an earlier cycle's LR) remain
**pre-registered selection & calibration runs** (header taxonomy), published
alongside the core grid with their trace, never journaled in section 9, which
is reserved for runs outside this declared procedure. An **OU** core-budget
band audit outside the band is instead journaled as the protocol amendment
prescribed in section 2 phase 4; it does not reopen the bounded loop.

### Honesty declarations

- (a) **Estimator/measurement gap.** The penalty optimizes `||.||_F`
  (Frobenius norm, an upper bound on the operator norm) at training time;
  the calibrated margin measured post-hoc is `sigma_max` (operator norm).
  This gap is declared, not hidden.
- (b) **Band rationale.** The damped pendulum's true modes at the frozen
  defaults (`zeta = 0.1`, `omega_0 = sqrt(g/L) ~= 3.1321`) are
  `lambda = -0.313209 +/- 3.116392i`, giving `|exp(dt * lambda)| =
  0.993755` at `dt = 0.02` — the true per-step decay sits INSIDE the band.
  Declared honestly: the band's lower edge (0.90 per step) permits a
  contraction log-rate ~16.8x the physical one (over a 50-step horizon the
  true modes decay to ~0.73, a 0.90-per-step map to ~0.005), so the band is
  NOT a claim that 0.90 is physically neutral. It is chosen wide for
  calibration reachability across four heterogeneous families; gross
  over-contraction inside it is the job of the spectral display/screening
  panel and the chaos metrics (README robustness bar #2), not of this
  margin.
- (c) **What "comparable" means here.** The band makes "constraint ON" a
  comparable *treatment* across families — each family is pushed toward the
  same measured contraction target — not an identical *intensity* of
  `mu_j`. The retained `mu_j` value differs by family; the achieved margin
  is what is held comparable.

### P1a endpoint — gauge-invariant drift (fixed at Wave 5)

The Fig. 1 / P1a endpoint is the free-running drift measured in
**true-state coordinates**, never in each run's own latent coordinates.
Rationale (adversarial review, Wave 5 round 1): each run trains its own
encoder, so raw-latent drift is not gauge-invariant — an equivalent linear
reparameterization `z' = S z` with `S = diag(100, 1)` changes the measured
curve from `[0.7071, 0.7071]` to `[0.99995, 0.0099995]`, both height and
shape, so an ON/OFF ordering read in raw latents can be a pure
encoder-coordinate artifact.

Declared endpoint, per run:

1. A linear readout (ordinary least squares with a bias term, float64) is
   fitted from the run's encoded latents to the true state, on the **val**
   split, on the encodings of the ground-truth trajectory (a coordinate
   change estimate — never a fit of the predictions).
2. Prediction and target BOTH pass through that same readout; the drift
   curve is then computed in true-state coordinates
   (`sld.audit.jacobian.fr_drift_curve`).
3. **Curve invariance and its domain (fixed at Wave 5 rounds 2-3).** On a
   **full-column-rank** affine design matrix (encoded latents plus the
   bias column), the OLS solution is unique and under any invertible
   linear reparameterization `z' = S z` the fitted readout becomes
   `W' = W S^{-1}`: the true-state estimates — and hence the drift curve —
   are exactly identical, up to rounding amplified by the design's
   conditioning. Locked by test
   (`tests/viz/test_viz.py::test_readout_drift_invariant_to_linear_reparam`)
   for adversarial bounded-conditioning gauges: rescaling
   (`S = diag(100, 1)`, the round-1 reproduction), a shear, and their
   composition.
4. **Eligibility under coordinate gauges (fixed at Wave 5 round 5).** The
   design `[z_enc | 1]` is column-equilibrated before OLS and before its
   rank/condition diagnostics: each column is divided by its own RMS. A run
   is classified **"readout indeterminate"** when this equilibrated design
   is column-rank-deficient or when `sensitivity = cond x float32-eps >
   1e-2`. This is a componentwise sensitivity measure of the OLS solution to
   **per-coordinate relative float32 storage noise**; it does not claim that
   a diagonal rescaling destroys information. The verdict is therefore
   **STABLE under every finite invertible diagonal (per-coordinate unit)
   gauge**, including `S = diag(1e9, 1)` and uniform `S = 2**20 I`, locked
   by `tests/viz/test_viz.py::test_fit_readout_stable_under_diagonal_gauge_round4`.
   There remains an unavoidable qualified limit: under a general
   **non-diagonal** invertible transform, componentwise storage noise itself
   changes coordinate meaning, so this componentwise finite-precision
   criterion need not be gauge-invariant. That is the only declared
   gauge-dependence; it is not an information-destruction interpretation of
   diagonal unit changes.
   An indeterminate run never silently enters a curve: in strict mode,
   rendering fails naming the run (after the per-run verdict CSV is
   written, so the verdict is durable); in lenient mode the run is
   excluded with an explicit warning. Because latent collapse is itself
   an object of study in this repo, this classification is a
   **reportable result**, not a hidden failure mode.
5. Declared limit: the curve invariance at point 3 is exact for invertible
   LINEAR reparameterizations on the full-rank domain; a nonlinear
   reparameterization is only approximately absorbed by a linear readout.
6. The readout's R² (gauge-invariant: fitted values are a projection onto
   the design's column span), rank, and determinability status are
   reported **per run, per render**, in the artifact
   `assets/fig1_readout_r2.csv` — written before any strict-mode failure,
   so a collapsed seed always leaves a durable record.

---

## 6. Absolute success thresholds (to be fixed at Wave 7, BEFORE any planning run)

**Placeholder.** Pendulum swing-up (angle threshold + hold-time threshold)
and CartPole swing-up (analogous thresholds) will be declared here, with
exact numeric values, before any planning run executes. Thresholds are
always absolute, never relative to the best-performing arm in the grid — an
arm that misses the threshold is reported as "does not reach threshold",
never as a ratio against the winner.

---

## 7. Planner budgets (to be fixed at Wave 7)

**Placeholder.** CEM population/elites/iterations and gradient-planner
step count/learning rate, both declared as fixed compute budgets (S, I, H
for CEM; k, learning rate for the gradient planner) before the planning
runs that feed Fig 4.

---

## 8. Invariance tolerances (to be fixed at Wave 4b)

**Placeholder.** Per-diagnostic tolerance on drift under compensated
reparametrization (`audit/invariance.py`) — the stress test every
diagnostic must pass before entering a figure or a conclusion (REGLES
section 4). Numeric tolerances per diagnostic (`r_ctrl`, `E_ctrl`, `jac_margin`,
...) will be declared here.

---

## 9. Exploratory-run journal (open since Wave 2)

Every run in the **exploratory** category of the header's three-category
taxonomy is logged here — whether it ran *before* the `pre-registration`
tag (spikes, smoke tests, dev iteration) or *after* it (anything tried
outside the declared sweeps of sections 2/5 and abandoned, debug runs, a
hunch). Runs in the other two categories — the **core grid** (sections 1,
3, 4: the factorial cells) and the **pre-registered selection &
calibration runs** (sections 2 and 5: the LR sweeps and the μ_J
calibration) — are reported in their own sections, never logged here as
exploratory. This restates the header's taxonomy at the point of use; it
is not a second, different rule.

| date | what ran | why | outcome |
|---|---|---|---|
| | | | |

---

## 10. Amendment 1 — repaired μ_J calibration (dated 2026-07-30)

This is the "new, dated, committed decision" that section 5 step 6 and section 2
phase 4 both prescribe as the ONLY way to resolve a terminal calibration state. It
amends the **compute constants** of the calibration procedure and **declares the
metric** in which its margin is measured. It does not rewrite section 5: the μ_J
grid, the selection-retention rule, the four refinement branches, the band
`[0.90, 1.00]`, the three-category run taxonomy, the P1-P5 predictions and the P1a
endpoint all stand **verbatim**. Every clause of section 5 not explicitly superseded
below remains in force.

### 10.1 Why (measured, before this amendment was written)

The pre-amendment calibration produced "band not reached" for 3 of 4 families. The
cause was diagnosed by offline probes on the retained checkpoints, and reviewed
adversarially (2 rounds, verdict SAME-PAGE: YES). The findings that force this
amendment:

1. **The calibration trained 10 optimizer steps per candidate.** The trainer takes
   one sampled batch and one `optimizer.step()` per epoch, so a 10-epoch budget is
   10 updates and the 50-epoch core budget is 50. Measured weight displacement over
   a 10-update calibration run: 0.3-0.8 % of the parameter norm.
2. **At 10 updates the sweep is entirely inside the transient**, where AdamW's
   partial loss-scale invariance makes `mu_j` inert: measured
   `cos(grad_total(mu_j=1), grad_total(mu_j=10))` = 0.9985 (mlp), 0.999996
   (min_ssm); `||theta(mu_j=10) - theta(mu_j=1e-3)|| / ||theta||` = 0.78 % (mlp),
   0.038 % (min_ssm). At a converged stationary point the dependence returns
   (`grad L_pred + mu_j * grad P = 0`, and `dP/dmu_j = -grad P^T H^-1 grad P <= 0`
   at a strict local minimum), so the fix is an adequate optimization budget, not a
   new mechanism.
3. **`step_subsample = 0.25` discards three quarters of the steps** and is the
   dominant variance source of the training-time estimator.
4. **`sigma_max(J^q)` was measured in undeclared coordinates.** Same dense
   Jacobians, read in the raw latent basis vs a whitened one: mlp 1.0309 / 1.0879,
   min_ssm 1.0439 / 22.94, koopman_ssm 0.9952 / 5.047, with
   `cond(Sigma)` = 62.7 / 1.5e7 / 1.3e5. The augmented state concatenates `h`
   (variance ~5e-2) with `m` (variance down to 2e-8). A margin whose verdict moves
   by a factor 23 under an equally defensible coordinate choice is not a measurement.
   `REGLES.md` section 4 already forbids publishing a diagnostic that fails the
   reparameterization stress test ("réparé ou retiré"); contraction theory
   (Lohmiller & Slotine, *Automatica* 1998, and the control/neural contraction-metric
   literature that follows) makes declaring a metric a theoretical obligation, since
   contraction is defined relative to a metric and not by the Jacobian alone.

### 10.2 Superseded compute constants

Section 5's calibration procedure keeps its steps 1-6 verbatim, with these three
constants replaced:

- **Budget.** Section 5 step 2's "calibration budget = 10 epochs" is superseded by
  **200 optimizer updates per candidate**, and the same 200-update budget applies to
  the LR sweeps of section 2 and to the retained/core achieved-margin runs, so that
  every arm still receives an identical tuning budget (the per-arm fairness law,
  `REGLES.md` section 4, is preserved). Budgets are henceforth expressed in
  **optimizer updates**, not epochs, and the LR scheduler's horizon is expressed in
  updates. 200 is the smallest horizon at which contraction was *measured* to be
  materially controllable for both `mlp` and `min_ssm`; 10 demonstrably is not, and
  50 has no family-wide supporting evidence. This is declared as a minimum
  diagnostic budget, not a claim of convergence.
- **`step_subsample`: 0.25 -> 1.0.** Every teacher-forced step is kept, which makes
  the Bernoulli step-mask deterministic and removes the dominant variance floor.
- **`k_probes`: unchanged at 2.** Measured, not assumed: at `step_subsample = 1.0`,
  with parameters and batch held fixed, the estimator's coefficient of variation
  over 256 independent draws is 0.075 % (mlp) and 0.139 % (min_ssm) — two orders of
  magnitude inside the 2.5 % requirement. Raising `k_probes` buys no measured
  variance reduction while multiplying the threaded-state path's cost by up to ~50x.

Both constants are now config keys (`k_probes`, `step_subsample`) and participate in
the run identity hash whenever `mu_j > 0`; when `mu_j == 0` they are canonicalized to
their defaults before hashing, because the penalty branch is not evaluated at all.

Measured cost of the amended calibration track: **2.71 CPU-hours** at
(`k_probes = 2`, `step_subsample = 1.0`, 200 updates). This category is published
alongside the core grid and never counted in it (header taxonomy), so `README.md`'s
core-grid budget is unaffected.

The μ_J penalty is no longer evaluated on the validation split. It was recomputed
there every epoch, unused, at 23-38 % of epoch wall-clock, and it advanced the global
RNG that the training-time probes draw from. Consequence, declared: `val_loss` in
`metrics.csv` is now TF + FR + VICReg only and never includes `mu_j * jac_pen`.

### 10.3 Declared metric (new)

The calibration margin is the median of the **metric-normalized** operator norm of
the dense one-step Jacobian on the **complete** augmented state `q = (h, m)`:

    sigma_max( S_out^-1 . J_t^{q,excl} . S_in )

applied **identically at training time and at audit time**. `J^{q,excl}` is the
Jacobian with the previously declared immutable identity carry-forward rows excluded
(section 5, "Identity-block exclusion"), and nothing else: every memory input column
and every transformed or newly written memory output row stays constrained.
Constraining a sub-block of the Jacobian remains forbidden (`REGLES.md` section 7 and
the Wave 4a gold counter-examples), as does using the spectral radius as a
certificate.

**Reference scales.** `S` is a **frozen** diagonal matrix of centered per-coordinate
standard deviations, estimated in float64 on the retained phase-OFF, seed-0,
OU-regime teacher-forced validation trajectories, **before** the μ_J sweep runs, and
held fixed for the whole sweep. It is never re-estimated from a penalized run.

**Growing memory.** For a family whose memory grows across the rollout, `S` is a
position-indexed family `{S_t}`, not one constant-length vector. Cache coordinates
are labelled structurally (layer, K/V, head, absolute position, feature); a
coordinate's scale is estimated across reference episodes at the step that creates
it, and reused unchanged while that coordinate persists. A coordinate that does not
yet exist needs no scale. The frozen reference rollout must cover every position that
is trained or audited.

**Indeterminacy, not repair.** A position beyond the reference rollout, a zero
reference variance, or a non-finite one yields the verdict **"metric indeterminate"**,
reported verbatim. No epsilon floor is substituted and no scale is extrapolated —
the same discipline the repo already applies to a non-determinable readout.

**Declared gauge domain.** Under `q' = D q` for any finite invertible diagonal `D`
preserving persistent coordinate identity, `S' = |D| S`, so the normalized Jacobian
changes only by left/right sign matrices and its singular values are unchanged. The
declared invariance is therefore over **all finite invertible diagonal
reparameterizations**, verified under signed and extreme rescalings — the same
invariance class, and the same declared limitation (not invariant under shears or
rotations), already adopted for the section 5 P1a endpoint. Full covariance whitening
is not adopted: it is singular or impractical for a growing cache. A true-state
pullback is not adopted: it is undefined on `m` and so cannot certify the complete
augmented state.

### 10.4 Pre-registered acceptance test (new)

The question this amendment leaves open is whether an ordinary monotone penalty with
a swept weight suffices once the budget, the estimator and the metric are repaired,
or whether a mechanism with a fixed point at the target (a hinge, or a dual/Lagrangian
controller) is required. That question is decided by the following test, **fixed here
before the amended calibration runs**, so that it cannot be chosen after seeing the
numbers.

Per family, **no controller is required only if all three hold**:

1. the four primary-grid dense medians are **non-increasing** in `mu_j` (equality
   allowed);
2. section 5's selection rule plus its single deterministic refinement returns an
   in-band candidate;
3. that retained candidate's 200-update OU-regime achieved-margin audit is also in
   `[0.90, 1.00]`.

A flat curve already inside the band passes. A flat curve outside it fails criterion
2. No post-hoc widening of the grid, the band, or the tolerance is permitted. Any
family failing any criterion enters the "controller required" state, whose resolution
is itself a further dated amendment — never a choice made inside the procedure.

### 10.5 Status of the runs already executed

The 24 selection and calibration runs executed under the pre-amendment constants keep
their category — **pre-registered selection & calibration runs** — and are published
with their trace, marked as executed under the superseded constants of section 5, as
that section already prescribes for superseded intermediate calibrations. They are
not retroactively reclassified as exploratory, and their "band not reached" verdicts
are not silently deleted: they are the measurement that produced this amendment.

---

## 11. Amendment 2 — withdrawal of P1(b) (dated 2026-07-31)

This amendment **withdraws prediction P1(b)** — "at fixed constraint, the explicit
spectral parametrization beats the generic families" — from the flagship's claim set.
It also leaves **P1(a) unresolved** and puts **P3** under a dependency audit, because both
are stated in terms of an ON arm that no longer has an admissible definition; the exact
status of every prediction is in 11.5. Nothing already measured is deleted; the runs, the
verdicts and the reasoning that led here stay published.

It is written and committed **before** the study that replaces it runs (section 11.6),
as sections 5 and 10.4 require of any resolution of a terminal calibration state.

### 11.1 Withdrawn, not falsified — and why the distinction is the whole point

P1(b) compares architectures **at a fixed constraint level**. That clause is
load-bearing: without a constraint level held common across families, "the spectral
parametrization wins" has no defined comparison. The band `[0.90, 1.00]` on the
calibrated margin was the operationalization of "fixed" (section 5, honesty
declaration (c)).

Three of four families never reached that band. There is therefore **no level at which
the comparison was ever made** — no measurement exists that could confirm P1(b) or
refute it. The prediction is **unidentified by the design as executed**.

Declaring P1(b) false would be a stronger claim than the data supports, and a
dishonest one: we did not observe generic families matching or beating the structured
one at a common constraint. We observed that a common constraint was never achieved.
Withdrawal is the accurate verdict.

### 11.2 What was actually tested

1. **Pre-registered calibration, 24 runs** (section 5, original constants): "band not
   reached" for 3 of 4 families; medians nearly insensitive to μ_J across four decades.
2. **Diagnosis by offline probes** on the retained checkpoints (section 10.1): the
   calibration trained **10 optimizer updates** per candidate (the trainer takes one
   `optimizer.step()` per epoch); `step_subsample = 0.25` discarded three quarters of
   the steps; and `sigma_max` was read **in undeclared coordinates** — the same dense
   Jacobians give 1.0439 raw and 22.94 whitened for `min_ssm`, a factor-23 swing in the
   verdict under an equally defensible coordinate choice.
3. **Amendment 1** (section 10): 200 optimizer updates, `step_subsample = 1.0`, and a
   declared metric `sigma_max(S_out^-1 J^{q,excl} S_in)` with `S` frozen on the
   phase-OFF, seed-0, OU arm.
4. **Repaired sweep**, the only campaign valid for this decision:

   | family | μ_J=1e-3 | 1e-2 | 1e-1 | 1 | 10 | verdict |
   |---|---|---|---|---|---|---|
   | `mlp` | 1.0301 | **0.9995** | 0.9049 | 0.8755 | — | selected (μ_J = 0.01) |
   | `transformer` | 7.2022 | 6.3212 | 5.9156 | 5.8217 | 5.8315 | band not reached |
   | `min_ssm` | 1.3299 | 1.1551 | 1.0767 | 1.0644 | 1.0639 | band not reached |
   | `koopman_ssm` | 1.0326 | 1.0340 | 1.0348 | 1.0348 | 1.0347 | band not reached |

   **One family, at one seed, reached the band.** The μ_J axis is not inert: at seed 0,
   `mlp` and `min_ssm` show an **observed decreasing response** where 10 updates gave
   medians identical to six digits. That is a single-seed observation, not a replicated
   family-level result, and it is not reachability of a common target.

**Declared defect of this campaign.** The repaired sweep ran under an amendment
(section 10) that was written but **not yet committed** at run time. It is therefore
**not pre-registered evidence**. It is reclassified here as a **post-hoc repair study**
and published as such. Committing section 10 now does not cure this retroactively, and
this amendment does not pretend otherwise.

### 11.3 What the calibration failure exposed

This section is deliberately **not** titled "why it did not work". Only item 2 was ever a
candidate numeric mechanism, and it has since been refuted. What follows is a list of what
the failure exposed, each with its evidential status stated.

1. **DESIGN LIMITATION — cross-family physical comparability was never validated.** What
   is established: each family is measured against its own OFF-derived reference; the
   declared whole-experiment diagonal invariance does hold; physical equivalence across
   those four references was never checked. What is **not** established is that the four
   measurements are intrinsically non-comparable, nor that normalization *caused* the band
   failures — that would need an external common-space validation. Section 5's honesty
   declaration (c) called this "the same measured contraction target"; that was an
   operational convention presented as if it were comparability.

   **Measured consequence** (offline probe, 2026-07-31, `_private/research/
   gauge-endpoint-unblock/probe_ab_report.json`, step-0 self-check reproducing the stored
   medians at absolute error 0.0): re-estimating the gauge per cell instead of freezing it
   flips a pre-registered acceptance criterion. `min_ssm` under the frozen gauge is
   monotone decreasing (1.3299 → 1.0639, criterion 1 satisfied); under its own per-cell
   gauge it is **non-monotone** (1.6209 → 2.0692 → 1.6653 → 1.5946 → 1.6096) and at a
   different level entirely. A verdict that moves with a normalization choice declared to
   be a convention is the concrete cost of the unvalidated construct.
2. **REFUTED — spectrum redistribution.** The general possibility is DERIVED: training
   minimizes a normalized Frobenius-squared estimate while acceptance reads the top
   singular value, and singular values (1,1) → (1.1, 0) reduce `||.||_F^2` from 2 to 1.21
   while raising `sigma_1` from 1 to 1.1. The `koopman_ssm`-specific account is **refuted
   by matched measurement**: on identical Jacobians, states, normalization and
   aggregation, its Frobenius norm **rises** with `sigma_max` (2.8097 → 2.8154) and its
   stable rank is flat (7.4028 → 7.4031). No redistribution occurs; the whole operator
   grows slightly. For `mlp` and `min_ssm` both norms fall together.
3. **ESTABLISHED construct limitation — the transformer's audited object.** The full
   augmented Jacobian has `sigma_max >= 1` by construction (the KV cache's identity
   carry-forward rows). Section 5 excludes those rows, so the transformer's measured
   quantity is a **new-write sensitivity**, not contraction of the complete augmented
   state. This is a limitation of the construct, **not** an explanation of the
   transformer's numeric non-response.
4. **ESTABLISHED confounding — `koopman_ssm`'s ON arm.** Only this family changes two
   things at once between OFF and ON: `mu_j > 0` **and** the parametrization (`alpha` free
   → `alpha = -exp(nu)`). Any ON/OFF contrast for it is a package effect. This is
   confounding, **not** an explanation of the positive-dose curve.

**What the measurement leaves standing, unexplained:** for `koopman_ssm` the penalty is
**inert**. Four decades of `mu_j` move every norm of its Jacobian by roughly 0.2 %, upward,
under both the frozen and the per-cell gauge, while the same penalty moves `mlp` from
1.0301 to 0.8755 and `min_ssm` from 1.3299 to 1.0639. A candidate — untested — is the
gradient-ratio measurement taken earlier for another purpose: `||grad P|| / ||grad L_pred||`
at init is 1.4 for `koopman_ssm` against 16 (`mlp`), 15 (`transformer`) and 314
(`min_ssm`). Whether low penalty leverage explains the inertia is an open question, not a
claim.

**Explicitly refuted, so it is not re-dug:** the hypothesis that freezing `S` on the OFF
arm breaks the declared invariance, or mechanically drives the margin upward. Two
independent grounds. Algebraic: under `q' = D q` the normalized Jacobian is conjugated by
sign matrices, so singular values are preserved, and a fixed `S` carries no monotone sign.
Measured: with each cell's own gauge `koopman_ssm`'s curve is unchanged to three decimals
(1.0327 → 1.0350 → 1.0347), no band verdict changes, and the leading singular direction is
**not** aligned with the coordinates where `S_cell / S_OFF` differs most (median
participation <= 0.0034 against a 0.25 uniform baseline). Note the asymmetry with item 1:
the gauge choice does not explain `koopman_ssm`, and it does move `min_ssm`'s verdict.
Those are compatible and both are measured.

**Declared exclusion:** the transformer was excluded from this probe on cost grounds (its
dense growing-cache Jacobian audit did not finish). Its numbers are therefore unexamined,
and that is a declared gap, not a silent omission.

### 11.4 What would make P1(b) pass, and why none of it is done here

- **(i) A dual/Lagrangian controller on the audited quantity.** One candidate, conditional
  on the target being feasible at all — a controller cannot make an infeasible target
  feasible, and a direct spectral objective or an architectural change could alter
  feasibility instead. It must control the **median of `sigma_max`** — what acceptance
  reads — not the mean Frobenius norm the current penalty minimizes; a naive two-sided
  hinge is already refuted (for stochastic `X`,
  `grad E[(L-X)+ + (X-U)+] = E[(-1{X<L} + 1{X>U}) grad X]`: both tails fire even when
  the true value is in band, because a zero gradient **per sample** is not a zero
  gradient **in expectation**). It therefore needs differentiable top-singular
  estimation, a median surrogate, multiplier schedules and feasibility handling.
  **Not done:** this is a mechanism change, with its own calibration, its own
  pre-registration and its own failure modes. It cannot be built, calibrated and
  validated inside the D1 window, and building it under deadline is precisely how one
  ships a result whose flaw is visible on the first page.
- **(ii) A per-family band.** Reaching a per-family target abandons the cross-family
  common level — which is the one thing P1(b) requires. **Not done:** it does not
  rescue the prediction, it removes its subject.
- **(iii) Widening the band until all four families fit.** **Not done — prohibited.**
  Section 10.4 forbids post-hoc widening of the grid, the band or the tolerance, and
  it is dishonest by construction.
- **(iv) A metric comparable across families by construction.** The candidate examined
  was a true-state pullback through the per-run linear readout already used by the P1a
  endpoint. **Rejected on the merits, not on cost:** the readout is non-injective, it
  ignores its own kernel, it ignores `m` entirely and the `h`↔`m` couplings; a
  "physical block plus memory term" that repaired those omissions would reconstruct the
  full augmented Jacobian anyway. The repo's own counter-example
  (`tests/audit/test_doctrine.py`) applies directly.

The deeper reason for stopping: every further run spent chasing the band is spent on
the **instrument** rather than on the **phenomenon**. The replacement study measures
the phenomenon.

### 11.5 What survives, and what is reclassified

**Status of every prediction** (corrected 2026-07-31 after review round 2 — an earlier
draft of this section wrongly asserted that P1(a) "stands unchanged"):

- **P1(b)** — withdrawn as unidentified.
- **P1(a)** — **UNRESOLVED**, not surviving. As published it is an ON-versus-OFF
  contrast, and no admissible ON cell exists: the calibration that was to define it
  terminated for three families, and section 5 forbids resolving a terminal state inside
  the procedure. Section 11.6 proposes a different, multi-dose estimand but does not yet
  supply a falsification rule for it. Until that rule is written and committed, P1(a) is
  open. For `koopman_ssm` any future version must additionally declare that it measures a
  package, not the penalty alone.
- **P3** — **DEPENDENCY AUDIT REQUIRED.** It is stated in terms of "the constrained
  predictor", which is the same undefined ON arm. It cannot be assumed untouched.
- **P2, P4, P5** — retained **only** insofar as their executable arms do not depend on the
  unresolved ON selection. Each must be checked before it is run, not assumed.

**Reclassified.**
- The repaired μ_J sweep → **post-hoc repair study** (11.2), published with its trace.
- **The OFF grid → post-hoc diagnostic, pending re-run.** An earlier draft called it "a
  complete and valid measurement"; that was false on the repo's own rule. The 40 cells ran
  at 17:14 on 2026-07-31; the amendment authorizing their 200-update budget was committed
  at 18:48 the same day. Section 11.2 states that later commitment does not retroactively
  pre-register a run, and `REGLES.md` requires a post-tag budget change to be committed
  before execution. The rule applies to this repo's own convenience too: the artifacts are
  preserved as diagnostics and the grid is re-run against the committed amendment **in a
  separate result root**, so sentinels cannot resume the pre-amendment artifacts.
  Seventeen minutes of compute is not a reason to carry a provenance defect.
  One cell failed (`action_conditioned` / `min_ssm` / seed 4, non-finite training loss at
  update 46) — see 11.7, and note that one cell out of 40 supports the cell-level fact and
  nothing more.
- The four core ON runs already on disk (`mlp` μ_J = 0.01 and `koopman_ssm`
  μ_J = 0.1, seed 0, both regimes) → **protocol-violation traces, never core
  evidence**. Two independent reasons: they were trained against the **raw** penalty
  because the factorial runner never supplied the frozen gauge, and `koopman_ssm`'s
  μ_J = 0.1 was written into the manifest from a **terminal "band not reached"**
  verdict by a patcher that ignored the verdict. Both defects are fixed in the same
  commit as this amendment; the corrupted manifest value itself is cleared by hand,
  because the repaired patcher refuses to write but does not rewrite what it already
  wrote. The runs are kept and marked, not deleted.

### 11.6 What replaces P1(b) — a dose-response study (declared, not yet run)

A **secondary** study, committed before it runs. It is **not** a rescue of P1(b) and
cannot be one: without a common estimand there is nothing for a cross-family comparison
to be *of*. What it can legitimately deliver is exactly three things — a **within-family
dose response**, a **sensitivity of every conclusion to the normalization choice**, and
**evidence that families respond differently to the same intervention**.

**11.6.1 Design.**
- Fresh seeds (5-9), disjoint from the calibration seeds. **Seed is the inferential
  unit.**
- `mu_j = 0` plus an exact, enumerated dose grid — the doses are written into the
  committed manifest, never "the full declared grid" resolved later.
- **Learning rate is held FIXED within each (family, regime) across all doses.** An
  LR re-selected per dose turns the curve into an optimized envelope over
  `(mu_j, LR)`, not a dose response: identical tuning budgets establish fairness, they
  do not isolate `mu_j`. If a future variant does tune per dose, it must be named
  **"dose-conditioned optimized performance"** and report the selected LR as a second
  treatment component.
- Jointly reported outcomes per cell: achieved margin (both metrics, see 11.6.2),
  true-state FR drift, validation loss. The margin is a **measured covariate**, never a
  controlled quantity.
- `koopman_ssm`: the arms are `(free alpha, mu_j = 0)`, `(constrained alpha, mu_j = 0)`
  and `(constrained alpha, mu_j > 0)`. This is a **path decomposition, not a 2×2**: the
  `(free alpha, mu_j > 0)` cell is absent, so **no structure-by-penalty interaction may be
  estimated or claimed**. Either that fourth arm is added, or the claim stays limited —
  declared here so the limitation cannot be forgotten at analysis time.
- The transformer's excluded-identity quantity is reported under its accurate name,
  **new-write sensitivity**, never as contraction of the complete augmented state.

**11.6.2 Metric — decided here, asymmetrically.**
Both metrics are reported for every cell, but they are **not interchangeable endpoints**:

- **Primary: frozen `S_OFF`.** One reference per `(family, regime, seed)`, estimated from
  that seed's own `mu_j = 0` teacher-forced validation trajectory, **before** any
  positive-dose training, and reused unchanged for every dose in that paired chain. For
  `koopman_ssm` the unconstrained `mu_j = 0` arm is the path reference — changing the
  reference at the structural-only arm would destroy the common meaning along the path.
- **Mandatory sensitivity: per-cell `S_cell`.** Offline **audit only**. It may not train,
  may not select a dose, may not define overlap, may not gate inclusion.

Pre-declared before any positive-dose run, so that nothing can be chosen after seeing
numbers: (1) the exact doses; (2) the gauge-producing arm, split, trajectories,
coordinates, cache positions, centering, float precision and zero-variance handling;
(3) a **digest of each frozen gauge recorded before positive-dose execution**;
(4) identical Jacobian samples and aggregation order on both audits; (5) seed as the
inferential unit, and the interval/test procedure; (6) `S_cell` is audit-only as above;
(7) both metrics appear for every cell; (8) **interpretation rule** — if the direction or
the family ordering changes between the two gauges, the result is reported as
**"metric-sensitive"** and no robust directional claim is made; (9) no cross-family
superiority claim under either normalization; (10) fixed-LR dose response versus
optimized envelope, named; (11) the exact `koopman_ssm` contrasts and the divergence
handling.

This is what stops "report both" from becoming "keep whichever looks better": only
`S_OFF` answers the primary fixed-reference question, and `S_cell` answers a separately
named sensitivity question. The rule that already bit — `min_ssm`'s monotonicity flips
between the two (11.3, item 1) — is precisely the case clause (8) exists for.

**11.6.3 What may never be claimed from this study.**
**No cross-family superiority claim, under either normalization, including inside
overlapping achieved-margin support.** Numeric overlap does not repair the construct: two
family-specific, post-treatment quantities can share values without sharing physical
meaning, and the transformer's quantity is explicitly a different object. No
interpolation between dose points. The words "Pareto frontier" are not used. The single
most likely empirical outcome is disjoint support — the observed ranges are already
separated (transformer ~5.8-7.2 against ~0.9-1.3 for the others) — which would leave a
useful within-family result and no frontier at all. That outcome is anticipated here, not
discovered later.

### 11.7 Divergence policy for core cells (new)

Section 5 had no policy for a core cell whose training diverges. Declared here, before
the figures are read.

A cell whose loss becomes non-finite **aborts without success artifacts** (already the
trainer's behaviour) and is recorded as a **divergence** — a result about that
(family, regime, seed, LR), on the same footing as a "readout indeterminate" verdict.

**It must be durable, and that requires artifacts, not prose.** A partial run directory
with no completion sentinel is otherwise indistinguishable from a cell that never ran,
and a resume would silently retrain it. Therefore: a terminal `failure.json` is written
atomically into the run directory carrying the run identity, the epoch/update index, the
reason, the offending values and the exception class; the index classifies such a cell as
**`divergence`**, distinct from both `ok` and `missing`; resume **preserves** it and does
not retrain; strict-mode rendering names it.

**Replay is permitted, substitution is not.** An exact diagnostic replay of the same
configuration, in a **separate namespace**, is allowed and encouraged — an absolute
"never re-run" rule would create an incentive not to distinguish a genuine numerical
instability from infrastructure corruption. What is forbidden is substituting a different
seed or a different LR into the primary result, or letting a replay overwrite the
original artifact.

**No determinism claim is made.** An earlier draft asserted that re-running reproduces the
divergence. The runs are seeded, but deterministic algorithms across execution
environments have not been established in this repo, so that claim is withdrawn until it
is tested.

**Inference from a divergence is bounded.** A diverged cell supports the cell-level fact
and nothing more. One cell out of forty is not evidence that a family is generally
unstable; reporting "39 complete, one cell failed" without inference is the correct
treatment.

---

## 12. Amendment 3 — endpoint conventions and positive-control gate (dated 2026-08-02)

### 12.0 Scope, and why it is committed now

The refocus replaces the architecture × extrinsic-constraint design with a crossed
factorial **backbone × structure** (3 × 2). That design needs two things this contract did
not yet declare: the **endpoint** (how a rollout is scored) and the **positive-control
gate** (what "this cell learned" means). Both are declared here, in full, **before the
module that computes them is written** and long before any weight of any predictor exists.

This is the ordering section 5 and section 10.4 require of any threshold, and it is the
ordering whose violation cost P1(b): a level chosen after seeing numbers is not a
pre-registered level.

Nothing here reopens sections 1–11. Section 11's divergence policy is used as written.

### 12.1 The normalized state error `e_h`

For a batch of `B` evaluation episodes, a true state of dimension `d`, and per-coordinate
standard deviations `σ_i^train` computed on the **train split only** and frozen before any
evaluation:

```
e_h = sqrt( (1 / (B·d)) · Σ_b Σ_i ( (x̂_{b,h,i} − x_{b,h,i}) / σ_i^train )² )
```

`x̂` is the free-running rollout decoded to true-state coordinates through the frozen
readout; `x` is the dataset's true state, generated by the RK4 oracle. `h = 1 … H` indexes
physical steps after the initial state; `h = 0` is not scored (the rollout starts from the
true state, so `e_0 = 0` by construction and would fabricate one free step).

**Provenance.** This is the NRMSE of Vlachas et al. (2020), *Backpropagation algorithms and
Reservoir Computing in Recurrent Neural Networks for the forecasting of complex
spatiotemporal dynamics*, Neural Networks 126:191–217, arXiv:1910.05266 —
`NRMSE(ô) = sqrt(⟨(ô − o)²/σ²⟩)` with `σ` the standard deviation of **each state
component** and `⟨·⟩` the state-space average. We add the episode average inside the root
and take `σ` on the train split rather than in time; the normalization is otherwise the
published one. Per-coordinate normalization — not a global RMS amplitude — is what makes
the quantity invariant to a change of units per coordinate.

### 12.2 The VPT (horizon-to-threshold)

```
VPT_τ(e) = max { h ∈ {0, 1, …, H} : e_j < τ for all j ≤ h }
```

the **length of the maximal valid prefix**. Consequences, each declared rather than left
to an implementation:

- **Validity is strict `<`.** A step with `e_h ≥ τ` is not valid. This is the published
  definition (`NRMSE(o_t) < ϵ, ∀ t ≤ t_f`) and we adopt it unchanged. It has no effect on
  any measured run — exact equality has probability zero in floating point — but it fixes
  the value of the hand-built test vector, which is the only place the convention is
  observable.
- **`VPT = 0` when `e_1 ≥ τ`**: zero valid steps, not one. A "first-crossing index" would
  report 1 here and read as "one valid step".
- **Never crossed on the available horizon**: the value is `inf` and the status is
  `censored`, with the censoring horizon `H` recorded alongside. This is right-censoring:
  we did not observe when validity ended. Substituting `H`, `H+1`, or any other finite
  number fabricates a measurement that was not taken, and `H` in particular is
  indistinguishable from a crossing exactly at `H`.
- **`τ = 0.5`** — the value used by Vlachas et al. on the same rooted NRMSE. The paper
  states `ϵ = 0.5` without derivation; we adopt it for comparability with the reference,
  and this contract says so rather than implying a justification that the source does not
  contain. A `0.4` threshold also circulates, applied to the **squared** normalized error;
  it is a different convention (`√0.4 ≈ 0.63` on our scale) and mixing the two is an error
  this paragraph exists to prevent.
- **The Lyapunov normalization stays outside the VPT.** Vlachas reports `VPT/Λ₁` because
  the systems are chaotic. The pendulum and the coupled oscillators are not: `Λ₁ ≤ 0` and
  the division is meaningless. The module returns a step count; the division by a
  system-specific time scale is the caller's business, which is exactly what makes the
  extension to Lorenz a wiring change and not a rewrite.
- **Module contract**: the reducer consumes an array whose last axis is the horizon
  (`(…, H)`), an already-normalized error curve, and a threshold; it returns a value and a
  status. It performs **no** normalization and **no** aggregation over episodes or seeds.
  Whether the primary statistic is the VPT of the episode-averaged curve or an aggregate of
  per-episode VPTs is **not decided here** — see 12.5.

### 12.3 The positive-control gate

A cell-seed enters the comparison only if **both** bounds hold:

```
G  =  [ e_{H_gate} < min(e_{H_gate}^pers, e_{H_gate}^lin) / γ ]   ∧   [ e_1 < τ₁ ]

  baselines, same coordinates, same episodes, same σ^train:
      x̂_h^pers = x_0                     x̂_h^lin = x_0 + h (x_0 − x_{−1})
```

Both comparisons are **strict `<`**.

| symbol | value | status |
|---|---|---|
| `H_gate` | **25** steps, common to every system and regime | project judgment |
| `γ` | **5** | project judgment |
| `τ₁` | **3 · e_recon**, per stratum | project judgment (the factor); measured (the floor) |

**`H_gate = 25`, common.** Measured on the pendulum at `dt = 0.02` (default IC regime,
`u = 0`): the persistence baseline rises to `2.07` at `h = 25` and then saturates and falls
back (`1.74` at `h = 50`) because the pendulum oscillates. `h = 25` is therefore where the
trivial reference is at its most demanding; beyond it the reference stops rising while the
model's error keeps growing, which makes a larger horizon *mechanically* harder without
being more informative. It is common to all cells because the whole point of the 3 × 2
design is that the six cells are compared on one identical yardstick — a per-system horizon
would leave a defensible objection that the yardstick was chosen per arm. A silent
`min(H, T)` is forbidden; the horizon is this number or a committed amendment.

**`γ = 5`.** This factor cannot be derived. The only measurement that would inform it is the
performance of the models, and looking at that before fixing the threshold is precisely
what pre-registration forbids. It is a judgment, and it is recorded as one. The reasoning:
the persistence baseline saturates near `2`, so `γ = 2` admits a cell whose error is still
about one state standard deviation — a cosmetic gain — while `γ = 10` demands
`e_25 ≲ 0.21`, which turns a **control** into a **performance criterion** and risks emptying
the grid of the cells the thesis exists to compare. `γ = 5` (`e_25 ≲ 0.41`) excludes the
cosmetic gain while leaving the gate a control. The gate's job is to prevent comparing four
equally bad models; it is not a state-of-the-art bar.

**`τ₁ = 3 · e_recon`.** `e_recon` is the reconstruction error of the frozen measurement
chain — `readout(encoder(o_t))` against `x_t`, in the same normalized coordinates, on the
evaluation split — computed once per stratum at milestone D2, **with zero predictors
trained**. It is the score a *perfect latent predictor* would obtain: a predictor that
outputs exactly the encoder's embedding of the next true state inherits the readout's own
error and nothing else. `τ₁` therefore reads as "the predictor is within a factor `k` of the
best score the measurement chain permits".

- Why not the RK4 oracle, as earlier drafts said: **measured, the RK4 discretization floor
  at `dt = 0.02` is `4.0e-8`** in these normalized coordinates at one step (`1.5e-6` at
  50 steps), against a persistence baseline of `0.134` at one step. Six decades separate
  them. A bound referenced to the integrator would be unreachable by any float32 network and
  would classify all six cells as "does not learn" by construction. The integrator is
  published here as context — proof that it is never the binding constraint — not as the
  bound.
- Why `k = 3`: `k = 1` demands that a trained model match the ideal predictor exactly, which
  no finite-sample model does, so the gate would be unpassable; `k = 10` permits an order of
  magnitude worse than the ideal, which in some strata exceeds the persistence baseline
  itself and stops being a ceiling. On a quantity whose natural scale is logarithmic,
  `√10 ≈ 3` is the midpoint of that interval. **No standard exists for this factor**: a
  literature search for an acceptance threshold on prediction error relative to a
  reconstruction or noise floor returns the general practice of pre-registered go/no-go
  plausibility gates but no conventional value. This contract states the absence rather than
  borrowing authority it does not have.
- `e_recon` is an **empirical** floor, not a proven lower bound: a predictor could in
  principle exploit a bias of the readout and score below it. The claim made here is only
  that it is the ideal latent predictor's score, which is what `τ₁` is calibrated against.
- Because the encoder and readout are shared and frozen across all six cells of a stratum
  (section 12.4 of the refocus design), `e_recon` is a **common** floor. A weak encoder
  makes the gate more permissive for every cell equally; it cannot advantage one arm.

**Pre-registered sanity condition, checked at D2 before any predictor is trained.** For each
stratum, verify `3 · e_recon < e_1^pers`. If it fails, the ceiling is looser than the trivial
baseline and the gate is meaningless for that stratum: that fact is **published**, the
stratum is reported as ungated, and **the numbers are not adjusted**. Discovering that a
declared threshold does not discriminate is a result; silently retuning it is what this
contract exists to prevent.

**No post-hoc loosening.** If every cell fails `τ₁`, the published verdict is "no cell learns
at the declared level", with the measured values. `γ`, `H_gate`, `τ₁` and `k` are not
revisited after any number is seen. Changing one requires a dated amendment committed before
the run it authorizes, and the prior reading stays published beside the new one.

### 12.4 Seeds, and what a failure must say

**All five seeds must pass individually.** No median, no `k`-of-5, no reducer. A cell with
one failing seed is published as failing, named, and takes no cell-level average. This
follows the rule already in force — "a cell that fails is published as *does not learn*,
named, never averaged, never silently excluded" — and it is the only option that leaves no
room for a post-hoc choice of reducer.

**A seed whose run terminates in `divergence` (section 11) does not pass the gate**: it
produced no eligible `e_h` curve. Section 11 governs everything else about it — the terminal
`failure.json`, the `divergence` classification distinct from `ok` and `missing`, the
permission to replay diagnostically in a separate namespace, the prohibition on substituting
a different seed into the primary result, the withdrawn determinism claim, and the bound on
what a divergence licenses as inference.

**A failure is reported with its numbers, never as a bare label.** The terminal artifact of
a non-passing cell-seed carries at minimum:

- which bound failed — lower, upper, or both;
- the measured `e_1`, `e_{H_gate}`, `e^pers_{H_gate}`, `e^lin_{H_gate}`, `τ₁` and `e_recon`;
- the **margin ratios** `e_{H_gate} · γ / min(e^pers, e^lin)` and `e_1 / τ₁`, so the distance
  from passing is readable, not inferred;
- the full `e_h` curve and the VPT value with its status;
- for a divergence, the update index, the offending values and the exception class (already
  required by section 11).

"It did not work" is not a result. "It missed the lower bound by 1.8× while passing the
upper bound, at these values" is.

### 12.5 What this amendment does **not** fix

Declared open, to be committed before the milestone that consumes each:

- **Episode aggregation of the VPT** — VPT of the episode-averaged curve, or an aggregate of
  per-episode VPTs. The two differ, and the second is closer to the published metric. Fixed
  at D0. The module contract in 12.2 is deliberately neutral so that this choice is a caller
  decision and not a rewrite.
- **Treatment of censored VPTs in cell-level statistics** — how an `inf` enters a mean or a
  figure. Fixed at E0.
- **Gate budget** — epochs/updates, batch, device, wall-clock ceiling. Fixed at D0.
- **Whether the D pilot may become part of E.** Fixed at D0; the standing recommendation is
  no.

### 12.6 Honesty declaration

Required by section 5's precedent, which is where an operational convention was once
presented as a physical fact.

- **Taken from the literature, cited, unmodified**: the shape of `e_h` (per-coordinate
  `σ` normalization inside a root), the VPT as a maximal valid prefix with strict `<`
  validity, and `τ = 0.5`. Source: Vlachas et al. 2020, arXiv:1910.05266. The source states
  `ϵ = 0.5` **without justification**; adopting it buys comparability with a widely cited
  reference, not a derivation, and this contract does not pretend otherwise.
- **Project judgments, not derived from anything**: `γ = 5`, `H_gate = 25`, `k = 3`. Each is
  stated above with its reasoning and with the consequence declared in advance if it turns
  out to be the wrong choice.
- **Measured, and published as context rather than as a bound**: the RK4 discretization floor
  (`4.0e-8` at one step, `1.5e-6` at fifty, `dt = 0.02`) and the persistence baseline
  (`0.134`, `1.20`, `2.07`, `1.74` at `h = 1, 10, 25, 50`). Both were measured on the
  pendulum in the default IC regime with `u = 0` and `σ` taken over initial conditions —
  an order-of-magnitude probe, not the final `σ^train`. The six-decade separation they
  establish does not depend on that reservation; no threshold in 12.3 is set from them.

## 13. Amendment 4 — the structure axis becomes architectural (dated 2026-08-04)

### 13.0 Scope, and why it is committed now

Phase C builds the arms this amendment describes. Rule R4 requires an amendment to be
committed **before** the execution it authorizes, and this repo has already paid for
breaking that rule once (11.5: the OFF grid ran 1h34 before its amendment). Nothing in
phase C trains anything, so this amendment authorizes no run. It fixes three things that
phase C's code would otherwise decide silently: what a structured arm *is*, what
`structure: on` *means*, and which result root the eventual 3x2 grid writes to.

Three decisions below are the project owner's, taken 2026-08-04, not derived from any
measurement: the strong-factorial option, a free `alpha` on the ON cells, and a fresh
result root for the full grid.

### 13.1 Decision C0 — three structured arms, one shared transition

The open decision was: build a third structured arm under the common wrapper (**strong
factorial**), or keep the historical `koopman_ssm` class as the SSM structured cell and
weaken the claim. **The strong factorial is chosen.**

Every structured arm runs literally the same transition; only the context producer
`kappa_b` differs between MLP, transformer and SSM:

```
h_0     = psi(z_0)
(c_t, m_{t+1}) = kappa_b(h_t, m_t; {(h_s, u_s)}_{s<t})
(alpha_t, omega_t, vec B_t) = theta_0 + W_theta c_t
G_{t,i} = [[alpha_{t,i}, -omega_{t,i}], [omega_{t,i}, alpha_{t,i}]]
A_bar_t = blkdiag_i exp(Delta G_{t,i})
B_bar_t = blkdiag_i (int_0^Delta exp(s G_{t,i}) ds) B_t
h_{t+1} = A_bar_t h_t + B_bar_t u_t
z_hat_{t+1} = pi(h_{t+1})
```

Fixed with it: `u_t` is excluded from `c_t` (otherwise the cell stops being linear in the
action and the arms stop being one treatment); the head `theta_0 + W_theta c` is affine and
unique, so a neutral context is exactly `W_theta = 0`; the ZOH is the exact input integral,
not a directly parametrized discrete `B`; and the lifted width `r` is identical across the
three structured arms.

**The reason this option was chosen over the cheaper one.** Only under the common wrapper is
"identical treatment, three backbones" a true sentence. Under the minimal option, an observed
gap between `(ssm,on)` and `(mlp,on)` would confound backbone with transition machinery, and
the claim would have to be weakened to "three models that contain 2x2 blocks" — which is not
the claim this project exists to make.

**The historical class is retained, not reinterpreted.** `KoopmanSSMPredictor` stays
constructible and loadable exactly as it is, under the family name `koopman_ssm`, so every
published artifact keeps its constructor. It is **never** a cell of the 3x2 factorial. The
registry therefore holds seven families while the factorial uses six; that asymmetry is
deliberate and a test enforces it — a factorial arm naming `koopman_ssm` must be refused
before any run.

### 13.2 What `structure: on` now means, and what it stopped meaning

Until this amendment, `structure: on` meant `koopman_ssm(constrained=True)`: the stability
axis constrained to `alpha = -exp(nu) < 0`, contractive by construction
(`koopman_ssm.py:78`). **From this amendment, `structure: on` means the structured
transition above — lift, 2x2 rotation-scaling blocks, exact ZOH, projection — with `alpha`
FREE.**

`alpha` is free deliberately. The OFF arms have no `alpha` at all, so constraining it on the
ON arms would make one axis carry two changes at once: structured transition AND contraction.
A gap measured across such an axis is not attributable to either. That confound is exactly
what the mu_J withdrawal (section 11) cost this project the most to identify, and it is not
being reintroduced under a new name. Stability is therefore **measured** on the ON cells, not
imposed.

**What this invalidates: nothing that exists.** Measured before deciding: every executed core
cell is an OFF cell (30 of them, `results_amended/pendulum_core/`), and no `structure: on`
cell has ever been run anywhere in this repo. The redefinition changes no published number.

**What it does withdraw.** `pendulum_core_partial.yaml` declared 40 cells, of which the 10
`(ssm, on)` cells were never executed. Those 10 cells named the OLD definition of ON. They are
**withdrawn as never-executed and superseded by this amendment**, and the arm is removed from
that file, which now records exactly the 30 cells that ran. This is a removal of a declaration,
never of a result.

**P3 is affected and stays open.** Section 11.5 already flagged P3 as requiring a dependency
audit because it is stated in terms of "the constrained predictor". The new ON arm is not that
predictor: it is structured, and it is not contraction-constrained. P3 therefore cannot be
resolved by the 3x2 grid as currently worded. Restating it or withdrawing it is due at E0,
before the grid runs, and this amendment does not pretend to do it here.

### 13.3 The 3x2 grid is a new pre-registration, under a new result root

The eventual factorial is 6 arms x 2 datasets x 5 seeds = **60 cells**, written to a **fresh
result root** (`results_factorial/`). `results_amended/` is **frozen**: its 30 OFF cells stay
as the phase-B trace, are never merged into the new root, and no resume sentinel may adopt
them. This is section 11.5's own rule applied to this repo's convenience again.

**The 30 OFF cells are re-run rather than reused.** They were executed before any structured
code existed. Reusing them would put cells produced by two code versions, under two separate
pre-registrations, side by side in one figure — a provenance defect the figure would then have
to declare. Compute is not the constraint here: the measured cost of the 40-cell OFF grid was
~17 minutes on 8 cores (mlp ~5 s, min_ssm ~10 s, koopman ~28 s, transformer ~60 s per run at
200 updates).

**What this amendment does NOT fix, and must not be read as fixing**: the grid's training
budget (epochs/updates, batch, device), which depends on what phase D measures and is due at
D0/E0; and the config file that declares the 60 cells, which does not exist yet. This section
fixes the boundary and the root, nothing about the runs themselves. No cell of this grid may
execute until that budget is committed.

### 13.4 Vocabulary

When `alpha_t`, `omega_t`, `B_t` depend on `c_t`, the operator is **LPV/cocycle**. It is not
Koopman, in the code, the docstrings, the notebooks or the README (`REGLES.md` section 3).
The `_k` suffix on `mlp_k`/`transformer_k`/`ssm_k` marks the structured transition; it grants
no right to write "Koopman" anywhere. `A_bar h + B_bar u` is **affine**; "bilinear" requires
the explicit interaction term `sum_j u_j B_bar_j h`, which the structured arms do not have.

### 13.5 Honesty declaration

- **Decided, not derived**: the strong-factorial option, `alpha` free on ON, and the fresh
  result root. Each is the project owner's call, taken on the reasoning stated above and
  recorded before the code that depends on it exists.
- **Measured before deciding**: 30 executed OFF cells and 0 executed ON cells; the ~17 min /
  40 cells compute cost. The first is what makes the redefinition free of consequence for
  published numbers; the second is what makes re-running the OFF arms cheaper than carrying a
  provenance defect.
- **Deferred, and named rather than omitted**: the grid budget, the phase-E config, and the
  restatement or withdrawal of P3.

### 13.6 Declared initialization of the structured head (added 2026-08-04)

Section 13.1 fixed the head as `theta_0 + W_theta c`. It said nothing about how that head is
initialized, and phase C measured why that gap matters. This subsection closes it. It authorizes
no run; it is committed before the code that depends on it, per R4.

**Measured, before deciding.** With a default random `W_theta`, the three arms do not start from
comparable operators. Median context norm at initialization, over 20 seeds, `d_z=4`, `r=2`,
`width=16`: `mlp_k` **0.68**, `ssm_k` **0.60**, `transformer_k` **3.45** — the transformer's
context is ~5x larger, because its output is a sum of residual branches. That propagates straight
through the head: median `max|alpha|` **0.217 / 0.182 / 0.496**. And because `c_t` is an
unnormalized read of `h_t` while `alpha` is free (13.2), the three quantities close a runaway
loop: `|h|` up, `|c|` up, `alpha` up, `|h|` up faster. Free-running rollouts over `H = 50` from
the real entry point `h_0 = lift(z_0)` go non-finite on 1/20 seeds at `Delta = 0.10`
(`transformer_k`), and on 5/20, 12/20, 7/20 seeds at `Delta = 0.50` for `mlp_k`,
`transformer_k`, `ssm_k` respectively.

An unequal starting operator across arms of a factorial whose whole claim is "identical
treatment, three backbones" is a confound, not a detail. It is the same shape of defect as the
one section 11 withdrew a prediction over.

**Declared.** `W_theta` is initialized to **exactly zero**, and `theta_0` is drawn from the ranges
this repo already uses for the same three quantities in `RotationScalingBlocks` and
`KoopmanSSMPredictor` — no new constant is introduced:

- `alpha_0 ~ U(-1.5, -0.5)` — the `nu` range of `RotationScalingBlocks`, whose own docstring calls
  it "contractive-ish and oscillatory by declaration, not by tuning";
- `omega_0 ~ U(0.1, 2.0)` — the `omega` range of the same module;
- `vec B_0 ~ N(0, 0.01^2)` — the `B_cont` initialization of `KoopmanSSMPredictor`.

**What this buys, measured.** At initialization `alpha = theta_0` exactly, so the context-norm gap
no longer reaches the operator at all: the three arms start from the same LTI system drawn from
the same law. Free-running rollouts become **20/20 finite on every arm at `Delta = 0.02`, `0.10`
and `0.50`**. As a side effect the initial state of every structured arm **is** the neutral-context
configuration (`W_theta = 0`) that the phase-C oscillator gold requires, which makes that gold a
statement about the model as built rather than about a hand-patched variant.

**What this does NOT do, so it is not read as doing it.** `alpha` stays **free** — unbounded, able
to become positive during training, exactly as 13.2 declares. Only the *initialization* is
contractive, on the same declared basis the existing blocks already use. This is not the
contraction constraint returning under another name: nothing at any later step bounds `alpha`, and
stability remains something the grid measures rather than something the architecture imposes. Nor
does it equalize the arms during training: the context-norm gap still reaches gradients on
`W_theta`, and 13.2's honesty applies — what is removed is a first-order confound at the starting
point, not every difference between backbones.

### 13.7 Budget allocation inside the structured arms (added 2026-08-05)

Section 1 fixes the fairness contract as a matched TOTAL parameter count (+/-5%). Phase C
measured that hitting it forces an allocation choice inside the structured arms, and this
subsection declares the choice rather than leaving it to whatever a search happens to return.

**Measured.** With a single width knob and `psi_hidden = pi_hidden = (width,)`, at
`target = 10 000`, `d_z = 4`, `d_u = 1`, `r = 4`, `transformer_k` has **no rung inside the
contract**: `width = 18` gives 9442 (**-5.58%**) and `width = 20` gives 11448 (**+14.48%**). The
attention context costs ~`24 width^2` and `width` must stay divisible by `n_heads`, so the ladder
steps by ~18% of the target near it. `mlp_k` and `ssm_k` do have a rung (`-0.39%` and `+0.12%`).
This is a property of a quadratic knob at this scale, not of this particular target.

**Declared.** `match_budget` searches the three structured arms in **two stages, identically for
all three**: a coarse `width` (last rung at or below target), then a fine `psi_hidden`/`pi_hidden`
(~26 params per unit, ~0.3% resolution). All six factorial cells land within +/-5%, verified
against `sum(p.numel() for p in model.parameters() if p.requires_grad)` and never against
`count_params` itself.

**The consequence, stated rather than hidden.** The arms then differ in how the shared budget is
split. Measured at `target = 10 000`, `d_z = 4`, `d_u = 1`:

| arm | psi + pi | head | context |
|---|---|---|---|
| `mlp_k` | 20.4% | 12.5% | 67.1% |
| `transformer_k` | 10.3% | 3.0% | 86.7% |
| `ssm_k` | 15.7% | 7.7% | 76.6% |

The **head** spread is not an allocation choice: the head is `Linear(d_context, 2r + 2r d_u)`, its
output is the same 16 numbers in every arm, and only its input width follows `d_context` — which
IS the backbone axis. The **psi/pi** spread (2x between `mlp_k` and `transformer_k`) is a real
allocation difference, and it means "identical treatment, three backbones" holds for the *form* of
psi, the head and the ZOH, not for the *capacity* of psi and pi.

**Why the alternative was rejected, measured before deciding.** Pinning `psi_hidden`/`pi_hidden` to
one common value across the three arms and letting `d_ff` absorb the transformer's granularity does
hit +/-5% (checked at common hidden 32, 48, 64, 80). But it buys equal `psi`/`pi` at the price of a
degenerate attention context: the search lands on `d_model = 4` — two heads of dimension 2 — with
`d_ff` near 450. That is an MLP behind a 4-dimensional attention bottleneck, and it would bias the
backbone effect the grid exists to measure. `psi` and `pi` are near-linear maps from `d_z = 4` to
`2r = 8`; their capacity is very unlikely to be the bottleneck, while a 4-wide attention certainly
would be. The asymmetry is therefore accepted in the place where it costs least.

**What phase E must carry.** The backbone comparison **within the ON row**
(`ssm_k` vs `transformer_k` vs `mlp_k`) is a comparison at matched total budget and **unmatched
psi/pi capacity**. Any statement of the backbone effect on that row declares this table. The
structure effect (`X-K` vs `X` at fixed backbone) is unaffected: it compares an arm against its own
OFF counterpart, both at the same total budget.

## 14. Amendment 5 — the phase-D gate budget and its execution contract (dated 2026-08-05)

### 14.0 Scope, and why it is committed now

Phase D runs this repo's first trainings under the 3x2 factorial. Rule R4 requires the
amendment to precede the execution it authorizes, and section 12.5 named four things as
"declared open, fixed at D0": the episode aggregation of the VPT, the gate budget, whether
the D pilot may become part of E, and (deferred to E0, untouched here) the treatment of
censored VPTs. Section 13.3 repeated the budget as due and stated that no cell of the 60-cell
grid may execute until it is committed.

This section fixes them, plus three things phase D's design surfaced that no earlier section
covers: the capacity and data budget, the learning rate of the three structured arms, and the
execution mechanics (crash-resume, logging, sharded concurrency) that the runs depend on.

**Nothing in section 12.3 moves.** `H_gate = 25`, `gamma = 5`, `tau_1 = 3 * e_recon` with
strict `<`, `tau = 0.5`, and the five-seeds-pass-individually rule of 12.4 are unchanged and
are not reopened by anything below. This section fixes what is spent, never what is required.

### 14.1 The gate budget

- **Device: CPU, float32.** Not a preference — a statement of what exists. There is no device
  placement code anywhere in `src/` (no `torch.device`, no `.to(...)`); `Trainer`'s docstring
  already declares "Runs on CPU, float32, by default". MPS is additionally forbidden for the
  audit module by its own doctrine (`audit/_common.py`), and at this model scale (~10^4
  parameters, batch of 8 episodes, a 100-step sequential rollout) kernel-launch overhead makes
  it an unlikely win. Any future GPU execution is a port, declared and measured, not a flag.
- **Budget: 1000 optimizer updates**, ceiling 2000. In this repo `epochs` **is** the update
  count: `Trainer.fit` takes one batch and one `optimizer.step()` per epoch. The 200 updates of
  the phase-B grid are retained as the phase-B trace and are not this budget.
- **`batch_episodes = 8`**, unchanged. One budget knob moves at a time; batch size and learning
  rate interact, and this project has already paid once for a dose-crossed-with-LR design
  (11.6).
- **The same budget for all six arms**, in every stratum and seed. This is what protects the
  comparison; equal outcomes are not promised by it and are not expected from it.
- **`CosineAnnealingLR(T_max=epochs)`**: a short run is **not** a prefix of a long one. A budget
  change is a new experiment, never a truncation or an extension of an existing one.
- **Wall-clock ceiling**: 30 min per run, 8 h per campaign, as monitoring numbers. Exceeding
  either stops the campaign for a re-plan; it is not a silent kill and not a terminal verdict.

### 14.2 Capacity, data, and the learning rate of the structured arms

- **`target_params = 10 000`** is retained as the entry value. Section 1 called it a default and
  fixed only the +/-5% matching *procedure*; nothing pre-registers the number. It is probed
  (14.3) rather than assumed.
- **`n_episodes = 256`**, raised from 64. At the `(0.8, 0.1, 0.1)` split this is 208 training
  episodes x T=100 = 20 800 transitions against ~14 600 trainable weights (10 000 predictor +
  4 612 encoder), where 64 episodes gave 5 200 against the same 14 600. The measured evidence
  from the 40 executed phase-B runs shows `val_loss ~ train_loss` throughout, never
  `val >> train` — no overfitting signature at 64 episodes — and this raise moves further from
  the regime where one could appear. It does **not** change the update count, which is 14.1's
  budget, and it does not change per-update cost, which is set by `batch_episodes`.
- **The encoder stays outside the matched budget**, as `budget.py` already declares. This is a
  deliberate scope, not an oversight: phase D freezes ONE encoder per stratum, shared by all six
  arms with an identical digest, so the six see identical embeddings and the encoder cannot
  carry a between-arm effect. The Wave-3 reviewer's capacity confound (`wave3_round1_report.md`,
  closed as a declared deferral) is answered by that design, not by counting. A later study with
  a jointly trained per-arm encoder is a different experiment, and it would have to match total
  capacity or declare that it does not.
- **Learning rate is a property of the backbone, shared by its OFF and its ON arm**: `mlp` and
  `mlp_k` at 1e-4, `transformer` and `transformer_k` at 1e-4, `ssm` and `ssm_k` at 3e-4, the
  values selected in section 10. The three `_k` arms never had a rate selected, and selecting one
  per ON arm would confound the structure effect (`X-K` vs `X` at fixed backbone) with a tuning
  effect. If the probe of 14.3 shows an ON arm badly served by its backbone's rate, that is
  **reported with its numbers**, not silently repaired: repairing it means changing the OFF arm
  too, which is a new pre-registration.

### 14.3 The calibration probe — exploratory, named, and bounded

A budget chosen by assertion is a guess. The values in 14.1 and 14.2 are entry values, measured
before D3 by a probe declared here in advance, under section 9's exploratory category:

- **Namespace**: `results_probe_d/`, never merged into any gate or grid root.
- **Scope**: one stratum, one seed, `mlp_k` (cheapest arm) with confirmation on `transformer_k`
  (costliest). Four axes, one at a time: updates `{500, 1000, 2000, 4000}`; `h_fr` `{5, 10}`;
  `target_params` `{10 000, 40 000}`; `n_episodes` `{64, 256}`. Plus one sanity rung on the
  learning rate, `{1e-4, 3e-4}` on `mlp_k`, for 14.2's last paragraph.
- **`h_fr` is on the list because training free-runs 5 steps and the gate scores 25.** That gap
  has never been measured and is the parameter most likely to decide the gate.
- **What the probe may move**: the four entry values of 14.1/14.2, once, in a dated addendum
  committed before D3. Any value it moves is applied **uniformly to all six arms**.
- **What the probe may not move**: anything in section 12.3 or 12.4. Thresholds are not informed
  by model performance, which is the whole point of having declared them first.
- **What it is for**: not tuning. It converts "the models were probably big enough" into a
  measured statement — capacity probed at 4x, data at 4x, budget at 8x — so that a
  `does_not_learn` verdict cannot be dismissed as, and is not mistaken for, a sizing artifact.

### 14.4 Near-miss policy

The gate verdict stays mechanical. The **margin ratios are the ones section 12.4 already
defines**, in 12.4's direction — distance from passing, so that **`< 1` is a pass** and a
larger number is a worse failure:

    m_low = e_{H_gate} * gamma / min(e^pers, e^lin)          m_high = e_1 / tau_1

This section does not introduce a second pair. An earlier draft of it defined the reciprocals
under the same name; that draft was wrong, and it is recorded here rather than quietly removed
because two quantities sharing one name is exactly the defect this protocol exists to prevent.
Any code carrying these fields carries 12.4's direction.

A cell is labelled `does_not_learn (near_miss)` when it fails and **every bound it fails fails
by at most 1.5x** — that is, the maximum of the ratios of the failing bounds lies in `(1, 1.5]`.
A cell that misses one bound by 1.1x and the other by 5x is not a near miss. It is published
with both ratios, both curves and its terminal state.

A near miss **does not** license moving a threshold: `gamma` does not become 4.5 because a cell
landed at 4.7. It licenses exactly one dated decision by the project owner, among: accept it as
a published failure and continue; re-declare the budget as a new named experiment (14.1, a
budget change is a new experiment); or stop. A silent rerun is not among the options, per
section 7's own rule.

The distinction this makes explicit, because phase D contains both kinds of number:
**pre-registered public thresholds** (12.3, 12.4 — frozen) versus **internal engineering
tolerances** (the `< 1e-3` of the learned-oscillator control, the `1e-12` of closed float64
constructions — adjustable with a measured justification, recorded where they live). The first
kind carries a claim; the second is instrumentation.

### 14.5 Episode aggregation of the VPT

Fixed as required by 12.5: **VPT is computed per episode**, and the cell publishes the full
per-episode vector plus its median. Non-crossing stays `inf` with status `censored` (12.2), and
the count of censored episodes is published alongside. **No mean is taken at D** — how an `inf`
enters a mean or a figure is 12.5's other open item and remains fixed at E0.

This choice is the one 12.5 called closer to the published metric. It is also low-stakes here:
the gate of 12.3 consumes `e_25` and `e_1`, never the VPT, which is a reported diagnostic in
phase D and nothing more.

### 14.6 The D artifacts are not reused in E

Fixed as required by 12.5, and it confirms that section's standing recommendation.

- `results_probe_d/` — the exploratory probe of 14.3.
- `results_gate_d/` — the learned-oscillator control and the six-cell positive gate.
- `results_pilot_d/` — the D5 pilot.
- `results_factorial/` — phase E only (13.3), which re-runs every cell.

No sentinel under a D root may be adopted by E, and no D root is merged into another. This is
13.3's own rule applied to this repo's convenience a third time: the positive gate is 60 cells
at the E budget, and it would be tempting to keep them. The reason not to is that if the gate
finds a cell that does not learn and anything changes as a result, a merged grid would be
invalidated in full, with no way to tell which cells predated the change.

### 14.7 Execution mechanics that the runs depend on

Declared here because they change what a killed run means, and a contract that ignores that
would be a contract about an execution nobody performed.

- **Crash-resume.** A training checkpoint is written atomically every 100 updates, carrying
  model, optimizer, scheduler, epoch counter and RNG state; `fit` resumes from it when it exists
  and its identity matches, and refuses it loudly when it does not. The requirement is
  bit-identity: a run of N updates interrupted at N/2 and resumed must produce the same
  `metrics.csv` bytes as an uninterrupted run. Anything less makes resume a second, undeclared
  experiment. `ckpt.pt` and `run_complete.json` keep their existing meanings — a checkpoint is
  still not a completion signal.
- **No new hashed configuration key.** Only `results_root` is excluded from the config hash;
  adding a knob for the checkpoint or log interval would change every `config_hash`, therefore
  every run directory, therefore make every sentinel already on disk unreachable. Both intervals
  are module constants.
- **Per-run log.** Each run writes `train.log` in its own run directory: the resolved
  configuration, `config_hash`, run directory and budget first, then one line per epoch
  (`epoch, train_loss, val_loss, lr, elapsed, ETA`). This exists so a long run can be diagnosed
  and cut early rather than waited out, and so a stall is distinguishable from slow progress.
  `train.log` joins the stale-artifact cleanup: a killed run's log must not survive into the next
  attempt at the same identity.
- **Sharded concurrency.** Cells may be executed by 2-3 concurrent processes via `--shard i/k`.
  Expansion and the collision preflight stay **global** in every shard, so the uniqueness check
  keeps seeing the whole grid and disjointness is guaranteed by construction rather than by
  correctly hand-editing several config files. `index.csv` becomes an atomic write and every
  shard writes the **full** grid, since a shard writing only its own subset would silently drop
  the other shards' completed cells from the index — artifacts on disk that the index denies.
  A final unsharded pass, which skips every cell by sentinel, remains the authoritative index.

### 14.8 Honesty declaration

- **Decided, not derived**: 1000 updates with a 2000 ceiling, `n_episodes = 256`, the
  `[1/1.5, 1)` near-miss band, backbone-shared learning rates, and the four separate result
  roots. Each is the project owner's call, taken on the reasoning stated above and recorded
  before the code that depends on it runs.
- **Measured before deciding**: the per-update cost per family on this machine (mlp ~0.02-0.035
  s, min_ssm ~0.035-0.075 s, koopman_ssm ~0.10-0.215 s, transformer ~0.225-0.52 s at 200
  updates, reconstructed from artifact timestamps — the orchestrator prints a wall-clock and
  persists none, which is itself fixed by 14.7's log); the absence of an overfitting signature in
  the 40 executed phase-B runs; and the 5 200-transitions-against-14 600-weights ratio that
  motivated 14.2.
- **Inherited unchanged**: every threshold of 12.2, 12.3 and 12.4, and the +/-5% budget matching
  of section 1.
- **Known and left standing**: `min_ssm` shows recurring non-fatal `train_loss` excursions in the
  phase-B trace (up to 4.7e14 on individual minibatches) while its `val_loss` stays smooth. They
  are finite, so the divergence policy of 11.7 never fired. Raising the budget fivefold raises the
  number of draws that could produce a non-finite one, which makes `ssm` and `ssm_k` the most
  likely divergence candidates in D4. This is declared in advance so that a divergence there is
  read as a known risk realized, not as a surprise — and 11.7 already says what happens: cell
  level, no inference.

### 14.9 The pretraining objective of the shared representation (added 2026-08-06)

Phase D freezes one encoder and one state readout per stratum, shared by all six arms
(section 12.3's `e_recon` is measured on that readout, and `tau_1 = 3 * e_recon` depends on
it). The plan named the artifact and its manifest but never named the objective that produces
it, and the objective is a fairness decision, not an implementation detail: an encoder
pretrained with a predictor in the loop carries that arm's inductive bias into all six cells.

**Declared: the representation is pretrained with no predictor in the loop.** The objective is
the two terms the existing trainer already applies to the embedding itself — the VICReg
variance and covariance terms — plus the linear state readout's reconstruction error. Nothing
that depends on a transition model enters it. Consequences, stated rather than discovered
later:

- The latent is **not** shaped by any dynamics objective. It is a well-conditioned,
  decodable coordinate system, and that is all this phase claims for it. An encoder trained
  jointly with a predictor would very likely produce a latent on which prediction is easier;
  the number this phase reports is therefore a **lower** bound on what a jointly-trained
  system would reach, not an estimate of it.
- The comparison the grid exists to make is unaffected, because the handicap is identical for
  the six arms by construction — that is the entire reason for freezing.
- A later study with a jointly trained per-arm encoder is a different experiment. It would
  have to match total capacity across arms or declare that it does not, since section 14.2's
  scope exemption for the encoder rests on the encoder being shared and frozen.

**Decided, not derived.** No measurement informs this choice; it is taken for arm-neutrality
and recorded before the artifact exists. The alternative considered and rejected was
pretraining on one declared reference arm, which is cheaper and produces a better latent, and
which would have privileged one backbone inside a comparison of backbones.

### 14.10 The pretraining objective, second version — supersedes 14.9's loss (added 2026-08-06)

14.9's **scope** stands unchanged: no predictor in the loop, arm-neutrality is the reason, and
everything it declares about consequences still holds. What changes is the loss itself, at the
project owner's direction, before any representation artifact exists as a published result.

**Declared: a two-view self-distillation objective, DINO/BYOL-shaped, with no predictor.**

- Two **views** of the same underlying state: two frames drawn at a short, declared temporal
  offset from the same episode. No transition model is involved — this is an invariance
  objective, not a prediction one, and that is what keeps it arm-neutral.
- A **student** encoder and a **teacher** encoder, the teacher an EMA of the student, with a
  stop-gradient on the teacher branch. The EMA machinery already exists in this repo
  (`EMATarget`) and is reused rather than rebuilt.
- The alignment term is taken **directly between the two encoders' outputs**. This is the
  substantive change from 14.9: no predictor head sits between them.

**Why an anti-collapse term is nonetheless present, stated because it is the one place this
contract departs from the instruction as literally given.** Matching two encoder outputs under
an EMA teacher with a stop-gradient and *nothing else* is the configuration BYOL's own ablation
reports as collapsing to a constant: the predictor head is what prevents it there. DINO removes
that predictor and prevents collapse another way — centering and sharpening a softmax over
prototypes — which does not transfer to a continuous 4-dimensional latent. This project already
carries the standard non-contrastive anti-collapse mechanism: VICReg's **variance and
covariance** terms, whose stated purpose is exactly this. They are retained, and they stand in
for DINO's centering and sharpening.

So the objective is: invariance between the two encoders' outputs, plus VICReg variance and
covariance on the student embedding, plus the linear state readout's reconstruction term
retained from 14.9. The last is kept because `e_recon` — and therefore `tau_1` — is only
meaningful if the latent is linearly decodable to the true state; an objective that is
invariant but not decodable would publish every stratum ungated under 12.3's condition.

**Decided, not derived**, like 14.9 before it: no measurement informs the choice of alignment
term, and the temporal offset defining the two views is a judgment. What IS inherited from the
literature, and cited rather than invented, is the collapse behaviour of predictor-free BYOL and
VICReg's role as the non-contrastive anti-collapse regularizer.

14.9 is not deleted. It is the first version of this decision, and the repo keeps superseded
decisions visible rather than rewriting them out of the record.

### 14.11 The measured invariance/fidelity trade, and two probe axes it adds (added 2026-08-06)

14.10 declared a tension between the alignment term and the reconstruction term without
quantifying it. It is now measured, and the measurement is recorded here because the raw
training-time signal is misleading on its own.

**What looks alarming.** Under the shipped default weights (alignment, variance, covariance and
reconstruction all at 1.0), the alignment loss does **not** decrease over training — it rises
monotonically, by roughly 280x over 60 epochs on the development fixture. Read alone, that says
the alignment term is inert or actively losing. It is neither, and a future reader who finds
that number without what follows would draw the wrong conclusion.

**What the downstream quantity says.** `e_recon` measured on held-out data, same fixture, same
seed, varying only the alignment weight:

| `w_align` | 0 (no temporal term) | **1 (shipped)** | 10 | 100 |
|---|---|---|---|---|
| `e_recon` | 0.1176 | **0.0950** | 0.2064 | 0.6232 |

At the shipped weight the alignment term **improves** reconstruction relative to switching it
off. The rising alignment loss is reconstruction correctly winning the direct conflict — the
pendulum genuinely moves in one physical step, so a faithful latent must separate `t` from
`t+1` — while the term still acts as a useful regularizer on the latent's geometry. Damage
begins a full decade higher, at `w_align >= 10`.

**Why this conflict is structural on this system, stated so it is not rediscovered as a
surprise.** The pendulum is fully observed (`d_x = 2`, `obs: null`). There is no nuisance
variation for an invariance objective to discard; every direction the encoder could smooth away
is state the readout needs. A DINO-shaped objective can therefore only ever be a **mild
regularizer** here, never the primary signal — unlike its home territory of high-dimensional,
nuisance-heavy inputs. This is a property of the system, not a defect of the objective, and it
would change on a partially observed or higher-dimensional stratum.

**Two axes added to the 14.3 probe**, since the numbers above come from a development fixture
(8 episodes, T=20, 15 epochs) and are probe-quality signal, not a result:

- `w_align` in `{0, 1, 10}` — 0 being the pure autoencoder-plus-VICReg alternative, which is
  the serious competing design and deserves a measurement rather than an argument;
- `view_offset` in `{1, 2}` — **corrected 2026-08-06 from `{0, 1}`, which was wrong.** Offset 0
  does not disable the temporal term: the teacher is an EMA of the student, so pairing a frame
  with ITSELF still yields a non-zero alignment term pulling student toward teacher. That is
  BYOL on identical views — the collapsing configuration this section's own reasoning warns
  about — not the pure-autoencoder ablation. Worse, `x[:, :-0, :]` is an EMPTY slice in Python,
  so an unguarded offset 0 produces zero view-pairs rather than raising. The ablation was
  already covered by `w_align = 0`; the two axes were redundant at that point and wrong at it.
  The offset axis now asks the question only it can ask — whether the offset is too small to
  matter — and `view_offset < 1` is refused, in `pretrain_representation` and again in
  `_build_two_views`, each with its own test.

Scored on `e_recon`, on the latent's `cond(Sigma)`, and on `e_25` for the cheapest arm. The
retained values enter the same dated addendum as 14.3's other axes, committed before D3.
`cond(Sigma)` is on the list because this project has already been damaged once by
badly-conditioned latents: the mu_J campaign measured 62.7, 1.5e7 and 1.3e5 across families,
and per-coordinate normalization amplified the low-variance directions rather than taming them.

## 15. Amendment 6 — the phase-D gate repairs and the second probe (dated 2026-08-07)

### 15.0 Scope, and why it is committed now

Probe round 1 (14.3) trained fifteen cells and scored all fifteen against the 12.3 gate;
none passed (`results_probe_d/report.json`, read below). Rule R4 requires this amendment
to be committed **before** the runs it authorizes: the second probe round, D3, D4. It is
committed now, before any of the three executes.

**What does NOT change**, stated explicitly so nothing below is read as reopening it:
`gamma = 5`, `H_gate = 25`, `tau = 0.5`, the five-seeds-pass-individually rule of 12.4, the
`e_h` metric of 12.1, and all of section 13 and section 14 except where this section
expressly amends a specific paragraph (section 4's `h_fr`, in 15.2 below). The
lower bound of 12.3 (`e_{H_gate} < min(e_pers, e_lin) / gamma`) is untouched; only the
upper bound (`tau_1`) is superseded, and only for D3/D4 onward.

### 15.1 The upper bound, second version — supersedes 12.3's `tau_1` rule for D3/D4 onward

**The defect, derived a priori.** 12.3's `tau_1 = 3 * e_recon` was calibrated against "the
score a perfect latent predictor would obtain": a predictor that outputs exactly the
encoder's embedding of the next true state, which inherits the readout's own error and
nothing else. That premise assumes zero *dynamical* error. Any predictor that is actually
trained on a transition carries a component `delta` — the error of predicting the next
latent from the current one — that is independent of `e_recon` and does not shrink when
the encoder improves. The gate `e_1 < 3 * e_recon` therefore requires `delta <~ 2 *
e_recon` to pass, which means a *better* encoder (lower `e_recon`) makes the bound
*harder* for the predictor to clear, while a *worse* encoder loosens it. D2's own success
at shrinking `e_recon` tightens D4 toward a bound the predictor cannot reach, and a
degraded encoder would have made D4 easier — the incentive points the wrong way for a
quantity that is supposed to be a floor, not a target. The pre-registered sanity condition
of 12.3 (`3 * e_recon < e_1_pers`) only ever monitors one direction of this defect — a gate
that is too loose (the trivial baseline beats the ceiling) — and has no corresponding check
for the other direction, a ceiling so tight it is unattainable regardless of how well the
predictor learns.

**Causality, declared.** This amendment was motivated by a specification review conducted
after the 14.3 probe's fifteen cells failed the gate. The argument above is a priori; the
impulse to seek it was not, and this sentence exists so that no reader has to discover that
from the git history.

**The new rule.** `tau_1 = e_1_pers / gamma_1`, strict `<` comparison unchanged. `e_1_pers`
is measured per-stratum at D2, on the same held-out split as `e_recon` (**val** —
`pretrain.py`'s freeze step measures both `e_recon` and `e_1_pers` on the `val` split,
never `test`), with zero trained predictors: it is already the quantity the artifact
manifest carries (`FrozenRepresentationArtifact.manifest["e_1_pers"]`), so this rule adds
no new measurement to D2, only a new use of one already made.

**`gamma_1 = 2`, a declared judgment call**, bracketed the way `gamma` is bracketed in
12.3:

- `gamma_1 = 1` sets `tau_1 = e_1_pers` exactly — the gate would only require beating
  one-step persistence, which is a cosmetic bar (persistence is beaten by construction
  once a model does anything at all resembling the local dynamics).
- On the probed stratum, `gamma_1 >~ 4` puts the bound *below* the measurement chain's
  floor: `tau_1 < e_recon` once `gamma_1 > e_1_pers / e_recon`. Measured on the probe's
  baseline representation (`results_probe_d/representations/DampedPendulum__OURegime__
  a2aaac76__8012e421.pt`, the artifact `mlp_k__baseline` and ten of the fifteen probe-1
  cells share): `e_1_pers = 0.0391210...` and `e_recon = 0.0096466...`
  (`results_probe_d/report.json`, `mlp_k__baseline`), giving `e_1_pers / e_recon =
  4.0554...` — so `gamma_1 ~= 4` is the point past which the bound is unattainable by
  construction, independent of any predictor's quality.
- The log-scale midpoint of the interval `(1, 4)` is `sqrt(1 * 4) = 2`. As with `k` in
  12.3 and `gamma` in 12.3, **no literature standard exists for this factor**; it is
  recorded as a judgment with its bracketing, not derived.

**The two validity conditions, checked at D2, published, never adjusted:**

- *discriminance*: `tau_1 < e_1_pers`. True by construction as soon as `gamma_1 > 1` —
  12.3's old discriminance condition (the upper bound must sit strictly inside what
  trivial persistence already achieves) is absorbed by the form of this rule, not
  reasserted separately.
- *attainability* (the direction 12.3 was missing): `tau_1 > e_recon`, i.e. `gamma_1 *
  e_recon < e_1_pers`. A stratum that fails this is published **"ungated (upper bound
  unattainable)"**, with its numbers unadjusted — the same non-negotiable stance 12.3
  already takes for a failure of its own sanity condition.

**The effect on the current stratum, quantified.** On the probe's baseline stratum:

```
tau_1 (old, 12.3) = 3 * e_recon        = 3 * 0.0096466... = 0.028940 (~0.0289)
tau_1 (new, 15.1) = e_1_pers / gamma_1 = 0.039121... / 2   = 0.019561 (~0.0196)
```

`0.0196 < 0.0289`: the new rule **tightens** the bound on this stratum — a model needs a
*smaller* `e_1` to pass than under the old rule. This is stated explicitly so that no
reading of "the threshold was loosened after a failed probe" is possible; on the one
stratum measured so far, the opposite happened.

**The old reading stays published.** Section 12.3 is not edited by this section. The
fifteen probe-1 cells' verdicts computed under the 12.3 rule remain on record in
`phaseD-REVIEW-PACKET.md` section 5.3, untouched — the same precedent section 10 set for
section 5's superseded constants and 14.10 set for 14.9's superseded loss: the old reading
stays visible beside the new one, not overwritten.

### 15.2 The training free-running horizon — supersedes section 4's `h_fr` for phase-D trainings

`h_fr` goes from **5 to 25** (`= H_gate`) for all phase-D trainings from here on: probe
round 2, D3, D4. Uniform across all families and arms — this is section 4's own fairness
criterion ("no family gets a silently different schedule") applied to a knob section 4
itself fixed. Section 4 is not edited; it remains Wave 3's v0 choice, on record as what it
was. `E` is re-declared at `E0` if phase E inherits this budget.

**Legitimacy, three arguments:**

(i) **Ordering.** Section 4 (Wave 3) precedes the gate: `H_gate = 25` was fixed later, by
amendment 3 (section 12.3). Training a model free-running on 5 steps while scoring it on
25 is an inconsistency between two pre-registrations of different ages — the older one
simply predates the number it now looks miscalibrated against — not a value chosen after
seeing performance.

(ii) **The gap was named in advance.** Section 14.3 itself listed `h_fr` as an axis
"because training free-runs 5 steps and the gate scores 25... the parameter most likely to
decide the gate." The intent to measure this gap was committed before the probe ran; only
the authorization to *move* the entry value was missing (`h_fr` is not among the four
14.1/14.2 entry values 14.3 declared the probe allowed to move).

(iii) **The mechanism is measured.** The learned operator of the `mlp_k` arm (a
context-scheduled `Ā(c_t)` — LPV/cocycle, never "Koopman", REGLES section 3) exposes a
state-dependent field of local spectra via `local_operator`/`local_spectra_field`
(`sld.audit.spectral`), display/screening only, never a stability certificate (REGLES
section 5). Re-derived here, read-only, from the two checkpoints already on disk
(`results_probe_d/runs/mlp_k__baseline` at `h_fr=5`, `results_probe_d/runs/mlp_k__h_fr_25`
at `h_fr=25`), by sampling the field at every step of a genuine `H=25` free rollout on the
probe's held-out (test) split: the median `|eigenvalue|` (max over the two-block spectrum,
per sampled step) over that rollout goes from **1.229** at `h_fr=5` to **1.051** at
`h_fr=25`; the value read at the rollout's first step goes from 1.166 to 1.051. Both
readings support the same qualitative claim — a local spectral radius of roughly `~1.2` at
`h_fr=5` collapsing to roughly `~1.05` at `h_fr=25` — and the intermediate checkpoints
(`h_fr=10`, `h_fr=15`, also on disk) fall monotonically in between (median `1.093`,
`1.066`). The `h_fr=5` field is also far less well-behaved along the rollout: its own
maximum over the 25 sampled steps reaches `1.866`, against `1.053` at `h_fr=25` — training
free-running only 5 steps leaves the operator's local behavior beyond step 5 essentially
unconstrained, exactly what (i) and (ii) predict. This is a screening reading, not a
stability proof, and it is reported as one.

**Cost consequence, stated, not measured in this wave.** The free-running term grows from
5 to 25 steps per update; for the family whose per-step cost is itself non-constant in
sequence length (`transformer_k`), this is a superlinear increase in per-update cost, not
a 5x one. This wave does not run the probe, so the actual wall-clock is not measured here;
it is measured in probe round 2, against section 14.1's own monitoring ceilings (30
min/run, 8h/campaign) — never hand-estimated, and never used as anything but a monitoring
number per 14.1.

### 15.3 Divergence at evaluation time (fills the gap between 11.7 and 12.4)

Section 11.7 governs a training loss that becomes non-finite. It says nothing about an
*evaluation* rollout — one built from a checkpoint that finished training cleanly — that
itself leaves the finite domain during free-running scoring. That gap is declared closed
here.

**New terminal class, cell-seed level: `divergence (evaluation)`**, distinct from `ok`,
`missing`, and the training-time `divergence` of section 11.7.

- It **fails the gate**, in the spirit of section 12.4: an evaluation that never produced a
  finite `e_h` curve has no eligible curve to compare against `tau_1`/`gamma`. No
  substitution is allowed — the same rule section 12.4 already states for a diverged
  training run.
- Deterministic re-evaluation is allowed, because evaluation is a **pure function** of a
  fixed checkpoint: unlike a training re-run (11.7 withdraws any determinism claim there),
  two evaluations of the same checkpoint against the same split are measurable, not merely
  asserted, to produce identical bytes — this is a checkable property, not a promise.
- **Durable artifact.** The evaluator writes `eval_failure.json` atomically (tmp file +
  `os.replace`, the same pattern already used for `results_probe_d/report.json` and for
  section 14.7's checkpoints) into the run directory, carrying `config_hash`, `data_hash`,
  `representation_digest`, `horizon`, `first_nonfinite_step` (or `null` if the curve never
  went non-finite but some other refusal fired), `reason` (the exception message), and
  `rule: "15.3"`. Idempotent on re-evaluation — re-running against the same checkpoint
  overwrites the file with the same content, not a second, drifting copy.

**Motivation.** Probe round 1 already had to improvise this case three times without a
declared policy: `sld/experiments/probe.py`'s own `_fmt` docstring records that three
separate printers in that module crashed on a divergent cell's `None` fields before a
formatting helper was added, and `_run_cell` currently distinguishes a genuine rollout
divergence from a real defect by string-matching `"non-finite"` in the caught exception's
message — a single ad hoc filter standing in for a declared terminal class. That the gap
had to be patched three times on one fifteen-cell probe is itself the evidence that it
sits squarely on D4's path, not an edge case.

### 15.4 The probe: round-1 regularization and round 2

**(a) Round-1 regularization.** Section 14.3 committed `h_fr` in `{5, 10}`. The `mlp_k`
axis table that actually ran (`sld/experiments/probe.py::_MLP_AXES`) includes `h_fr` values
`{10, 15, 25}` in addition to the `h_fr=5` baseline — the `h_fr=15` and `h_fr=25` cells
(`mlp_k__h_fr_15`, `mlp_k__h_fr_25`) exceed what 14.3 pre-registered before they ran. This
ordering error — execution before authorization — is **noted and kept on record**, in the
style of section 14.4's own treatment of a wrong earlier draft: not hidden, not deleted.
Both cells are classified as valid exploratory results under section 9's journal, their
numbers published exactly as measured (`results_probe_d/report.json`); they are not
discarded, and they are the direct empirical input to 15.2(iii) above.

**(b) Probe round 2, declared.** Namespace `results_probe_d/` (unchanged), one seed,
exploratory under section 9 — it cannot move any threshold, per 14.3's own rule, carried
forward unchanged. **Nine cells, a closed list:**

| # | cell | knobs vs baseline |
|---|---|---|
| 1 | `mlp_k__hfr25_updates_2000` | h_fr=25, epochs=2000 |
| 2 | `mlp_k__hfr25_updates_4000` | h_fr=25, epochs=4000 |
| 3 | `mlp_k__hfr25_lr_3e-4` | h_fr=25, lr=3e-4 |
| 4 | `mlp_k__hfr25_updates_4000_lr_3e-4` | h_fr=25, epochs=4000, lr=3e-4 |
| 5 | `mlp_k__hfr25_walign_0` | h_fr=25, w_align=0 (pure autoencoder encoder) |
| 6 | `transformer_k__hfr25` | h_fr=25 |
| 7 | `transformer_k__hfr25_updates_2000` | h_fr=25, epochs=2000 |
| 8 | `ssm_k__baseline` | h_fr=5, lr=3e-4 (14.2: the ssm backbone's own rate; the missing ssm comparison point) |
| 9 | `ssm_k__hfr25` | h_fr=25, lr=3e-4 |

Cells 8-9 exist because section 14.8 already names `ssm`/`ssm_k` as the most likely
divergence candidate in D4 (repeated non-fatal `train_loss` excursions in the phase-B
trace); round 1 never touched this family, and D4 would otherwise discover that blind.

**Screening columns added to the report**: `rho_max_rollout` (the maximum, over the
rollout, of the spectral radius of the field of local spectra — vocabulary per REGLES
section 3, never "the spectrum" for a state-dependent operator) and `phi_norm_at_h` (the
2-norm of the composed propagator at the last finite step). Both are **declared
display/screening only, never a stability certificate**, per REGLES section 5 — the same
doctrine `sld.audit.spectral`'s own module docstring already states for every function in
that file.

**Pre-declared reading criterion**, fixed before any of the nine cells runs: does there
exist a cell with `m_low < 1` **and** `m_high < 1` under the committed gate (12.3 for the
lower bound, 15.1 for the upper)? Whatever the result — yes or no — it is published with
its margins, not with a bare label.

### 15.5 Honesty declaration

- **The `H_gate = 25` justification does not describe the OU stratum used by the probe.**
  Measured, read-only, on the probe's own baseline representation and dataset (true
  `sigma_train`, `DampedPendulum`/`OURegime`, `theta_ou=1.0`, the regime every phase-D
  probe cell actually trains and evaluates on): the persistence baseline's `e_h` rises to
  `e_25 = 0.7546` (matching `results_probe_d/report.json`'s own `e_25_persistence` for
  `mlp_k__baseline` exactly) and **keeps rising** past `h=25` — `e_50 = 1.2798`, `e_60 =
  1.3709` — with the maximum over `h in [1, 60]` occurring at `h=60`, not at `h=25`. There
  is no peak at 25 on this stratum, unlike the `u=0` default-IC regime 12.3's own
  justification measured (persistence rising to `2.07` at `h=25` and falling back to
  `1.74` at `h=50` — a genuine peak, because the undriven pendulum oscillates). `H_gate =
  25` is **kept** as a shared judgment call — one horizon, common to all six cells of every
  stratum, is still the point of the 3x2 design (12.3) — but 12.3's *justification* is
  re-scoped here to the regime where it was actually measured (`u = 0`); nothing in 12.3
  is edited, the correction lives here.
- **Split convention, declared.** `e_recon` and `e_1_pers` (hence `tau_1`) are measured on
  **val** (`sld/encoders/pretrain.py`'s freeze step, both quantities); the gate itself
  evaluates on **test** (`sld/experiments/gate.py::evaluate_run_free_rollout`, called from
  `probe.py::_run_cell` with `data["test"]`). These are two i.i.d. draws from the same
  distribution, not the same data scored twice — stated here so no future reader discovers
  it as an inconsistency between 12.1's split and 12.3's floor.
- **Decided, not derived**: `gamma_1 = 2`; `h_fr = 25` for phase D; the nine-cell list of
  15.4(b).
- **Measured before deciding**: `rho`/local-spectral-radius readings at the `h_fr=5` and
  `h_fr=25` checkpoints (15.2); the ratio `e_1_pers / e_recon ~= 4.055` on the probed
  stratum (15.1); the absence of a persistence peak through `h=60` on the OU stratum
  (above).

## 16. Amendment 7 — the entry values retained from the probe, and probe round 3 (dated 2026-08-07)

### 16.0 Scope, and why it is committed now

Round 2 of the calibration probe has run and been read under 15.4(b)'s pre-declared
criterion: **no cell passes both bounds**. The best cell,
`mlp_k__hfr25_updates_4000_lr_3e-4`, misses at `m_low = 2.3246` / `m_high = 2.8673`
(`results_probe_d/report.json`, 24 cells). Section 14.3 authorized exactly one
consequence: the probe may move the entry values of 14.1/14.2, once, in a dated addendum
committed before D3, applied uniformly to all six arms. Section 14.11 put the
representation weights on the same addendum. This is that addendum. Per rule R4 it
precedes every execution it authorizes — including probe round 3 (16.4), which it
declares.

The five author decisions it instantiates, each recorded 2026-08-07:

- **D-i (budget)**: entry value 4000 updates; one measured doubling rung at 8000 read by
  the pre-declared criterion of 16.1; ceiling = 2x the retained value.
- **D-ii (learning-rate policy)**: option (b) of the decisions packet — uniform
  re-selection at the D regime, all three backbones, by the declared procedure of 16.2.
  The author's stated ground: the section-10 rates were selected under a regime that no
  longer exists, and the goal is a system that trains, not fidelity to a pre-project
  number.
- **D-iii (ssm)**: covered by D-ii. If no listed rate trains the ssm family at the D
  regime, that is a published result, not a repair item — no silent third path.
- **D-iv (monitoring numbers)**: 30 min per run and 10 h per shard, 3 shards, at the
  4000-update budget; both double mechanically if 8000 is retained (16.1).
- **D-v (representation weights)**: neither keep `w_align = 1` by default nor retain
  `w_align = 0` on quarter-budget evidence — measure the missing full-budget cell first
  and decide by the two-sided criterion of 16.3.

**Nothing in section 12.3 or 12.4 moves** — `H_gate = 25`, `γ = 5`, `τ = 0.5`, 15.1's
`tau_1` rule with `gamma_1 = 2`, and the five-seeds-pass-individually rule are untouched,
per 14.3's own boundary ("thresholds are not informed by model performance"). Also
unmoved: the device (CPU, float32), `batch_episodes = 8`, `h_fr = 25` (15.2), the four
namespace roots of 14.6, and the two remaining entry axes of 14.2:

- **`target_params = 10 000` stands.** The probed alternative (40 000) left the gate
  unreadable — `divergence_rollout`, published in the report — and nothing else was
  measured; there is no basis for moving it.
- **`n_episodes = 256` stands.** The 64-episode encoder is worse at its own job on both
  of 14.11's encoder scores — `e_recon` 0.017597 vs 0.009647, `cond(Σ)` 4.899 vs 1.873 —
  and no `h_fr = 25` cell ran at 64 episodes, so no evidence at the retained regime
  weighs against 14.2's original data-to-parameter basis. The 64-episode cell's lower
  raw `e_25` at the `h_fr = 5` baseline config (1.4525 vs 5.1837, one seed, both far
  from the gate) is noted, not hidden; it does not outweigh the encoder-quality
  measurements.

### 16.1 The budget: entry value 4000, one measured rung at 8000, ceiling 2x

- **Entry value: 4000 optimizer updates** (was 1000). Measured basis, at `h_fr = 25`,
  one seed: `e_25` improves monotonically and does not saturate — 0.7393 → 0.6180 →
  0.4241 for 1000/2000/4000 (−16.4 %, then −31.4 %); `e_1` drops −44.9 % then −46.4 %
  over the same doublings; and the combined cell (4000 updates, lr 3e-4) improves BOTH
  bounds at once (`e_1` 0.0561, `e_25` 0.3084), retiring round 1's apparent bound
  conflict.
- **The author's question — "and if 4000 is not enough?" — is answered by measurement,
  not assertion.** Round 3 carries one budget rung: `mlp_k`, 8000 updates, `h_fr = 25`,
  at the mlp backbone's retained rate (16.2). Criterion, fixed now: **8000 becomes the
  entry value iff its `e_25`, read under 16.4's rules, improves on its 4000 twin by at
  least 10 % (relative); otherwise 4000 stands.** Basis for the 10 % (decided, not
  derived): the two measured doublings bought −16.4 % and −31.4 %; a doubling that buys
  less than 10 % does not justify doubling the wall-clock of the entire 60-run campaign.
- **Ceiling: 2x the retained value** (8000 if 4000 is retained, 16 000 if 8000 is) — the
  same headroom ratio 14.1 set with 1000/2000. A budget above the ceiling is a new
  amendment, and `CosineAnnealingLR(T_max=epochs)` still makes every budget its own
  experiment, never a truncation or an extension of another (14.1, unchanged).
- **Monitoring numbers (14.4's second category, re-declared with their measured basis)**:
  **30 min per run** and **10 h per shard, 3 shards**, at the 4000 budget — the shard
  ceiling supersedes 14.1's 8 h campaign ceiling, which 60 runs at the measured per-run
  cost cannot fit. Basis, re-derived from the run artifacts (`train.log`, not from a
  planning document): `mlp_k` at 4000 x `h_fr` 25 took 895.35 s and 861.59 s;
  `transformer_k` at 2000 x `h_fr` 25 took 673.87 s, extrapolating to ~22.5 min at 4000;
  60 runs x 15–25 min = 15–25 h sequential, 5–9 h on the proven `--shard i/k` mechanics
  (14.7). If 8000 is retained, both numbers double (60 min per run, 20 h per shard).
  Exceeding either stops the campaign for a re-plan; not a silent kill, not a terminal
  verdict (14.1, unchanged).

### 16.2 The learning rate: re-selected at the D regime, uniformly, by a declared procedure

**What is kept from 14.2**: the learning rate is a property of the **backbone**, shared
by its OFF and its ON arm, never a per-arm tuning — selecting per ON arm would confound
the structure effect with a tuning effect, exactly as 14.2 said. **What moves**: the
selection regime. Section 10's rates were selected at 200 updates, in the μ_J era,
without a frozen encoder, at `h_fr = 5` — a regime nothing in phase D uses. 14.2 itself
named the repair path ("repairing it means changing the OFF arm too, which is a new
pre-registration"); this subsection is that pre-registration.

The evidence that forces it, from the report:

- `ssm_k` dies **in training** at its section-10 rate (3e-4) in both round-2 cells —
  `failure.json` at epoch 30 (`h_fr = 5`) and epoch 36 (`h_fr = 25`), non-finite
  `train_loss`/`val_loss` — the risk 14.8 named (min_ssm excursions to 4.7e14), realized
  by the x20 budget. `h_fr = 25` does not save it.
- `mlp_k` at the D budget is better served by 3e-4 than by its section-10 1e-4 **on both
  bounds** (`e_25` 0.3084 vs 0.4241, `e_1` 0.0561 vs 0.1261, one seed).

**The procedure** (mechanical; every rule below is fixed before any round-3 run):

1. **Sweep** — nine runs: the three OFF arms (`mlp`, `transformer`, `min_ssm`) x lr in
   {1e-4, 3e-4, 1e-3}, at 4000 updates, `h_fr = 25`, the stratum's frozen shipped
   encoder, `n_episodes = 256`, `target_params = 10 000`, `batch_episodes = 8`, one
   seed.
2. **Selection criterion** — per backbone, the listed rate minimizing the OFF arm's
   `e_25` under 16.4's reading rules (val split — never test, never an ON arm). A run
   that fails to train or to evaluate finitely is excluded from the argmin; a diverged
   run cannot win.
3. **ON finitude screen** — declared as a screen, not a tuning: the backbone's ON arm
   must complete training (11.7's `failure.json` absent) at the selected rate, 4000
   updates, `h_fr = 25`, one seed. If the screen fails, the next lower listed rate is
   taken for that backbone — OFF and ON both, preserving the shared-rate rule — and the
   screen re-runs. No metric of the screen run is consulted for selection; only
   completion is.
4. **Exhaustion branch** — if the screen fails at 1e-4, the lowest listed rate, the
   backbone has no trainable rate at the D regime among the declared list. That is a
   result: D4 publishes it as such for that family (11.7 semantics, no inference), and
   the comparative grid carries the fact rather than hiding the family. No off-list rate
   is tried.
5. **No redundant re-runs** — a screen cell whose exact configuration already ran to
   completion in round 2 is satisfied by that artifact and is cited, not re-executed
   (e.g. `mlp_k__hfr25_updates_4000_lr_3e-4` if mlp's selected rate is 3e-4).

The selected rates apply to D3 and D4, OFF and ON arms both, and are recorded in the
round-3 report plus a dated line in the phase-D progress note — outputs of this
procedure, never new decisions.

### 16.3 The representation weights: the full-budget measurement decides `w_align`

The evidence on record, both stages, one seed: the `w_align = 0` encoder beats the
shipped default on both encoder scores (`e_recon` 0.008524 vs 0.009647, `cond(Σ)` 1.321
vs 1.873) and, at equal predictor config (`h_fr` 25, 1000 updates, lr 1e-4), on the
gate's primary score (`e_25` 0.5643 vs 0.7393) — the opposite of 14.10's expected
direction, on this fully observed stratum, as 14.11 itself named possible. Against it:
its `e_1` is worse (0.4693 vs 0.4273) and its `m_high` reaches 23.99 — at a quarter of
the D budget. The cell that would decide — `w_align = 0` at the full budget — does not
exist. Round 3 adds it rather than deciding on quarter-budget evidence (decision D-v).

- **The cell**: `mlp_k`, 4000 updates, `h_fr = 25`, the mlp backbone's retained rate
  (16.2), on the `w_align = 0` encoder already pretrained and digest-pinned under
  `results_probe_d/representations/`. Its **twin** is the same predictor config on the
  shipped (`w_align = 1`) encoder — the 16.2 screen cell or its round-2 equivalent.
- **Reading criterion, fixed now**: **`w_align = 0` is retained iff its cell completes
  (training and evaluation both finite) AND strictly improves `e_25` AND does not worsen
  `m_high`, versus the twin, under 16.4's reading rules. Mixed evidence keeps the
  shipped default.** Basis for the two-sided rule: phase D freezes ONE encoder per
  stratum, shared by all six arms (14.2) — an encoder that buys the long horizon by
  degrading the one-step bound degrades `m_high` for the entire campaign, not for one
  cell.
- **If retained**: the pretraining objective is, in effect, 14.9's loss again —
  reconstruction plus VICReg, no temporal term — and this section says plainly that the
  measurement, at both stages, did not support 14.10's direction on this stratum; the
  `view_offset` axis disappears with the term it parameterized. **If not retained**:
  `w_align = 1` and `view_offset = 1` stand as shipped, and the tension 14.11 declared
  stays declared.

### 16.4 Probe round 3 — declared, closed, bounded

- **Namespace** `results_probe_d/` (the four roots of 14.6 stay four), one seed,
  exploratory under section 9. It moves no threshold — 14.3's rule, carried forward
  unchanged. It is declared here, before any of its runs execute (R4).
- **The closed list, in stage order** — later stages take earlier stages' mechanical
  outputs (the retained rates) as inputs; the dependency is declared, never chosen after
  looking:

| stage | runs | cells |
|---|---|---|
| 1 | 9 | `{mlp, transformer, min_ssm}` x lr `{1e-4, 3e-4, 1e-3}`, 4000 updates, `h_fr = 25` (16.2 sweep) |
| 2 | ≤ 3 + re-screens | `{mlp_k, transformer_k, ssm_k}` at the selected rate, 4000 x `h_fr` 25 (16.2 screen; round-2 artifacts satisfy identical configs) |
| 3 | 2 | `mlp_k` 8000 x `h_fr` 25 x retained rate (16.1 rung); `mlp_k` 4000 x `h_fr` 25 x retained rate x `w_align = 0` encoder (16.3) |

- **Reading rules, fixed now**: every selection or retention reading in this amendment —
  16.1's rung, 16.2's argmin, 16.3's retention — is taken on the **val split**. Test is
  never consulted for a decision; test-split numbers are published as measured, as
  always, and decide nothing here. This extends 15.5's declared split convention: the
  gate scores on test; anything that *selects* reads val. Where a criterion compares a
  round-3 cell to a round-2 twin, the twin is **re-evaluated read-only on val from its
  committed checkpoint** — evaluation is a pure function of a fixed checkpoint (15.3);
  no twin is retrained.
- **Cost, measured basis**: stage 1 ≈ 9 x 15–25 min ≈ 3 h; stage 2 ≤ 1 h; stage 3 ≈
  45 min. Total ≈ 4–5 h CPU, within one session.
- **Report columns** unchanged from 15.4(b), including the two screening columns —
  display and screening only, never a stability certificate (REGLES section 5).

### 16.5 Honesty declaration

- **Decided, not derived**: the 10 % rung criterion (16.1); the rate list
  {1e-4, 3e-4, 1e-3}; the two-sided `w_align` rule (16.3); the 2x ceiling ratio; the
  30 min / 10 h x 3 monitoring numbers. Each carries its stated basis above; none is a
  measurement.
- **Measured before deciding**: every number cited in 16.0–16.3 was re-derived at the
  decision session from `results_probe_d/report.json` (24 cells) or the run artifacts
  themselves (`train.log` wall-clocks 895.35 s / 861.59 s / 673.87 s; the two ssm
  `failure.json` files at epochs 30 and 36; the encoder scores), never copied from a
  planning document (REGLES 7.2).
- **The chain**: 14.3 declared what the probe may move; 14.11 put the representation
  weights on the same addendum; the probe ran and was read under 15.4(b)'s pre-declared
  criterion (no cell passes; best margin 2.87x); this amendment converts that reading
  into entry values and mechanical criteria; and it is committed before round 3, D3, or
  D4 executes anything it governs. The values round 3 pins (the rates, 4000-vs-8000,
  `w_align`) are outputs of criteria fixed here — the entry values move once, by this
  amendment, and nothing is chosen after looking at the runs it authorizes.
- **Two known asymmetries, declared**: the rate sweep runs on the shipped encoder while
  16.3 may retain the `w_align = 0` encoder, and the rates are selected at 4000 updates
  while 16.1 may retain 8000; the rates are re-selected in neither event. One seed,
  exploratory: the rate is a backbone property measured under a frozen encoder either
  way, and each stage-3 comparison holds the rate fixed on both sides. If D4's measured
  behavior contradicts these assumptions, that is reported, not repaired.
- **The evaluation-path change this amendment requires is mechanical, not scientific**:
  `probe.py` currently passes `data["test"]` to the evaluation path; round 3's selection
  readings pass `data["val"]` to the same unchanged `gate.py` functions. No new
  measurement is defined.

### 16.4(bis) Dated precision to 16.4 — the val reading exists for factorial roots too (dated 2026-08-11)

A precision on an existing section, not an amendment (precedent: 20.4). 16.4 already declares
this change mechanical, in its own words: *"`probe.py` currently passes `data["test"]` to the
evaluation path; round 3's selection readings pass `data["val"]` to the same unchanged
`gate.py` functions. **No new measurement is defined.**"*

What that clause did not say is that the val reading existed **only inside `probe.py`**, whose
cell list is hard-coded to round 3 and whose output is `results_probe_d/report.json`. For a run
produced by `sld-factorial`, `e_25` on val was written **nowhere**: `sld-factorial` never calls
the gate, and `sld-gate` scored `datasets[...]["test"]` at a hard-coded index. 23.3's per-arm
rate selection needs exactly that quantity, on new roots. The gap was structural, not a typo,
and it was found before the sweep ran.

**What was added** — the same mechanical change, at the campaign level:

- `sld-gate` takes `--split {test,val}`, **default `test`**. Every existing gate reading is
  byte-identical; 15.5's rule (*"the gate scores on test; anything that selects reads val"*) is
  unchanged and is now expressible rather than only stated.
- The val path calls the **same** `evaluate_run_free_rollout`, with the **same** `tau_1`
  (`protocol.tau_1(e_1_pers)` off the same frozen representation artifact) and the same
  horizon. No second definition of anything.
- The artifact is **`gate_val.json`, never `gate.json`**. `sld.viz.ingest.load_gate` globs
  `gate.json` only, so a selection reading cannot be ingested as a verdict. One filename, one
  split, decided in one function (`gate.gate_filename`).
- The four column names a selection may request — `e_1_val`, `e_25_val`, `m_low_val`,
  `m_high_val` — are `probe.py::_val_reading`'s keys **verbatim**, so 23.3 minimises the same
  OBJECT round 3 minimised rather than a look-alike computed down a second path.

**The check that a swapped split is really a swapped split.** A `--split` argument that was
accepted and then ignored would be invisible in every green test. On the same run, the two
artifacts carry **different numbers**: `gate.json` (test) `e_h_gate = 0.8241093476571955`,
`gate_val.json` (val) `e_h_gate = 0.7841935405111540`. The readers additionally refuse, by name,
an artifact whose recorded `split` is not the one requested, and one scored at a horizon the
requested column does not name.

**No criterion changed.** 23.3's argmin is still *the listed rate minimising that arm's own
`e_25` on the val split*, and the exclusions (a run that did not finish, or whose criterion is
not finite, cannot win) are unchanged. What changed is that the quantity now exists for the
roots 23.3 applies to.

- **R4**: this precision is committed before the sweep's argmin is written into any campaign
  config.


## 17. Amendment 8 — the gradient guard, and the clipped re-run of round 3's ssm cells (dated 2026-08-07)

### 17.0 Scope, and why it is committed now

Round 3 published `min_ssm` diverging in training at all three listed rates (16.2's
sweep: epoch 2, `inf train_loss`, every rate). Before that line is interpreted as a fact
about the architecture, the author ordered a check: architecture, or harness? The answer
is measured first, corroborated second:

- **Measured, this repo, read-only instrumentation**: the global gradient norm at the
  FIRST optimizer step is **2.4e18** for `min_ssm` against **28.2** for `mlp` — same
  harness, same data, and a healthy epoch-0 forward loss (0.807 vs 1.547; `min_ssm`'s
  forward pass at initialization is fine). Death occurs in ONE AdamW step: the epoch-1
  loss scales quadratically with the learning rate (6.44e24 at 1e-4, 6.44e26 at 1e-3),
  the signature of a single oversized parameter step, not of a drifting optimization.
  The state transition itself is stable by construction (`A_bar = exp(delta * a_cont)`
  in `(0, 1)`, the standard exponential parameterization) — the **gradients** explode
  through the 25-step closed-loop BPTT, never the states.
- **The literature** treats an unclipped optimizer under closed-loop multi-step BPTT as
  an incomplete harness for any recurrent backbone: gradient-norm clipping is standard
  in every published SSM world model (PlaNet at 1000; the Dreamer family at 100; R2I at
  1000 for the world model; DRAMA at 100), the failure mode has a direct published
  precedent (TD-MPC "sometimes diverges due to exploding gradients" — the TD-MPC2
  paper, arXiv:2310.16828), and the theory for selective state-dependent gating losing
  the LTI stability guarantees exists (Zubic & Scaramuzza, arXiv:2505.11602). No
  published work trains an SSM's dynamics loss on a 25-step self-consumed rollout from
  initialization without such a guard.

This amendment adds ONE uniform optimizer guard (17.1), re-runs ONLY the ssm portion of
round 3 under it plus one no-op verification cell (17.2), and stops there. **Nothing in
section 12.3 or 12.4 moves.** Section 16's retained values — the rates, the 8000
budget, `w_align = 1` — are not reopened: they stand conditioned on 17.2's no-op check,
which is mechanical.

### 17.1 The guard: global gradient-norm clipping, uniform, hashed

- **`grad_clip = 100`**: the global L2 norm of the optimizer's parameter gradients is
  clipped to 100 (`torch.nn.utils.clip_grad_norm_`) immediately before every optimizer
  step.
- **Uniform**: all six arms, every stratum, every seed, D3/D4 onward, and every
  round-3c cell below. An optimizer guard is a budget/optimizer property, like the
  learning rate (14.1's own doctrine: one budget for all six arms) — never a per-arm
  tuning.
- **The value, decided not derived**: the published range is 20-1000; 100 is the
  Dreamer-family standard and sits more than 3x above the largest healthy-arm gradient
  norm this repo has measured (~28-30), so the guard is a measured no-op for the arms
  that already train and only intercepts pathological steps.
- **Run identity**: `grad_clip` is an OPTIONAL config key. Absent, there is no clipping
  and the canonical config — hence `config_hash` — is byte-identical to before this
  amendment: every existing artifact keeps its identity and its resume. Present, it is
  validated (finite, strictly positive, noisy refusal otherwise) and it is hashed: a
  clipped run is a different experiment and carries a different identity, exactly as a
  budget change does.
- **Measurement added**: the trainer records the PRE-clip global gradient norm each
  epoch as a metrics column, so the guard's activity is published wherever the run is —
  a guard whose bite is not measured would be a silent knob.

### 17.2 Round 3c — the clipped re-run, a closed list, and the no-op gate

Namespace `results_probe_d/` (unchanged), one seed, exploratory under section 9, the
reading rules of 16.4 verbatim (val decides, test published). The unclipped round-3
records remain published untouched — round-3c cells carry NEW cell_ids (`__r3c_`), so a
published divergence is never overwritten by its clipped sibling.

Execution order is part of the declaration:

| order | cell | knobs | role |
|---|---|---|---|
| 1 | `mlp__r3c_lr_1e-3` | = `mlp__r3_lr_1e-3` + grad_clip 100 | **no-op gate** |
| 2-4 | `min_ssm__r3c_lr_1e-4/3e-4/1e-3` | 4000 updates, h_fr 25, grad_clip 100 | clipped sweep |
| 5 | `ssm_k__r3c_screen_lr_<selected>` | 16.2(3) mechanics, grad_clip 100 | ON screen |

- **The no-op gate, pre-declared reading**: a clip that never binds multiplies
  gradients by exactly 1.0, so cell 1 must reproduce `mlp__r3_lr_1e-3` exactly — same
  logged loss trajectory, same `e_25_val`. **Equal → the no-op is proven and every
  section-16 retained value stands unconditionally. Not equal → STOP: cells 2-5 do not
  run, the discrepancy is published, and the author decides.** No third path.
- **ssm selection**: 16.2's procedure verbatim, restricted to the ssm backbone, over
  cells 2-4 (argmin of `e_25_val` on val among cells that trained and evaluated
  finitely; the screen's drop-down and exhaustion rules unchanged).
- **Outcomes, both declared**: if the screen passes, the ssm backbone's rate joins
  16.2's outputs and D3/D4 carry six trainable arms. If exhaustion repeats UNDER the
  guard, 16.2(4)'s line stands — now robust: the interpretation "no trainable rate at
  the D regime" henceforth carries "with the standard gradient guard in place", which
  is the defensible form of the claim.

### 17.3 Honesty declaration

- **Decided, not derived**: the norm 100; re-running only the ssm cells — the no-op
  gate is the declared bridge that lets every other section-16 value stand without a
  full re-run.
- **Measured before deciding**: 2.4e18 vs 28.2 (first-step gradient norms); the
  quadratic-in-lr epoch-1 losses; the healthy epoch-0 forward loss; the healthy arms'
  ~28-30 norms that anchor the choice of 100.
- **A known deviation, named and kept**: AdamW applies weight decay uniformly here,
  including to `a_log`/`W_delta`, where official Mamba practice exempts them
  (`A_log._no_weight_decay` in state-spaces/mamba). Kept deliberately: per-family
  optimizer recipes would break the uniformity this comparison rests on. Recorded so a
  reader sees a choice, not an oversight.
- **Not adopted, and why**: free-running weight/horizon curricula (they would reopen
  15.2's measured `h_fr = 25`) and structurally bounded latents (SimNorm and kin —
  representation design, 14.9/14.10 territory). Both remain available to a future
  amendment if the guard alone does not settle the ssm question.
- **The chain**: round 3 published the divergences; the author ordered the check; the
  mechanism was measured before any remedy was declared; this amendment precedes every
  run it authorizes (R4), and the unclipped records it responds to stay published
  exactly as measured.

## 18. Amendment 9 — the grid narrows to two backbones; the ssm family becomes a documented negative result (dated 2026-08-08)

### 18.0 Scope, and the decision it records

**Author's decision, this date**: the phase-D comparison proceeds on the `mlp` and
`transformer` backbones only — four arms (`mlp`, `mlp_k`, `transformer`,
`transformer_k`), the full backbone x structure 2x2. The ssm family (`min_ssm`,
`ssm_k`) is withdrawn from the competitive grid and carried as a **documented negative
result**, under the author's stated condition that the record shows what was tried,
what each attempt proved, and why the family ultimately does not train here.

The grounds, all measured and all committed BEFORE this amendment (the chain a reader
can audit):

1. **16.2's re-selection at the D regime**: the declared three-rate sweep found no
   trainable rate for `min_ssm` — training divergence at epoch 2, `inf train_loss`, at
   every listed rate (round 3, `results_probe_d/report.json`). `ssm_k` had already
   diverged at its own declared rate in round 2 (epochs 30 and 36).
2. **17's standard gradient guard did not rescue it**: with `grad_clip = 100` — proven
   a no-op on the healthy arms by 17.2's pre-declared equality gate (trajectories and
   `e_25_val` identical to the last digit) — every ssm rate still diverges at epoch 2
   (round 3c, published under distinct `__r3c_` cell ids).
3. **The mechanism is measured, twice over**: the first optimizer step's global
   gradient norm is 2.4e18 (vs 28.2 for `mlp`, same harness), and the epoch-1 loss is
   invariant to clipping (6.4413e24 clipped vs 6.4415e24 unclipped) while scaling
   quadratically with the learning rate. Adam's first step is magnitude-normalized
   (~lr x sign(g) per coordinate), so bounding the gradient could not and did not
   change the outcome: the 25-step closed-loop rollout is **hypersensitive to the
   parameters at initialization** — any first step of size lr explodes the forward
   pass. The state transition itself is stable by construction; the failure is a
   property of the closed loop.
4. **The literature context is on record** (17.0): no published SSM world model trains
   its dynamics loss on a long self-consumed rollout from initialization; the theory
   for selective state-dependent gating losing LTI stability guarantees exists
   (arXiv:2505.11602), and the recurrent-backbone divergence precedent is published
   (TD-MPC, arXiv:2310.16828).

**What this amendment does NOT claim**: that SSMs cannot learn latent dynamics. The
claim is regime-scoped — under THIS declared regime (`h_fr = 25` closed-loop from
initialization, one uniform optimizer for all arms), the family has no trainable rate
among the declared list, with the standard guard in place. The remedies that might
change that verdict (a teacher-forced dynamics loss, horizon curricula, structurally
bounded latents) were declared not-adopted with reasons in 17.3; adopting any of them
would change the regime for every arm and is a different pre-registered experiment,
not a repair of this one.

### 18.1 What changes, mechanically

- **The D4 grid**: 4 cells per stratum (`mlp`, `mlp_k`, `transformer`,
  `transformer_k`) x 2 strata x 5 seeds = **40 runs** (was 60). Everything else fixed
  by sections 14-17 is unchanged: budget 8000, the section-16 rates (1e-3 for both
  backbones), `grad_clip = 100`, `w_align = 1`, and **nothing in 12.3/12.4 moves**.
- **The D3 gate**: "the structured arms, each green under the committed thresholds"
  now quantifies over TWO arms (`mlp_k`, `transformer_k`) — 2/2, not 3/3. Nothing
  else about D3 changes.
- **M4 is restated over the narrowed grid**: every backbone-effect statement names the
  two-backbone scope; the structure effect (`X-K` vs `X` at fixed backbone) is
  untouched in form and gains nothing from the withdrawal.
- **The registry keeps `min_ssm`/`ssm_k` loadable** for the published artifacts and
  refuses them as grid cells — the same treatment section 13 gave `koopman_ssm`.
- **Every ssm record stays published exactly as measured** (rounds 2, 3, 3c): those
  runs are the negative result's evidence, not discarded work.

### 18.2 The publication duty this amendment creates

Wherever the 2x2 grid's results are published, the ssm withdrawal is published with
them: the declared re-selection that was tried, the standard guard that was added and
proven harmless elsewhere, the measured mechanism that explains the failure, and the
regime-scoped wording of the conclusion. A reader must be able to verify that the
withdrawal followed the evidence — dated, committed, in order — and not convenience.

### 18.3 Honesty declaration

- **Decided, not derived**: the withdrawal itself, and the choice to keep the current
  training regime for the four remaining arms rather than reopen it for all six.
- **Measured before deciding**: every number in 18.0's chain predates this amendment
  in committed history (amendments 6-8 and the probe reports they authorized).
- **The asymmetry, named**: the grid's backbone conclusions now cover two of the three
  families phase C built. The third is present as a bounded, mechanistically explained
  negative result — never silently absent.
- **R4**: this amendment precedes every 2x2 execution it authorizes (the D3 config,
  the D4 grid).

## 19. Amendment 10 — D3's measurement coordinates, the spectral and action bounds made well-posed, and the gate artifact (dated 2026-08-08)

### 19.0 Scope, and why it is committed now

Amendment 9 restated the D3 gate over two arms but did not say **in which coordinates**
D3's four bounds are measured, nor **which** learned eigenvalues the spectral bound
compares. Writing the D3 config surfaced two defects in the bounds as previously worded:
one makes the state bound unattainable by construction, the other leaves the spectral
bound undefined. Both are repaired here. R4 requires this section to be committed
**before** the runs it authorizes: the D3 control and the D4 grid. It is committed now,
before either executes.

**What does NOT change**: `gamma = 5`, `H_gate = 25`, `tau = 0.5`, `gamma_1 = 2` and the
`tau_1` rule of 15.1, the five-seeds-pass-individually rule of 12.4, the `e_h` metric of
12.1, the 8000-update budget and the 1e-3 D3 thresholds themselves, and every value fixed
by 16-18. **No threshold is loosened anywhere in this section.**

### 19.1 The defect that makes D3's state bound unattainable, and the repair

**Derived, not observed after a failure.** D3's `H=100` bound (`e_100 < 1e-3`) was written
with its rationale attached: `1e-3` sits "far enough above the float64 exact-ZOH errors to
measure learning and not arithmetic." That rationale contemplates exactly two error
sources, learning and floating-point arithmetic. Routing D3 through the shared frozen
representation introduces a third: the readout's own reconstruction error.

The floor is the one 15.1 already established for `tau_1`, applied one step further. A
predictor that is *perfect in the latent* — one that outputs exactly the encoder's
embedding of the next true state — decodes to `W . enc(x_1)`, whose error against `x_1` is
exactly `e_recon`. So `e_1 >= e_recon` for **any** predictor, and `e_h` cannot fall below
it either. Measured on the only stratum where the frozen representation has been built,
`e_recon = 0.0096466` (val, `results_probe_d/report.json`, `mlp_k__baseline`) — **about ten
times the 1e-3 bound**. Through the frozen chain, D3's state bound is therefore
unattainable no matter how well the predictor learns: it would score the representation,
not the optimizer, and would classify a perfectly-trained predictor as a failure. This is
the same defect 15.1 named — a bound referenced below its own measurement chain's floor —
in a second place it had not yet been looked for.

**The repair, and why it is not a relaxation.** D3 is a **capability control on the
predictor and the optimizer**, not a comparison between arms: it asks whether the
architecture that provably *contains* the system can be *found* by the declared optimizer
under the declared budget. The shared frozen encoder exists to make the arms mutually
comparable in D4; D3 compares no arms. Applying it there serves none of its purpose and
defeats D3's own.

**D3's control stratum therefore measures in state coordinates**: the representation is
the **exact identity** (`d_z = d_x = 4`, encoder a single linear layer with weight
`I` and bias `0`, readout `W = I`), so the reconstruction floor drops out of the chain
and the four bounds measure the predictor and the optimizer alone. **Measured, not
assumed**: `e_recon = 3.3e-8` — the float32 floor of the declared CPU policy of 14.1, not
zero, and it is measured by the *same* code path that measures it for a pretrained
artifact rather than written in as a constant. That is **four decades below** the 1e-3
bound, against the pretrained representation's `9.6e-3`, which sits one decade *above* it. The thresholds are untouched; only the confound is removed. Nothing about D4
changes — D4 keeps the pretrained frozen representation of 14.9/14.10 for every arm,
because there the shared latent is exactly what makes the comparison fair.

### 19.2 The spectral and action bounds, made well-posed

**The defect.** The phase-C gold proves the structured arms reproduce the physical
discrete operator *exactly* (`3.331e-16` one step, `3.886e-15` at `H=100`, `4.495e-16` on
eigenvalues) under a construction that requires `2r = d`: a planted change of basis `S`
block-diagonalises the physical generator, and the two spectra then coincide **as
multisets by similarity**, not by luck. The complementary negative (`r = 1`, `2r = 2 < d`)
is refuted by an Eckart-Young rank bound. But the general budget declaration of 14.2 fixes
`r = d_z = 4` for this system (1532), i.e. `2r = 8` against `d = 4`: **eight learned
eigenvalues against four physical ones, with no rule anywhere saying which four to
compare, nor what is expected of the four surplus modes.** As worded, the bound has no
truth value at `2r > d`.

**The repair.** D3's control cell is declared at **`r = 2`, so `2r = d = 4`** — the exact
configuration in which the phase-C gold's equality is a proven similarity, and the unique
value at which the bound is both defined (`2r = d`) and attainable (`2r < d` is refuted).
D3 is a control, not a grid cell: it is not bound by 13.7's budget-matching allocation,
which exists to equalise capacity *between arms* in D4.

**The two bounds are taken in STATE coordinates, on the model's one-step PREDICTIVE map**
— `x_hat_{t+1} = pi(A_bar psi(x_t) + B_bar u_t)`, i.e. lift, step once, project — never on
`A_bar`/`B_bar` read directly out of the block basis:

**What this map is and is not, stated because the distinction is real.** The free-running
rollout does **not** iterate this map: `project` is read-only by the predictor's own
contract, so the rollout carries the lifted state forward
(`h_{t+1} = A_bar h_t + B_bar u_t`) and projects only to *read* each output. The rollout
therefore realises `pi . A_bar^t . psi`, which equals `(pi A_bar psi)^t` only when
`psi . pi` acts as the identity on the reachable set — an **emergent** property of a
well-trained model, never a structural one. Bounds 2 and 3 thus characterise the one-step
operator the model applies to a *true* state, and bound 1 characterises the composed
100-step rollout. They are complementary, not two readings of one quantity, and the gate
requires **both**: a model can match the one-step operator and still drift over a hundred
compositions, or track a trajectory while carrying an operator that is not the system's.

- *Spectral*: the eigenvalues of `J_x = d x_hat_{t+1} / d x_t`, paired against
  `exp(Delta * (alpha +- i omega))` by **optimal pairing over all permutations**
  (`n = 4`; brute force, never sorting — this system's four eigenvalues share the same
  real part exactly, so a sorted comparison is genuinely ill-defined here, not merely
  fragile). Relative error `< 1e-3`.
- *Action*: `J_u = d x_hat_{t+1} / d u_t`, against the independent ZOH integral
  `Phi B = (int_0^Delta exp(tA) dt) B`. Relative error `< 1e-3`.
- The reference for both is built by a Van Loan augmented-matrix exponential on the
  hand-built physical `(A_x, B_x)` — independent of the closed form under test and of
  phase A's RK4 polynomial, exactly as the phase-C gold's reference is.

**Why state coordinates and not the block basis, measured rather than argued.** An earlier
draft of this section compared `B_bar` from `local_operator` directly against `Phi B`.
That is wrong, and building the evaluator proved it twice over:

1. *The block-basis comparison is basis-dependent and the training objective is blind to
   the basis.* For any invertible `T` respecting the block structure,
   `(psi, A_bar, B_bar, pi)` and `(T psi, T A_bar T^-1, T B_bar, pi T^-1)` produce
   **identical predictions**, so the loss is exactly flat along that family. `A_bar`'s
   spectrum is invariant under it, but `B_bar -> T B_bar` is not: a model that has
   genuinely learned the physics can fail a `B_bar`-against-`Phi B` bound by basis luck
   alone. `J_u = pi B_bar` is invariant (`pi T^-1 . T B_bar = pi B_bar`), and is the
   physically meaningful quantity — the state-space response to an input.
2. *No spectral comparison, in any basis, can see a rotation-sign flip.* Measured:
   flipping the rotation sign leaves the spectral error at `4.5e-16` in the block basis
   **and at `3.365e-16` on `J_x`**. This is not a defect of either formulation — it is a
   similarity invariant twice over: a conjugate pair's eigenvalue **set** is invariant
   under `omega -> -omega`, and passing to state coordinates is itself a similarity, so
   `J_x` inherits every invariant `A_bar` already had. The action bound is what catches
   it (`0.0358`).

   **The consequence is declared, not left to be discovered**: the spectral bound is
   *structurally* insufficient on its own. A model can carry the exact right spectrum and
   still be a different dynamical system — one rotating the other way. The four bounds
   are therefore a conjunction of genuinely complementary checks, and the spectral one is
   never to be reported as if it summarised the others. Both numbers are pinned as
   assertions in the test, so a future change that alters this blindness is flagged
   rather than silently absorbed.

The state-coordinate form also removes the dimensional objection this section opened with
**independently of `r`**: `J_x` is `d x d` whatever `2r` is. `r = 2` therefore stops being
a precondition for the bound to be well-posed and becomes what it should be — the choice
to keep the control in the phase-C gold's own proven configuration. Both reasons for the
repair are recorded because a reader is entitled to know the first formulation was wrong
and how that was found, rather than reconstructing it from a diff.

**The repair is verified, not asserted.** On a planted exact model: applying an
anisotropic basis change moves `A_bar` by `9.49e-2` and `B_bar` by `7.55e-3` while
`J_x`/`J_u` stay invariant to `3.24e-16` / `7.05e-19` — the basis-dependence is removed,
measured. The bounds evaluate identically at `r = 4` (`2r = 8 > d`, the shape at which the
original wording had no truth value): spectral `4.48e-16`, action `6.55e-16`, the same as
at `r = 2`. And the repealed block-basis design, re-run as a mutant on the identical exact
model, scores `1.176` where the state-coordinate form scores `6.55e-16` — the repair
changes the verdict on a model that is *correct*, which is exactly the failure mode it was
written to remove. The Jacobians themselves are cross-checked against central finite
differences (`1.00e-10` / `5.07e-11`), so autograd is not trusted against itself (R2).

**Vocabulary (REGLES section 3, eliminatory)**: `psi` and `pi` are MLPs, and the head is
contextual, so `J_x` is state-dependent: what the model carries is a **field of local
Jacobians**, and its spectral content a **field of local spectra**, never "the spectrum".
Both bounds are therefore evaluated over the held-out probe states and published as the
worst-case error **together with the field's dispersion**. On an LTI system a correctly
trained model should collapse that field to a point — that collapse is a **measured
outcome**, never an assumption, and any screening of it is display only, never a stability
certificate.

### 19.3 The cost of the frozen representation, measured rather than assumed

Because 19.1 removes the frozen representation from D3's gated path, the same four
quantities are **also** computed on the same trained cells through the frozen-chain path
and published as a **secondary, explicitly non-gating diagnostic**. This turns the
limitation into a number: it says what the shared representation costs on a system whose
exact answer is known, which is the quantity a reader of D4 — where every arm runs through
that chain — needs in order to interpret D4's own magnitudes. It decides nothing.

### 19.4 The gate artifact — `gate.json`

The positive-control verdict is persisted per run, by a pass **separate from training**,
so that a verdict is recomputable from disk artifacts alone and no checkpoint is ever
mistaken for a success:

- one `gate.json` per run dir, written **atomically** (temp file + `os.replace` in the same
  directory) and **last**; closed schema `gate/1`, refused on read if any key is missing,
  unknown, or non-finite where finiteness is required;
- it carries the identity, the digests **re-read from the run's own artifacts on disk**
  (never from caller memory), the split, and **every threshold actually used** — so that a
  threshold changed after a run is *detectable*, not merely forbidden;
- terminal states `learns` / `does_not_learn` / `divergence`. On divergence, the measures
  are `null` and `first_step` is recorded: 11.7 forbids fabricating a number for a rollout
  that left the finite domain, and a `0` or an `inf` written there would be exactly that;
- the campaign expands the **same** grid as `sld-factorial`, keeps expansion and the
  collision preflight **global** under `--shard`, and writes a campaign index covering the
  **full** grid, a cell never evaluated appearing as `missing`. A run that is not complete
  is a **named refusal**, never a silent skip;
- the gate scores on **test** (2321); anything that selects reads val.

### 19.5 D4's execution plan is measured, not assumed

The circulating "about 7-8 h in 3 shards" estimate is an artifact of dividing a sequential
total by the shard count with zero contention assumed. Measured from the probe's own
`train.log` files, D4 at 8000 updates is **21-28 h sequential** (`mlp_k` measured directly
at 8000: 1369 s; `transformer` and `transformer_k` extrapolated from measured 4000-update
points at the retained rate, with the wider band covering an unmeasured clip overhead and
the session-to-session contention the probe itself exhibits). The host is a 4P+4E
asymmetric machine and **no concurrent shard has ever been exercised in this repo** — the
partition is proven disjoint, its throughput never measured. The shard count and
`OMP_NUM_THREADS` are therefore **measured by a short calibration before the campaign is
committed to them**, and the measurement is published with the result. No number in this
paragraph gates anything; it exists so that the campaign's duration is a measurement
rather than a hope.

**The calibration, run 2026-08-08 on the real `transformer_k` workload** (the arm that
dominates the cost), as aggregate optimizer steps per second:

```
procs x threads:  1x8    1x4    2x4    3x2    4x1    5x1    6x1    7x1    8x1   10x1
aggregate:       23.9   26.9   48.0   74.4  102.7  112.2  126.1  136.7  136.2  145.8
```

Two results, both against the prior reasoning. **Intra-op threading is a net loss here**:
one process at 8 threads (23.9) is slower than the same process at 4 (26.9) — a
~10 000-parameter model gives the thread pool nothing to amortise its per-call overhead
against. And **single-threaded processes scale almost linearly to four** (102.7 = 4.3x,
each running at a lone process's speed), with diminishing returns after.

**D4 runs on five shards, one thread each**, and the deciding argument is not the clock.
Folding the throughput above into the per-arm costs gives a real makespan of 5.82 h at
k=4, **4.51 h at k=5**, 4.65 h at k=7 and 4.38 h at k=8 — k=5 and k=8 are within noise of
each other. What separates them: `cells[i-1::k]` at k=5 lands **exactly** on the five
seeds, so every shard receives precisely one seed of each (arm x stratum) and the balance
is **structural**, not an artifact of the cost estimates. If the extrapolated
`transformer_k` cost is wrong by a factor of two, k=5 stays perfectly balanced while every
other k skews. A schedule whose balance does not depend on an extrapolation being right is
worth more than twenty predicted minutes.

### 19.6 Honesty declaration

- **Decided, not derived**: that D3's control runs in state coordinates and at `r = 2`.
  Both are judgments about what a positive control should isolate; the facts that motivate
  them are measured, the choice to repair rather than to publish D3 as unattainable is
  mine.
- **Derived, not observed after a failure**: the `e_recon` floor argument and the `2r > d`
  ill-posedness are both a priori, and neither was found by looking at a D3 result — **no
  D3 run exists at the time this section is committed**. The impulse to look came from
  writing the config, and this sentence exists so no reader has to reconstruct that from
  the git history.
- **No threshold moved**: the four D3 bounds and every constant of 12-18 stand exactly as
  committed. What changed is *where the quantities are measured* and *which learned
  eigenvalues the spectral bound names* — in one case removing a confound the bound was
  never written to include, in the other supplying a definition that was missing.
- **The narrowing, named**: D3 no longer exercises the frozen-representation path in its
  gated result. That path is exercised by D2c's tests and by all forty D4 runs, and 19.3
  measures its cost on D3's own cells rather than leaving it unquantified.
- **R4**: this section precedes every execution it authorizes.

### 19.7 The two configurations, declared in full

Every value below is fixed here, before the runs. Where a value is inherited from an
earlier section, that section is named; where it is chosen here, the measurement that
chose it is given.

**D3 — the learned-oscillator control** (`learned_oscillator_gold.yaml`, root
`results_d3_oscillator/`, a fifth root disjoint from 14.6's four: a control is not a gate
and its sentinels must not be adoptable by one):

- *System*: `CoupledOscillators`, default parameters (`m1 = m2 = 1`, `k1 = 1`,
  `k_c = 0.5`, `k2 = 1.5`, `c = 0.1`, `u_max = 2`), fully observed (`obs: null`,
  `d_x = 4`). **Measured, because the phase-C gold's hypotheses are preconditions, not
  assumptions**: the continuous spectrum is two conjugate pairs at
  `-0.05 +- 1.518722i` and `-0.05 +- 1.090176i` — underdamped, and diagonalizable with
  `cond(V) = 1.52`. Both hypotheses hold.
  **A consequence worth naming**: all four eigenvalues share the *same* real part
  (`-0.05`), the damping being identical on both masses. The ordering ambiguity that
  19.2 requires optimal pairing for is therefore not a hypothetical of this system — it
  is its exact configuration, and a sorted comparison is genuinely ill-defined here.
- *Regime*: `OURegime` (`theta_ou = 1.0`). `NarrowPolicyRegime` is excluded for a stated
  reason, not silently: it requires `system.energy()`, which `CoupledOscillators` does
  not define, and inventing an energy for a damped coupled system is a physics decision
  no config is entitled to make.
- *`Delta = dt = 0.05`*, the phase-C gold's own step. **Measured**: the discrete
  eigenvalues of the RK4 propagator that generates the data differ from the exact-ZOH
  reference by `2.114e-08` relative — **4.7 decades below the 1e-3 bound**, so that bound
  measures learning and not the integrator, which is the rationale it was written with.
  The two rejected alternatives are recorded with their numbers: `dt = 0.02` puts the gap
  at `2.16e-10` but lets `H = 100` cover only 0.35 of a period, a degenerate test;
  `dt = 0.1` covers 1.7-2.4 periods but raises the gap to `6.78e-07`. At `0.05`, `H = 100`
  covers 0.87-1.21 periods.
- *`T = 120`*: `evaluate_free_rollout` requires `H + 2 = 102` states and `H + 1 = 101`
  actions per episode at `H = 100`; 120 leaves margin without a silent truncation ever
  being possible (a short split is a refusal, never a `min(H, T)`).
- *`n_episodes = 256`* (14.2), *seeds `{0,1,2,3,4}` passing individually* (12.4),
  *8000 updates* (16.1), *`lr = 1e-3`* (16.2), *`grad_clip = 100`* (17.1),
  *`h_fr = 25`* (15.2).
- *`d_z = 4`, `r = 2`, representation `identity`* (19.1, 19.2), and *`delta = dt = 0.05`*
  — the latent operator's ZOH step is the dataset's step by identity, since the predictor
  advances exactly one dataset step. It is a required key for structured arms (the trainer
  refuses the family without it) and is stated here so it is a declaration rather than a
  default nobody wrote down; D4 carries `delta = dt = 0.02` for the same reason, matching
  the probe's own value on those strata.
- **Declared now so it cannot be claimed later**: training uses `h_fr = 25` while bound 1
  scores at `H = 100`. The control therefore asks for extrapolation four times past the
  trained horizon. That is deliberate — a model that had merely memorised its training
  horizon would fail it — and it is stated here rather than discovered in the result.

**D4 — the positive-control gate** (`positive_control.yaml`, root `results_gate_d/`,
14.6):

- *Strata*: the two of the frozen record, re-declared at 14.2's episode count —
  `DampedPendulum` + `OURegime(theta_ou = 1.0)`, and `DampedPendulum` +
  `ActionConditionedRegime(K_gain = [[0.8, 0.4]], sigma = 0.3)`; `obs: null`, `d_x = 2`,
  `T = 100`, `dt = 0.02`, `n_episodes = 256`.
- *Arms*: the four of 18.1 — `mlp`, `mlp_k`, `transformer`, `transformer_k` — all at
  `lr = 1e-3` (16.2), `mu_j = 0`.
- *Budget*: 8000 updates (16.1), `grad_clip = 100` (17.1), `h_fr = 25` (15.2),
  `target_params = 10000`, `d_z = 4`, seeds `{0,1,2,3,4}` (12.4). **40 runs** (18.1).
- *Representation*: the pretrained artifact of 14.10 at the retained setting of 16.3 —
  `w_align = 1.0`, `view_offset = 1`, `w_var = w_cov = w_recon = 1.0`, `tau_ema = 0.99`,
  `d_z = 4`, `encoder_hidden = [64, 64]`, `epochs = 50`, `lr = 1e-3`, `seed = 0`. These
  are the probe's own values, reused unchanged so that D4 is scored against the same
  representation the retained entry values were selected under, not a variant of it.
- *Attainability, checked per stratum before scoring* (15.1): `gamma_1 * e_recon <
  e_1_pers`. A stratum that fails it is published **"ungated (upper bound
  unattainable)"**, its numbers unadjusted. Only one stratum has been measured so far
  (`e_1_pers / e_recon = 4.0554` against `gamma_1 = 2`); the second is measured when its
  representation is built, and its value is published whichever way it falls.

### 19.8 A wiring defect found while writing these configs, and its repair

17.1 declares `grad_clip = 100` uniform across every phase-D arm, and deliberately keeps
the key **out** of the trainer's `_DEFAULTS` so that existing runs' `config_hash` does not
move — a property pinned by a test on run `42cf513a`. But the runner derives the set of
keys a YAML may legally name **from** `_DEFAULTS`
(`factorial._trainer_allowed_keys`): `_DEFAULTS | (_REQUIRED_KEYS - runner-owned) |
family_keys`. Measured: `grad_clip` is absent from the 24 keys the four D4 arms accept.

So no grid config could declare the guard that 17.1 makes mandatory. The probe never met
this because it builds its trainer configs in Python and bypasses the schema entirely;
the two paths had drifted, and the only reason it was not a *silent* fault is that
`expand()` refuses an unknown key loudly rather than dropping it.

The repair makes `grad_clip` **declarable without becoming defaulted**: an explicitly
optional key, recognised by the validator, still absent from `_DEFAULTS`, so 17.1's
`config_hash` guarantee survives untouched and is re-tested as such. Recorded here rather
than fixed quietly, because a protocol value that no configuration could carry is exactly
the class of defect this document exists to make visible — a declaration is worth what its
wiring is worth (R7).

## 20. Amendment 11 — D3's verdict: the path is proven, exact identification is not reached, and why that does not block D4 (dated 2026-08-08)

### 20.0 What was run, and what it returned

The D3 control of 19.7 executed in full: **10 cells** (`mlp_k`, `transformer_k` x 5 seeds)
on `CoupledOscillators`, 8000 updates each, `results_d3_oscillator/`. **10/10 completed,
zero divergence**, scored by `sld-d3-gate` into one `d3_gold.json` per run.

| bound | threshold | result | median | spread |
|---|---|---|---|---|
| gain vs the two trivial baselines | `m_low < 1` | **10/10 PASS** | `0.156` | `0.148`-`0.170` |
| state error at `H=100` | `< 1e-3` | **0/10** | `0.189` | `0.120`-`0.492` |
| spectral, on `J_x` | `< 1e-3` | **0/10** | `0.112` | `0.070`-`0.158` |
| action, on `J_u` | `< 1e-3` | **0/10** | `0.198` | `0.141`-`0.340` |

`m_low = 0.156` means the models beat persistence and linear extrapolation by roughly
**32x**, against the `gamma = 5` the gate demands. They learn. They do not identify.

### 20.1 The failure is attributed, not assumed — four independent checks

Each candidate explanation was tested and eliminated by its own measurement:

1. **Not the data.** Ordinary least squares on the *same* training split (24 480
   transitions) recovers the operator: spectral error **2.11e-08**, action **1.58e-07**,
   free-rollout `e_100` **7.98e-14**. All three clear the 1e-3 bound by five to eleven
   decades. *(A cross-check falls out of this: the LS spectral error `2.1141e-08` lands on
   the RK4-vs-exact-ZOH gap 19.7 declared, `2.114e-08` — LS recovers the RK4 propagator to
   machine precision and its residual **is** the integrator gap, confirmed by a path
   entirely independent of the one that declared it.)*
2. **Not the architecture.** The phase-C gold, parameters planted, reproduces the exact ZOH
   solution at `1.9e-15` — the optimum exists inside the hypothesis class the runs used.
3. **Not the budget, at any proportionate scale.** The validation curve's decay flattens
   from `-1.66` decades per 1000 updates over epochs 0-1000 to `-0.023` over 6000-8000.
   Closing 1.24 decades at the terminal rate would take **50 000-100 000 further updates**
   — 7 to 13 times the declared budget — under a log-linear extrapolation that such curves
   flatter, not honour.
4. **Not over-parameterised state-dependence.** The hypothesis that the contextual head
   spends its freedom overfitting an LTI truth was **tested and refuted**: forcing
   `W_theta = 0` (the exactly-constant operator, the LTI form) makes both bounds *worse* —
   spectral `0.130 -> 0.190`, action `0.138 -> 0.947`. The learned state-dependence carries
   real signal, and the action response is almost entirely context-driven.

**What remains, by elimination and by measurement**: under the declared regime — AdamW at
`lr = 1e-3`, 8000 updates, batch of 8 episodes, the JEPA-style latent loss with
`h_fr = 25` — gradient descent converges to a solution roughly **200x** worse than one
ordinary least squares finds in closed form on the same data. **The threshold is not
mis-specified**: 19.7's `1e-3` is attainable, demonstrably, and is therefore left exactly
where it was. No threshold moves in this section.

### 20.2 The decision

**Author's decision, this date**: D3 is closed with its declared purpose met and its
exactness bounds published as a **documented negative result** — the same treatment
section 18 gave the `ssm` family, and for the same reason: what was tried, what each
attempt proved, and why it does not reach, are all on record before the conclusion is
drawn.

- **D3's declared purpose is met.** Section 6 of the phase-D plan states it: *"le chemin
  train→checkpoint→rollout est alors prouvé"*. It is — 10/10 complete, zero divergence,
  verdicts computed from disk artifacts and persisted atomically.
- **The exactness bounds are a named negative**, published with 20.1's four checks
  attached. They are not retried, not softened, and not quietly dropped.
- **D3 does not block D4**, and the reason is a difference in kind, not convenience:
  D4's gate is **relative** — beat the trivial baselines by `gamma`, and hold `e_1` under
  `tau_1` — while D3's failing bounds are **absolute** fidelity to an analytic solution.
  D3's own cells pass the relative criterion **10/10 with a 32x margin**, which is direct
  positive evidence on the very criterion D4 applies. A gate may proceed on a criterion its
  own control passed.
- **The caveat this creates is carried forward, not shed.** Every statement D4 and phase E
  make about the structured arms is a statement about what the declared optimizer reaches,
  never about what the structure can express. Wherever the 2x2 grid's results are
  published, this is published with them.

### 20.3 What this result is worth

It is not a disappointment to be managed. On a linear system whose exact solution is
representable by the architecture **and** recoverable from the data by ordinary least
squares, the declared training regime lands two decades away. That is a measured statement
about the optimisation landscape of structured latent dynamics: **the structural prior
contains the exact solution without making it reachable by gradient descent.** It bears
directly on the project's thesis, and it is the kind of negative that is only available to
someone who built the analytic reference first.

### 20.4 Two dated precisions to section 19

- **19.1's `e_recon = 3.3e-8`** was measured on the evaluator's fixture, not on this
  stratum. Measured on the oscillator stratum itself: `e_recon = 2.60e-08`,
  `e_1_pers = 0.0777`, attainability margin **1.5e6x**. The conclusion is unchanged and the
  precise number now belongs to the stratum it describes.
- **19.3's diagnostic is realised as the floor measurement.** "The same trained cells
  through the frozen chain" is not executable — a predictor trained in one latent cannot be
  evaluated in another. What carries 19.1's whole argument is the floor, and it was
  measured on the same oscillator stratum with D4's own declared representation
  hyperparameters: **`e_recon` identity `2.60e-08` versus pretrained `0.0288`**. The frozen
  chain's floor sits **28.8x ABOVE** D3's 1e-3 bound on this system. Had D3 run through it,
  as the plan literally prescribed, the milestone would have been arithmetically impossible
  to pass and the repo would have published "the structured predictors do not learn" when
  the measured fact is "the shared representation does not reconstruct this finely". This
  is the single most expensive error the amendment avoided, and it is recorded as such.

### 20.5 Honesty declaration

- **Decided, not derived**: that the exactness failure is published rather than chased, and
  that it does not block D4. The evidence in 20.1 is measured; the decision on what to do
  about it is mine.
- **A hypothesis of mine was refuted by its own test** (20.1, check 4) and the refutation is
  recorded rather than the hypothesis quietly dropped.
- **No threshold moved.** Attainability was established *before* deciding what to do, which
  is what makes leaving `1e-3` in place an argument rather than a stubbornness.
- **R4**: this section precedes the D4 execution that its "does not block" clause
  authorizes.

## 21. Amendment 12 — D4's reading: an upper bound that is not arm-neutral, a rate asymmetry, and one declared re-selection (dated 2026-08-09)

### 21.0 What D4 returned

The 40 runs of 19.7 executed: **40/40 complete, zero divergence, zero execution failure**,
scored by `sld-gate` into one `gate.json` per run. Per cell, medians over the five seeds,
with 12.4's rule (all five pass individually):

| stratum | arm | `e_1` | `m_low` | `m_high` | lower | upper | **learns** |
|---|---|---|---|---|---|---|---|
| OU | `mlp` | 0.0106 | 0.505 | 0.540 | 5/5 | 5/5 | **5/5** |
| OU | `transformer` | 0.0229 | 0.723 | 1.172 | 5/5 | 1/5 | 1/5 |
| OU | `mlp_k` | 0.0520 | 1.148 | 2.656 | 1/5 | 0/5 | 0/5 |
| OU | `transformer_k` | 0.0492 | 1.112 | 2.516 | 1/5 | 0/5 | 0/5 |
| AC | `mlp` | 0.0119 | 0.966 | 0.777 | 3/5 | 5/5 | 3/5 |
| AC | `transformer` | 0.0297 | 0.854 | 1.934 | 5/5 | 0/5 | 0/5 |
| AC | `mlp_k` | 0.0438 | 1.690 | 2.857 | 0/5 | 0/5 | 0/5 |
| AC | `transformer_k` | 0.0436 | 1.502 | 2.841 | 0/5 | 0/5 | 0/5 |

**One cell of eight passes.** `m_low < 1` and `m_high < 1` are the passing conditions.

### 21.1 The upper bound is not arm-neutral, and the measurement that shows it

`tau_1 = e_1_pers / gamma_1` (15.1) is calibrated on stratum-level quantities, and 15.1's
attainability condition guards it against one floor only: `e_recon`, the **shared
readout's** reconstruction error. That implicitly assumes every arm's sole reconstruction
cost is the shared one. **The structured arms carry a second, architecture-intrinsic
cost**: their state passes through `psi` (lift `d_z=4 -> 2r=8`) and back through `pi`, two
independently-learned MLPs that are not inverses of one another. Even with a *perfect*
transition operator, the prediction comes back deformed.

Measured on the 20 structured runs, in true-state coordinates normalised by the frozen
`sigma_train` — the same metric `e_1` uses — with the "perfect-operator floor" defined as
the one-step error the arm would obtain if its transition operator were exact:

| stratum | arm | perfect-operator floor | `tau_1` | **floor / `tau_1`** |
|---|---|---|---|---|
| OU | `mlp_k` | 0.0728 | 0.019561 | **3.72x** |
| OU | `transformer_k` | 0.0617 | 0.019561 | **3.16x** |
| AC | `mlp_k` | 0.0666 | 0.015342 | **4.34x** |
| AC | `transformer_k` | 0.0525 | 0.015342 | **3.42x** |

The **minimum over the five seeds** stays 2.31x-3.49x above `tau_1`. The measurement chain
was validated first: `e_recon` recomputed independently reproduces the artifact's stored
value **bit-exactly on both strata**, and `e_1` recomputed through the production path
matches the persisted `gate.json` **bit-exactly on all 40 runs**.

**Conclusion, and it is not about learning**: no transition operator, however perfect,
could have passed the upper bound for the structured arms. The gate disqualified them for
a property of the architecture, on the very axis the grid exists to measure.

**The repair applies 15.1's own rule at the granularity it should always have had.** 15.1
already prescribes the treatment for a bound that sits below its own floor: the subject is
published **"ungated (upper bound unattainable)"**, with its numbers unadjusted, never
counted as a failure. That rule was written per stratum; it is hereby checked **per (arm,
stratum)** against the arm's own floor. **No threshold moves**, and this makes nobody pass:
the structured arms still fail the lower bound, which is arm-neutral by construction.

### 21.2 The rate asymmetry, named

16.2 selected the rate by the argmin of the **OFF** arm's `e_25_val` and applied it to the
backbone's ON arm, which received only a finitude screen — "no metric of the screen run is
consulted for selection; only completion is". 14.2's reason was that tuning per ON arm
would confound the structure effect with a tuning effect.

**The measured situation inverts that reasoning.** As executed, the treatment is
asymmetric: each OFF arm runs at a rate selected *for it*, each ON arm at a rate selected
for a *different architecture*. The confound 14.2 sought to avoid is the one now present,
pointing against the ON arms. Applying 16.2's **identical** procedure to the structured
arms does not create an asymmetry; it removes one.

14.2's and 16.2's text stands unedited, as section 10's did when 16.2 superseded it. What
changes is stated here and nowhere else: **the shared-rate rule is withdrawn for the
structured arms**, replaced by the same selection every other arm already received.

### 21.3 The re-selection, pre-registered in full

Fixed here, before any run it authorizes:

1. **Sweep** — six runs: `mlp_k` and `transformer_k` x `lr` in **{1e-4, 3e-4, 1e-3}**
   (16.2's list, unchanged), **4000 updates**, `h_fr = 25`, `grad_clip = 100`, the OU
   stratum's frozen shipped representation, `n_episodes = 256`, `target_params = 10 000`,
   `batch_episodes = 8`, `r = d_z = 4`, **one seed (0)**. Identical in every respect to
   16.2's sweep except the arm.
2. **Criterion** — per arm, the listed rate minimising that arm's own `e_25` on the **val**
   split (16.4's reading rules; never test). A run that fails to train or to evaluate
   finitely is excluded from the argmin; a diverged run cannot win.
3. **Exhaustion branch** — if no listed rate trains, that arm has no trainable rate at the
   D regime and D4 publishes the fact, exactly as 16.2(4) prescribes. No off-list rate is
   tried.
4. **Re-run** — the 20 structured cells (2 arms x 2 strata x 5 seeds) at the selected
   rates, everything else identical to 19.7, into a **new root `results_gate_d2/`**. The
   original `results_gate_d/` is **frozen** and stays published: it is the measurement of
   what the inherited rate produces, which is itself a result about the rate's effect on
   the structure axis.
5. **ONE selection, ONE re-run, published whatever it returns.** This is not a retry until
   green. If the re-run still fails either bound, that outcome is the phase's result and is
   published as measured. No third rate list, no further re-selection, no change of budget,
   loss, or any other declared value. This clause is the binding part of this section.

**The composed grid, and its provenance.** The final 2x2 grid takes the OFF cells from
`results_gate_d/` (already at their own 16.2-selected rate, regime unchanged by this
section) and the ON cells from `results_gate_d2/`. Every cell is therefore at a rate chosen
by **one identical procedure applied to itself**, which is the homogeneity that matters;
the two roots are named per cell wherever the grid is published, so provenance is auditable
rather than implied.

### 21.4 M4's status

**M4 is not crossed.** Phase D's gate is the positive control cell by cell, and seven cells
of eight do not pass it as measured. No GPU rental and no phase-E comparative claim is
authorized by this section. What phase D has established stands: the pipeline, the metric,
the frozen representation, the verdict artifacts, D3's control, and the two negative
results now on record.

### 21.5 Honesty declaration

- **Decided, not derived**: that the rate asymmetry is repaired rather than published as-is,
  and that the structured arms are re-run once. The measurements in 21.1 are facts; what to
  do about them is the author's call, taken on 2026-08-09 with the numbers in hand.
- **The repair makes nobody pass.** 21.1's per-arm attainability rule reclassifies *why* the
  structured arms fail the upper bound; they still fail the arm-neutral lower bound
  (`m_low` 1.11-1.69 against `< 1`, i.e. 4.4x better than the trivial baselines where 5x was
  required). Nothing here converts a failure into a success.
- **No threshold moved.** `gamma = 5`, `gamma_1 = 2`, `tau_1`'s formula, the budget, the
  loss: untouched. `gamma = 5` in particular is what the structured arms miss by 11-15% on
  OU, and it is left exactly where 12.3 pre-registered it — moving it now, with the result
  visible, is the one thing this document exists to forbid.
- **The failure mode this section is most exposed to** is re-running until green. 21.3(5)
  binds against it in advance and in writing.
- **R4**: this section precedes the sweep and the re-run it authorizes.

### 21.6 The 21.3(5) re-run — cancelled unexecuted (dated 2026-08-17)

21.3(5) authorised one LR re-selection and one 20-cell re-run into `results_gate_d2/`,
and 22 restated that plan after 21.1's conclusion was withdrawn. The re-run never
started: as of this date `results_gate_d2/` holds zero runs — one cached representation
file and nothing else. It is **cancelled rather than left pending**. The reason is 23.4:
the comparative programme moved to the oscillator roots, the pendulum gate path is
parked, and a re-run whose sole purpose was to feed that gate would measure something no
live section reads. Cancelling an unexecuted authorisation removes no measurement, and
nothing on disk is deleted. If the pendulum path resumes, a new dated section must
re-authorise a re-run rather than revive this one.

## 22. Amendment 13 — withdrawal of 21.1's conclusion (dated 2026-08-11)

### 22.0 What is withdrawn

Section 21.1 concludes that *"no transition operator, however perfect, could have passed the
upper bound for the structured arms"*. **That conclusion is withdrawn.**

It is not supported by its own table. The "perfect-operator floor" 21.1 reports is **above** the
`e_1` those same arms actually measured in 21.0, in **four cells out of four**:

| stratum | arm | 21.1 floor | 21.0 measured `e_1` |
|---|---|---|---|
| OU | `mlp_k` | 0.0728 | **0.0520** |
| OU | `transformer_k` | 0.0617 | **0.0492** |
| AC | `mlp_k` | 0.0666 | **0.0438** |
| AC | `transformer_k` | 0.0525 | **0.0436** |

A floor that every run already clears is not a floor. Section 12.3 had already warned that
`e_recon` is *"an empirical floor, not a proven lower bound"*; the same caution applies to a
floor built on top of it.

### 22.1 Why no replacement claim is made

The counterfactual 21.1 evaluates — an exact transition operator inserted into a **trained**
`psi`/`pi` pair — is out of distribution. `psi` and `pi` were fitted jointly with the learned
operator, so whether they partially compensate its errors is a hypothesis that must be
**measured**, not asserted. Asserting it here would repeat the mu_J fault one level up: an
inference drawn from a chain that has just been declared suspect. **No causal statement replaces
the withdrawn one.**

The hypothesis is nameable and testable, and it is recorded here as an open question rather than
as a finding: `psi` and `pi` are two independently parametrized MLPs with no invertibility
constraint and no reconstruction term, so `pi . psi` is not the identity and the loss is
invariant along a continuous family of `(psi, A_bar, pi)` triples. Whether that degeneracy
affects what gradient descent reaches is exactly the kind of claim this repository does not make
without a measurement.

### 22.2 What is unaffected

Section 21.3 **stands and is executed as written**. It repairs the rate asymmetry named in 21.2,
which is an independent defect, measured and not disputed here. The 20 structured cells re-run
into `results_gate_d2/` under 21.3(5)'s binding clause — one selection, one re-run, published
whatever it returns.

### 22.3 Honesty declaration

- **Withdrawn, not falsified.** The measurement in 21.0 and the floor computation in 21.1 both
  stay on record, unedited. What is removed is the inference drawn from comparing them.
- **Decided, not derived**: that the withdrawal is published rather than repaired by a new
  counterfactual. The arithmetic in 22.0 is a fact; what to do about it is the author's call.
- **No threshold moves**, and nothing here converts a failure into a success: the structured arms
  still fail the arm-neutral lower bound.
- **R4**: this section precedes any run it would authorize. It authorizes none.

## 23. Amendment 14 — the oscillator campaigns: a three-level structure factor, a sample-size axis, and a closed-form misspecification axis (dated 2026-08-11)

### 23.0 Scope, and why it is committed now

Phase D measured the structure axis ON against OFF on `DampedPendulum` only (21.0), and
measured the structured arms alone, with no OFF cell, on `CoupledOscillators` (19.1: *"D3 is a
capability control on the predictor and the optimizer, not a comparison between arms"*, and
*"D3 compares no arms"*). **The contrast this repository exists to measure therefore has no
measurement on the one system where the rotation-scaling prior is exactly correct.** This
section declares the campaigns that take it, in full, before any of them runs.

Three roots, all NEW and all **self-sufficient** — each re-runs its own reference cells, so no
result is read across roots and 14.6 is not violated. This is stated rather than assumed
because a cross-root reading would be illegal under 14.6 even when it is arithmetically
harmless.

| root | campaign | cells |
|---|---|---|
| `results_ssm_delta/` | C — the ssm family's trainable-rate question, re-opened | 20 |
| `results_osc_n/` | A — gain against sample size, prior exact | 45 (+15 conditional) |
| `results_osc_delta/` | B — gain against prior misspecification | 45 |

### 23.1 The structure factor has three levels

`off` · `on-free` · `on-stable`.

`on-free` is `structure: on` exactly as 13.2 defines it: `alpha` FREE, discrete radius
`exp(Delta*alpha)` unbounded. `on-stable` reparametrises the SAME cell as
`alpha = log(sigmoid(rho)) / Delta`, so the radius is `sigmoid(rho)`, strictly inside `(0, 1)`
for every real `rho`, **with no side condition**. `omega`, the exact ZOH integral `Phi`,
`B_bar`, the LTI gold and every contract test are untouched, and the head emits the same two
numbers per block — so 13.7's fairness argument holds unamended, and the two levels are
initialised into the same radius band (13.6's `alpha_0 ~ U(-1.5, -0.5)` and its sigmoid
pre-image).

**`on-stable` imposes stability rather than measuring it.** That is exactly what 13.2 declined
to do, and it is why this is a THIRD LEVEL measured beside `on-free`, never a repair applied to
it. Its justification comes from outside the world-model literature: an unbounded-radius
forward-Euler discretisation of the same 2x2 generator is the documented cause of divergence
under long closed rollouts in the spiking-neuron literature (SE-adLIF, Nature Communications
2025; Balanced Resonate-and-Fire, ICML 2024; LinOSS, ICLR 2025), which fixes it by
construction. The symplectic-Euler form those papers use was considered and **rejected here**:
its `sqrt(alpha*beta) < 1` bound holds only while the eigenvalues stay complex, and outside
that regime the radius is unbounded (measured: up to 368 over 2e5 draws). A guarantee that
needs a watched side condition is not the right object for an axis this factorial has to
trust.

### 23.1(bis) Dated precision to 23.1 — what the bound guarantees, and at which precision (dated 2026-08-11)

A precision on an existing section, not an amendment: the precedent is 20.4, *"Two dated
precisions to section 19"*. **Nothing else in 23.1 moves** — the three levels, their
justification, the rejection of the symplectic-Euler form, and 13.7's fairness argument all
stand as written.

23.1 states that `on-stable`'s radius is `sigmoid(rho)`, *"strictly inside `(0, 1)` for every
real `rho`, **with no side condition**"*. Two of those claims are exact and one is a wording
error, and the distinction is worth the paragraph because the arm's whole purpose rests on it.

- **The guarantee is `r <= 1`**, hence `r**H` bounded for every horizon `H`. That is what this
  level is recruited to provide, against the documented divergence of an unbounded-radius
  forward-Euler discretisation. It is **measured**: 400 001 values of `rho` swept over
  `[-50, 50]`, at **both** precisions, **zero radii above 1**.

| precision | radius > 1 | radius == 1 | first saturating `rho` | max radius |
|---|---|---|---|---|
| float32 (the Trainer's working precision, CPU) | **0** | 261 372 | **17.329** | 1 |
| float64 | **0** | 100 562 | 37.430 | 1 |
| `on-free`, float32, same input | 4000 / 4000 | — | — | **12.18** |

- **"Strictly inside `(0, 1)`" holds in exact arithmetic and not at the working precision.**
  `sigmoid(rho)` rounds to exactly `1.0` once `1 - sigmoid(rho)` falls below half an ulp at 1
  — above `rho ~ 17.33` in float32, `rho ~ 37.43` in float64. This is a property of
  representing `1 - epsilon`, not of the map.
- **The saturation is benign, and it is tested rather than asserted.** `r = 1` gives
  `r**H = 1`: bounded at `H = 25` and `H = 100`. It is the *opposite* of the failure mode
  this level exists to exclude, and it makes the contrast with the rejected symplectic-Euler
  form (radius measured up to 368 outside the oscillatory regime) **sharper**, not softer.
- **"With no side condition" stands**, unqualified. That was the whole reason the
  symplectic-Euler form was rejected, and nothing here weakens it.

**How this was found, and what was NOT done about it.** `tests/test_structured_radius.py`
asserted the strict form and went red on 4 tests the first time it was ever executed (it was
written by an agent without torch). The test was corrected — to `r <= 1` **and** finite **and**
non-negative everywhere, plus three new tests: one that LOCALISES the saturation point at the
two measured values, one that shows `r**H` bounded at `H = 25` and `H = 100`, and one that
keeps the `r = sigmoid(rho)` identity on the domain the precision can carry. The corrected file
is **stronger in substance** than the one it replaces, and the mutant test that goes red when
the division by `Delta` is dropped is unchanged. Verified by mutation: removing the bound turns
16 of its 25 tests red.

**This is a bug fix, not a threshold move**, and the distinction is the one this document
exists to police. A threshold here is a decision criterion on an experimental result — `gamma`,
`gamma_1`, `tau_1`, the `1e-3` bound. A unit assertion on the image of a reparametrisation is
not one: no result depends on its wording, and the arm's measured behaviour is unchanged by
correcting it. **No campaign number, no reading rule, and no threshold moves.** The `on-stable`
level ships in the campaigns as declared.

- **R4**: this precision is committed before campaigns A and B execute a single `on-stable`
  cell.


### 23.2 Backbone frozen, budget NOT matched

`d_context`, `psi_hidden` and `pi_hidden` are identical across all arms of these roots, and the
realised parameter count is **published per arm**. 13.7's budget matching is suspended here, on
the precedent 19.2 already set (*"D3 is a control, not a grid cell: it is not bound by 13.7's
budget-matching allocation"*). The reason is mechanical: `match_budget` owns the width knob, so
any change to the head's output would make it shrink the backbone — and the campaign would then
be measuring width while calling it structure. That is the mu_J fault, and it is the one this
section must not reproduce. Since `radius` does not change the head's output size at all, the
three levels are in fact parameter-identical; the frozen backbone makes that true by
construction rather than by coincidence.

### 23.3 Rate selection, per arm, for itself

16.2's procedure, unchanged, applied to **each arm on its own** — which is precisely the repair
21.2 prescribes for the asymmetry it named. Sweep: `lr` in {1e-4, 3e-4, 1e-3}, 4000 updates,
one seed, argmin of that arm's own `e_25` on the **val** split. A run that fails to train or to
evaluate finitely is excluded from the argmin; a diverged run cannot win. If no listed rate
trains, that arm has no trainable rate at this regime and the fact is published, exactly as
16.2(4) prescribes. **No off-list rate is tried.**

### 23.4 These roots have NO gate

12.4 (*"the five pass INDIVIDUALLY; no reducer, no averaging"*) is a **gate** rule and does not
apply here, because there is no gate to pass. Stated explicitly rather than violated quietly.
A median may DISPLAY in a figure; it decides nothing. Per-seed values are published alongside
it in every panel.

21.4's clause (*"no phase-E comparative claim is authorized by this section"*) is **lifted for
these three roots, and for them only**, by this section, dated. The campaigns below ARE
comparative claims; running them under a protocol that forbids them would be the first thing a
reviewer of a dossier sold on methodological honesty would find.

### 23.5 The primary reading rules, written before the numbers

| campaign | primary estimand | pre-registered prediction | what kills it |
|---|---|---|---|
| **A** | the SIGN of the interaction `gain(structure) x log n_episodes` | the gain of `on-stable` over `off` DECREASES as `n` grows | no detectable interaction at 5 seeds -> published as such |
| **B** | the SIGN of the slope of `gain(structure)` against `delta(c)` | the gain DECREASES as `delta` grows | a null or opposite slope falsifies the conditional form of the thesis, and it is written that way |
| **C** | does the `ssm` family have a trainable rate at `delta_max = 4*dt`? | yes | no -> 18.0's negative result is strengthened and becomes citable as measured |
| **on-free vs on-stable** | does the gain come from the structure or from the bound? | **not predicted — this is the open question** | nothing: both outcomes are the result |

Effect sizes are published with the sign. **No pass/fail verdict is issued on any of these
roots.**

Three prohibitions, binding:

1. no additional level of `n` or of `c` after the numbers are seen;
2. no third rate sweep, whatever the outcome (21.3(5)'s clause, applied here too);
3. no figure ships whose recomputation disagrees with the table it cites — **the discrepancy
   becomes the result**, the treatment 19.8 already applied to the `grad_clip` wiring defect.

### 23.6 Campaign C — the ssm family, re-opened on a physical argument

18.0 records that the `ssm` family diverges at every declared rate under `h_fr = 25`, with
gradient norm 2.4e18 against 28.2 for `mlp`, and a first-epoch loss **quasi-invariant to the
clip** (6.4413e24 clipped against 6.4415e24 unclipped). A loss invariant to gradient clipping is
not an optimiser step that is too large: it is a forward map that diverges.

`SelectiveSSMPredictor` bounds its diagonal transition by construction
(`a_cont = -exp(a_log)`, so every entry of `A_bar` lies in `(0, 1)`), so the divergence cannot
come from the recurrence's spectrum. What is NOT bounded is the selective gate:
`delta_t = clamp(softplus(W_delta(c)), 1e-4, 10.0)` and `B_bar_t = delta_t * b_t` with `b_t`
unconstrained. **`delta_max = 10.0` is 200x the pendulum's physical step and 500x the
oscillators'**; at that value `A_bar -> 0` (the state is forgotten entirely) while the input
term is amplified by 10, and in a CLOSED rollout that input is the model's own prediction.

The declared factor is `delta_max` at two levels, `10.0` (the value under which the family was
declared dead) and `4*dt` (0.20 here). The second is a **physical** argument — a latent
discretisation step should not exceed a few physical steps — not a tuned value, and it is fixed
here and does not move. Both outcomes are reportable and neither is a repair of the other: if
the family trains at the bounded level, 18.0's sentence is amended to name the configuration it
was measured under; if it does not, 18.0 stands and is no longer attributable to a
configuration choice.

### 23.7 Campaign B — misspecification with a closed-form axis

`CoupledOscillators` takes its viscous damping `c` as a constructor parameter. For the frozen
`m1 = m2 = 1, k1 = 1, k_c = 0.5, k2 = 1.5`, the stiffness matrix has eigenvalues
`kappa = {1.191, 2.309}` and each mode obeys `lambda^2 + c*lambda + kappa = 0`, so a mode is
oscillatory while `c < 2*sqrt(kappa)`; the two thresholds are `2.183` and `3.039`.

A rotation-scaling block imposes `lambda = alpha +- i*omega`: either a conjugate pair or a
**double real root**. Two DISTINCT real eigenvalues therefore cannot be represented, and the
best block sets `alpha` to their mean and `omega = 0`, at spectral distance exactly
`|lambda_a - lambda_b| / 2`. That is the declared misspecification metric `delta(c)`, computed
in closed form and not estimated:

| `c` | eigenvalues of `A` | regime | `delta` |
|---|---|---|---|
| 0.10 | -0.050 +- 1.519i, -0.050 +- 1.090i | two conjugate pairs | **0.000** (prior exact) |
| 2.30 | -1.150 +- 1.010i, -0.762, -1.488 | one pair + two reals | **0.363** |
| 2.60 | -1.300 +- 0.812i, -0.594, -2.006 | one pair + two reals | **0.706** |
| 3.00 | -1.500 +- 0.278i, -0.471, -2.529 | one pair + two reals | **1.029** |

**`c` does not go past 3.039.** Beyond it all four eigenvalues are real and `delta` DECREASES
(0.468 at c = 3.05, 0.188 at c = 4.0); the axis would stop being monotone. The four values above
are the whole axis and no fifth is added later.

**The pairing convention, declared (added 2026-08-11).** `delta(c)` is a distance from a spectrum
to the nearest block-diagonal rotation-scaling spectrum, so it depends on WHICH eigenvalues are
paired into which block. The four values above hold under the convention **a conjugate pair is
never split**: each surviving oscillatory pair takes one block, and the remaining real
eigenvalues pair among themselves. It is the physically meaningful convention here -- an
oscillatory mode of the true system is exactly what a rotation-scaling block exists to carry --
and it is the one the campaign uses.

It is not the only one, and the difference is not negligible: under **free pairing**, which
minimises the distance over all pairings without regard to physical modes, `c = 2.60` gives
**0.529** rather than 0.706. Free pairing also makes `delta` DISCONTINUOUS at the second
threshold `c* = 3.039`, where a conjugate pair collapses to a double real root and the optimal
pairing jumps; the declared convention is continuous there. Both quantities are computed and
published in `notebooks/00_dynamique_latente_structuree.ipynb` section 3, so the reading rule of
23.5 applies to a number whose definition is on the record rather than implied.

Nothing in 23.5's prediction depends on which convention is used -- it is a statement about the
SIGN of a slope, and `delta` is monotone under both on the declared interval. The convention is
declared because a metric on a figure's x-axis must have one definition, not because the result
turns on it.

### 23.8 Honesty declaration

- **Decided, not derived**: the three levels, the two extra axes, the `4*dt` bound, and the four
  values of `c`. Each has an argument above; none is a measurement.
- **A parametrisation was rejected on its own test.** The symplectic-Euler form was the first
  candidate and its bound was checked before adoption, not after; it failed outside the
  oscillatory regime and was dropped. The check is recorded because the failure is what selects
  the shipped form.
- **Nothing here converts a past failure into a success.** 21.0's result stands as measured, and
  these campaigns are new measurements on new roots, never a re-reading of it.
- **R4**: this section precedes every run it authorizes.

## 24. Amendment 15 — the ssm family trains under a physically bounded latent step; 18.0's sentence names its configuration (dated 2026-08-11)

### 24.0 What ran, and what it returned

Campaign C of 23.6 executed in full: **20 cells** (`min_ssm`, `ssm_k` x `delta_max` in
{10.0, 0.20} x 5 seeds) on `CoupledOscillators`, 8000 updates each, `results_ssm_delta/`,
`lr = 3e-4` and `grad_clip = 100` unchanged from 16.2/17.1. The declared factor moved and
nothing else did.

| `delta_max` | family | completed | dies at epoch | first-step grad norm | median `val_loss` | median `e_25` (test) | median `m_low` |
|---|---|---|---|---|---|---|---|
| 10.0 (= 500x `dt`) | `min_ssm` | **0 / 5** | 0, 0, 0, 0, 0 | *none — no step completed* | — | — | — |
| 10.0 | `ssm_k` | **0 / 5** | 28, 36, 37, 56, 56 | **1.807 - 2.355** | — | — | — |
| **0.20 (= 4x `dt`)** | `min_ssm` | **5 / 5** | — | **22.40 - 27.86** | 0.1995 | 0.0484 | 0.150 |
| **0.20** | `ssm_k` | **5 / 5** | — | **1.806 - 2.355** | 0.0253 | 0.1128 | 0.350 |

**The primary result is the first two columns and it is unanimous**: zero of ten cells complete
at `delta_max = 10.0`, ten of ten at `4*dt`. Per-seed values are in
`_private/AGENT_JOURNAL.md`; no median decides anything here (23.4).

### 24.1 The two families fail differently, and the first step is not the discriminator

This is where the campaign contradicts what was expected of it, so it is stated before the
amendment it supports.

- **`min_ssm` at 10.0 fails immediately**: non-finite `val_loss` at **epoch 0**, in all five
  seeds, before a single row reaches `metrics.csv`. There is no trajectory and no gradient norm
  to report, and that absence is the measurement.
- **`ssm_k` at 10.0 fails slowly, from an ordinary start**: its first optimizer step is
  **indistinguishable** from the bounded arm's. Seed-matched, the two levels' first-step
  gradient norms agree to three decimals (1.807/1.806, 2.068/2.067, 2.167/2.167, 2.217/2.217,
  2.355/2.355), and their loss trajectories first differ at **epoch 1**, by ~1e-5. The clamp
  binds from the first update; its consequence is cumulative, and takes 28 to 56 updates to
  become terminal.

**Therefore the first-step gradient norm is NOT what separates the two levels on this root**,
and 18.0's `2.4e18` cannot be placed beside these numbers as if it were: that figure was
measured on `min_ssm`, on `DampedPendulum`, in `results_probe_d/` — a different family, a
different system and a different root, and 23.0 requires this root to be self-sufficient. The
mechanism 18.0 identified (a forward map that diverges at initialization) is what `min_ssm`
does here; it is **not** what `ssm_k` does here.

**How this was caught.** 24.0's first table wrote *"—"* for `ssm_k` at 10.0, because the
tabulation only read `grad_norm` from cells that carried a completion sentinel. The figure's
reader takes it from `metrics.csv` regardless, so figure E plotted five points where the table
claimed none. Under 23.5(3) the discrepancy is the result: the table was wrong, the figure was
right, and the mechanism sentence this section first carried was wrong with it. Corrected here,
with the disagreement on the record rather than silently reconciled.

### 24.2 The amendment to 18.0

18.0's sentence — *"the family ultimately does not train here"* — **is amended to name the
configuration it was measured under**:

> The ssm family has no trainable rate at the D regime **under `delta_max = 10.0`, i.e. 200x
> the pendulum's physical step and 500x the oscillators'**. Under a physically bounded latent
> step (`delta_max = 4*dt = 0.20`) it trains: on `CoupledOscillators`, 10/10 cells complete
> 8000 updates with finite losses, against 0/10 at `delta_max = 10.0`, every other knob held.

The supporting magnitude, within this root: `min_ssm` at the bounded level takes its first
optimizer step at a gradient norm of **22.40 - 27.86** — an ordinary number for this harness —
where at 10.0 it takes no step at all.

### 24.3 What this does NOT establish

**Scope, per 25.6**: every statement in this section holds **under the declared regime**
(`h_fr = 25`). Root 1 of amendment 16 has since measured that the sign of the structure
effect is unchanged at `h_fr = 5` (25.8), so the scope is narrower than the claim would be
without it and wider than "at one horizon"; it is stated rather than assumed either way.


Named explicitly, because the distance between "trains" and "works" is where this repository
would most easily overclaim.

- **"Trains" is not "learns", and neither is a gate verdict.** These roots have **no gate**
  (23.4). `gate.json` is written on test for the figures and its `status` column decides
  nothing here. For the record and not as a criterion: 10/10 bounded cells clear `m_low < 1`
  (0.115 - 0.562, i.e. they beat both trivial baselines), and 2 of 10 also clear the upper
  bound — one of them at `m_high = 0.999515`, a margin of 5e-4. A verdict that turns on the
  fourth decimal is exactly why 23.4 declares these roots ungated.
- **The withdrawal from the competitive grid is NOT reversed.** 18.0's decision was the
  author's, taken on grounds that included cost and the phase-D calendar, not only
  trainability. This amendment corrects one clause of one sentence; it re-admits no arm to any
  grid.
- **No claim is made about the pendulum.** Campaign C ran on `CoupledOscillators`. 18.0's
  measurements were taken at the D regime on `DampedPendulum`. The bound `4*dt` was declared in
  23.6 as a physical argument for both, but only the oscillator has been measured, and 14.6
  forbids reading one root through another.
- **The mechanism is not identified, only bounded.** That the clamp binds and that the
  unbounded arm dies are both measured. *Why* `ssm_k` needs 28-56 updates to die while
  `min_ssm` dies at once is not, and no explanation is offered here.
- **`delta_max` was not tuned.** Two levels, declared in 23.6 before the runs, `4*dt` fixed by
  a physical argument. No third value was tried and none will be.

### 24.4 Honesty declaration

- **Withdrawn by measurement, not repaired.** 18.0's grounds 1-4 stand on the record, unedited,
  and were correct about what they measured. What is corrected is the *scope* of the sentence
  they support: a property attributed to a family belonged to a configuration of it. That is
  the same class of error 22.0 corrected in 21.1, and it is recorded the same way.
- **This section's own first draft was wrong, and its figure caught it.** Two errors, both
  recorded above rather than quietly fixed: a rounding that put two of four cited bounds
  outside the values they claimed to contain, and a missing column that carried a false
  mechanism sentence. 23.5(3) found both before either shipped.
- **This was the pre-declared outcome, not a rescue.** 23.6 wrote both branches before the runs
  ("if the family trains at the bounded level, 18.0's sentence is amended to name the
  configuration it was measured under; if it does not, 18.0 stands"). The branch taken is the
  one the numbers selected.
- **No threshold moved**, no level was added, no rate was re-swept. `lr`, `grad_clip`, `gamma`,
  `gamma_1`, `tau_1` and the budget are exactly what 16.2/17.1 fixed.
- **R4**: this amendment follows the runs it reports and authorizes none.

## 25. Amendment 16 — two roots declared together: the training-horizon regime, then the decomposition (dated 2026-08-12)

### 25.0 Scope, and the correction it starts from

**Author's decision, this date.** Campaign A returned a structure penalty at every sample
size (24 is campaign C; A's table is in the run journal and figure C). Before that is read
as "explicit dynamical structure does not pay", one variable shared by every negative
result this repository holds must be measured rather than assumed: **the free-running
horizon `h_fr` is pinned at 25 from the first optimizer step, with no ramp.**

Two corrections to how that argument was first stated, both found by checking the harness
rather than trusting the memory of it, and both recorded because the first version would
have entered this document as fact:

1. **Teacher forcing is ACTIVE, at full weight.** Every campaign cell carries
   `w_tf = 1.0` and `w_fr = 1.0`: the loss is a teacher-forced term threaded over the true
   prefix PLUS a 25-step free-running rollout, equally weighted (4, and
   `trainer._composite_loss`'s two branches). The shared variable is therefore **not**
   "a closed rollout without teacher forcing" — it is the free-running horizon, and its
   fixity.
2. **17.0 is NOT cited in support of this.** Its sentence — *"No published work trains an
   SSM's dynamics loss on a 25-step self-consumed rollout from initialization **without
   such a guard**"* — concerns the GRADIENT GUARD, which 17.1 then added (`grad_clip =
   100`). It establishes nothing about curricula. The argument below rests on the author's
   declaration and on the two levels' own internal history, and on nothing else.

**Two roots are declared here, together, before either runs.** Execution order is part of
the declaration: the regime root first, because it is decisive for how the second is read.

### 25.1 Root 1 — `results_osc_hfr/`, the regime root (20 cells)

| axis | levels |
|---|---|
| `h_fr` | **5**, **25** |
| `structure` | `off`, `on-stable` |
| seeds | 0-4 |

`n_episodes = 32`, `CoupledOscillators` at `c = 0.10`, every other knob exactly as campaign
A. Self-sufficient (14.6): it re-runs its own `h_fr = 25` reference cells rather than
reading campaign A's.

**Zero new code, and that is the point.** `h_fr` is already a scalar config key, already
hashed into the run identity, already validated. **Both values are already this
protocol's**: 4 fixed `h_fr = 5` as the v0 choice; 15.2 raised it to 25. The contrast is
therefore INTERNAL — between two horizons this document has itself declared — and no
curriculum from the literature is cited, needed, or implied.

**Why two levels and not a ramp.** A ramp varies the horizon AND the trajectory the
horizon takes; two fixed levels isolate the variable. If an effect exists, the shape of a
schedule is a question for after it, not a question to entangle with it now.

**One confound, named before the runs.** 23.3's sweep selected `lr = 1e-3` for all three
arms **at `h_fr = 25`**, and 23.5(2) forbids a third sweep whatever the outcome. The
`h_fr = 5` cells therefore run at a rate selected under the OTHER level of the very factor
being varied. Its direction is unknown and is not guessed at. What this does and does not
damage is precise:

- **Within** each `h_fr` level, `off` and `on-stable` run at the SAME rate, so the
  structure contrast — the estimand of 25.2 — is fair at each level;
- **Across** levels, any difference in the size of the structure effect is confounded with
  the rate's suitability. The reading rule below is about the **sign** at each level and
  its change between them, which is the reading least exposed to this.

### 25.2 Root 1's reading rule, written before the numbers

| outcome | what follows |
|---|---|
| the sign of the structure effect **inverts** between `h_fr = 5` and `h_fr = 25` | the training free-running horizon was the active variable. Every negative result in this repository is republished as **conditional on `h_fr = 25`**, in this document, in the README, and in the caption of every figure already rendered |
| the sign is **preserved** | the negative is robust to the horizon, and it becomes the repository's principal result |
| an effect of `h_fr` **without** a change of sign | published as such, with its effect size |

No pass/fail. This root has no gate. No threshold moves. **No third level of `h_fr`,
whatever the outcome.**

### 25.3 Root 2 — `results_osc_decomp/` (30 cells), and the regime it runs under

The question: is campaign A's penalty attributable to the structured TRANSITION, or to the
lift/projection pair that accompanies it?

| axis | levels |
|---|---|
| `structure` | `off`, `on-stable`, **`on-stable-nolift`** |
| `n_episodes` | 8, 256 |
| seeds | 0-4 |

`on-stable-nolift` is the structured transition acting directly on `z`, with
`psi = pi = identity`.

**This one DOES need new code, and the amendment says so rather than discovering it at
launch.** `d_z = 4` and `2r = 4`, so the lift preserves dimension — necessary, not
sufficient: `psi` and `pi` are always built by `_make_mlp`, and `_kwargs_for` passes
`psi_hidden = (w,)`, so they are one-hidden-layer GELU MLPs; even `hidden = ()` yields a
learned `nn.Linear`, never the identity. An explicit identity option is required, hashed
into the run identity, and verified not to move the `config_hash` of any existing run —
the same check the 17.1 guard's key received.

**The regime it runs under is fixed HERE, before root 1's numbers exist**: if root 1
inverts the sign, the decomposition runs at `h_fr = 5`; in **both** other branches it runs
at `h_fr = 25`. No third possibility, and no choice made after looking.

**Reading rule, written before the numbers:**

| outcome | attribution |
|---|---|
| `nolift` ~ `off` | the penalty is attributable to `psi`/`pi` |
| `nolift` ~ `on-stable` | the penalty is attributable to the transition |
| between the two | partial attribution, published with its effect size |

No pass/fail; this root has no gate either.

### 25.4 Why the decomposition is worth its 30 runs

**Reference verified by the author, cited as such** (this agent has not read it; only the
fields the author verified are reproduced): *Koopman Dreamer: Spectrally Constrained Latent
Dynamics for Stable World-Model Imagination*, Li, Zhang, Xie, Lan, Pan, Xu,
arXiv:2607.19719, 22 July 2026. It uses a latent transition in **2x2 rotation-scaling
blocks with a BOUNDED radius** — the same parametrisation as this repository's `on-stable`
level — and reports improvement over DreamerV3 on **8 of 9** proprioceptive DMC tasks, and
UAV-LiDAR navigation from **53.8 % to 73.8 %**. Its section VI-D "Structural Ablation"
ablates only its own components, and it contains no sample-size study.

The same parametrisation helps there and costs here. The two settings differ in more than
one way — there, a jointly learned representation; here, an identity representation and an
unconstrained `psi`/`pi` pair — so the disagreement is not itself evidence for either.
**This root turns that disagreement into a measurement instead of an explanation**, which
is the only reason it is authorized.

### 25.5 Binding clause

**These are the LAST TWO roots of this cycle.** No further root is declared after them,
whatever they return.

Campaign B's status under this amendment is fixed now: it is **stopped**, not abandoned —
26 completed cells and 2 divergences stand as measured under `h_fr = 25`. It resumes ONLY
if root 1 preserves the sign. If root 1 inverts it, B is re-declared under the new regime
and its already-run cells are **published as measured under `h_fr = 25`, never merged with
the new ones**.

### 25.6 A scope correction to every statement already written

Until root 1 has spoken, every claim this repository makes about the structure axis carries
**"under the declared regime"**. This is not a stylistic hedge: it is the true scope of what
was measured. It applies to 24, to the captions of figures A, B, C and E, and to the README.

### 25.7 Honesty declaration

- **Decided, not derived**: that the regime is measured before the decomposition; the two
  levels of `h_fr`; the two levels of `n`; the regime under which root 2 runs in each
  branch. Each has an argument above; none is a measurement.
- **Two errors in this section's own first framing were corrected before it was written**,
  by checking the harness: teacher forcing is active, and 17.0's sentence is about the
  gradient guard. Both are recorded in 25.0 rather than quietly fixed.
- **A confound is named that cannot be removed**: the rate was selected at `h_fr = 25` and
  23.5(2) forbids re-selecting it. The reading rule was chosen to be the one least exposed
  to it, and the exposure is stated rather than minimised.
- **No threshold moves**, no level is added to an existing root, and campaign A's result is
  not re-read by either root — both are new measurements on new roots.
- **R4**: this section precedes every run it authorizes, including root 1's first cell.

### 25.8 What root 1 returned, and three things recorded rather than chased (dated 2026-08-13)

`results_osc_hfr/`, **20/20 completed, zero divergence at either level**, scored 20/20 on
test. `m_low` per cell -- lower is better, `< 1` beats both trivial baselines:

| `h_fr` | arm | the 5 seeds | median | IQR | range |
|---|---|---|---|---|---|
| 5 | `off` | 0.1501 0.1044 0.1184 0.1174 0.1388 | **0.1184** | [0.1174, 0.1388] | [0.1044, 0.1501] |
| 5 | `on-stable` | 3.1925 3.1302 3.7970 2.2951 1.3954 | **3.1302** | [2.2951, 3.1925] | [1.3954, 3.7970] |
| 25 | `off` | 0.2857 0.2624 0.2760 0.2151 0.2209 | **0.2624** | [0.2209, 0.2760] | [0.2151, 0.2857] |
| 25 | `on-stable` | 0.3911 0.6149 1.0380 0.9029 0.9183 | **0.9029** | [0.6149, 0.9183] | [0.3911, 1.0380] |

**25.2's estimand -- the SIGN of the structure effect at each level**: `on-stable` is worse
than `off` at **both** levels (ratio **26.44** at `h_fr = 5`, **3.44** at `h_fr = 25`).

**The sign is PRESERVED. 25.2's second branch fires: the negative is robust to the training
horizon, and it becomes this repository's principal result.** 25.2's third branch fires with
it -- there IS an effect of `h_fr`, on the effect SIZE, and it is published here with that
size. Consequences, both fixed before the numbers existed: root 2 runs at **`h_fr = 25`**
(25.3), and campaign B **resumes** (25.5).

**(1) An interaction, recorded as an OBSERVATION and not pursued.** The two arms move in
OPPOSITE directions when the horizon shortens: `off` improves **2.2x** (0.2624 -> 0.1184)
while `on-stable` degrades **3.5x** (0.9029 -> 3.1302). That is not a common regime effect
with a shared sign; it is an interaction between the horizon and the structure axis.

Its confound is named: `lr = 1e-3` was selected at `h_fr = 25` (23.3), and 23.5(2) forbids
re-selecting it, so **both arms are off-rate at `h_fr = 5`**. Within each level the two arms
share the rate, so the sign at each level is clean; the comparison of effect SIZES across
levels is not. But the confound does not, by itself, explain a **divergence of direction**:
being off-rate is expected to hurt both arms, and here it coincides with one improving and
the other degrading. **No cause is advanced, and no experiment is declared to pursue this.**
The two-root ceiling of 25.5 holds.

**(2) The structured arm is admissible only at the horizon it is evaluated at.** At
`h_fr = 25` its median `m_low` is **0.9029**, below 1: it beats persistence and linear
extrapolation. At `h_fr = 5` it is **3.1302**, with **all five seeds above 1**: trained at a
horizon shorter than the one it is scored at, it does not beat the trivial baselines at all.

**(3) An instrument validation, published as one.** By 14.6's self-sufficiency requirement
this root re-runs its own `h_fr = 25` reference cells, which are the same cells campaign A
ran at `n = 32`. They reproduce them: `m_low` **identical to 1e-12 on 10/10 cells**, and the
saved **weights are bit-identical -- maximum absolute difference exactly 0.0** across all 16
predictor and encoder tensors. `run_dir`, `config_hash` and `ckpt_sha1` all differ, because
`exp_name` and `results_root` enter the run identity; the checkpoint hash differs by
serialized metadata alone, verified by loading both and comparing tensors. **14.6's
self-sufficiency costs compute and nothing in consistency.**

The first probe of this control returned 3/10 and was wrong: it matched cells on
`h_fr = 25` without matching on `n = 32`, and all three of campaign A's datasets carry
`h_fr = 25`. Re-run on the correct pairing. The corrected probe is reported with its
original error because a probe whose correction is stated is a probe that can be relied on.

## 26. Two dated precisions after the last root (dated 2026-08-13)

Precisions on existing sections, not amendments, on 20.4's precedent. **No root is declared
here and none is authorized.** 25.5's ceiling of two roots holds; neither precision below
runs anything -- the first is a closed-form computation, the second a re-reading of numbers
already published.

### 26.0 Why `on-stable-nolift` is worse than both arms: it is refused by the coordinates, not by the optimizer

25.3's reading rule placed `nolift` between `off` and `on-stable`. It landed above both
(3.1x `on-stable` at `n = 8`, 13.1x at `n = 256`). The reason is structural and computable
in closed form.

**The claim.** In STATE coordinates the true generator
`A = [[0, I], [-K, -c*I]]` does not belong to the block-diagonal rotation-scaling class for
ANY pairing of the four coordinates. Two immediate counterexamples, verified:

- pairing `(1,2)(3,4)` requires `A[1,3] = A[3,1] = 0`; measured **1.0** and **-1.5**;
- pairing `(1,3)(2,4)` requires `a11 = a22` inside the block; measured **0.0** against
  **-c**.

**Quantified**, the same way 23.7 quantified `delta(c)`: the minimal Frobenius distance from
`A` to the set of block-diagonal rotation-scaling generators, minimised over the three
pairings AND over each block's `(alpha_i, omega_i)`. For a pairing, the best block for
`[[a, b], [c, d]]` is `alpha = (a+d)/2`, `omega = (c-b)/2`, residual
`((a-d)^2 + (b+c)^2)/2`; off-block entries must vanish and contribute in full.

| `c` | `delta(c)` (spectral, 23.7) | distance in the IDENTITY basis | `‖A‖_F` | relative |
|---|---|---|---|---|
| **0.10** | **0.000** | **1.0654** | 2.9614 | **36 %** |
| 2.30 | 0.363 | 2.5328 | 4.3966 | 58 % |
| 2.60 | 0.706 | 2.8080 | 4.7191 | 60 % |
| 3.00 | 1.029 | 2.9580 | 5.1720 | 57 % |

**And in the best basis it is exactly zero.** At `c = 0.10` the four eigenvalues are two
conjugate pairs, so the real Jordan form IS block-diagonal rotation-scaling: measured
distance **1.97e-15**, with blocks `(-0.05, +-1.5187)` and `(-0.05, +-1.0902)`.

So at the configuration where campaign B's declared axis reads `delta = 0.000` -- "the prior
is exact" -- the prior is exact **in a basis**, and that basis is not the identity. `nolift`
is forced into the identity basis and therefore faces a misspecification of **1.0654**,
where `on-stable` faces **0**.

**The bounded conclusion.** `psi` and `pi` are not an overhead on top of the structured
transition: they ARE THE CHANGE OF BASIS the structure requires. `_private/PLAN_REFOCUS.md`
section 7 already declared that exact representability assumes `A` diagonalisable -- i.e.
that a basis EXISTS. This section measures that the basis is not the identity, and how far
from it. **25.3's branch 1 is refuted for a structural reason, not an empirical one**: it
could not have held.

**What this suggests about campaign A, written as the most explanatory CANDIDATE and not as
established**: the penalty is neither `psi`/`pi` alone nor the transition alone, but their
COUPLING -- the structure demands a basis, the basis has to be learned jointly with the
operator, and 20.1 already measured that the joint problem is not reached by gradient
descent even where ordinary least squares recovers the operator to 2.11e-08. **No experiment
is declared to test this**, and 25.5's ceiling is why.

### 26.1 Campaign B: the ratio confirmed the predicted sign, and hid the mechanism

23.5's estimand for campaign B is the SIGN of the slope of the structure gain against
`delta(c)`, and the measured slope confirms the prediction: the cost ratio rises **+9.61**
(`on-free`) and **+10.52** (`on-stable`) per unit of `delta`. **That reading stands.** What
follows does not overturn it; it names what the ratio was made of.

**The absolutes, which the ratio does not show:**

| `delta` | `off` | `on-free` | `on-stable` | persistence `e_25` | linear `e_25` |
|---|---|---|---|---|---|
| 0.000 | **0.2624** | 1.0790 | 0.9029 | 1.3925 | 1.5323 |
| 0.363 | 0.0891 | 1.1690 | 1.0294 | 2.1225 | 2.4175 |
| 0.706 | 0.0845 | 0.9716 | 1.1200 | 2.1489 | 3.0029 |
| 1.029 | **0.0715** | 1.1122 | 1.0572 | 2.1867 | 4.0641 |

- `on-stable` spans **x1.24** across the whole axis, `on-free` **x1.20** -- **flat**. Both
  sit PINNED at `m_low ~ 1`, i.e. they beat the two trivial baselines by very nearly exactly
  the factor `gamma = 5` the gate demands, and by no more, at every level of misspecification
  including `delta = 0`.
- `off` spans **x3.67**, and in the direction that matters: it IMPROVES as `c` grows.

**So the growth of the ratio is a DENOMINATOR effect.** The structured arms do not degrade
with misspecification; the unstructured arm improves. The pre-registered sign is confirmed
and **its mechanism is not the one the prediction supposed** -- the prediction assumed a
structure whose advantage erodes as the prior degrades, and what is measured is a structure
whose absolute performance is insensitive to the prior at all.

The baselines belong to this reading because `m_low = e_25 * gamma / min(persistence, linear)`
and they move with the damping: persistence `e_25` goes 1.39 -> 2.19, linear 1.53 -> 4.06.
`m_low` is therefore a ratio of ratios, and both terms move.

23.5's "step, not ramp" observation stands and is reinforced: the ratio jumps between
`delta = 0` and `delta = 0.363` and is flat after, which is what a denominator effect looks
like when the denominator makes its move early.

### 26.2 Honesty declaration

- **Neither precision runs anything.** 26.0 is closed-form arithmetic on a frozen system;
  26.1 re-reads published numbers. 25.5's ceiling holds.
- **26.0's conclusion is bounded and 26.1's is a correction of mechanism, not of sign.** The
  candidate explanation for campaign A is labelled a candidate.
- **A metric caveat, stated rather than smoothed**: `delta(c)` is a SPECTRAL distance and
  26.0's is a FROBENIUS distance on the generator. They are not commensurable term by term;
  the comparison that carries the argument is within ONE metric -- identity basis **1.0654**
  against best basis **1.97e-15**, at the same `c`.
- **Third time in this cycle that an aggregate hid a mechanism** (the two earlier ones are on
  record in `_private/`). Disaggregation is not a reading precaution here; it is the
  measurement.


### 26.3 A trace limit, recorded (dated 2026-08-17)

Sections 21 through 26 entered git as **one retrospective commit** (`752502d`,
2026-08-14), unlike every other amendment in this file, which carries its own commit.
Their internal declaration-before-run ordering is therefore attested by the dated text
alone; git cannot independently confirm, for that batch, that 23.6 preceded 24.0. This
is recorded as a **known limit of the trace, not repaired**: rewriting history to
manufacture per-section commits after the fact would be worse than the gap it papered
over. From 27 onward the one-commit-per-declaration discipline resumed and is
git-verifiable (27's declaration `3a7d8f0` precedes its results `38829d1` in commit
order).

## 27. Amendment 17 — the ceiling of 25.5 is lifted for ONE root: the backbone axis, never measured (dated 2026-08-14)

### 27.0 Scope, and the measurement gap that justifies lifting a binding clause

25.5 declared two roots the last of that cycle. Both ran. This section lifts that ceiling
for **exactly one further root** and for one reason, which is a gap and not a hope.

**Every cell of the oscillator cycle ran with the `mlp` backbone.** Verified by scan over
the six roots' own `config.yaml` files, by two independent methods that agree: the distinct
values of `family` across `results_sweep_osc/`, `results_osc_n/`, `results_osc_delta/`,
`results_osc_decomp/` and `results_osc_hfr/` are `mlp` and `mlp_k` and nothing else.
`results_ssm_delta/` adds `min_ssm` and `ssm_k`, on its own one-factor question. The
family `transformer` has **never** run on `CoupledOscillators` at all; `transformer_k` ran
five cells there once, as amendment 10's D3 control, never as a comparison.

So this repository's principal result — that explicit structure in the latent predictor
does not pay on `CoupledOscillators` — is measured **on one backbone**. The design this
project was built around is `3 backbones x 2 structures`. That factorial has never been
run on the system where the arms demonstrably learn. It is the one measurement whose
absence the six roots cannot repair by any re-reading, and no amount of further analysis
of the existing 164 cells produces it.

**This is a lift, not a repeal.** 25.5's ceiling is replaced by this section's ceiling:
one root, declared in full below, and no other. Campaign B's status under 25.5 is
untouched.

**The author's decision, dated 2026-08-14**, taken with the six roots' results in hand.
Decided, not derived.

### 27.1 The root — `results_primary_factorial/`, 30 cells

A **separate results root**, named so that no cell of it can ever be confused with, or
merged into, a cell of the six roots that precede it. 14.6's self-sufficiency applies in
full: this root re-runs its own reference cells and reads nothing out of
`results_osc_hfr/` or any other root, even where a cell would be nominally identical.

One stratum, six arms, five seeds:

| axis | levels | count |
|---|---|---|
| stratum | `CoupledOscillators`, `OURegime(theta_ou = 1.0)`, `n = 32`, `T = 120`, `dt = 0.05` | 1 |
| backbone | `mlp`, `transformer`, `ssm` | 3 |
| structure | `off`, `on` | 2 |
| seed | 0, 1, 2, 3, 4 | 5 |

`N_expected = 1 x 3 x 2 x 5 = 30`. The six arms resolve to exactly
`{mlp, mlp_k, transformer, transformer_k, min_ssm, ssm_k}`. `koopman_ssm` is not a cell
of this grid, as 13.x already fixed.

Held fixed, all inherited and none tuned here: `d_x = d_z = 4`, `d_u = 1`,
`target_params = 10000` (14.2), `epochs = 8000` (16.1), `h_fr = 25` (15.2),
`grad_clip = 100` (17.1), `step_subsample = 1.0`, `representation.kind = identity`
(so `e_recon = 2.60e-08` and the measurement scores the predictor, 19.1), `mu_j = 0.0`.

The `on` treatment is **one wrapper, identical across the three backbones**:
`delta = 0.05`, `radius = bounded`, `r = 2`. `bounded` is the level under which 25.8's
principal result was measured (`on-stable`), and it is carried unchanged rather than
re-chosen.

The `ssm` arms additionally carry `delta_max = 0.20`. This is not a tuning choice: 24.0
measured **0/10 cells complete at `delta_max = 10.0` and 10/10 at `4*dt = 0.20`**, and
23.6 fixed `4*dt` as a physical bound before that measurement. Running the ssm arms at
the value under which 18.0 declared the family dead would measure the clamp, not the
backbone.

**Budget parity, verified before this section was committed**, by an independent parameter
count against `sum(p.numel())` at `d_z = 4`, `d_u = 1`, `target_params = 10000`:
`mlp` 10074 (+0.74 %), `mlp_k` 9997 (-0.03 %), `transformer` 10324 (+3.24 %),
`transformer_k` 10006 (+0.06 %), `min_ssm` 10069 (+0.69 %), `ssm_k` 10003 (+0.03 %).
**6/6 within the +-5 % of 13.7**, which that section recorded as the fragile condition.

### 27.2 The rates, and the confound they carry — named, not removed

23.3 selected `lr` **per arm, for itself**, by an argmin of that arm's own `e_25` on the
val split. That sweep covered the `mlp` arms only. This root does **not** re-run it and
does not extend it. Each family enters at the rate under which it is already known to
train on this system, and nowhere else does a rate come from:

| arms | `lr` | provenance |
|---|---|---|
| `mlp`, `mlp_k` | `1e-3` | 23.3's argmin, applied unchanged (23.5(2) forbids re-selection) |
| `transformer`, `transformer_k` | `1e-3` | the rate under which `transformer_k` completed 5/5 on this system as amendment 10's D3 control |
| `min_ssm`, `ssm_k` | `3e-4` | 24.0's rate, inherited from 16.2/17.1, under which the family completed 10/10 at `delta_max = 4*dt` |

**The confound this creates, stated in full.** The three backbones do not share a rate,
and no rate on this grid was selected by a sweep run on this grid. Two consequences,
which point in opposite directions and are both real:

1. **Within a backbone, the structure contrast is clean.** `off` and `on` of the same
   backbone share their rate, their dataset, their seed and their budget. The ON-OFF
   contrast is therefore fair for each backbone taken alone, and that is the contrast
   prediction 3 of 27.3 is about.
2. **Across backbones, the comparison is confounded with rate suitability.** A backbone
   ordering read off this root cannot separate "this backbone is better here" from "this
   backbone's inherited rate happens to suit this stratum better". Predictions 1 and 2 of
   27.3 are exposed to this and their verdicts carry it in every statement.

The alternative — a per-arm sweep on this grid — was considered and refused: it triples
the root's cost, and 23.3's own rule already returned its argmin **at the edge** of the
declared list `{1e-4, 3e-4, 1e-3}` with the criterion still decreasing, so a sweep over
the same list would inherit that defect rather than fix it. **Naming the confound is
cheaper and more honest than a sweep that would not resolve it.** No off-list rate is
tried, here as in 23.3.

### 27.3 The reading rule, written before the numbers

**The primary statistic is `VPT` at `tau = 0.5`**, as 12.x pre-registered it from Vlachas
et al. 2020 (arXiv:1910.05266): the first horizon at which the normalized state error
crosses `tau`, on the free rollout in true-state coordinates, higher is better. The
per-run value is the **median over the run's evaluation episodes**, which is what
`gate.json` already writes as `vpt_median`. The per-cell value is the mean over the five
seeds, with `SE = std(ddof = 1) / sqrt(5)`.

**One endpoint, one set of bytes.** The `e_h` curve of the figures, the VPT of the
ordering, and `m_low` are all read out of the same `gate.json` produced by one free
rollout per run. No second rollout, no per-run OLS readout, no auto-decoded
reconstruction. `m_low` is reported alongside for continuity with the six roots but is
**not** the ordering statistic here.

**Censoring is conservative.** A run whose error never crosses `tau` within the scored
horizon has `vpt_status = "censored"`. Censored values are published, counted, and shown
in the figure. **Any order comparison whose outcome depends on a censored value is
recorded `indeterminate`** — never resolved in either direction, never replaced by
`Hmax + 1`, never dropped. Same for an exact tie.

**The verdict is descriptive and pre-registered, not confirmatory.** With `n = 5` this
section declares no test, no alpha and no multiplicity correction, and **no `p`-value will
be computed or reported for these orders**. A relation is `supported` when the paired
contrast has the predicted strict sign and its `+-1 SE` interval excludes zero;
`contradicted` when it has the opposite strict sign under the same condition;
`indeterminate` otherwise, including every censored or tied case.

**Eligibility.** A cell is eligible only with **5/5 seeds terminal and `ok`**. If any of
the five is `does_not_learn`, `divergence` or `missing`, the cell is named in the table
and the figure with its five statuses and **no mean, SE or aggregated VPT is computed for
it**. It is never silently excluded and never averaged over the survivors. A divergence
is a result (15.3), not a gap.

**Pairing.** An ON-OFF contrast uses the same seed, the same stratum, the same dataset
hash and the same representation. Pairing by file order or by group mean is forbidden.

**The nine predicted relations**, taken unchanged from `_private/PLAN_REFOCUS.md` and not
modified here, read in the stratum:

1. structured: `ssm_k > transformer_k > mlp_k`;
2. unstructured: `min_ssm > transformer > mlp`;
3. structure effect: `mlp_k > mlp`, `transformer_k > transformer`, `ssm_k > min_ssm`;
4. **no order is predicted** between `min_ssm` and `mlp_k`, and none will be reported.

Nine relations, three within each of the first three groups. The pair in 4 is recorded
`not_tested` and no verdict is written for it.

**One leg of prediction 3 is already known to fail on this system, and it is left
standing.** 25.8 measured `on-stable` worse than `off` on `mlp` at both training
horizons, and 23.5 measured it worse at every sample size and every misspecification
level. `mlp_k > mlp` is therefore expected to be `contradicted` here. It is **not**
withdrawn, softened or rewritten: a prediction that has lost once is worth more standing
than quietly replaced, and the question this root actually asks is whether the negative
**generalizes across backbones** or is a property of the `mlp` dorsal alone. Both answers
are publishable and neither repairs the other.

### 27.4 The figures, engaged before the data

Three figures, fixed now, with no panel to be added or removed after the numbers exist:

- **Fig. 1** — `e_h` curves in true-state coordinates, mean `+-1 SE` over five seeds, six
  cells, non-eligible cells named rather than drawn as a curve;
- **Fig. 2** — VPT, the **five raw per-seed values** per cell with their statuses, censored
  values visibly marked;
- **Fig. 3** — the three paired ON-OFF contrasts, five seeds each.

Each figure writes a sidecar CSV and a hash, and each caption names system, regime, `H`,
`sigma_train`, `tau`, the censoring convention, `n = 5`, and every non-eligible cell.
A figure that does not reproduce the table it cites does not ship, and the gap is the
result (23.5(3)).

### 27.5 What this root does NOT do, and the ceiling it leaves behind

It does not measure the pendulum. 21.4 stands unmodified: **M4 is not crossed**, seven of
eight positive-control cells on `DampedPendulum` do not pass, and **no claim of this root
extends to that system**. Every statement produced here carries "on `CoupledOscillators`,
`OURegime`, `n = 32`, `h_fr = 25`", exactly as 25.6 requires.

It does not test 26.0's coupling candidate, and it does not touch
`representation.kind`. 20.4's measurement — that the pre-trained representation's
`e_recon` floor is 28.8x above the `1e-3` bound on this system — is why the second is not
attempted incidentally.

No threshold moves. `gamma = 5`, `gamma_1 = 2`, `tau_1`, the `1e-3` bound, the 8000-update
budget and the loss are untouched.

**This is the LAST root declared. No further root follows it, whatever it returns.** The
ceiling of 25.5 is restored by this sentence, over this section.

### 27.6 Honesty declaration

- **Decided, not derived**: that the ceiling is lifted at all; that it is lifted for one
  root and not a programme; the single stratum and its `n = 32`; the inherited per-family
  rates; `radius = bounded` as the common treatment. Each has an argument above; none is a
  measurement.
- **The confound that cannot be removed is named twice** (27.2), once for what it spares
  and once for what it costs, and the rejected alternative is stated with its reason.
- **A prediction expected to fail is published unmodified** (27.3), because the value of
  this root is the same whichever way it goes.
- **A risk is declared rather than hedged**: the ssm family's 10/10 completion was measured
  at `n = 256` (24.0) and this root runs at `n = 32`. 24.0 attributes the flip to
  `delta_max` and records that nothing else moved, but `n` was not itself varied there. If
  the ssm arms diverge at `n = 32` **that is published as measured** and no rate, clamp or
  sample size is changed to rescue them.
- **Budget parity was verified before this section was written**, not after, and the six
  numbers are in 27.1.
- **R4**: this section precedes every run it authorizes, including this root's first cell.

### 27.7 Dated precision to 27.3 — the primary statistic is `e_H`, because VPT is censored by construction on this system (dated 2026-08-14)

27.3 named **VPT at `tau = 0.5`** the primary ordering statistic. That choice is a defect
of this section, and it is corrected here **before a single datum of this root exists**:
at the moment this precision is written, `results_primary_factorial/` contains **0
`gate.json` and 0 `run_complete.json`**. Nothing of this root has been read, because
nothing of it has been produced.

**The measurement that forces the correction**, taken on the roots that already exist and
not on this one: across the **162 scored runs** of `results_osc_n/`,
`results_osc_delta/`, `results_osc_decomp/`, `results_osc_hfr/` and
`results_ssm_delta/`, **1179 of 1346 evaluation episodes are censored — 87.6 %** — and
only **30 of 162 runs** reach `e_h >= 0.5` at all within `H = 25`. The normalized state
error on `CoupledOscillators` at this horizon has a median `e_h_gate` of **0.1592**, a
factor of three below `tau`.

VPT is therefore not censored by accident here; it is censored **by construction**. Under
27.3's own conservative censoring rule — any comparison depending on a censored value is
`indeterminate` — a VPT-primary reading would return `indeterminate` on very nearly every
one of the nine relations, whatever the models did. A statistic that cannot separate the
arms is not a conservative choice; it is an empty one.

**The correction.** The primary ordering statistic becomes **`e_H` at `H = 25`**, the
field `measures.e_h_gate` of the `gate/1` artifact, **lower is better**. This is option
(b) of the phase's own open decision on the primary statistic, which listed it as
admissible alongside VPT; it is not a new invention. Per-cell value: mean over the five
seeds, `SE = std(ddof = 1) / sqrt(5)`, unchanged.

**What does not change.** The endpoint is still one free rollout in true-state
coordinates, and `e_h_gate` is read from the **same bytes** as the `e_h` curve of Fig. 1
— 27.3's one-endpoint rule is strengthened by this change, not weakened, since `e_h_gate`
IS a point of that curve. `m_low` continues to be reported. The eligibility rule, the
pairing rule, the descriptive verdict without a `p`-value, and the nine relations with
their one untested pair are untouched.

**VPT is not dropped.** It was pre-registered and it stays published: Fig. 2 shows the
five raw per-seed VPT values per cell **with their censoring status and the censored
count**, exactly as 27.4 engaged. What changes is that the ORDER VERDICTS no longer rest
on it. Removing a pre-registered quantity because it turned out uninformative is the
fault this repository exists to forbid; publishing it as measured, and saying why it
could not carry the ordering, is not.

**Note on direction.** `e_H` is an error: **lower is better**, the opposite of VPT. Every
predicted relation `X > Y` of 27.3 means "X is better than Y", so on this statistic it is
tested as `e_H(X) < e_H(Y)`. The sign convention is fixed here, before the numbers, and a
reading that inverts it is a defect the negative tests must catch.

**R4**: this precision precedes every reading it governs. It does not authorize any run,
and it changes no threshold: `tau = 0.5`, `gamma = 5`, `gamma_1 = 2`, `tau_1`, the budget
and the loss are untouched.

### 27.8 What the primary factorial returned (dated 2026-08-14)

`results_primary_factorial/`, **30/30 cells complete, zero divergence**, scored 30/30 on the
test split. Index: 30 rows, 30 unique identities, `missing = 0`, duplicates `0`, verified by
two independent methods that agree — the on-disk sentinels and `index.csv`.

**The measurement**, primary statistic `e_H` at `H = 25` (27.7), **lower is better**, mean
over five seeds with `SE = std(ddof = 1) / sqrt(5)`:

| family | `e_H` mean | SE | `m_low` median | gate verdict |
|---|---|---|---|---|
| `mlp` | **0.0702** | 0.0040 | 0.2624 | `learns` |
| `transformer` | 0.1197 | 0.0106 | 0.4464 | `learns` |
| `mlp_k` | 0.2153 | 0.0329 | 0.9029 | `does_not_learn` |
| `min_ssm` | 0.3353 | 0.0542 | 1.4218 | `does_not_learn` |
| `transformer_k` | 0.3681 | 0.0673 | 1.3101 | `does_not_learn` |
| `ssm_k` | **0.6189** | 0.2021 | 1.5229 | `does_not_learn` |

#### 27.8.1 The pre-registered verdict, as 27.3 declared it

**One relation `contradicted`, eight `indeterminate`, one `not_tested`.**

`transformer > mlp` is `contradicted`: paired difference **+0.0496 +- 0.0097**, i.e.
`transformer` is *worse*, where the prediction had it better. The other eight relations each
involve at least one cell that is not eligible under 27.3 — 20 of the 30 runs carry the gate
verdict `does_not_learn` — and 27.3's rule sends every such comparison to `indeterminate`.
`min_ssm` vs `mlp_k` stays `not_tested`, and no verdict is written for it.

**That is the confirmatory result of this root, and it is reported first because it is the
one that was declared first.**

#### 27.8.2 Why the cells fail, disaggregated — the failure is itself ordered

`does_not_learn` covers two different things, and the ventilation separates them:

| family | lower bound (arm-neutral) | upper bound (`tau_1`) | `near_miss` |
|---|---|---|---|
| `mlp` | **5/5** | **5/5** | 0/5 |
| `transformer` | **5/5** | **5/5** | 0/5 |
| `mlp_k` | 4/5 | 0/5 | 3/5 |
| `transformer_k` | 2/5 | 0/5 | 0/5 |
| `min_ssm` | 2/5 | 0/5 | 0/5 |
| `ssm_k` | **0/5** | 0/5 | 0/5 |

`ssm_k` fails even the arm-neutral lower bound on every seed: it does not beat the trivial
baselines at all. `mlp_k` passes that bound on 4 of 5 seeds and falls only on the upper
bound — the one 21.4 records as failed by seven of eight cells and whose surrounding
conclusion 22.0 withdrew. **The two labels are the same word and not the same fact**, and a
reading that did not disaggregate would have said so.

#### 27.8.3 The question 27.0 asked, and its answer

27.0 opened this root on one gap: the repository's negative was measured on one backbone.
**It generalizes to all three.** The paired ON-OFF contrast of `e_H` — the one comparison
27.2 certifies as confound-free, since off and on of a backbone share rate, dataset, seed
and budget — is positive on every backbone, meaning structure *costs*:

| backbone | contrast `on - off` | SE | five per-seed contrasts |
|---|---|---|---|
| `mlp` | **+0.1451** | 0.0351 | 0.0294, 0.0982, 0.2122, 0.1915, 0.1942 |
| `transformer` | **+0.2484** | 0.0670 | 0.1119, 0.2596, 0.2667, 0.4822, 0.1213 |
| `ssm` | **+0.2836** | 0.2379 | 0.0016, 1.2141, -0.0005, 0.2412, -0.0384 |

The `mlp` negative was not a property of the `mlp` dorsal. **The structured arm is worse on
three backbones out of three**, and the two whose rate provenance is strongest (`mlp`,
`transformer`) carry it with intervals well clear of zero.

#### 27.8.4 An exploratory reading, labelled as such and carrying no confirmatory weight

**This subsection is post-hoc.** It was written after the numbers existed, it was NOT
pre-registered, and under section 9 it is exploratory. It changes no verdict of 27.8.1.

27.3's eligibility rule conflates two things that this grid shows to be different:
*"the run produced a valid measurement"* and *"the run passed the two-sided positive
control"*. Every one of the 30 runs did the first — 30 complete sentinels, 30 `gate.json`,
zero divergence, a readable `e_h` curve everywhere. 20 did not do the second. The rule I
wrote discards the second group entirely, which is why eight relations are `indeterminate`.

Had eligibility been defined on the first, the nine paired differences are:

| relation | paired diff | SE | `+-1 SE` excludes 0 |
|---|---|---|---|
| `ssm_k > transformer_k` | +0.2508 | 0.2071 | yes |
| `transformer_k > mlp_k` | +0.1528 | 0.0661 | yes |
| `ssm_k > mlp_k` | +0.4036 | 0.2148 | yes |
| `min_ssm > transformer` | +0.2155 | 0.0588 | yes |
| `transformer > mlp` | +0.0496 | 0.0097 | yes |
| `min_ssm > mlp` | +0.2651 | 0.0534 | yes |
| `mlp_k > mlp` | +0.1451 | 0.0351 | yes |
| `transformer_k > transformer` | +0.2484 | 0.0670 | yes |
| `ssm_k > min_ssm` | +0.2836 | 0.2379 | yes |

**All nine positive, all nine excluding zero: all nine relations contradicted, and the
predicted ordering exactly inverted on both axes.** Predicted `ssm > transformer > mlp`
among the structured arms and again among the unstructured; measured
`mlp < transformer < ssm` in error, in both rows, with `off` beating `on` in every column.

**This is not promoted to a finding.** It is recorded because suppressing it would be worse
than labelling it, and because the honest object of this root is the contrast in 27.8.3,
which does not depend on it.

#### 27.8.5 Honesty declaration

- **The pre-registered rule was not moved to get a result.** 27.3's eligibility clause turns
  out to be the wrong instrument for this grid, and it is left exactly where it was written.
  The reading it produces — one relation out of nine — is published as the primary verdict.
- **This is the SECOND defect of my own section 27.3**, after the VPT choice corrected in
  27.7. The first was caught before any datum existed and was therefore repaired; this one
  was not visible until the cells were scored, and is therefore *recorded*, not repaired.
  The distinction between the two is the whole of R4 and it is why 27.7 is an amendment and
  27.8.4 is an observation.
- **Decided, not derived**: that the exploratory reading is published alongside rather than
  omitted or promoted.
- **The confound of 27.2 stands on 27.8.1 and 27.8.4 and NOT on 27.8.3.** The backbone
  ordering is confounded with rate suitability, and every ordering statement above carries
  it. The paired within-backbone contrast does not, and that is why it, and not the
  ordering, is what this root is reported as having measured.
- **Scope, per 25.6**: everything here holds under `CoupledOscillators`, `OURegime`,
  `n = 32`, `h_fr = 25`, `delta_max = 0.20` on the ssm arms, and the inherited rates of
  27.2. **21.4 is untouched: M4 is not crossed and no statement here reaches the pendulum.**
- **No threshold moved.** `gamma = 5`, `gamma_1 = 2`, `tau_1`, `tau = 0.5`, the 8000-update
  budget and the loss are exactly where they were.
- **The ceiling of 27.5 holds.** This root is the last one declared, and nothing here
  declares another.

### 27.9 Dated precision to 27.8.3 — what the three-backbone statement covers, and where it stops (dated 2026-08-15)

27.8.3 reports the paired ON-OFF contrast as "positive on every backbone" and the structured
arm as "worse on three backbones out of three". Both sentences are true of the means. Neither
holds at the same strength across the three columns, and this precision says where the
difference lies. **It changes no verdict**: 27.8.1 stands unmodified, 27.8.4 stands
unmodified, and every number below is already published in 27.8.3 and in
`assets/fig3_factorial_effects.csv` — read differently, not recomputed.

**The sign, counted per seed.** A descriptive count, not a test: 27.3's prohibition on
`p`-values is untouched and none is computed here.

| backbone | seeds with a positive contrast | mean | median | mean without the extremum |
|---|---|---|---|---|
| `mlp` | **5/5** | +0.1451 | +0.1915 | — |
| `transformer` | **5/5** | +0.2484 | +0.2596 | — |
| `ssm` | **3/5** | +0.2836 | **+0.0016** | +0.0510 |

On `ssm` the mean is set by one seed. Seed 1 contributes **+1.2141**; the other four are
+0.0016, -0.0005, +0.2412, -0.0384 — two of them within `1e-3` of zero, one negative.
Fig. 3 already draws all five points in every column, so the figure never claimed more than
this. The prose of 27.8.3 did.

**Why that column was expected to be uninformative, declared before the runs.** 27.6 records
the risk in writing: the ssm family's 10/10 completion was measured at `n = 256` (24.0) and
this root runs at `n = 32`, and "if the ssm arms diverge at `n = 32` **that is published as
measured**". 27.8.2 then measures the outcome — `ssm_k` fails the arm-neutral lower bound on
**5/5** seeds, `min_ssm` on **3/5**. The `ssm` column is therefore a contrast between two arms
of which neither reliably left the trivial baseline, and two of its five differences sitting
within `1e-3` of zero is what that looks like from the outside.

**The bounded statement, which replaces 27.8.3's last sentence as the citable one.** The
paired contrast is positive on all three backbones. It is carried by every seed on `mlp` and
by every seed on `transformer` — the two whose rate provenance 27.8.3 already names as the
strongest. On `ssm` it is not carried by the seeds and **is not counted as a third
independent confirmation**. What this root establishes is that the repository's negative
generalizes across **two** backbones; the third is uninformative, for a reason 27.6 declared
before the first cell ran.

**The same qualification inside 27.8.4.** Of its nine exploratory differences, three sit
between 1.19 and 1.88 SE of zero — `ssm_k > transformer_k` (1.21), `ssm_k > mlp_k` (1.88),
`ssm_k > min_ssm` (1.19) — and all three involve `ssm_k`. The remaining six are at 2.31 SE
and above. "All nine exclude zero at `+-1 SE`" is literally true and is the pre-registered
wording of 27.3; the spread behind it is published here so the sentence is not read as nine
equally supported statements. That subsection remains exploratory and is still promoted to
nothing.

**R4 note.** This precision is written after the numbers, like 27.8.4 and unlike 27.7. It
therefore *records* rather than repairs, and it is deliberately confined to **weakening a
descriptive claim of my own**. No eligibility rule, threshold, statistic or verdict moves. A
precision that strengthened a claim after the fact would not be admissible under 9; this one
only takes back reach that the seeds do not support.

## 28. Amendment 18 — the ceiling of 27.5 is lifted a second time, for the one experiment 26.0 named and declined (dated 2026-08-15)

### 28.0 What is being overridden, and the argument for overriding it

27.5 closes with: "**This is the LAST root declared. No further root follows it, whatever it
returns.** The ceiling of 25.5 is restored by this sentence, over this section." This section
overrides that sentence, once, and gives the argument in full — because a ceiling lifted
without an argument was never a ceiling.

**What the ceiling exists to forbid.** 25.5 and 27.5 exist to stop one specific failure: an
author who keeps opening roots, each with one more knob, until one returns the sign he
wanted. The test of whether *this* section is that failure is not the author's account of his
own motives. It is a question about the object: **does the new root re-ask a question that
already has an answer?**

It does not. The six oscillator roots and 27's factorial all ask one question in one set of
coordinates — *in the identity basis, does the structured class beat the unstructured one?*
That question is answered, and answered negatively: at `n` in {8, 32, 256}, at `delta(c)` from
0.000 to 1.029, at `h_fr` in {5, 25}, and on two backbones carried by every seed (27.9).
**Nothing below re-opens that question, and nothing below can repair it.**

26.0 asks a different question, and its own closing sentence records that it is not testing
it:

> the structure demands a basis, the basis has to be learned jointly with the operator, and
> 20.1 already measured that the joint problem is not reached by gradient descent even where
> ordinary least squares recovers the operator to 2.11e-08. **No experiment is declared to
> test this**, and 25.5's ceiling is why.

The question below was therefore named, argued, and deliberately left unmeasured for a
**governance** reason and not a scientific one. Running the experiment that an earlier section
had already specified, and had declined only because of a self-imposed stop, is not the
failure the ceiling guards against. Re-running a losing comparison with a new knob would be,
and that is not what follows.

**Decided, not derived**: that the ceiling is lifted at all; that it is lifted a second time
after 27.5 declared it final; that it is lifted for ONE root and not a programme; that the
root is `mlp`-only. The author takes this decision on 2026-08-15 **with 27's numbers already
in hand**, and that fact is itself a reason to distrust the decision. It is why 28.4 —
what would falsify the prediction, and the commitment to publish that outcome with the same
prominence — is written here, before the first cell runs, and not after.

### 28.1 The question, stated so that both answers are publishable

26.0 measured, in closed form, that at `c = 0.10` — the configuration where campaign B's
declared axis reads `delta(c) = 0.000`, "the prior is exact" — the true generator sits at
Frobenius distance **1.0654** from the block-diagonal rotation-scaling class *in the identity
basis*, which is **36 %** of `‖A‖_F = 2.9614`; and at distance **1.97e-15** from that same
class in the basis of its own real Jordan form, with blocks `(-0.05, +-1.5187)` and
`(-0.05, +-1.0902)`.

The declared misspecification axis of 23.7 therefore moved **one** of two misspecifications
and held the other fixed at 36 % throughout. Every arm carrying `lift = "mlp"` had to discover
that change of basis by gradient descent, jointly with the operator, from the one-step loss
folded through a 25-step closed rollout.

**The question.** Is the penalty this repository has measured seven times a property of the
structured class, or a property of *asking gradient descent to find the basis and the operator
at the same time*?

The root answers it by removing exactly one thing — the requirement to discover the basis —
and changing nothing else.

### 28.2 The root — `results_basis/`, 15 cells, three arms x five seeds

**Stratum, inherited verbatim from 27.1 so the contrasts are commensurable with 27.8.3**:
`CoupledOscillators`, `OURegime(theta_ou = 1.0)`, `c = 0.10`, `n = 32`, `h_fr = 25`, `H = 25`,
8000-update budget, identity shared representation, the `mlp` rate of 27.2 (`1e-3`). One
dataset build, one `data_hash`, shared by all fifteen runs.

**One backbone, `mlp`, and the reason.** 27.9 leaves `mlp` as the column where the negative is
carried by 5/5 seeds with the tightest SE (0.0351) and the strongest rate provenance. A
mechanism claim is tested where the effect it must explain is best measured, not where the
measurement is weakest. `transformer` and `ssm` are **not** run here, and no statement of this
root extends to them.

**The three arms:**

| arm | `structure` | `lift` | `psi` / `pi` |
|---|---|---|---|
| `off` | off | n/a | none |
| `on-learned` | on | `"mlp"` | learned jointly, exactly 27's `mlp_k` |
| `on-oracle` | on | `"frozen"` | `psi = T`, `pi = T^-1`, **fixed, no gradient** |

`T` is the similarity transform to the real Jordan form of the true generator `A` at
`c = 0.10`, computed in closed form by `real_modal_basis`, and written to the root as an
artifact with its own hash before any run starts. `T = P^-1` where `P` is the SAME matrix
`real_modal_distance` builds, so the two routines cannot disagree about which basis is meant.

**A numerical discrepancy, named here rather than left to be discovered.** `real_modal_basis`
reaches the block form by inverting `P`; 26.0's `real_modal_distance` reached it by solving
`P x = A P`. The two agree to machine precision but not bit-for-bit: the residual distance is
**1.41e-15** through the inversion path against the **1.97e-15** 26.0 published through the
solve path. Both are zero at float64 and neither is a measurement — the citable statement
remains "exactly zero in the best basis, against 1.0654 in the identity basis". No threshold
depends on which of the two is quoted, and `tests/test_frozen_lift.py` pins both under
`< 1e-12` rather than pinning either value. `lift = "frozen"`
is a new third level beside `"mlp"` and `"identity"`, validated by refusal like the other two
(`_VALID_LIFT`), and requires `2r == d_z` for the same reason `"identity"` does.

**`T` is frozen as a literal, not recomputed at run time, and the normalisation is a
DECISION (added 2026-08-15, before any run).** Two facts force this, both measured rather
than assumed:

- `numpy.linalg.eig` does not guarantee eigenvector **scaling** or ordering across LAPACK
  builds. Recomputing `T` on another machine yields a different matrix realising the same
  basis, so a recomputed basis would make this root non-reproducible across platforms while
  looking deterministic.
- The block form is **invariant** under per-block rescaling of `T` — verified: rescaling by
  `diag(3, 3, 0.2, 0.2)` leaves `T A T^-1` block-diagonal rotation-scaling with the same two
  pairs — but the **latent scale is not**: the same rescaling moves `E‖T z‖ / E‖z‖` by a
  factor of 2.1. This is the ψ/π scale gauge this project already met, reappearing on the
  oracle side.

The matrix actually used therefore ships in `basis_root.yaml` as a literal with its
`sha256 = e026f402194929b09fdb6c7c88fe801e5d945131daa537947cd13ac7b8736be5` over its float64
bytes, and its properties are **recomputed in `tests/test_basis_root.py` rather than
trusted**: block form to `1.3e-15`, blocks `(-0.05, +-1.5187)` and `(-0.05, +-1.0902)`,
singular values `[1.1966, 1.3494, 1.4894, 1.8208]`, `cond = 1.5216`, and
`E‖T z‖ / E‖z‖ = 1.476`. The normalisation is numpy's unit-norm eigenvectors, which is
**decided, not derived** — it is one representative of a gauge orbit, chosen because it
leaves the latent scale within 1.5x of the identity's and therefore does not hand the oracle
arm a conditioning advantage on top of the basis. The dynamics themselves are scale-invariant
(a global rescaling of `h` is a similarity transform and leaves `A_bar`'s spectrum fixed), so
this choice affects the optimisation landscape and nothing about what the arm can represent.

**`on-oracle` is an ORACLE arm and is labelled as one everywhere it appears.** `T` is derived
from the true `A`, which no arm is otherwise allowed to see. This root therefore measures a
**ceiling** — what the structured class achieves when the basis is given — and `on-oracle`
is not a fair competitor to `off`. Any sentence that reports 28.3's P3 without the word
"oracle" is a misreport of this section.

**Budget — matching is SUSPENDED, not applied. Corrected 2026-08-15, before any run.**

*What this paragraph said when 28 was first committed (`bf901bc`), and why it was wrong:* it
said `match_budget` sizes the backbone against `on-oracle`'s zero-parameter lift "exactly as
25.3's width note prescribes". 25.3 and 23.2 prescribe the opposite, and the code implements
the opposite. `budget.py` suspends matching for a zero-parameter lift precisely so that the
matcher cannot widen a backbone to compensate — "measuring width while calling it structure,
that is the mu_J fault, and it is the one this section must not reproduce". Sizing
`on-oracle` up to `on-learned`'s parameter count would have made the two arms differ in TWO
ways and attributed nothing. The error was mine, it was caught while scoping the
implementation, and no cell had run.

*What holds instead.* `on-oracle` inherits `on-learned`'s `width` and `psi_hidden`
**exactly**; its backbone and head are parameter-identical; its `psi`/`pi` are replaced by
frozen linear maps carrying **zero trainable parameters** and `2 * d_z^2` non-trainable ones
(`T` and `T^-1`). The arm therefore comes out smaller in trainable parameters by exactly the
learned lift pair — **and that difference is the quantity this root exists to attribute**, so
removing it would remove the measurement. This is the same treatment `on-stable-nolift`
receives under 23.2, for the same stated reason.

The three realised counts — total and trainable, per arm — are written into this section
**before** the first run, not after.

**The three realised counts (added 2026-08-17, before any run — the sentence above, kept).**
Measured by construction and pinned by `tests/test_basis_root.py`
(`test_the_frozen_lift_oracle_arm_inherits_on_learned_s_budget_exactly`,
`test_the_budget_tolerance_exempts_the_frozen_arm_by_name_not_by_widening`):

- `off` (`mlp`): total = trainable = **10074** (+0.7% against `target_params = 10000`, inside
  the ±5% contract that applies to matched arms).
- `on-learned` (`mlp_k`): `width = 85`, `psi_hidden = (87,)`, total = trainable = **9997**
  (−0.03%).
- `on-oracle` (`mlp_k`, `lift="frozen"`): `width = 85` and `psi_hidden = (87,)` inherited from
  `on-learned` exactly; total **8455**, trainable **8423**. The 32 non-trainable entries are
  `2 * d_z^2` — precisely `T` and `T^-1`, nothing else. The trainable gap to `on-learned` is
  **1574**, exactly the learned `psi`/`pi` pair, which is the quantity this root exists to
  attribute. The arm sits 15.8% below target in trainable parameters and raises nothing: the
  ±5% contract exempts it **by name** (`budget.py`'s `zero_param_lift`), not by a widened
  tolerance — an ordinary arm that misses its target still raises, and the suite pins that
  contrast.

**Cost is measured, not estimated.** One cell of `on-oracle` runs first, alone; its wall-clock
and its seconds-per-update are recorded here; the remaining fourteen are launched only after
that number exists. This clause is in response to a measured failure of estimation elsewhere
in this project and is not decorative.

### 28.3 The reading rule and the three predictions, written before the numbers

**Primary statistic: `e_H` at `H = 25`**, in true-state coordinates on the free rollout, lower
is better, read out of the same `gate.json` as 27.7's — one endpoint, one set of bytes, no
second rollout. Per-cell value is the mean over five seeds with `SE = std(ddof = 1) / sqrt(5)`.

**Eligibility — the 27.8.4 repair, applied prospectively.** A cell is eligible when its five
runs produced a **valid measurement**: terminal sentinel, `gate.json` written, a readable
`e_h` curve, no divergence. The gate's positive-control verdict (`learns` /
`does_not_learn`) is published per cell beside the number and is **never** used to exclude a
cell from a contrast. 27.3 tied eligibility to the positive control; 27.8.4 measured that this
conflates "the run produced a valid measurement" with "the run passed the two-sided control"
and sent eight of nine relations to `indeterminate`. That defect was **recorded** in 27.8.5
rather than repaired, because it surfaced after scoring. Here it is repaired, in a section
written before any datum exists. That ordering is the whole difference and it is why this
clause is admissible.

**Pairing.** Every contrast is within-seed: same seed, same `data_hash`, same rate, same
budget, same stratum. Pairing by file order or group mean is forbidden.

**Verdict.** As 27.3: `supported` when the paired contrast has the predicted strict sign and
its `+-1 SE` interval excludes zero; `contradicted` on the opposite strict sign under the same
condition; `indeterminate` otherwise. No test, no alpha, no multiplicity correction, and **no
`p`-value will be computed or reported**. In addition, and declared now as a consequence of
27.9: **the per-seed sign count is published for every contrast**, beside its mean and SE, so
that no mean set by a single seed can be read as a carried effect.

**The three predictions:**

- **P1 — `on-learned` > `off`** (structure costs when the basis must be learned). This
  replicates 27.8.3's `mlp` column, `+0.1451 +- 0.0351`, inside a fresh root. Expected
  `supported`. It is a control on this root, not a discovery: if P1 fails to replicate, the
  root's other two contrasts are uninterpretable and 28.4 says what happens then.
- **P2 — `on-oracle` < `on-learned`** (giving the basis away helps the structured arm). This
  is 26.0's coupling candidate stated as a measurable sign, and it is the prediction this
  root exists to test.
- **P3 — `on-oracle` < `off`** (with the basis given, the structured class finally beats the
  unstructured one). This is the strong claim. It can fail while P2 holds, and 28.4 declares
  in advance how that outcome is reported.

### 28.4 What falsifies, and what each outcome is allowed to be called

Written before the runs, and binding on the reading:

- **P2 `contradicted` or `indeterminate`** — handing over the basis does not help.
  **26.0's coupling candidate is refuted as the explanation of campaign A's penalty**, and
  that sentence is published in the README head with the same prominence any positive would
  have received. This is the outcome that costs the author the most and it is the reason the
  clause exists.
- **P2 `supported`, P3 `contradicted` or `indeterminate`** — the basis is a real part of the
  penalty and not the whole of it. Reported as: *the coupling is a contributing mechanism; it
  is not sufficient to reverse the sign.* It is **not** reported as "structure works when you
  fix the basis". This is, on the author's own reading of 20.1, the most likely outcome, and
  naming it in advance is what stops it from being spun on arrival.
- **P2 and P3 both `supported`** — the diagnosis holds end to end: the class was never the
  problem, the joint discovery of basis and operator was. Even here the claim is bounded by
  the word **oracle**: it says the structured class is reachable when the basis is given, on
  one system, one backbone, one stratum. It does not say structured predictors win in
  practice, and 28.5 restates why.
- **P1 not replicated** — the root is declared **uninterpretable** and P2/P3 are not read at
  all. No repair, no re-run under a different rate, no rescue. The failure to replicate is
  itself published, and 27.8.3 is left standing as measured.

**This root cannot repair 27.** Whatever it returns, 27.8.1's verdicts, 27.8.3's contrast and
27.9's bounded statement stand exactly as written. A positive here does not retroactively make
the identity-basis negative into a positive; it explains it.

### 28.5 What this root does NOT do

It does not measure the pendulum. **M4 is not crossed**; 21.4 stands unmodified; no statement
here reaches `DampedPendulum`.

It does not run `transformer` or `ssm`. It does not vary `n`, `h_fr`, `c`, the rate, the
budget or the loss. It does not touch `representation.kind`: the shared representation stays
the identity one, so `T` acts on true-state coordinates and means what 26.0 measured it to
mean.

**No threshold moves.** `gamma = 5`, `gamma_1 = 2`, `tau_1`, `tau = 0.5`, the `1e-3` bound,
the 8000-update budget and the loss are exactly where they were.

**The ceiling is restored, and hardened.** This is the last root of this repository. A third
lift is not available to the author on his own signature: any further root requires a written
external objection, from a named reader who is not the author, recorded in this file with its
date. That is a deliberately higher bar than the one this section just cleared.

### 28.6 Honesty declaration

- **The ceiling was lifted a second time, after being declared final.** That is recorded here
  in the first line of 28.0 rather than softened, and the argument that distinguishes it from
  knob-chasing is given in the same paragraph, where a reader can reject it.
- **The author had the numbers before deciding.** Declared in 28.0. The mitigation is 28.4,
  written before the runs, which fixes the report of every outcome including the two that
  embarrass the decision.
- **An eligibility rule is repaired here that was only recorded in 27.8.5.** The repair is
  admissible solely because this section precedes its own data; the same change made to 27.3
  today would not be, and 27.3 is left exactly where it stands.
- **The oracle character of `on-oracle` is named three times** — in 28.2, in 28.3's P3, and in
  28.4's best-case clause — because it is the single most likely thing to be dropped when the
  result is summarized by someone in a hurry, including the author.
- **A prediction expected to hold is nonetheless run as a control** (P1), and its failure
  voids the root rather than being explained away.
- **R4**: this section precedes every run it authorizes, including this root's first cell.
