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
