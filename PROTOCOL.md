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
