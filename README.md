# structured-latent-dynamics

> When does an explicit dynamical prior in a JEPA world model's latent predictor help — and when it does not, why not.

**The short answer, on the one system where the prior is exactly correct: it does not help, and the reason is that the prior is a statement about a *basis*, not about coordinates.** Three measurements, in order.

| | | |
|---|---|---|
| ![representable](assets/figA_d3_exactness.png) | ![the contrast](assets/fig3_factorial_effects.png) | ![the basis](assets/figD_misspecification.png) |
| **1. The optimum is inside the class.** A planted rotation-scaling operator is recovered to `1.9e-15`, and ordinary least squares finds it from data to `2.11e-08`. The theory is not the failure. | **2. Yet the structured arm loses.** Paired within backbone — same seed, dataset, rate and budget, the one confound-free contrast — structure *costs* `e_H`: `+0.1451 ± 0.0351` on `mlp` (5/5 seeds), `+0.2484 ± 0.0670` on `transformer` (5/5 seeds). | **3. Because the declared axis moved one of two misspecifications.** Where the spectral axis reads `δ(c) = 0.000` — "the prior is exact" — the true generator still sits `1.0654` from the class *in the identity basis*: **36 % of ‖A‖**. In its own real Jordan basis: `1.97e-15`. |

Gradient descent lands ~200× from an optimum that least squares reaches, and the isolating
ingredient is the **25-step closed rollout**, not the one-step loss — established by
elimination, not asserted (`PROTOCOL.md` 20.1). So the structured arm was never only fitting an
operator: it was being asked to discover a change of basis and the operator at once, from a
loss folded through 25 steps.

**The thesis this leaves standing:** a structural prior of the Koopman family is a claim about
a basis. Imposing the family in an arbitrary basis is not imposing the prior.

**Read next:** [`FINDINGS.md`](FINDINGS.md) — the study in six pages: question, design, the three
results above with their numbers, the diagnosis, and an explicit list of what it does not show.
[`PROTOCOL.md`](PROTOCOL.md) is the audit trail behind every figure quoted here; it opens with a
reading map, and is meant to be queried rather than read.

## Scope — what is measured, and what is not claimed

Every sentence above holds under: `CoupledOscillators`, `OURegime`, identity shared
representation, one rate list, one optimizer, `n ∈ {8, 32, 256}`, `δ(c) ∈ [0.000, 1.029]`,
`h_fr ∈ {5, 25}`. It is 194 cells across seven self-sufficient roots; `PROTOCOL.md` 23–28
carry every pre-registration, reading and confound.

- **Measured robust**: the sign is unchanged between `h_fr = 5` and `25` (25.8), across all
  three sample sizes and all four misspecification levels (23.5), and across two backbones
  each carried by 5/5 seeds (27.8.3, 27.9).
- **Measured, and explicitly *not* a third confirmation**: the `ssm` column's contrast
  (`+0.2836 ± 0.2379`) is set by a single seed against a median of `+0.0016`, in a backbone
  whose both arms fail the arm-neutral lower bound. 27.6 declared that risk before the runs;
  27.9 reports it rather than counting it.
- **Not a claim about structured world models in general.** 25.4 records a published result
  using the *same* parametrisation and reporting a gain, under a jointly learned
  representation. 26.0 measures the difference that most plausibly separates the two settings
  and names it a candidate — not as established.
- **Not a claim about the pendulum.** M4 is not crossed; 21.4 stands; seven of eight
  positive-control cells on `DampedPendulum` do not pass and nothing here reaches that system.

## The one experiment this diagnosis demands, declared and open

If the penalty is the *joint* discovery of basis and operator, then handing the basis over
should remove it. `PROTOCOL.md` **28** pre-registers that test — `off` vs `on-learned` vs an
**oracle** arm whose `ψ` is frozen at the closed-form real Jordan transform — with its three
predicted signs, its falsification clause, and a commitment written **before** the runs to
publish a refutation with the same prominence as a confirmation. Section 28.0 states in its
first line that this lifts a ceiling 27.5 had declared final, and gives the argument a reader
can reject.

**Status: declared, not yet run. No result from it is reported anywhere in this repository.**

## How this repository is meant to be read

The pre-registrations are load-bearing and the corrections are the point. Two conclusions were
withdrawn after measurement (22.0, 25.3's branch 1), two figures corrected their own readings
(24, 26.1), and one eligibility rule was found defective *after* scoring and left standing
rather than repaired (27.8.5) — because a rule moved with the numbers in hand is not a rule.
`REGLES.md` R4 (the contract precedes the run) and the correction-over-honesty hierarchy are
what the section numbering exists to enforce.

A withdrawn result is not a falsified one. Both are on the record, dated, beside the
measurement they revise.

## Motivation

When does explicit dynamical structure in a JEPA world model's latent predictor improve rollout stability, state sufficiency, and planning — at equal compute cost? Generic predictors (MLP, causal transformer) can fit the one-step loss well and still drift catastrophically under free-running rollout; structured predictors (2×2 rotation-scaling blocks, selective SSM, Koopman-style lifting) impose an explicit transition law on the latent state, at the cost of expressivity. This repo runs that comparison as a controlled factorial rather than a leaderboard.

**Calibrating that penalty to a margin comparable *across* families was attempted and failed.** Cross-family physical comparability of the calibrated margin was never validated, and the pre-registered common band was reached by one family at one seed. What follows from that — which predictions survive, which are withdrawn, which are unresolved — is in "What didn't work" below and in `PROTOCOL.md` section 11. Read that before reading any number in this repo as a cross-family comparison.

The design crosses **backbone** (`mlp`, `transformer`, `ssm`) with **structure** (`off`/`on` — whether the backbone runs its structured variant, an architectural choice, never a numeric penalty) and with data regime, on four custom dynamical systems (damped pendulum, coupled oscillators, Lorenz-63, CartPole). This is the current primary axis; the paragraph above documents an earlier attempt to encode structure as a calibrated total-Jacobian penalty μ_J instead — that attempt failed and μ_J is now a secondary/regression study only (`src/sld/objectives/jac_reg.py`, `spectral_reg.py`, Fig. 2), never load-bearing for the thesis. An audit panel separates what is displayed from what is measured: the spectrum (or field of local spectra) is display/screening only, never a stability certificate; stability is read off the total Jacobian on the augmented state; identifiability is read off R²/Procrustes recovery against the true state. The baseline every arm has to beat is val loss alone.

## Reproduce

```bash
make setup          # uv sync + editable install
make test           # full test suite
make smoke          # end-to-end factorial smoke (< 5 min)
uv run sld-factorial --config src/sld/experiments/configs/pendulum_core_partial.yaml   # the 3 executed OFF arms of the 3x2 factorial (30 runs, ~17 min)
make figures        # fig1 (e_h curves), fig2 (VPT per seed), fig3 (paired ON-OFF contrast) -- PROTOCOL 27.4
```

All **6** `backbone` x `structure` pairs now resolve to a predictor family: `(mlp,on)` → `mlp_k`, `(transformer,on)` → `transformer_k`, `(ssm,on)` → `ssm_k`, the three structured arms sharing one transition (`src/sld/predictors/structured.py`, `PROTOCOL.md` section 13.1). The registry holds **seven** families for those six cells: `KoopmanSSMPredictor` stays constructible and loadable so every published artifact keeps its constructor, and is never a cell of the factorial — an arm resolving to it is refused before any run.

`pendulum_core.yaml` is still **not runnable**: it is a frozen historical config kept on the retired `family`/`constraint` schema (the μ_J-era axis), and the backbone/structure validator refuses it outright. `pendulum_core_partial.yaml` above is the frozen record of the 30 OFF cells that ran. The full 3x2 grid (6 arms x 2 datasets x 5 seeds = 60 cells) is a **new pre-registration under a new result root**, and its training budget is not committed yet — no cell of it may execute until it is (`PROTOCOL.md` section 13.3).

Full core grid: ~550 runs, ~30-55 CPU-hours on a laptop — frugal by design.

## Status: the sections below document the retired μ_J campaign (historical)

Everything from "Pre-registered predictions" through "What didn't work" was written for, and measured under, the μ_J-calibration design (architecture × constraint, `PROTOCOL.md`'s original taxonomy) — **not** the current primary axis (backbone × structure, above). That design is retired: the numbers, verdicts and reasoning below are kept exactly as measured (`REGLES.md`'s correction-over-honesty hierarchy forbids rewriting them), but they are secondary/regression material now, never the primary thesis. The full trace — every `config.yaml`, `metrics.csv`, `audit.json`, `index.csv`, selection report and failure record from the `pendulum_core_selection`, pre-amendment `pendulum_core`, amended `results_amended/pendulum_core`, and `smoke` runs — is published at `reports/historical/mu_j/traces/` (**922 files, 5,043,481 bytes**); the **228 `.pt` checkpoints (~19 MB) are deliberately excluded**, so weights are not reproducible from the trace alone (see `reports/historical/mu_j/README.md`, verified against `manifest.sha256`). `experiments/selection.py` (the μ_J selector) no longer exists in `src/`; it is historical, not an active mechanism.

## Pre-registered predictions (P1–P5)

Committed and tagged `pre-registration` before any run of the core grid or of the pre-registered LR/μ_J selection and calibration sweeps; both start counting from the tag (`PROTOCOL.md`'s three-category taxonomy: core grid, pre-registered selection & calibration runs, exploratory). Exploratory runs, before or after the tag, are logged in `PROTOCOL.md` section 9, never in either of the other two counts.

- **P1 — both halves are currently unavailable, for different reasons.** ~~(a) *constraint effect*: within each family, the ON cell produces more stable free-running rollouts than the OFF cell.~~ **Unresolved**: P1(a) as published is an ON-vs-OFF contrast, and no admissible ON cell exists — the calibration that was to define it terminated without one for three families, and the protocol forbids resolving that inside the procedure. A replacement estimand (multi-dose, `PROTOCOL.md` section 11.6) is declared but not yet run, and it needs its own falsification rule before it can carry P1(a). ~~(b) *architecture effect*: at fixed constraint, the explicit spectral parametrization beats the generic families.~~ **Withdrawn as unidentified** — the constraint level it compares families *at* was never achieved, so no measurement of it exists. Withdrawn is not falsified. See "What didn't work" and `PROTOCOL.md` section 11.
- **P2 (sufficient state, falsifiable)** — the state deficit measured **before** moving to the contextual encoder (R²/Procrustes recovery against the true state) predicts **the order** of the contextual encoder's gains, announced in advance, on the available system × regime pairs (~8 at flagship scale — limited power, declared; the dozen pairs is the PhD-scale target). Killed if this order prediction does not beat the val-loss baseline.
- **P3 (directional co-adaptation)** — co-trained with the constrained predictor, the encoder produces latents with better recovery (R², Procrustes, modal separation) than with the generic family at the same budget. The direction is predicted, not just a change. **Dependency flagged**: "the constrained predictor" is the same undefined ON arm that leaves P1(a) unresolved. P3 cannot be executed as written until section 11.6 defines it; the dependency audit is open, not silently assumed away.
- **P4 (interventional)** — task pairs on the same environment and the same model, costs oriented toward well- vs. poorly-controllable directions, **constant oracle difficulty** (a CEM on the true RK4 dynamics succeeds equally on both orientations); prediction: the success gap is ordered by $r_{\mathrm{ctrl}}$ and $E_{\mathrm{ctrl}}$ (invariant forms, declared $R_u$ metric). Killed if the intervention does not produce the predicted order. The post-hoc correlational scatter can illustrate, never prove (endogenous).
- **P5 (stratification)** — gains from the structured arm (stability, planning, transition recovery) concentrate outside the OU regime (narrow-policy, anisotropic, action-conditioned). Under isotropic OU, **the exact zero expected concerns the orientation-identifiability gain** ($\bar A = \alpha I$, no reduced gauge) — not variance or stability, which can persist. Predicting zero everywhere would kill P5 for nothing.

## Robustness bars

Pre-registered decisions, not causal conclusions:

1. gains vanish under strict matching (params, FLOPs, updates, tuning) → **the claim is withdrawn** — we do not conclude "capacity effect", the experiment does not establish it;
2. stability via over-contraction (slow modes killed) → caught by the spectral panel + chaos metrics; the contraction-improves-MSE demo on Lorenz was to join Fig 1 at D2 (superseded by Fig 1's phase-D retirement, status note at the top of this README);
3. better MSE but degraded planning → metric-objective divorce, publishable on its own; the repo switches to planner-facing metrics;
4. the contextual encoder explains everything, across all families → the bottleneck was the state; the secondary hypothesis wins and "What didn't work" says so.

Causal language, when it appears, is earned by the P4 intervention and by the factorial's cells — never by a correlation.

## D1 grid arithmetic

| Factor | Values |
|---|---|
| System | pendulum |
| Regime | OU, action-conditioned (2) |
| Predictor family | 4 |
| Constraint | OFF, ON (2) |
| Seeds | 5 |
| **Total** | **1 × 2 × 4 × 2 × 5 = 80 runs (D1)** |

Status: the 40 OFF cells are trained (39 complete, 1 divergence). The 40 ON cells are held pending `PROTOCOL.md` section 11.6 — the constraint level they were to be run at was never calibrated across families.

This is the **core grid** count only (`PROTOCOL.md`'s taxonomy). The pre-registered LR sweeps (OFF and ON cells) and μ_J calibration runs (`PROTOCOL.md` sections 2 and 5) are separate, preserved alongside these results in the same historical trace (`reports/historical/mu_j/`, see the status note above), and not counted in the 80.

The full core grid (~550 runs, plan §3.9) arrives at D2. Models are ~10k predictor params, budget-matched ±5% (`PROTOCOL.md`).

## Results

**OFF half of the D1 grid: trained, but not yet valid evidence.** 4 families × 2 regimes × 5 seeds at the amended 200-update budget, 39 cells complete, 1 cell failed. It ran at 17:14 on 2026-07-31; the amendment authorizing its 200-update budget was committed at 18:48 the same day. Later commitment does not retroactively pre-register a run — the repo applies that rule to itself here. These artifacts are kept as post-hoc diagnostics and the grid is re-run against the committed amendment in a separate result root (`PROTOCOL.md` section 11.5). Seventeen minutes of compute is not a reason to carry a provenance defect.

The failed cell — `action_conditioned` / `min_ssm` / seed 4, non-finite training loss at update 46 — aborted without success artifacts. It is reported as a cell-level fact and **no inference is drawn from it**: one cell out of 40 supports nothing about unconstrained `min_ssm` in general.

**ON half: not run.** It waits on the dose-response study declared in `PROTOCOL.md` section 11.6. The four ON runs currently on disk are protocol-violation traces, kept and marked, never core evidence (section 11.5).

Fig. 2 lands here once the replacement study completes; nothing is reported before it does. Fig. 1 was a separate, more fundamental blocker at the time this section was written — retired pending a shared, frozen readout. **That blocker is since resolved and the sentence is left standing as written**: `assets/fig1_rollout.png` now ships from the oscillator roots under the identity shared representation (`PROTOCOL.md` 19.1, 27.4), which is the frozen readout this paragraph was waiting for. Nothing about the retired μ_J campaign below is revised by it.

## What didn't work

**The matched-margin calibration — and it is the most informative thing this repo has produced so far.**

The design crosses architecture with a constraint that had to be *comparable* across families. That comparability was operationalized as a common band `[0.90, 1.00]` on a calibrated stability margin. It failed, in three stages, each of which the instrument caught itself:

1. **The first calibration measured almost nothing.** The trainer takes one optimizer step per epoch, so a "10-epoch" calibration budget was 10 updates — entirely inside the transient, where AdamW's partial loss-scale invariance makes the penalty weight inert. Measured: `cos(grad(μ_J=1), grad(μ_J=10))` = 0.9985, and a weight displacement of 0.3–0.8 % across four decades of μ_J.
2. **The first metric was not a measurement.** `σ_max` was read in undeclared coordinates. The same dense Jacobians give 1.0439 raw and 22.94 whitened for `min_ssm` — a factor-23 swing in the verdict under an equally defensible choice of basis. `REGLES.md` requires a diagnostic that fails the reparameterization stress test to be repaired or withdrawn; it was repaired, and a metric was declared (`PROTOCOL.md` section 10.3).
3. **With budget and metric repaired, the band was still reached by one family at one seed.** `mlp` selected μ_J = 0.01; `transformer` sat at 5.8–7.2, `min_ssm` at 1.06–1.33, `koopman_ssm` at 1.0326–1.0348 with a 0.2 % non-monotone response.

**What the failure exposed.** The band asked four families to agree on one number while each family's margin is normalized by scales estimated on its *own* representation. Cross-family physical comparability was never validated — a design limitation, and the load-bearing one. It is not a matter of argument: with the gauge re-estimated per cell instead of frozen, `min_ssm`'s response stops being monotone (1.62 → 2.07 → 1.67 → 1.59 → 1.61 against a frozen-gauge 1.33 → 1.06), so a pre-registered acceptance criterion flips on a normalization choice that was presented as a convention.

Two mechanisms were tested against the checkpoints and **both were refuted**. Spectrum redistribution — the penalty minimizes a Frobenius norm while acceptance reads the top singular value, so the former can fall while the latter rises — does not happen here: for `koopman_ssm` the matched Frobenius norm *rises* with σ_max (2.8097 → 2.8154) and the stable rank does not move. A foreign reference frame — the gauge frozen on the OFF arm, applied to a structurally different ON arm — does not explain it either: with each cell's own gauge the curve is unchanged to three decimals, and the leading singular direction is not aligned with the mismatched coordinates (participation ≤ 0.0034 against a 0.25 uniform baseline).

What is left is simpler and was hiding behind both hypotheses: **for `koopman_ssm` the penalty is inert.** Four decades of μ_J move every norm of its Jacobian by ~0.2 %, upward. The same penalty moves `mlp` from 1.0301 to 0.8755 and `min_ssm` from 1.3299 to 1.0639. Consistent with a gradient-ratio measurement taken earlier for a different purpose — ‖∇penalty‖/‖∇prediction‖ at init is 1.4 for `koopman_ssm` against 16, 15 and 314 for the others — but that link is a candidate, not a tested claim.

For the transformer, separately, the audited quantity is a new-write sensitivity rather than contraction of the complete augmented state: an established construct limitation, not an explanation of any number.

**Consequence.** P1(b) compares architectures *at a fixed constraint*; no fixed constraint was ever achieved, so P1(b) is **withdrawn as unidentified** rather than declared false. What replaces it is a dose-response study that reports the achieved margin as a measured covariate instead of a controlled quantity (`PROTOCOL.md` section 11.6). What is *not* done, and why, is written out in section 11.4 — including the one mechanism that could have made the band reachable (a dual controller on the audited quantity) and the reason it is out of scope here rather than merely hard.

A gate was allowed to fail. The runs, the superseded verdicts and the reasoning stay published — see the status note above for exactly where (`reports/historical/mu_j/traces/`) and what that does and does not include.

## Scope and limitations

- One core model size — a Pareto *slope* claim needs ≥ 3 sizes; that is Year-2 PhD work, not this repo.
- ATM is the Year-1 PhD baseline, out of flagship scope — the gap is declared, not measured here.
- CartPole here is a controlled-dynamics and planning bench, not an RL benchmark (distinct from the author's `rl_from_scratch` repo).
- ~8 system × regime pairs at flagship scale — limited power, declared.
- Spectral quantities are display/screening only; stability claims go through the total Jacobian audit.

## License

MIT — see [LICENSE](LICENSE).
