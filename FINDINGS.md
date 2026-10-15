# Structured latent dynamics in JEPA world models

**A pre-registered factorial study, and a negative result with a diagnosis.**

Farès Zaghouane · August 2026
<br>Full audit trail: [`PROTOCOL.md`](PROTOCOL.md) · Code and artifacts: this repository

---

## Summary

In a JEPA-style world model the encoder learns *what* to represent; the predictor, which moves latent states forward in time, is usually a generic network with no transition structure. A natural conjecture is that giving it an explicit dynamical structure — the Koopman-style family of 2×2 rotation-scaling blocks under exact zero-order-hold discretization — should improve long-horizon rollout stability.

This study tests that conjecture on the system where it is *most* favourable: coupled linear oscillators, whose generator lies in the structured family by construction. **The conjecture fails there.** Across 194 cells and seven independent result roots — 209 and eight counting the oracle root below — the structured arm is worse at every sample size, every level of prior misspecification, both training horizons measured, and on two backbones independently — in a design where the structured and unstructured arms of a backbone share rate, dataset, seed and budget, so the contrast carries no confound. That eighth root, 15 cells, then hands the structured arm the true basis as a frozen oracle: most of the penalty disappears on every seed, without putting the structured arm ahead of the unstructured one (§9).

The study then identifies why, and the reason is not that the prior is wrong. The optimum sits inside the model class and ordinary least squares recovers it from data to `2.1e-08`; gradient descent stops around `0.1`, some six orders of magnitude further, and about 100× above the pre-registered `1e-3` bound. Four checks rule out the data, the architecture, the budget and the contextual head; what is left is the training regime itself, a 25-step rollout loss under AdamW and mini-batches. A toy model shows the closed rollout can be enough to open such a gap; the real runs have not isolated it. And a closed-form measurement shows that the declared misspecification axis had been moving one of *two* misspecifications: at the configuration where the spectral axis reads "the prior is exact", the true generator is still `36 %` of its own norm away from the model class **in the identity basis**, and at machine precision from it in its own real Jordan basis.

**A structural prior of the Koopman family is a statement about a basis, not about coordinates.** Imposing the family in an arbitrary basis is not imposing the prior.

---

## 1. The question, and why it is not obvious

Generic predictors — an MLP, a causal transformer, a selective SSM — can fit a one-step prediction loss well and still drift catastrophically once they are rolled out freely, because nothing constrains the composition of many steps. Structured predictors impose an explicit transition law on the latent state. The 2×2 rotation-scaling block is the natural minimal choice: with

```
G = αI + ωJ,    J² = −I,    exp(ΔG) = e^{Δα} R(Δω)
```

the discrete transition has spectral radius `r = e^{Δα}` exactly, so stability is a property of one scalar per block rather than an emergent accident. Block-diagonalising `r` such blocks gives the real Jordan form of any generator with `2r` distinct complex eigenvalues — the family is not arbitrary, it is *the* linear family for oscillatory dynamics.

The cost is expressivity. So the question is a trade, and the trade is empirical:

> Does explicit dynamical structure in a JEPA predictor buy rollout stability, at equal compute budget, and if so where?

The reason to expect *yes* is strong enough that concurrent work reaches for the same parametrisation and reports a gain ([`PROTOCOL.md` §25.4](PROTOCOL.md)). The reason this study still matters is that the published setting differs from this one in a specific way, named in §7 below, which turns out to be the whole story.

---

## 2. Design: making the question answerable

Three design decisions do most of the work, and each was fixed before the runs it governs.

**The system is chosen so that a negative is strong.** All measurements are on `CoupledOscillators`, a four-dimensional linear system whose generator is

```
A = [[0, I], [−K, −cI]],    K = [[1.5, −0.5], [−0.5, 2.0]],    c = 0.10
```

Its eigenvalues are two conjugate pairs, so **the rotation-scaling prior is exactly correct** — the model class contains the truth. A negative result on a system where the prior is wrong would say nothing; a negative result *here* is informative, because the prior has every advantage.

**Structure is architectural, never a penalty.** The factor is `off` / `on`: whether the backbone runs its structured variant. It is never a numeric regularisation weight. An earlier design encoded structure as a calibrated Jacobian penalty; that design failed its own calibration, and the failure and its consequences are on the record rather than deleted ([§11](PROTOCOL.md)). Making the axis architectural is what removed the last free parameter from the comparison.

**Comparisons are paired within backbone.** Each backbone's `off` and `on` arms share learning rate, dataset build, seed and parameter budget (target 10 000, parity verified 6/6 within ±5 %). The *within-backbone* contrast is therefore free of confounds. The *across-backbone* ordering is not — the three families enter at the rate under which each is already known to train on this system (`1e-3` for MLP and transformer, `3e-4` for the SSM family), and that confound is named in the protocol twice, once for what it spares and once for what it costs ([§27.2](PROTOCOL.md)). Every ordering claim in this document carries it; the structure contrast does not.

The shared representation is the identity map, so the measurement scores the predictor rather than a learned readout (`e_recon = 1.87e-08`). Budget, optimizer, loss and the 8000-update schedule are common to all arms.

---

## 3. Result 1 — the optimum is inside the class, and least squares finds it

Before asking whether training reaches a good structured predictor, the study asks whether one exists and whether it is identifiable at all ([§20](PROTOCOL.md)).

| what | value |
|---|---|
| planted operator, recovered exactly | `1.9e-15` |
| same operator, recovered from data by ordinary least squares | `2.11e-08` |
| same operator, reached by gradient descent under the training objective | **`0.112`** (spectral error, median of 10 runs; ~100× above the `1e-3` bound) |

The first two lines say the theory is sound: the target is representable in the class, and it is recoverable from the observations the model actually receives. The third line says the problem is elsewhere. This distinction — *expressible* versus *reachable* — organises everything that follows, and it is the reason the negative result below is not a refutation of the prior.

---

## 4. Result 2 — and yet the structured arm loses

The primary factorial ([§27](PROTOCOL.md)) crosses three backbones with structure on/off, five seeds each: 30 cells, all complete, zero divergence. Primary statistic is `e_H`, the free-rollout error at horizon 25 in true-state coordinates; **lower is better**.

The paired within-backbone contrast, the one comparison certified free of confounds:

| backbone | contrast `on − off` | SE | seeds with a positive contrast |
|---|---|---|---|
| `mlp` | **+0.1451** | 0.0351 | **5 / 5** |
| `transformer` | **+0.2484** | 0.0670 | **5 / 5** |
| `ssm` | +0.2836 | 0.2379 | 3 / 5 |

Positive means structure *costs*. It costs on the MLP backbone and on the transformer backbone, carried by every seed in both.

**The `ssm` column is reported and not counted.** Its mean is set by a single seed contributing `+1.2141` against a median of `+0.0016`; two of its five differences sit within `1e-3` of zero. Both of its arms fail the arm-neutral lower bound — `ssm_k` on 5/5 seeds, `min_ssm` on 3/5 — so the column compares two arms that never left the trivial baseline. The protocol declared this risk in writing *before* the runs ([§27.6](PROTOCOL.md)) and the outcome is published as measured ([§27.9](PROTOCOL.md)). **What this root establishes is that the negative generalizes across two backbones; the third is uninformative for a pre-declared reason.**

This is not an isolated measurement. Six earlier roots, 164 further cells, put the same sign under variation:

- **sample size** — structure loses at `n = 8`, `32` and `256` ([§23.5](PROTOCOL.md));
- **prior misspecification** — it loses at every level of a closed-form spectral axis `δ(c)` from `0.000` to `1.029` ([§23.7](PROTOCOL.md));
- **training horizon** — the sign is unchanged between `h_fr = 5` and `h_fr = 25` ([§25.8](PROTOCOL.md)), so it is not an artifact of that choice within the pair.

---

## 5. Result 3 — the mechanism, isolated by elimination

Knowing *that* structure costs is worth little without knowing *why*. The study eliminates candidates rather than proposing one ([§20.1](PROTOCOL.md)).

Four candidates were each tested by their own measurement ([§20.1](PROTOCOL.md)). **Not the data**: least squares on the same training split recovers the operator (spectral error `2.11e-08`, free rollout `e_100 = 7.98e-14`). **Not the architecture**: with parameters planted by hand the same class reproduces the exact solution to `1.9e-15`. **Not the budget**: the validation curve has flattened, and closing the gap at its terminal rate would take 7 to 13 times the declared number of updates. **Not an over-free contextual head**: forcing the operator to be constant makes both bounds worse, not better.

What remains is the training regime as a whole: AdamW at `1e-3`, 8000 updates, batches of 8 episodes, and a latent loss rolled out over 25 steps. Inside that regime the closed rollout is the leading candidate. A controlled toy in [`notebooks/04`](notebooks/04_ce_que_la_grille_a_dit.ipynb) (§2.4–2.5, marked illustrative) shows that it *can* be enough: holding the linear structured model, the optimizer and the synthetic data fixed between the two objectives, Adam reaches machine precision on the one-step loss and stalls about thirteen orders of magnitude higher on the closed 25-step loss. The toy has no encoder, no nonlinear lift and no contextual head, so it does not show that the rollout is what blocks the real runs. No run of the study trains the same arm on the one-step loss alone against the rollout loss alone; that pair is the measurement that would settle it, and it is listed as open (§8).

A second structural finding came from the same family of measurements. A decomposition arm that keeps the structured transition but removes the lift/projection pair (`ψ`, `π`) landed **above both** the structured and unstructured arms — 3.1× worse at `n = 8`, 13.1× at `n = 256` — which the pre-registered reading rule had predicted would sit *between* them. That prediction was withdrawn, and the reason turned out to be structural rather than empirical, which is §7.

---

## 6. What the study says about its own instruments

Three findings concern the measurement rather than the model, and they are reported because omitting them would make the rest less trustworthy.

**A gauge degeneracy of the lift.** The triple `(ψ, Ā, π)` is invariant under `(cψ, Ā, π/c)` and under `(Rψ, Ā, πRᵀ)`, giving `2r` exactly flat directions in the loss. The obvious fix — constraining `π = ψ⁺` — removes **neither** gauge: `(cP)⁺ = P⁺/c` keeps the scale orbit inside the constrained set, and `(RP)⁺ = P⁺Rᵀ` keeps the rotation orbit inside it too, so the constraint ties `π` to `ψ` without picking a representative of either orbit. Both survivals are demonstrated numerically in the accompanying notebooks. This is a full refutation of a repair the author had proposed. (An earlier version of this paragraph claimed the constraint removes the rotation gauge; the notebook's own computation contradicts that, and the sentence was corrected on 2026-08-18.)

**A stability guarantee that did not hold.** A symplectic-Euler discretisation was proposed on the grounds that `det(M) = αβ` exactly, and therefore that the spectral radius would be bounded. Sweeping 200 000 draws showed a maximum radius of **368** — the determinant identity holds, but `|λ| = √(αβ)` only in the oscillatory regime, and the analytic supremum is `e⁶ ≈ 403`. The guarantee was withdrawn and replaced by a reparametrisation that is unconditional: `α = log σ(ρ)/Δ` gives `r = σ(ρ) ∈ (0,1)` for every `ρ`, verified over 400 001 values at both float32 and float64 with zero radii above 1 ([§23.1](PROTOCOL.md)).

**An eligibility rule that was wrong, and was left standing.** The primary factorial's pre-registered eligibility clause required 5/5 seeds to pass a two-sided positive control; 20 of 30 runs did not. Under that rule the confirmatory verdict is one relation contradicted, eight indeterminate, one untested. The rule conflates *"the run produced a valid measurement"* — which all 30 did — with *"the run passed the positive control"*. Repairing it after scoring would have converted one supported relation into nine, all in the pre-registered direction's favour or against it uniformly. **It was not repaired.** The defect is recorded, the exploratory reading is published beside the confirmatory one and promoted to nothing, and the distinction between a defect visible *before* data (repaired, [§27.7](PROTOCOL.md)) and one visible only *after* scoring (recorded, [§27.8.5](PROTOCOL.md)) is stated explicitly. A rule moved with the numbers in hand is not a rule.

---

## 7. The diagnosis

The misspecification axis of §4 measures a *spectral* distance: how far the true eigenvalues sit from any achievable rotation-scaling spectrum. At `c = 0.10` that distance is exactly `0.000` — the axis reads "the prior is exact".

A closed-form computation of the *other* distance ([§26.0](PROTOCOL.md)) gives:

| basis | Frobenius distance from `A` to the block rotation-scaling class | relative to `‖A‖_F = 2.9614` |
|---|---|---|
| identity | **1.0654** | **36 %** |
| the real Jordan basis of `A` | `1.97e-15` | ~0 |

The two 2×2 blocks recovered in that basis are `(−0.05, ±1.5187)` and `(−0.05, ±1.0902)` — the exact modal decomposition of the system. So at the configuration the protocol called "the prior is exact", the prior is exact **in a basis**, and that basis is not the identity. Every structured arm carrying a learned lift had to discover that change of basis by gradient descent, jointly with the operator, from a loss folded through twenty-five steps — the training regime §5 could not rule out.

This also explains the decomposition result of §5 without any new experiment: the arm without a lift is *forced* into the identity basis and therefore faces a misspecification of `1.0654` where the lifted arm faces `0`. It could not have landed between the two arms. The pre-registered prediction was refuted for a structural reason, not an empirical one.

**The bounded conclusion.** `ψ` and `π` are not an overhead sitting on top of the structured transition — they *are* the change of basis the structure requires. A structural prior of the Koopman family is a claim about a basis, not about coordinates, and the declared misspecification axis had been varying one of two misspecifications while holding the other fixed at 36 %.

---

## 8. What this does not show

Stated at the same volume as the results, because a scope claim that is easy to miss is a scope claim that will be misused.

- **It does not reach the pendulum.** On `DampedPendulum`, seven of eight positive-control cells do not pass ([§21.4](PROTOCOL.md)). No statement here extends to that system, and the protocol marks that boundary explicitly rather than letting it blur.
- **It is one system, one representation, one optimizer, one rate list.** Coupled oscillators, an identity shared representation, Adam, and a rate list whose argmin sat at its edge. Every claim inherits those.
- **It is not a claim about structured world models in general.** Concurrent work — *Koopman Dreamer*, arXiv:2607.19719, July 2026 — uses the *same* 2×2 parametrisation and reports a gain, under a **jointly learned** representation ([§25.4](PROTOCOL.md)). §7 measures the difference that most plausibly separates the two settings and names it a candidate, not an established cause.
- **The across-backbone ordering is confounded** with rate suitability, and the one confirmatory verdict the primary factorial produced (`transformer` worse than `mlp`, `+0.0496 ± 0.0097`) carries that confound. The unconfounded result is the paired contrast, which is descriptive.
- **The `ssm` family is largely broken at this sample size**, as §4 records and as the protocol predicted it might.
- **The mechanism of §5 is not isolated.** The four checks eliminate candidates; they do not show that the 25-step rollout, rather than the regime around it, is what blocks gradient descent. The one-step-only against rollout-only comparison has not been run.
- **The oracle arm of §9 is a ceiling, not a competitor.** Its basis is computed from the true generator, which no other arm may see. §9 measures what the structured class reaches when the basis is *given*; it says nothing about what it reaches when the basis must be learned, which is the setting every practical model is in.

---

## 9. The experiment this diagnosis demands

If the penalty is the *joint* discovery of basis and operator, then handing the basis over should remove it. That test was pre-registered in [`PROTOCOL.md` §28](PROTOCOL.md) — three arms on one backbone (`off`, `on-learned`, and an **oracle** arm whose `ψ` is frozen at the closed-form real Jordan transform with zero trainable parameters), five seeds each, in the stratum of §4, with three predicted signs, a control prediction whose failure voids the root, and a falsification clause binding the reporting of all four outcomes before any cell executed. **It ran on 2026-08-19, and the outcome is the one §28.4 had named in advance as most likely ([§28.7](PROTOCOL.md)):**

**The coupling is a contributing mechanism; it is not sufficient to reverse the sign.** Handing the basis over helps on every seed (`on-oracle − on-learned` = `−0.1302 ± 0.0340`, 0/5 seeds positive) and returns the structured arm to the positive-control gate that the learned-basis arm fails 5/5 (`on-oracle` passes 5/5); it does not put the structured arm ahead of the unstructured one (`on-oracle − off` = `+0.0149 ± 0.0189`, 3/5 seeds positive — indeterminate). The control contrast replicates §4's `mlp` column exactly (`+0.1451 ± 0.0351`, 5/5) — a determinism check on the pipeline, not an independent draw. The claim stays bounded by the word *oracle*: one system, one backbone, one stratum, basis given rather than learned.

Opening it required overriding a ceiling that a previous section had declared final. That override is argued in the first line of §28.0 where a reader can reject it, and the ceiling is restored afterwards at a higher bar: a third lift requires a written external objection, not the author's signature.

---

## 10. How this repository is meant to be read

Four layers, deliberately.

- **`README.md`** — the project and its result in ninety seconds.
- **This document** — the study, its scope and its limits, in the order a reader wants them.
- **`notebooks/`** — the course: `00` through `06`, from the problem and the instrument to the figures. It is meant to be worked through in order, not queried.
- **`PROTOCOL.md`** — the audit trail: 28 sections, 18 dated amendments, every pre-registration committed before the runs it authorises. It is long because it is complete, and it is not meant to be read front to back. Its reading map is at the top.

Two conclusions in this repository were withdrawn after measurement, and one prediction known in advance to fail was published unmodified rather than quietly replaced. **A withdrawn result is not a falsified one**; both are on the record, dated, beside the measurement they revise. That convention, and the rule that the contract precedes the run, are what the section numbering exists to enforce.

## 11. Reproducing

```bash
make setup                 # uv sync + editable install
make test                  # 1130 tests
make figures               # figA-figG from their result roots
make fig1 fig2 fig3        # the primary-factorial trio (PROTOCOL 27.4), standalone
uv run sld-factorial --config src/sld/experiments/configs/primary_factorial.yaml --dry-run
```

Figures regenerate from committed sidecar CSVs with their hashes, and a figure that does not reproduce the table it cites does not ship. Result roots are gitignored by convention — the scientific content lives in `PROTOCOL.md` and in the figure sidecars, both versioned.
