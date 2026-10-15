# structured-latent-dynamics

A pre-registered factorial study of one question: does an explicit dynamical prior in
the latent predictor of a JEPA world model buy rollout stability? The study measures
the answer on the one system where the prior is exactly correct by construction — and
the answer is no. Structure costs accuracy on two backbones, on every seed. The study
then isolates why, and confirms the mechanism with an oracle experiment whose reading
rules were declared before its runs.

This is original research in progress, not a finished artifact and not a reproduction
of a paper: the question, the design, the measurement contract and the diagnosis are
the project. It is also small by design: every run fits on a laptop CPU.

## Why this project exists

I wanted to do one piece of research end to end with the discipline I admire in good
experimental work: freeze the thresholds before the runs, publish the negatives with
the same weight as the positives, and leave an audit trail someone else could attack.
The literature reports a gain for exactly this kind of structured predictor. Instead
of reproducing that result, I built the cleanest test of it I could: a system whose
true generator lies inside the structured family, so that a negative answer would
mean something.

A sibling project, `constrained-trace-ebm`, studies the other half of the same
world-model question: there the dynamics are exact and only the cost is learned;
here the cost is fixed and the dynamics model is the object of study.

## The question and the design

A JEPA world model encodes observations into a latent state `z` and moves `z`
forward in time with a predictor that never decodes back. Generic predictors — an
MLP, a small causal transformer, a Mamba-style selective SSM — can fit the one-step
loss well and still drift once rolled out freely, because nothing constrains the
composition of many steps. The structured alternative routes the transition through
2×2 rotation-scaling blocks under exact zero-order-hold discretization: the natural
linear family for oscillatory dynamics, in the spirit of Koopman-style lifted models.

```
backbone ∈ {mlp, transformer, ssm}   ×   structure ∈ {off, on}

paired within backbone: same seed, same data, same learning rate,
same parameter budget (±5%) — the on−off contrast carries no confound
```

All published comparisons run on coupled linear oscillators, whose generator belongs
to the structured family by construction. The prior has every advantage there, so a
negative result is informative rather than trivial. The primary statistic is `e_H`,
the free-rollout error at horizon 25 in true-state coordinates, read under rules
committed before the runs.

## What is measured

Both halves are published with the same weight.

**The optimum is inside the class, and the right tool finds it.** A planted
structured operator is recovered exactly (`1.9e-15`), and ordinary least squares
recovers it from the training data to `2.11e-08`. Gradient descent under the actual
training objective stops around `0.1`, six orders of magnitude further. The theory is
not the failure; reachability is.

**And yet the structured arm loses.** The paired on−off contrast on `e_H` is
`+0.1451 ± 0.0351` on the MLP backbone and `+0.2484 ± 0.0670` on the transformer,
positive on every seed in both — positive means structure costs. The ssm column is
reported and not counted: its mean is set by a single seed and both of its arms fail
an arm-neutral baseline, a risk declared in writing before the runs.

**The diagnosis: the prior is a statement about a basis, not about coordinates.**
Where the declared misspecification axis reads "the prior is exact", the true
generator still sits 36 % of its own norm away from the model class in the identity
basis — and at machine precision (`1.97e-15`) in its own real Jordan basis. Imposing
the family in an arbitrary basis is not imposing the prior.

**An oracle experiment confirms the mechanism, and bounds it.** Handing the
structured arm the true basis — `ψ` frozen at the closed-form real Jordan
transform — removes most of the penalty on every seed and returns the arm to a
positive control that the learned-basis arm fails 5/5. It does not put the
structured class ahead of the unstructured one. The pre-registered reading: the
coupling is a contributing mechanism; it is not sufficient to reverse the sign.

## How the study is run

- Thresholds, predictions and reading rules are committed before the runs they
  govern; every later change is a dated amendment committed before the execution it
  authorizes. [`PROTOCOL.md`](PROTOCOL.md) — 28 sections, 18 amendments, a reading
  map at the top — is that audit trail.
- Two conclusions withdrawn after measurement stay published, and an eligibility
  rule found defective after scoring was left standing and documented rather than
  repaired: a rule moved with the numbers in hand is not a rule.
- 1130 tests run beside the code, including mutation tests and a two-check rule:
  every mathematical object carries a positive and a negative test against an
  independent reference.
- Every published figure traces back to a committed CSV with its hash, and runs are
  deterministic end to end.

[`FINDINGS.md`](FINDINGS.md) is the study in six pages — question, design, results,
diagnosis, and an explicit list of what it does not show.

## Repository structure

```text
src/sld/
  systems/        ground-truth dynamics (pendulum, coupled oscillators, Lorenz-63, CartPole)
  data/           data regimes (OU, action-conditioned) and dataset generation
  predictors/     the backbones and the structured wrapper (2×2 blocks, ψ/π lifts)
  objectives/     the JEPA loss (EMA target, stop-gradient)
  training/       trainer and parameter-budget matching (±5%)
  experiments/    factorial orchestrator, gates, protocol constants
  audit/, viz/    diagnostics and figure generation

notebooks/        a seven-notebook course (in French), 00 → 06
assets/           published figures with their CSV sidecars and hashes
FINDINGS.md       the study in six pages
PROTOCOL.md       the measurement contract and audit trail
```

## The notebooks

Seven notebooks form a course, in French, meant to be read in order: `00` the
problem and the instrument, `01` from the neuron to the operator, `02` the factorial
arms and budget fairness, `03` training and measurement, `04` what the grid said,
`05` the oracle experiment, `06` the figures. They rebuild the project's reasoning
rather than paraphrasing the code, and they verify the package against the course
with `np.allclose` parity cells.

## Quick start

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q                                     # 1130 tests
uv run --with jupyter jupyter lab notebooks/  # the course
```

Result roots (`results_*/`) are gitignored and stay local: the published numbers
live in `PROTOCOL.md`, `FINDINGS.md` and the hashed sidecars in `assets/`. The few
notebook cells that read result roots say so, and fail with a clear message on a
fresh clone.

## Status

Work in progress. The first pre-registered campaign — the factorial grid, the
diagnosis and the oracle confirmation — is closed and published; that is what this
repository documents today. The diagnosis opens the next questions rather than
closing them: what accounts for the part of the penalty the basis does not explain,
and whether a jointly learned representation — the one setting where the literature
reports a gain — changes the picture. The next cycles will follow the same
discipline: contract first, then runs.

## Limitations

- One system family carries every published comparison (coupled linear
  oscillators). The pendulum enters only as a positive control, which mostly fails
  and is reported as such; Lorenz-63 and CartPole are implemented but have no
  published measurement.
- One model size (~10k predictor parameters) and one shared representation (the
  identity map): the published contrasts score the predictor, not a learned encoder.
- Not a claim about structured world models in general. The literature contains a
  published gain with the same parametrisation under a jointly learned
  representation; the study names that difference as a candidate explanation and
  does not settle it.
- Small scale by design: five seeds per cell, CPU only.

## License

MIT — see [LICENSE](LICENSE).
