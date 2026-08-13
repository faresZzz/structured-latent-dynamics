# The neuron as an operator

**How the subthreshold LIF, the adaptive LIF, and the 2×2 rotation-scaling block of this
repository are the same object — and what that buys.**

*Public note, `structured-latent-dynamics`. Companion to the course notebook
`notebooks/01_du_neurone_a_l_operateur.ipynb` (French), which contains the runnable
derivations and every figure referenced here.*

---

## Summary

The `on` level of this repository's `structure` factor is a block-diagonal latent
transition whose blocks are 2×2 rotation-scaling operators,
$\bar{A} = \exp(\Delta G)$ with $G = \alpha I + \omega J$. This note establishes one fact
and draws one design decision from it.

**The fact.** That operator is not *inspired by* a neuron. It **is** one: the exact
discretisation of a subthreshold adaptive integrate-and-fire (adLIF) neuron. The
correspondence is an algebraic identity — a change of basis written out in §2 — not an
analogy. A one-dimensional leaky integrate-and-fire (LIF) neuron is, in the same exact sense,
a single channel of a diagonal state-space model with every coefficient frozen.

**The decision.** Once the identity holds, the spiking-neuron literature becomes usable as a
literature *on discretisation* — exactly where this repository has a measured problem. Under
the pre-registered protocol the spectral radius $r = e^{\Delta\alpha}$ is free, hence
unbounded, hence gradient amplification over a closed-loop rollout of $H$ steps is $r^{H}$.
§3 derives a reparameterisation bounding $r$ to $(0,1)$ unconditionally at the cost of one
line and no parameter count, shows why the obvious alternative — symplectic Euler, as used by
SE-adLIF — does not deliver an unconditional bound, and states why the bounded radius must be
a **third measured level of the structure factor** rather than a fix applied to the free one.
§4 proves why spiking is out of scope: a hard reset zeroes the gradient path through the
rollout, which is the quantity this project's central claim depends on.

### Convention on numbers

Two statuses, never mixed.

- **measured** — produced by a campaign of this repository or recorded in its verification
  annex; the source is named at each use.
- **reproduced** — recomputed by the companion notebook at each execution. Values below are
  from one run on `float64`; the notebook recomputes them and prints a comparison table.

---

## 1. The subthreshold LIF is a one-dimensional diagonal SSM

### 1.1 The membrane equation

$$\tau \frac{dV}{dt} = -\bigl(V(t) - V_{\text{rest}}\bigr) + R\,I(t).$$

$V$ is the membrane voltage (the *state*), $V_{\text{rest}}$ the resting potential (the
equilibrium under zero input), $\tau = RC$ the membrane time constant (typically 10–30 ms),
$R$ the membrane resistance, and $I$ the input current. The term $-(V - V_{\text{rest}})$
is a restoring force proportional to displacement — the *leak*. Below threshold, this is
all a LIF neuron is: an RC circuit.

### 1.2 Exact solution over one step

We do not approximate the derivative. Because the equation is **linear in $V$**, it solves in
closed form over $[t_k, t_k+\Delta]$ under one assumption on the *input*: $I(t) = I[k]$ is
constant on the interval. This **zero-order hold** (ZOH) is not an approximation of the
dynamics but a definition of the sampled signal — the same assumption used to discretise S4,
S4D and Mamba. *(A first-order hold is more faithful to a sampled physical signal but
introduces a second term $\bar{B}_1(I[k+1]-I[k])$, hence a dependence on the immediate
future, hence an acausal shift inside a self-consuming rollout: rejected on causality
grounds, not accuracy grounds.)*

Normalise and introduce the input-conditional equilibrium
$V_\infty := V_{\text{rest}} + R I[k]$ (obtained by setting $dV/dt = 0$):

$$\frac{dV}{dt} + \frac{1}{\tau}V = \frac{V_\infty}{\tau}.$$

Multiply by the integrating factor $\mu(t) = e^{t/\tau}$, which is the solution of
$\mu' = \mu/\tau$; the left-hand side becomes an exact derivative by the product rule:

$$\frac{d}{dt}\Bigl(e^{t/\tau}V(t)\Bigr) = e^{t/\tau}\frac{V_\infty}{\tau}.$$

Integrate over the step. The $\tau$ produced by $\int e^{s/\tau}ds$ cancels the $1/\tau$
prefactor:

$$e^{(t_k+\Delta)/\tau}V(t_k+\Delta) - e^{t_k/\tau}V(t_k) = V_\infty\Bigl(e^{(t_k+\Delta)/\tau} - e^{t_k/\tau}\Bigr).$$

Multiplying by $e^{-(t_k+\Delta)/\tau}$ isolates the next state:

$$\boxed{\;V[k+1] = a\,V[k] + (1-a)\bigl(V_{\text{rest}} + R\,I[k]\bigr), \qquad a = e^{-\Delta/\tau}.\;}$$

**Assumptions used:** $\tau > 0$, constant coefficients, $I$ constant on the step. **Nothing
was assumed about the size of $\Delta$** — the recurrence is exact at $\Delta = 10^{-3}\,$ms
and at $\Delta = 10^{3}\,$ms alike. That property is what distinguishes an exact
discretisation from a numerical scheme, and it is the whole subject of §3. Both limits
degenerate correctly: $\Delta \to 0 \Rightarrow a \to 1$ (nothing moves);
$\Delta \to \infty \Rightarrow a \to 0$ (the state forgets everything and jumps to
equilibrium).

### 1.3 The check that settles it

Agreement at $10^{-3}$ would only show that the recurrence is a good approximation. The
claim is stronger: it is *the solution*. The companion notebook integrates the same ODE with
a fine RK4 solver (100 substeps per step, 400 steps, ZOH input) in dimensionless units, and
compares.

| quantity | value |
|---|---|
| max absolute error, exact recurrence vs RK4 | $8.9\times10^{-16}$ *(reproduced)* |
| the same, in units in the last place | 4 ulp |
| repository verification annex | $5.83\times10^{-16}$ *(measured)* |
| machine epsilon, `float64` | $2.22\times10^{-16}$ |

Both figures say the same thing, and the differing final digits are rounding noise from two
different computation paths converging on the same real number. The consequence is that
statements proved about the recurrence transfer without reservation to the continuous ODE,
and conversely.

### 1.4 Placing it in the SSM family

A continuous linear state-space model $\dot{x} = Ax + Bu$ discretises under ZOH to
$x[k+1] = \bar{A}x[k] + \bar{B}u[k]$ with $\bar{A} = e^{\Delta A}$ and
$\bar{B} = \bigl(\int_0^{\Delta}e^{sA}ds\bigr)B$. **S4D** imposes diagonal $A$, so each
channel is an independent scalar recurrence. Term by term:

| S4D channel | subthreshold LIF |
|---|---|
| $A_{ii}$, learned, complex | $-1/\tau$, **fixed** by physiology, real negative |
| $\bar{A}_{ii} = e^{\Delta A_{ii}}$ | $a = e^{-\Delta/\tau}$ |
| $B_i$, learned | $R/\tau$, hence $\bar{B} = R(1-a)$, **fixed** |
| $\Delta$, learned per channel | $\Delta$, **fixed** simulation step |
| — | affine offset $(1-a)V_{\text{rest}}$ |

The input gain falls out of the general formula rather than a convenient rewriting:
$\int_0^{\Delta}e^{-s/\tau}ds \cdot R/\tau = \tau(1-e^{-\Delta/\tau})R/\tau = R(1-a)$. The
notebook checks that a "S4D channel" written in SSM vocabulary and a LIF produce the same
trajectory to $1.1\times10^{-16}$ *(reproduced)*.

**What Mamba adds.** Mamba keeps $\bar{A} = \exp(\Delta A)$ diagonal but makes $\Delta$, $B$
and $C$ functions of the current input, so $\bar{A}_t = \exp(\Delta_t A)$ is time-varying and
the forget rate becomes a *selectivity* mechanism. The LIF has neither ingredient: $a$ does
not depend on $I[k]$, $\Delta$ is chosen rather than optimised, and classically $\tau$ is not
learned either. Hence

$$\text{subthreshold LIF} \subsetneq \text{S4D} \subsetneq \text{Mamba},$$

with the neuron the **most constrained** member. That is what makes the statement
interesting: the claim is not that the neuron is powerful, but that it is already inside the
family, at the bottom.

### 1.5 What explicit Euler changes

Most SNN libraries integrate with explicit Euler, which gives

$$V[k+1] = \Bigl(1 - \tfrac{\Delta}{\tau}\Bigr)V[k] + \tfrac{\Delta}{\tau}\bigl(V_{\text{rest}} + R I[k]\bigr).$$

Writing $h := \Delta/\tau$, this is the first-order truncation of the exact coefficient:
$e^{-h} = 1 - h + h^2/2 - \dots$, so $a_{\text{Euler}} - a = -h^2/2 + O(h^3)$ and the
relative gap is $\tfrac{h^2}{2}e^{h} + O(h^3)$. For $\tau = 20$ ms, $\Delta = 1$ ms
($h = 0.05$) this predicts $1.31\times10^{-3}$, against $1.292\times10^{-3}$ *(reproduced)*
and $1.29\times10^{-3}$ *(measured, repository annex)*. The expansion explains the number.

Two points that are usually missed. First, **the input gain diverges more than the leak
does**: the exact scheme multiplies the input by $1-a = h - h^2/2 + \dots$, Euler by $h$, a
relative gap of $\approx h/2 = 2.5\%$ — twenty times the gap on $a$. The commonly inspected
coefficient is the wrong one. Second, **the stability domain is an artifact of the scheme**:
$a = e^{-h} \in (0,1)$ for **every** $h > 0$, so the exact recurrence is unconditionally
contractive, whereas $|1-h| < 1$ only for $h < 2$ and $1-h < 0$ already for $h > 1$, where the
voltage alternates sign every step — a numerical artifact with no physical referent.

At physiological settings this is harmless, which is why the practice persists. It stops
being harmless once $\tau$ is **learned**: a learned time constant can descend toward
$\Delta$, exactly where $1-h$ leaves the region in which it approximates $e^{-h}$. The bias
is also systematic — $a_{\text{Euler}} < a$ always — so it compounds as
$(a_{\text{Euler}}/a)^{H}$ over a rollout.

---

## 2. Two dimensions: adLIF and resonate-and-fire

### 2.1 Why a second variable

$a = e^{-\Delta/\tau}$ is real and positive, so a scalar LIF can only relax monotonically: it
cannot oscillate, resonate, or encode a phase. The constraint is algebraic, not biological —
eigenvalues of a **real** matrix come in conjugate pairs, so a non-real eigenvalue needs at
least a 2×2 block. The **adaptive LIF** supplies that second variable with a physiological
meaning: an adaptation current $w$ that accumulates as the membrane depolarises and brakes it
in return — delayed negative feedback, which is what makes a system oscillate.

$$\tau_u \dot{u} = -(u - u_{\text{rest}}) - w + R I, \qquad \tau_w \dot{w} = -w + a\,(u - u_{\text{rest}}).$$

With $x = (u - u_{\text{rest}},\; w)^{\top}$ this is $\dot{x} = Gx + bI$ with

$$G = \begin{pmatrix} -1/\tau_u & -1/\tau_u \\ a/\tau_w & -1/\tau_w\end{pmatrix}, \qquad b = \begin{pmatrix} R/\tau_u \\ 0\end{pmatrix}.$$

Sanity: at $a = 0$, $G$ is triangular with real eigenvalues $-1/\tau_u, -1/\tau_w$ — no
coupling, no oscillation, back to a driven LIF; at $\tau_w \to \infty$, $w$ freezes and we
recover a shifted LIF. The *resonate-and-fire* neuron is the same structure written directly
in resonance coordinates.

### 2.2 The underdamped condition, exactly

For a 2×2 matrix, $\lambda^2 - T\lambda + D = 0$ with $T = \operatorname{tr}G$ and
$D = \det G$. Here $T = -(1/\tau_u + 1/\tau_w)$ and $D = (1+a)/(\tau_u\tau_w)$ — note that
the coupling $a$ enters only through $D$. Eigenvalues are complex conjugate iff $T^2 < 4D$;
writing $p = 1/\tau_u$, $q = 1/\tau_w$,

$$(p+q)^2 < 4(1+a)pq \iff (p-q)^2 < 4a\,pq \iff \boxed{\;a > \frac{(\tau_w-\tau_u)^2}{4\,\tau_u\tau_w}.\;}$$

Read plainly: the coupling must be strong enough to overcome the asymmetry of the two time
scales. In that regime, with $\lambda_\pm = \alpha \pm i\omega$,

$$\alpha = -\tfrac{1}{2}\Bigl(\tfrac{1}{\tau_u} + \tfrac{1}{\tau_w}\Bigr), \qquad \omega = \tfrac{1}{2}\sqrt{4D - T^2}.$$

Two consequences used later: $\alpha$ depends only on the time constants while $\omega$ also
depends on $a$, so damping and frequency are **not** freely decoupled in the biophysical
parameterisation — which is what SE-adLIF's reparameterisation addresses; and $\alpha < 0$
always, so the biophysical neuron is intrinsically contractive. That last property is lost
the moment $\alpha$ is learned freely, which is §3.

### 2.3 Conjugacy to the rotation-scaling block

**Proposition.** Let $G \in \mathbb{R}^{2\times2}$ have eigenvalues $\alpha \pm i\omega$
with $\omega \neq 0$. Then there is $P \in GL_2(\mathbb{R})$ with
$P^{-1}GP = \alpha I + \omega J$, where $J = \begin{pmatrix}0 & -1\\ 1 & 0\end{pmatrix}$.

**Proof.** Let $v = p + iq$ be an eigenvector for $\lambda = \alpha + i\omega$. Then
$G(p+iq) = (\alpha p - \omega q) + i(\omega p + \alpha q)$, and since $G$ is real, $Gp$ and
$Gq$ are real, so real and imaginary parts may be identified separately:

$$Gp = \alpha p - \omega q, \qquad Gq = \omega p + \alpha q.$$

Set $P := (\,q \;|\; p\,)$. Then $GP = (\,\omega p + \alpha q \;|\; \alpha p - \omega q\,)$,
and $PC$ for $C = \alpha I + \omega J$ gives the same two columns. $P$ is invertible: if $p$
and $q$ were collinear, $v$ would be a complex multiple of a real vector, forcing $\lambda$
real and contradicting $\omega \neq 0$. $\blacksquare$

**Failure case, and it matters.** If $\omega = 0$ the construction collapses: the eigenvalues
are real and $G$ is either diagonalisable over $\mathbb{R}$ (two independent relaxation
modes) or a Jordan block. Neither has the form $\alpha I + \omega J$. **The rotation-scaling
block does not contain real distinct eigenpairs** — this is the representability defect this
repository quantifies as $\delta(c)$, and the reason $T^2 < 4D$ is a modelling hypothesis
rather than a technicality.

Because $\alpha I$ and $\omega J$ commute and $J^2 = -I$ (so the exponential series of
$\theta J$ splits into the cosine and sine series), the discrete operator has a closed form:

$$\bar{A} = \exp(\Delta G') = e^{\Delta\alpha}R(\Delta\omega), \qquad \text{eigenvalues } re^{\pm i\varphi},\quad r = e^{\Delta\alpha},\ \varphi = \Delta\omega.$$

This is, verbatim, the `rotation_scaling_zoh` block of this repository. The notebook verifies
$\lVert P^{-1}GP - (\alpha I + \omega J)\rVert_{\max} = 3.1\times10^{-17}$ *(reproduced)* and
that the free trajectory's modulus follows $r^{k}$ to $2.7\times10^{-15}$ *(reproduced)*.

### 2.4 What the identity implies, and what it does not

**It implies** that the entire literature on the discretisation and parameterisation of
adaptive neurons applies directly to this repository's latent predictor, with no analogical
transfer. A guarantee proved on a discretised adLIF is a guarantee on the block.

**It does not imply** four things that are tempting and false. *No extra capacity:* the class
$\{e^{\Delta\alpha}R(\Delta\omega)\}$ is identical whether called a rotation-scaling block or
a subthreshold adLIF — two real parameters per block either way, and renaming adds no
expressivity. *No biological-plausibility argument:* resembling a neuron says nothing about
performance, and neuroscience is treated here as a source of hypotheses, never an authority.
*No energy argument:* the energy case for spiking networks rests on the **spike** —
event-driven computation, sparse activity — not on the subthreshold regime; a dense
subthreshold block costs what a 2×2 matrix costs. *No validation of the prior:* if the target
dynamics have a real distinct eigenpair, no rotation-scaling block contains them (§2.3).

**What it does buy: a literature on discretisation.** The open problem is not "which
structure?" but "how should this structure be parameterised so that it trains?" — worked on,
with tested answers, since 2024.

---

## 3. Discretisation decides the radius

### 3.1 The radius is free, hence unbounded

This repository parameterises the continuous generator and discretises exactly, so
$r = |\lambda(\bar{A})| = e^{\Delta\alpha}$. `PROTOCOL.md §13.2` deliberately leaves $\alpha$
**free** — *"no `-exp(nu)`, no `-softplus`"* — so that stable **and** unstable modes stay
representable and stability is *measured, not imposed* (§3.5). But $\alpha \in \mathbb{R}$
means $r \in (0,+\infty)$. Under a **closed-loop** rollout of $H$ steps,
$\partial x_H/\partial x_0 = \bar{A}^{H}$ with $\lVert\bar{A}^H\rVert \sim r^{H}$; and since
$\partial x_H/\partial\alpha = H\Delta\,x_H$, the gradient carries $r^{2H}$.

| $r$ | $r^{25}$ | $r^{50}$ (gradient scale) |
|---|---|---|
| 1.05 | 3.4 | 11.5 |
| 1.20 | 95 | $9.1\times10^{3}$ |
| 1.50 | $2.5\times10^{4}$ | $6.4\times10^{8}$ |

*(Rows for $r^{25}$ match the values recorded in the repository review; $H = 25$ since
amendment 6. Reaching $r = 1.05$ from $r = 1$ requires moving $\alpha$ by $0.049$.)*

### 3.2 The measured failure, and how to read it

The following are **measured** in this repository (§17.0/§18.0), same harness for both
families:

| observation | `ssm` family | `mlp` family |
|---|---|---|
| gradient norm at the first step | $2.4\times10^{18}$ | $28.2$ |
| epoch-0 loss | healthy | healthy |
| epoch-1 loss, **with** gradient clipping | $6.4413\times10^{24}$ | — |
| epoch-1 loss, **without** gradient clipping | $6.4415\times10^{24}$ | — |

The `ssm` family was declared to have no trainable learning rate after divergence at epoch 2
at every rate tried.

The decisive line is the last pair, and the inference runs as follows. *Hypothesis A: the
optimiser step is too large.* Clipping bounds the gradient norm, hence bounds the parameter
displacement, hence bounds the degradation at the next step; under A, epoch-1 loss should
fall by orders of magnitude when clipping is switched on. *Observation:* the two losses agree
to five significant digits, a relative difference of $3\times10^{-5}$. *Refutation:* a loss
invariant to clipping cannot be caused by step size, because clipping acts on step size and
produced no effect. *Hypothesis B: the forward map diverges.* If $r > 1$ then
$\lVert x_H\rVert \sim r^{H}\lVert x_0\rVert$ is already astronomical **at frozen
parameters**; the loss is essentially a function of $r$, and reducing it requires moving
$\alpha$ a *finite* distance (across $\alpha = 0$) that no bounded-step budget covers in one
epoch. B predicts exactly what is observed, including a giant first-step gradient because
$\lVert\nabla L\rVert \sim r^{2H}$.

As a magnitude check, $\ln(2.4\times10^{18})/50 = 0.845$, so $r \approx 2$ suffices to
produce that gradient norm at $H = 25$; no exotic pathology is needed. The notebook
reproduces the *mechanism* on a two-parameter toy with analytic gradients (verified against
finite differences to $4.6\times10^{-10}$ relative): measured gradient norm follows $r^{2H}$
across the range, and clipped and unclipped epoch-1 losses stay at the same level. The toy
does not reproduce the five-digit agreement and does not claim to; it reproduces the
diagnostic.

`§17.0` records that no published work trains an SSM dynamics loss on a 25-step
self-consuming rollout from initialisation without a guard. "The SSM family does not train"
is therefore not a fact about structural priors but about **this discretisation at this
radius in this regime**.

### 3.3 Symplectic Euler: the obvious answer, and its trap

SE-adLIF (arXiv:2408.07517, *Nature Communications* 2025) answers by changing the
integration scheme. Semi-implicit ("symplectic") Euler updates the second equation using the
**already updated** first variable. With $\alpha := 1 - \Delta/\tau_u$ and
$\beta := 1 - \Delta/\tau_w$ (both in $(0,1)$ whenever $\Delta < \min(\tau_u,\tau_w)$),
substituting the $u$-update into the $w$-update yields the transition matrix

$$M(\alpha,\beta,a) = \begin{pmatrix} \alpha & -(1-\alpha) \\[2pt] (1-\beta)a\alpha & \beta - (1-\beta)a(1-\alpha)\end{pmatrix}.$$

**Determinant.** Directly:

$$\det M = \alpha\bigl(\beta - (1-\beta)a(1-\alpha)\bigr) + (1-\alpha)(1-\beta)a\alpha = \alpha\beta - \alpha(1-\alpha)(1-\beta)a + \alpha(1-\alpha)(1-\beta)a = \alpha\beta.$$

The two terms in $a$ are identical up to sign and cancel: $\det M = \alpha\beta$ **exactly,
for all $(\alpha,\beta,a)$**, not merely to leading order in the step. Since for a complex
conjugate pair $\det M = |\lambda|^2$, this gives $|\lambda| = \sqrt{\alpha\beta} < 1$ for
$\alpha,\beta \in (0,1)$ — stability by construction, with the coupling $a$ (hence the
frequency) left free.

**The trap.** $|\lambda| = \sqrt{\det M}$ holds **only if the eigenvalues are complex**, i.e.
only under $\operatorname{tr}(M)^2 < 4\alpha\beta$. And the trace *does* depend on $a$:

$$\operatorname{tr}M = \alpha + \beta - (1-\beta)\,a\,(1-\alpha),$$

decreasing linearly in $a$ and unbounded below. Past the parabola the eigenvalues turn real
and $|\lambda|_{\max} = \bigl(|\operatorname{tr}| + \sqrt{\operatorname{tr}^2 - 4\det}\bigr)/2$,
which tends to $|\operatorname{tr}|$ as $\det = \alpha\beta \to 0$ and is therefore linear
in $a$.

Concretely, at $\alpha = 0.9$, $\beta = 0.8$ (so $\det = 0.72$ exactly, independently of
$a$):

| $a$ | $\operatorname{tr}M$ | $\det M$ | $\lvert\lambda\rvert_{\max}$ | regime |
|---|---|---|---|---|
| 0.5 | $+1.690$ | 0.7200 | 0.8485 | complex |
| 3 | $+1.640$ | 0.7200 | 0.8485 | complex |
| 30 | $+1.100$ | 0.7200 | 0.8485 | complex |
| **300** | $\mathbf{-4.300}$ | 0.7200 | **4.1255** | **real — bound lost** |

Over 200 000 draws of $(\alpha,\beta) \in (0,1)^2$ and $a \in [e^{-6},e^{6}]$, the repository
records a maximum radius of **368** *(measured)*; the notebook, over five replicates,
reproduces maxima in **371–391**, with 68% of draws in the real regime and 27% exceeding
$|\lambda| = 1$. The sampled maximum is an extreme-value statistic, so the digit depends on
the draw; the *supremum* is analytic — as $\alpha,\beta \to 0$, $\det \to 0$ and
$|\lambda|_{\max} \to a \to e^{6} \approx 403.4$. The order of magnitude is the result.

This is not a criticism of SE-adLIF, whose physiological parameter range stays oscillatory;
it is a criticism of using it as a *guarantee* where $a$ is learned unconstrained.
**A guarantee conditioned on an inequality that a test must police is the wrong object for an
experimental axis** — it would require constraining $a$, testing the condition, defining
crossing behaviour and documenting one more edge case, in a repository already carrying four
uncontrolled confounders.

### 3.4 The retained parameterisation

Keep the continuous generator and the exact discretisation. Replace **only the production of
$\alpha$**:

$$\boxed{\;\alpha := \frac{\log\sigma(\rho)}{\Delta}, \qquad \rho \in \mathbb{R}\ \text{free}, \qquad \sigma(\rho) = \frac{1}{1+e^{-\rho}}.\;}$$

**Proposition.** For every $\rho \in \mathbb{R}$ and every $\Delta > 0$, $r = \sigma(\rho) \in (0,1)$
strictly, with no side condition.

**Proof.** (i) $e^{-\rho} > 0$ for all real $\rho$, so $1 + e^{-\rho} > 1$ and
$\sigma(\rho) < 1$; and $\sigma(\rho) > 0$ as the reciprocal of a finite positive number.
Moreover $\sigma' = \sigma(1-\sigma) > 0$ and $\sigma$ is onto $(0,1)$, so
$\rho \mapsto \sigma(\rho)$ is a **bijection** $\mathbb{R} \to (0,1)$ — no part of the
interval is unreachable. (ii) Hence $\log\sigma(\rho)$ is finite and negative, and
$\alpha \in (-\infty, 0)$. (iii) By §2.3, $r = e^{\Delta\alpha} = \exp\bigl(\Delta\cdot\log\sigma(\rho)/\Delta\bigr) = \sigma(\rho)$;
the $\Delta$ cancels exactly, which is why the definition divides by it. $\blacksquare$

**Failure cases, named.** *(i) Saturation.* As $\rho \to +\infty$, $r \to 1^-$ with
$\partial r/\partial\rho = \sigma(1-\sigma) \to 0$; as $\rho \to -\infty$, $r \to 0^+$ and the
state is erased in one step. Neither breaks stability; both create optimisation plateaus.
That is the price of the bound, to be measured rather than assumed away. *(ii) Floating
point.* $r < 1$ is strict over $\mathbb{R}$, but $\sigma(\rho)$ rounds to exactly $1.0$ once
$\sigma(-\rho) < \varepsilon/2$ — around $\rho \gtrsim 37$ in `float64` (measured at $37.6$)
and $\rho \gtrsim 17$ in `float32`. The degeneracy is benign, and the reason matters: it
yields $r = 1$, hence $r^{H} = 1$, hence **no amplification**. The bound can degrade to
neutrality, never to divergence — unlike the symplectic map, whose degeneracy yields
$|\lambda| = 368$.

| $\rho$ | $-6$ | $-3$ | $-1$ | $0$ | $1$ | $3$ | $6$ | $12$ |
|---|---|---|---|---|---|---|---|---|
| $\sigma(\rho) = r$ | 0.002473 | 0.047426 | 0.268941 | 0.500000 | 0.731059 | 0.952574 | 0.997527 | 0.999994 |
| $r^{25}$ | $6.7\!\times\!10^{-66}$ | $8.0\!\times\!10^{-34}$ | $5.5\!\times\!10^{-15}$ | $3.0\!\times\!10^{-8}$ | $4.0\!\times\!10^{-4}$ | 0.297 | 0.940 | 0.99985 |

**What stays identical.** Generator, discretisation $\bar{A} = \exp(\Delta G)$, the angular
frequency $\omega$, the closed-form ZOH integral $\Phi = \int_0^{\Delta}e^{sG}ds$ (checked
against fine quadrature to $4.9\times10^{-11}$, a quadrature error, *reproduced*),
$\bar{B} = \Phi B$, and the parameter count — two numbers per block, $(\rho,\omega)$ instead
of $(\alpha,\omega)$. The protocol's fairness contract (`§13.7`, *"the SAME 16 numbers in
every arm"*) holds without amendment, and the `free` vs `bounded` contrast is as clean as it
can be made: one reparameterisation, everything else matched.

**One numerical requirement.** Compute $\log\sigma(\rho) = -\operatorname{softplus}(-\rho)$
via a `logsigmoid`/`logaddexp` primitive, never as `log(sigmoid(rho))`. Composing naively
forms $e^{-\rho}$ first, which **overflows** for $\rho \lesssim -89$ in `float32` (the usual
training precision); $\sigma$ then evaluates to exactly $0$, $\log 0 = -\infty$, and `nan`
propagates through the model in one update. That left tail is not exotic: $\rho = -89$
corresponds to $r \approx 10^{-39}$, the over-contracted corner the optimiser drifts toward
when it cannot otherwise reduce a rollout error. The stable form never materialises
$e^{-\rho}$ for negative $\rho$ and is asymptotically exact ($\log\sigma(\rho) \to \rho$).

### 3.5 Honesty statement

Bounding the radius **imposes** stability instead of measuring it — exactly what
`PROTOCOL.md §13.2` refused on purpose. Treating the bounded radius as a *fix* to the free arm
would conflate "structure" with "contraction", making a win uninterpretable, and would
rewrite a measurement, which this repository's rules forbid. It is therefore a **third level
of the `structure` factor** — $\{\texttt{off},\ \texttt{on-free},\ \texttt{on-stable}\}$ —
measured **alongside** the free arm, never applied **to** it. The question becomes
identifiable: *where on that axis does the gain live?* Both outcomes are publishable, which
is the quality test for a pre-registered prediction.

Nothing here is novel in isolation: bounded radii appear in Koopman Dreamer, and an earlier
cell in this repository (`koopman_ssm(constrained=True)`) already bounded $r$ via
$\alpha = -e^{\nu}$. What is new is the three-level contrast with everything else matched, and
the gain curve against sample size and against the representability defect $\delta(c)$. *(The
$-e^{\nu}$ form is set aside on coverage, not correctness: it drives $r \to 0$ very fast, so
most of the parameter mass over-contracts, whereas $\sigma$ is a bijection onto all of
$(0,1)$ at identical cost.)*

---

## 4. What spiking would cost here, and why we stop below threshold

Add threshold and hard reset to the LIF of §1:

$$V_{\text{free}}[k+1] = aV[k] + (1-a)(V_{\text{rest}} + RI[k]), \quad s[k+1] = \mathbf{1}\bigl[V_{\text{free}}[k+1] \ge V_{\text{th}}\bigr], \quad V[k+1] = (1-s[k+1])\,V_{\text{free}}[k+1].$$

The system stays **piecewise linear** — between spikes it is exactly the affine recurrence of
§1 — but the map $V[k] \mapsto V[k+1]$ becomes **discontinuous** at
$V_{\text{free}} = V_{\text{th}}$, with a jump of $-V_{\text{th}}$ that does not shrink.

**The decisive computation.** Almost everywhere (off the measure-zero set where
$V_{\text{free}} = V_{\text{th}}$ exactly), the indicator is locally constant, so

$$\frac{\partial V[k+1]}{\partial V[k]} = (1-s[k+1])\,a = \begin{cases} a & \text{no spike},\\ \mathbf{0} & \text{spike}.\end{cases}$$

Hence along a trajectory

$$\frac{\partial V[T]}{\partial V[0]} = \prod_{k=1}^{T}(1-s[k])\,a = \begin{cases} a^{T} & \text{if no spike occurs},\\ 0 & \text{as soon as one spike occurs}.\end{cases}$$

This is annihilation, not attenuation.

**Why that is fatal for this project specifically.** The central link of the research
statement is that at planning time — on the order of $3\times10^{5}$ predictor calls per
decision in a realistic CEM — the predictor dominates cost, and that *if its action
gradients are well conditioned through the rollout, one can plan by gradient descent instead
of sampling*, worth one to two orders of magnitude. The controllability audit
$\partial x_H/\partial u$ is therefore the *access condition* to the efficient regime, not a
thermometer beside it. And that quantity factorises by the chain rule into a product of
state Jacobians:

$$\frac{\partial x_H}{\partial u_t} = \Bigl(\prod_{k=t+1}^{H-1}\frac{\partial x_{k+1}}{\partial x_k}\Bigr)\frac{\partial x_{t+1}}{\partial u_t}.$$

A hard reset puts an exact zero into that product at every spike. **The spike severs the
gradient path precisely where this thesis needs it well conditioned.** The claim is not "this
is hard"; it is "the measured quantity becomes identically zero".

The notebook demonstrates this on a neuron that actually fires — the drive is chosen so that
$V_\infty > V_{\text{th}}$, checked by assertion: 9 spikes over 200 steps, inter-spike
interval 22, matching the analytic
$\lceil \ln\bigl(V_\infty/(V_\infty-V_{\text{th}})\bigr)/(\Delta/\tau)\rceil = 22$
*(reproduced)*. Without reset the Jacobian product decays smoothly as $a^{T}$
($4.5\times10^{-5}$ at $T=200$); with hard reset it is **exactly zero** from step 22 on.

**The standard remedy, and its cost.** Surrogate gradients keep the indicator in the forward
pass and replace its derivative in the backward pass with a smooth bump, restoring a non-zero
path at the explicit price of a **biased estimator** — the computed gradient is the gradient
of no function the forward pass implements. Here that adds a confounder (the surrogate's
shape and width) to a study already carrying four uncontrolled ones; an irreducible
attribution ambiguity (dynamics, or estimator bias?); and an incompatibility with the primary
instrument, since $\kappa(\partial x_H/\partial u)$ stops being an audit if the measured
Jacobian is not the model's.

**Scope, stated positively.** The subthreshold regime is what serves the thesis today: a
linear operator, differentiable everywhere, whose spectrum and conditioning are auditable in
closed form. Spiking becomes relevant when **energy is the measured object** — a legitimate
question with its own shape ("does a natively dynamic unit reduce cost per successful
decision at planning time?") requiring an energy bench, a corpus and a protocol. A project,
not an arm variant.

**Hodgkin–Huxley is excluded, and not as a value judgement.** Four coupled **stiff** ODEs per
neuron ($V$ plus three gating variables, time constants two orders of magnitude apart) mean
no BPTT path that scales to rollouts of tens of steps; four states and several exponentials
per neuron per step against two numbers for the 2×2 block, for an operator class with no
closed-form spectral characterisation; and no connection to the question asked here, which is
about the structure of the latent transition operator. It is good computational
neuroscience — in a JEPA dossier it would weaken the argument.

---

## Related work

The spectral constraint that the world-model community is rediscovering in July 2026 is the
one the neuron community proved stable in 2024–2025. Four references, verified:

- **Koopman Dreamer: Spectrally Constrained Latent Dynamics for Stable World-Model
  Imagination** — [arXiv:2607.19719](https://arxiv.org/abs/2607.19719), 22 July 2026.
  2-D rotation-scaling blocks with **bounded radius** inside a Dreamer; better than DreamerV3
  on **8 of 9** proprioceptive DMC tasks; UAV-LiDAR navigation from **53.8% to 73.8%**.
- **SE-adLIF — Advancing spatio-temporal processing through adaptation in spiking neural
  networks** — [arXiv:2408.07517](https://arxiv.org/abs/2408.07517), *Nature Communications*
  2025. Symplectic-Euler discretisation of adLIF; damping and frequency decoupled. Source of
  the map $M(\alpha,\beta,a)$ analysed in §3.3.
- **Balanced Resonate-and-Fire Neurons** —
  [arXiv:2402.14603](https://arxiv.org/abs/2402.14603), ICML 2024. Divergence-boundary-aware
  optimisation, with stable BPTT convergence over hundreds of steps.
- **Oscillatory State-Space Models (LinOSS)** —
  [arXiv:2410.03943](https://arxiv.org/abs/2410.03943), ICLR 2025 (oral). Universality for
  oscillatory SSMs — the operator class of §2.3 — which is what prevents reading the
  constraint as a loss of expressivity.

*(Background mentioned without any figure taken from them: S4D — Gu, Gupta, Goel, Ré — for
the diagonal form; Mamba — Gu & Dao — for selectivity; Izhikevich for the resonate-and-fire
neuron.)*

**What this repository measures next** (developed in §5 of the companion notebook): free
versus bounded radius with **encoder, decoder and backbone matched**, which the Koopman
Dreamer ablation does not do; the **gain-versus-sample-size** curve, which nobody publishes
and which matters because an inductive bias buys sample efficiency and has nothing to buy in
an over-determined regime; and this bridge inside a **predictive-latent (JEPA)** world model
rather than a reconstructive one. That last sign is **not predicted** — both outcomes are the
result.
