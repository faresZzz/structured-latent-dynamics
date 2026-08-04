"""SPIKE (throwaway): total-Jacobian penalty via JVP probes.

STATUS (jalon B2, demotion explicite) -- SECONDARY -- not primary thesis.
This spike produced the numbers in reports/spike_jacreg.md for the
now-retired mu_J penalty line. Historical trace: reports/historical/mu_j/.

De-risks `TotalJacobianPenalty(k_probes=2, step_subsample=0.25)`: a training-time
regularizer that estimates ||J^q||_F^2 of a rollout step function w.r.t. the augmented
state q = (h, m), using `torch.func.jvp` probes.

Estimator (the load-bearing correctness point):
    for v ~ Uniform(S^{d-1}),  E[v v^T] = I/d,  so  E||J v||^2 = ||J||_F^2 / d.
    => the UNBIASED estimator of ||J||_F^2 is   (d / k) * sum_v ||JVP(step_fn, q, v)||^2.
The `d` factor is verified numerically below (Experiment 1), not assumed.

Run:  uv run python scripts/spike_jacreg.py   (CPU, < ~2 min)

This is a spike. No package code, nothing committed. The real `sld` implementation
comes later, informed by these numbers.
"""

from __future__ import annotations

import statistics
import time

import torch
import torch.nn as nn

torch.set_num_threads(torch.get_num_threads())  # keep default; documented in report
DEVICE = torch.device("cpu")  # spike runs on CPU per the brief


# ----------------------------------------------------------------------------------
# The spike's penalty. Small on purpose. `probe` estimates ||J||_F^2 at ONE state via
# k fresh sphere probes; the rollout logic (Bernoulli step subsampling) lives in the
# training loop that uses it. Fresh probes every call, reproducible via an explicit
# torch.Generator (requirement c).
# ----------------------------------------------------------------------------------
class TotalJacobianPenalty:
    def __init__(self, k_probes: int = 2, step_subsample: float = 0.25,
                 generator: torch.Generator | None = None):
        self.k = k_probes
        self.p = step_subsample
        self.g = generator

    def _sphere(self, batch: int, d: int, dtype: torch.dtype) -> torch.Tensor:
        # Uniform on S^{d-1}: normalize an isotropic Gaussian. One fresh vector per row.
        v = torch.randn(batch, d, generator=self.g, dtype=dtype, device=DEVICE)
        return v / v.norm(dim=-1, keepdim=True)

    def probe(self, step_fn, q: torch.Tensor, d: int) -> torch.Tensor:
        """Per-sample estimate (shape (B,)) of ||J_i||_F^2 at each state q_i.

        step_fn processes rows independently (batched MLP / linear map), so
        jvp(step_fn, (q,), (v,)) returns J_i v_i per row.
        """
        batch = q.shape[0]
        acc = torch.zeros(batch, dtype=q.dtype, device=DEVICE)
        for _ in range(self.k):
            v = self._sphere(batch, d, q.dtype)
            _, jv = torch.func.jvp(step_fn, (q,), (v,))
            acc = acc + jv.pow(2).sum(dim=-1)
        return (d / self.k) * acc

    def step_penalty(self, step_fn, q: torch.Tensor, d: int) -> torch.Tensor:
        """Scalar penalty at one rollout step: mean over the batch."""
        return self.probe(step_fn, q, d).mean()


def rel_err(est: float, truth: float) -> float:
    return abs(est - truth) / abs(truth)


# ==================================================================================
# Experiment 1 — Unbiasedness on a known linear map (J = A exactly).
# ==================================================================================
def experiment_1() -> dict:
    print("=" * 78)
    print("EXPERIMENT 1 — Unbiasedness on a known linear map (d=16, float64)")
    print("=" * 78)

    d = 16
    seed = 0
    g_data = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(d, d, generator=g_data, dtype=torch.float64)
    analytic = (A * A).sum().item()  # ||A||_F^2

    # For a linear map J = A is constant in q, so any fixed q works.
    q = torch.randn(d, generator=g_data, dtype=torch.float64)
    step_fn = lambda x: A @ x  # noqa: E731

    def one_draw(k: int, gen: torch.Generator) -> float:
        # (d/k) * sum_v ||A v||^2 at a single q (batch of 1 collapsed to a vector).
        acc = 0.0
        for _ in range(k):
            v = torch.randn(d, generator=gen, dtype=torch.float64)
            v = v / v.norm()
            _, jv = torch.func.jvp(step_fn, (q,), (v,))
            acc = acc + jv.pow(2).sum()
        return ((d / k) * acc).item()

    g = torch.Generator(device="cpu").manual_seed(seed + 1)

    # --- Single-draw estimate at k=8 ---
    single_k8 = one_draw(8, g)
    print(f"\nAnalytic ||A||_F^2                     : {analytic:.6f}")
    print(f"Single draw, k=8                       : {single_k8:.6f}  "
          f"(rel err {rel_err(single_k8, analytic):.3%})")

    # --- Mean over 200 independent draws at k=8 ---
    draws_k8 = [one_draw(8, g) for _ in range(200)]
    mean_k8 = statistics.mean(draws_k8)
    std_k8 = statistics.stdev(draws_k8)
    err_mean_k8 = rel_err(mean_k8, analytic)
    relstd_k8 = std_k8 / analytic
    print(f"\nk=8, mean over 200 draws               : {mean_k8:.6f}  "
          f"(rel err of mean {err_mean_k8:.3%}, ACCEPT if < 5%)")
    print(f"k=8, single-draw std across 200        : {std_k8:.6f}  "
          f"(rel std {relstd_k8:.3%})")

    # --- k=2 (training config): single-draw noise, and mean unbiasedness over 500 ---
    draws_k2_200 = [one_draw(2, g) for _ in range(200)]
    std_k2 = statistics.stdev(draws_k2_200)
    relstd_k2 = std_k2 / analytic
    print(f"\nk=2, single-draw std across 200        : {std_k2:.6f}  "
          f"(rel std {relstd_k2:.3%})  <-- training-time per-step noise")

    draws_k2_500 = [one_draw(2, g) for _ in range(500)]
    mean_k2 = statistics.mean(draws_k2_500)
    err_mean_k2 = rel_err(mean_k2, analytic)
    print(f"k=2, mean over 500 draws               : {mean_k2:.6f}  "
          f"(rel err of mean {err_mean_k2:.3%}, ACCEPT if < 5%)")

    ok = err_mean_k8 < 0.05 and err_mean_k2 < 0.05
    print(f"\nVERDICT: d-scaling unbiasedness {'PASS' if ok else 'FAIL'} "
          f"(k=8 mean err {err_mean_k8:.3%}, k=2 mean err {err_mean_k2:.3%})")

    return dict(analytic=analytic, single_k8=single_k8, mean_k8=mean_k8,
                err_mean_k8=err_mean_k8, relstd_k8=relstd_k8, relstd_k2=relstd_k2,
                mean_k2=mean_k2, err_mean_k2=err_mean_k2, pass_=ok)


# ==================================================================================
# Experiment 2 — Differentiability THROUGH the jvp, w.r.t. module params.
# ==================================================================================
def experiment_2() -> dict:
    print("\n" + "=" * 78)
    print("EXPERIMENT 2 — Differentiability through the JVP (MLP 16->64->16, tanh)")
    print("=" * 78)

    d = 16
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, d)).double()

    # Functional closure: close over the module and CALL it inside the jvp'd function.
    # No torch.func.functional_call needed for a stateless MLP (no batchnorm/dropout).
    # jvp does forward-mode AD over q only; params flow as ordinary graph tensors, so
    # the returned jvp output stays connected to params for reverse-mode .backward().
    step_fn = lambda q: mlp(q)  # noqa: E731

    g = torch.Generator(device="cpu").manual_seed(1)
    pen = TotalJacobianPenalty(k_probes=4, step_subsample=1.0, generator=g)

    q = torch.randn(8, d, dtype=torch.float64)  # batch of states
    penalty = pen.step_penalty(step_fn, q, d)
    print(f"\nPenalty value (backprop target)        : {penalty.item():.6f}")
    print(f"Penalty has grad_fn                    : {penalty.grad_fn is not None}")

    penalty.backward()

    named = list(mlp.named_parameters())
    with_grad = [(n, p) for n, p in named if p.grad is not None]
    no_grad = [n for n, p in named if p.grad is None]
    all_finite = all(torch.isfinite(p.grad).all().item() for _, p in with_grad)
    any_nonzero = any(p.grad.abs().sum().item() > 0 for _, p in with_grad)
    grad_norm = sum(p.grad.pow(2).sum().item() for _, p in with_grad) ** 0.5

    # The Jacobian J = W2 diag(tanh'(W1 q + b1)) W1 contains W1, W2, b1 but NOT the
    # output bias b2 (an additive output constant vanishes under d/dq). So ||J||_F^2
    # is mathematically INDEPENDENT of b2: its grad is legitimately None. That is a
    # property of the penalty, not a failure of jvp backprop. Verify the only params
    # without grad are exactly the output-layer bias.
    out_linear = [m for m in mlp if isinstance(m, nn.Linear)][-1]
    out_bias_names = {n for n, p in named if p is out_linear.bias}
    no_grad_expected = set(no_grad) == out_bias_names
    print(f"Params WITH grad                       : {[n for n, _ in with_grad]}")
    print(f"Params WITHOUT grad                    : {no_grad}  "
          f"(expected = output bias {sorted(out_bias_names)}: {no_grad_expected})")
    print(f"All present grads finite               : {all_finite}")
    print(f"At least one grad nonzero              : {any_nonzero}")
    print(f"Total grad norm                        : {grad_norm:.6f}")

    # Backprop THROUGH the jvp works iff: penalty has grad_fn, every param that can
    # influence J received a finite grad, at least one is nonzero, and the only params
    # left ungraded are the output bias (provably outside J).
    diff_ok = (penalty.grad_fn is not None and all_finite and any_nonzero
               and grad_norm > 0 and no_grad_expected)
    if not diff_ok:
        # STOP condition: genuine inability to backprop to Jacobian-relevant params.
        print("\n*** STOP CONDITION: cannot backprop to params through jvp. ***")
        return dict(pass_=False, first=None, last=None)

    # 30-step Adam minimizing ONLY the penalty -> must decrease (regularizer can shape J).
    torch.manual_seed(0)
    mlp2 = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, d)).double()
    step_fn2 = lambda q: mlp2(q)  # noqa: E731
    g2 = torch.Generator(device="cpu").manual_seed(2)
    pen2 = TotalJacobianPenalty(k_probes=4, generator=g2)
    opt = torch.optim.Adam(mlp2.parameters(), lr=1e-2)
    q_fixed = torch.randn(8, d, dtype=torch.float64)

    first_val = last_val = None
    for step in range(30):
        opt.zero_grad()
        loss = pen2.step_penalty(step_fn2, q_fixed, d)
        loss.backward()
        opt.step()
        if step == 0:
            first_val = loss.item()
        last_val = loss.item()
    print(f"\nAdam(30) minimizing only the penalty   : "
          f"first {first_val:.4f} -> last {last_val:.4f}  "
          f"({'DECREASED' if last_val < first_val else 'did NOT decrease'})")

    ok = diff_ok and (last_val < first_val)
    print(f"\nVERDICT: differentiability {'PASS' if ok else 'FAIL'}")
    return dict(pass_=ok, grad_norm=grad_norm, first=first_val, last=last_val,
                no_grad=no_grad, all_finite=all_finite, any_nonzero=any_nonzero)


# ==================================================================================
# Experiment 3 — Overhead at the training config (k=2, subsample=0.25, d=32).
# ==================================================================================
class Predictor(nn.Module):
    def __init__(self, d=32, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, d))

    def forward(self, q):
        return self.net(q)


def _train_step(model, q0, target, d, pen: TotalJacobianPenalty | None,
                subsample_gen: torch.Generator, H: int):
    """One realistic toy training step: H-step rollout, MSE loss, (optional) penalty, backward."""
    model.zero_grad(set_to_none=True)
    q = q0
    step_fn = lambda x: model(x)  # noqa: E731  (functional closure, same pattern as Exp 2)
    penalties = []
    for _ in range(H):
        q = model(q)
        if pen is not None:
            keep = torch.rand(1, generator=subsample_gen, device=DEVICE).item() < pen.p
            if keep:
                penalties.append(pen.step_penalty(step_fn, q, d))
    loss = ((q - target) ** 2).mean()
    if penalties:
        loss = loss + torch.stack(penalties).mean()
    loss.backward()
    return loss


def _median_time(fn, reps=50, warmup=5) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def experiment_3() -> dict:
    print("\n" + "=" * 78)
    print("EXPERIMENT 3 — Overhead at training config (MLP 32->128->32, B=64, H=20)")
    print("=" * 78)

    d, hidden, B, H = 32, 128, 64, 20
    torch.manual_seed(0)
    model = Predictor(d, hidden)
    q0 = torch.randn(B, d)
    target = torch.randn(B, d)
    reps, warmup = 60, 5

    g_probe = torch.Generator(device="cpu").manual_seed(10)
    g_sub = torch.Generator(device="cpu").manual_seed(20)

    def baseline():
        _train_step(model, q0, target, d, None, g_sub, H)

    def make_penalized(k, p):
        pen = TotalJacobianPenalty(k_probes=k, step_subsample=p, generator=g_probe)
        return lambda: _train_step(model, q0, target, d, pen, g_sub, H)

    t_base = _median_time(baseline, reps, warmup)
    t_primary = _median_time(make_penalized(2, 0.25), reps, warmup)
    ratio_primary = t_primary / t_base

    print(f"\nProtocol: CPU, {torch.get_num_threads()} threads, median of {reps} reps "
          f"after {warmup} warmup.")
    print(f"\n{'config':<34}{'median (ms)':>14}{'ratio':>10}")
    print("-" * 58)
    print(f"{'baseline (rollout+loss+backward)':<34}{t_base*1e3:>14.3f}{1.0:>10.2f}")
    print(f"{'penalty k=2, subsample=0.25 (PRIMARY)':<34}{t_primary*1e3:>14.3f}"
          f"{ratio_primary:>10.2f}")

    out = dict(t_base=t_base, t_primary=t_primary, ratio_primary=ratio_primary)

    print(f"\nPRIMARY ratio = {ratio_primary:.2f}x  (target < 2.5x): "
          f"{'PASS' if ratio_primary < 2.5 else 'EXCEEDS TARGET'}")

    # Always measure the two declared mitigations so the report is complete regardless.
    t_m1 = _median_time(make_penalized(1, 0.25), reps, warmup)
    t_m2 = _median_time(make_penalized(2, 0.10), reps, warmup)
    r_m1, r_m2 = t_m1 / t_base, t_m2 / t_base
    print(f"{'mitigation (i)  k=1, subsample=0.25':<34}{t_m1*1e3:>14.3f}{r_m1:>10.2f}")
    print(f"{'mitigation (ii) k=2, subsample=0.10':<34}{t_m2*1e3:>14.3f}{r_m2:>10.2f}")
    out.update(t_m1=t_m1, t_m2=t_m2, ratio_m1=r_m1, ratio_m2=r_m2)

    return out


def main():
    t0 = time.perf_counter()
    r1 = experiment_1()
    r2 = experiment_2()
    r3 = experiment_3()

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"E1 unbiasedness (d-scaling verified): {'PASS' if r1['pass_'] else 'FAIL'} "
          f"| k=8 mean err {r1['err_mean_k8']:.2%}, k=2 mean err {r1['err_mean_k2']:.2%}, "
          f"k=2 single-draw rel std {r1['relstd_k2']:.1%}")
    print(f"E2 differentiability                : {'PASS' if r2['pass_'] else 'FAIL'} "
          f"| penalty {r2.get('first')} -> {r2.get('last')}")
    print(f"E3 overhead (primary k=2, ss=0.25)  : {r3['ratio_primary']:.2f}x "
          f"(target <2.5x) | mit(i) {r3['ratio_m1']:.2f}x, mit(ii) {r3['ratio_m2']:.2f}x")
    print(f"\nTotal wall time: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
