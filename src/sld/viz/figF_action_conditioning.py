"""Figure F — the conditioning of the action Jacobian per arm at H = 25.

`kappa(d x_H / d u)`: how the state reached after 25 free-running steps responds
to the action sequence that drove it there, and how invertible that response is.
A large condition number means some action direction has almost no effect on the
final state -- the quantity a planner would have to invert.

Torch-free, like every module under `sld.viz`: the conditioning is computed by
`sld.audit.scan_action_jacobian` against each run's committed checkpoint and
persisted as `action_jacobian.json`; this module reads JSON. Every value drawn
here was cross-checked against centred finite differences to better than 1e-8
relative BEFORE it was written -- the audit raises rather than return an
unchecked Jacobian, so an unchecked number cannot reach this figure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from sld.viz.style import INK_FAINT, INK_MUTED, INK_PRIMARY, STRUCTURE_COLORS, apply_style, save_fig

DEFAULT_ROOT = "results_osc_n"
DEFAULT_OUT = "assets/figF_action_conditioning.png"
ARMS = ("off", "on-free", "on-stable")


def load_conditioning(root: str | Path):
    root = Path(root)
    exp = next(p for p in root.iterdir() if p.is_dir())
    idx = {r["run_dir"].split("/")[-1]: r["regime"] for r in csv.DictReader((exp / "index.csv").open())}
    cells, worst_fd = {}, 0.0
    for d in sorted(exp.iterdir()):
        if not d.is_dir() or not (d / "action_jacobian.json").exists():
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        arm = ("off" if cfg["family"] == "mlp"
               else ("on-stable" if cfg.get("radius") == "bounded" else "on-free"))
        payload = json.loads((d / "action_jacobian.json").read_text())
        worst_fd = max(worst_fd, payload["fd_max_rel_error"])
        cells.setdefault((idx.get(d.name, "?"), arm), []).append(payload["cond"])
    return cells, worst_fd


def make_figF(root: str | Path = DEFAULT_ROOT, out: str | Path = DEFAULT_OUT) -> Path:
    apply_style()
    cells, worst_fd = load_conditioning(root)
    datasets = ["n8", "n32", "n256"]

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    for slot, ds in enumerate(datasets):
        for k, arm in enumerate(ARMS):
            vals = sorted(cells.get((ds, arm), []))
            if not vals:
                ax.annotate("no cell", xy=(slot + (k - 1) * 0.24, 10), ha="center",
                            fontsize=7, color=INK_MUTED, style="italic")
                continue
            x = slot + (k - 1) * 0.24
            v = np.asarray(vals)
            ax.vlines(x, v.min(), v.max(), color=STRUCTURE_COLORS[arm], linewidth=1.0, alpha=0.55, zorder=2)
            q1, q3 = np.percentile(v, [25, 75])
            ax.vlines(x, q1, q3, color=STRUCTURE_COLORS[arm], linewidth=4.5, alpha=0.28, zorder=2)
            ax.scatter(x + np.linspace(-0.055, 0.055, len(v)), v, s=26,
                       color=STRUCTURE_COLORS[arm], edgecolor="white", linewidth=0.5, zorder=4)
            ax.hlines(np.median(v), x - 0.075, x + 0.075, color=INK_PRIMARY, linewidth=1.8, zorder=5)

    ax.set_yscale("log")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(["8", "32", "256"])
    ax.set_xlabel("n_episodes")
    ax.set_ylabel(r"$\kappa(\partial x_H / \partial u)$ at $H = 25$")
    ax.set_title("Action-Jacobian conditioning per arm", loc="left")
    ax.set_xlim(-0.5, len(datasets) - 0.5)
    ax.legend(handles=[plt.Line2D([], [], color=STRUCTURE_COLORS[a], marker="o", linestyle="none",
                                  markersize=5, label=a) for a in ARMS] +
                      [plt.Line2D([], [], color=INK_PRIMARY, linewidth=1.8, label="median"),
                       plt.Line2D([], [], color=INK_MUTED, linewidth=4.5, alpha=0.28, label="IQR"),
                       plt.Line2D([], [], color=INK_MUTED, linewidth=1.0, alpha=0.55, label="range")],
              fontsize=7.5, loc="upper right", framealpha=0.95)

    fig.text(0.0, -0.07,
             f"PROTOCOL 25.6's scope applies: measured UNDER THE DECLARED REGIME (h_fr = 25).  Root: "
             f"results_osc_n/, 44 of 45 cells -- the diverged cell has no checkpoint and is not "
             f"fabricated.  Probe state and action sequence are drawn from ONE fixed seed for every arm, "
             f"so what differs between arms is the model.  Every value was cross-checked against centred "
             f"finite differences before being written; worst relative disagreement over all 44 runs: "
             f"{worst_fd:.1e}, against the 1e-8 the audit refuses above.  CONFOUND: the structure axis "
             f"moves a composite (transition + psi + pi, 13.7), so kappa is the composite's conditioning, "
             f"not the transition's.  No gate on this root (23.4).",
             fontsize=6.6, color=INK_MUTED, va="top", wrap=True)
    out = Path(out)
    save_fig(fig, out)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    make_figF(**vars(ap.parse_args()))


if __name__ == "__main__":
    main()
