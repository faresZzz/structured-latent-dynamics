"""Figure D — the structure axis against prior misspecification (PROTOCOL 23.5,
23.7 and 26.1).

TWO panels, and the order matters. The LEFT one carries the absolutes: the three
arms' `m_low` and, on the same axes, the two trivial baselines the metric is
normalised by. The RIGHT one carries the ratio -- never on its own.

That layout is the finding, not a style choice. 23.5's pre-registered estimand is
the SIGN of the slope of the structure gain against `delta(c)`, and the ratio
confirms it: +9.61 and +10.52 per unit of `delta`. But the absolutes show the
ratio is a DENOMINATOR effect -- the structured arms are PINNED at `m_low ~ 1`
across the whole axis (x1.24 and x1.20) while `off` improves x3.67. The
predicted sign is confirmed and its mechanism is not the one the prediction
supposed. A ratio of ratios is exactly where a mechanism hides, which is why
both terms are drawn.

The x axis is `delta(c)`, computed in closed form by 23.7 under its declared
pairing convention (a conjugate pair is never split), not estimated.

Torch-free (REGLES section 2).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from sld.viz.style import INK_FAINT, INK_MUTED, STRUCTURE_COLORS, apply_style, save_fig

DEFAULT_ROOT = "results_osc_delta"
DEFAULT_OUT = "assets/figD_misspecification.png"

# 23.7's closed-form axis, declared before the runs and not re-derived here.
DELTA_OF = {"c010": 0.000, "c230": 0.363, "c260": 0.706, "c300": 1.029}
ARMS = ("off", "on-free", "on-stable")


def load_cells(root: str | Path):
    root = Path(root)
    exp = next(p for p in root.iterdir() if p.is_dir())
    idx = {r["run_dir"].split("/")[-1]: r["regime"] for r in csv.DictReader((exp / "index.csv").open())}
    cells, base = {}, {}
    for d in sorted(exp.iterdir()):
        if not d.is_dir() or not (d / "config.yaml").exists():
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        arm = ("off" if cfg["family"] == "mlp"
               else ("on-stable" if cfg.get("radius") == "bounded" else "on-free"))
        level = idx.get(d.name, "?")
        if (d / "failure.json").exists() or not (d / "gate.json").exists():
            cells.setdefault((level, arm), []).append(None)
            continue
        m = json.loads((d / "gate.json").read_text())["measures"]
        cells.setdefault((level, arm), []).append(m["m_low"])
        base.setdefault(level, []).append((m["persistence_e_h"][24], m["linear_e_h"][24]))
    return cells, base


def make_figD(root: str | Path = DEFAULT_ROOT, out: str | Path = DEFAULT_OUT) -> Path:
    apply_style()
    cells, base = load_cells(root)
    levels = ["c010", "c230", "c260", "c300"]
    xs = [DELTA_OF[c] for c in levels]

    fig, (ax_abs, ax_ratio) = plt.subplots(1, 2, figsize=(12.2, 4.5))
    fig.subplots_adjust(wspace=0.42)

    # --- LEFT: absolutes, with the baselines the metric divides by ----------
    for arm in ARMS:
        med, seeds_x, seeds_y = [], [], []
        for x, c in zip(xs, levels):
            vals = [v for v in cells[(c, arm)] if v is not None]
            med.append(float(np.median(vals)))
            seeds_x += [x] * len(vals)
            seeds_y += vals
        ax_abs.scatter(seeds_x, seeds_y, s=20, color=STRUCTURE_COLORS[arm],
                       alpha=0.55, edgecolor="white", linewidth=0.4, zorder=3)
        ax_abs.plot(xs, med, color=STRUCTURE_COLORS[arm], linewidth=1.8, marker="o",
                    markersize=5, label=arm, zorder=4)
    ax_abs.axhline(1.0, linestyle=":", linewidth=1.0, color=INK_MUTED, zorder=1)
    ax_abs.annotate("the two structured arms are PINNED here, across the whole axis",
                    xy=(0.03, 1.0), xycoords=("axes fraction", "data"), xytext=(0, 13),
                    textcoords="offset points", fontsize=7, color=INK_MUTED)
    ax_abs.set_yscale("log")
    ax_abs.set_xlabel(r"$\delta(c)$   (closed form, 23.7)")
    ax_abs.set_ylabel(r"$m_{low}$   (lower is better)")
    ax_abs.set_title("Absolutes, and what normalises them", loc="left")

    # the baselines, in ink: they are not arms
    twin = ax_abs.twinx()
    pers = [float(np.median([p for p, _ in base[c]])) for c in levels]
    lin = [float(np.median([lin_e for _, lin_e in base[c]])) for c in levels]
    twin.plot(xs, pers, color=INK_MUTED, linewidth=1.1, linestyle="--", marker="s", markersize=3.5)
    twin.plot(xs, lin, color=INK_FAINT, linewidth=1.1, linestyle="-.", marker="^", markersize=3.5)
    twin.set_ylabel(r"baseline $e_{25}$", color=INK_MUTED, fontsize=8)
    twin.tick_params(axis="y", labelcolor=INK_MUTED, labelsize=7)
    ax_abs.legend(
        handles=[plt.Line2D([], [], color=STRUCTURE_COLORS[a], marker="o", linewidth=1.8, label=a) for a in ARMS]
        + [plt.Line2D([], [], color=INK_MUTED, linestyle="--", marker="s", markersize=3.5, label="persistence (right axis)"),
           plt.Line2D([], [], color=INK_FAINT, linestyle="-.", marker="^", markersize=3.5, label="linear (right axis)")],
        fontsize=7, loc="center left", framealpha=0.95)

    # --- RIGHT: the ratio, second and never alone ---------------------------
    for arm in ("on-free", "on-stable"):
        ratio = []
        for c in levels:
            off = float(np.median([v for v in cells[(c, "off")] if v is not None]))
            arm_med = float(np.median([v for v in cells[(c, arm)] if v is not None]))
            ratio.append(arm_med / off)
        slope = np.polyfit(xs, ratio, 1)[0]
        ax_ratio.plot(xs, ratio, color=STRUCTURE_COLORS[arm], linewidth=1.8, marker="o",
                      markersize=5, label=f"{arm}   slope {slope:+.2f}")
    ax_ratio.set_xlabel(r"$\delta(c)$")
    ax_ratio.set_ylabel(r"$m_{low}$(structured) / $m_{low}$(off)")
    ax_ratio.set_title("The ratio: predicted sign confirmed, step not ramp", loc="left")
    ax_ratio.legend(fontsize=7.5, loc="lower right", framealpha=0.95)
    ax_ratio.annotate("the jump is between the first two levels;\nflat after -- what a denominator effect looks like",
                      xy=(0.5, 0.12), xycoords="axes fraction", ha="center", fontsize=7, color=INK_MUTED,
                      bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                            "edgecolor": INK_FAINT, "linewidth": 0.7})

    fig.text(0.0, -0.06,
             "PROTOCOL 23.5 / 23.7 / 26.1. Root: results_osc_delta/, 60 cells, 58 complete + 2 divergences "
             "(on-free seed 3 at c=0.10 and c=2.60); 23.5's stop rule -- more than 2 of 3 cells at a level of c -- "
             "is reached at NO level.  m_low = e_25 * gamma / min(persistence, linear), so both terms move with the "
             "damping and the left panel draws both.  CONFOUNDS: the structure axis moves a composite (transition + "
             "psi + pi, 13.7), and the 23.3 argmin sat at the EDGE of the declared rate list with the criterion still "
             "decreasing, so all three arms may be under-tuned in the same direction.  No gate (23.4).",
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
    make_figD(**vars(ap.parse_args()))


if __name__ == "__main__":
    main()
