"""Fig B -- the D4 grid, read across the horizon instead of at the gate.

Section 21.0 reports the gate margins at a single horizon and concludes that
`structure: on` costs 1.54x-2.27x. That reading is correct and it is not the
whole content of the artifacts: the same `gate.json` files carry `e_h` for
every h in 1..25, and the shape of those curves says something the scalar
cannot. The structured arms start with a larger one-step error and COMPOUND it
more slowly -- which is, word for word, the repository's stated research
question.

No new run. This module reads the 40 artifacts already on disk.

Two things the caption must name, because the reading is confounded by both
and neither is repairable from these artifacts:

1. the rate asymmetry of section 21.2 -- each OFF arm ran at a rate selected
   for itself, each ON arm at a rate selected for a different architecture;
2. `structure: on` is a composite -- the structured transition AND the lift
   psi AND the projection pi, which together take 10.3%-20.4% of the parameter
   budget (section 13.7). The panels cannot separate the three.

Medians are over the five seeds. This root has a gate, so section 12.4 forbids
a reducer from DECIDING anything; a median may still DISPLAY, and the per-seed
points are drawn alongside it so the spread is never hidden behind it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sld.viz.ingest import arm_label, load_gate
from sld.viz.style import (
    CONSTRAINT_LINESTYLES,
    FAMILY_COLORS,
    INK_FAINT,
    INK_MUTED,
    INK_PRIMARY,
    apply_style,
    save_fig,
)

DEFAULT_ROOT = "results_gate_d"
DEFAULT_OUT = "assets/figB_horizon.png"

STRATUM_TITLES = {"OU": "OU stratum", "AC": "action-conditioned stratum"}
ARM_ORDER = ["mlp", "mlp_k", "transformer", "transformer_k"]


def make_figB(root: str | Path = DEFAULT_ROOT, out: str | Path = DEFAULT_OUT, *, dpi: int | None = None) -> Path:
    """Render Fig B from the `gate.json` artifacts under ``root``."""
    import matplotlib.pyplot as plt

    scalars, curves = load_gate(root)
    for frame in (scalars, curves):
        frame["arm"] = [arm_label(b, s) for b, s in zip(frame.backbone, frame.structure, strict=True)]

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1), constrained_layout=True)
    strata = ["OU", "AC"]

    for ax, stratum in zip(axes[:2], strata, strict=True):
        sub = curves[curves["stratum"] == stratum]
        # the two trivial baselines: identical for every arm by construction,
        # so they are drawn once, in ink, never in a series color
        base = sub.groupby("h")[["persistence_e_h", "linear_e_h"]].median()
        ax.plot(base.index, base["persistence_e_h"], color=INK_FAINT, linewidth=1.2, zorder=1)
        ax.plot(base.index, base["linear_e_h"], color=INK_FAINT, linewidth=1.2, linestyle=(0, (4, 2)), zorder=1)
        ax.annotate("persistence", xy=(25, base["persistence_e_h"].iloc[-1]), xytext=(3, 0),
                    textcoords="offset points", fontsize=7, color=INK_MUTED, va="center")
        ax.annotate("linear extrap.", xy=(25, base["linear_e_h"].iloc[-1]), xytext=(3, -6),
                    textcoords="offset points", fontsize=7, color=INK_MUTED, va="center")

        for arm in ARM_ORDER:
            a = sub[sub["arm"] == arm]
            if a.empty:
                continue
            structure = a["structure"].iloc[0]
            med = a.groupby("h")["e_h"].median()
            ax.plot(med.index, med.values, color=FAMILY_COLORS[arm], linewidth=2.0,
                    linestyle=CONSTRAINT_LINESTYLES[structure], zorder=3)
            for seed, s in a.groupby("seed"):
                s = s.sort_values("h")
                ax.plot(s["h"], s["e_h"], color=FAMILY_COLORS[arm], linewidth=0.6, alpha=0.28, zorder=2)
            # direct label: the two light hues fall below 3:1 on white, so the
            # legend alone would not carry identity
            ax.annotate(arm, xy=(25, med.iloc[-1]), xytext=(3, 0), textcoords="offset points",
                        fontsize=7.5, color=FAMILY_COLORS[arm], va="center", fontweight="bold")

        ax.set_yscale("log")
        ax.set_xlabel("free-running horizon $h$", color=INK_MUTED)
        ax.set_ylabel(r"normalised state error $e_h$", color=INK_MUTED)
        ax.set_title(STRATUM_TITLES[stratum], color=INK_PRIMARY)
        ax.set_xlim(1, 31)
        ax.set_xticks([1, 5, 10, 15, 20, 25])

    # --- panel 3: the compounding rate rho = e_25 / e_1 ---------------------
    ax = axes[2]
    e1 = curves[curves["h"] == 1].set_index(["stratum", "arm", "seed"])["e_h"]
    e25 = curves[curves["h"] == 25].set_index(["stratum", "arm", "seed"])["e_h"]
    rho = (e25 / e1).rename("rho").reset_index()

    xticklabels, xpos = [], []
    for i, stratum in enumerate(strata):
        for j, arm in enumerate(ARM_ORDER):
            x = i * (len(ARM_ORDER) + 1) + j
            r = rho[(rho.stratum == stratum) & (rho.arm == arm)]["rho"].to_numpy()
            if not len(r):
                continue
            ax.scatter(np.full(len(r), x) + np.linspace(-0.13, 0.13, len(r)), r, s=30,
                       color=FAMILY_COLORS[arm], edgecolor="white", linewidth=0.8, zorder=3)
            ax.plot([x - 0.24, x + 0.24], [np.median(r)] * 2, color=FAMILY_COLORS[arm],
                    linewidth=2.4, solid_capstyle="round", zorder=4)
            xpos.append(x)
            xticklabels.append(f"{arm}\n{stratum}")
    ax.set_xticks(xpos)
    ax.set_xticklabels(xticklabels, fontsize=7.5, color=INK_PRIMARY)
    ax.set_ylabel(r"$\rho = e_{25}\,/\,e_{1}$   (error compounding)", color=INK_MUTED)
    ax.set_title("How fast the error compounds", color=INK_PRIMARY)
    ax.set_yscale("log")

    fig.suptitle(
        "D4, read across the horizon. The structured arms start worse at one step and compound more slowly "
        "in 3 of the 4 (backbone, stratum) pairs -- 40 runs already on disk, no new training.",
        fontsize=9.5, color=INK_PRIMARY,
    )
    fig.text(
        0.5, -0.09,
        "Solid = structure ON, dashed = OFF. Thin lines are the five seeds, thick line their median (display only: "
        "section 12.4 forbids a reducer from deciding). TWO CONFOUNDS, neither repairable from these artifacts: "
        "(1) the rate asymmetry of section 21.2 -- OFF arms ran at a rate selected for themselves, ON arms at a rate "
        "selected for a different architecture; (2) 'structure: on' is a composite of the structured transition, the "
        "lift psi and the projection pi, which take 10.3-20.4% of the parameter budget (section 13.7)."
        " Scope (25.6): measured UNDER THE DECLARED REGIME (h_fr = 25); root 1 of amendment 16 measured the sign of the structure effect unchanged at h_fr = 5 (25.8).",
        ha="center", fontsize=7.5, color=INK_MUTED, wrap=True,
    )
    out = Path(out)
    save_fig(fig, out, dpi=dpi)
    return out


def main() -> None:
    print(f"wrote {make_figB()}")


if __name__ == "__main__":
    main()
