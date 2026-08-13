"""Fig A -- the D3 control: the models learn, and they do not identify.

Four panels, four separate y-axes, deliberately. PROTOCOL.md section 19.2 fixes
that "the spectral bound is never to be reported as if it summarised the
others"; putting the four bounds on one axis would do exactly that. The gain
panel is on a linear scale and the three exactness panels are logarithmic,
because the quantity they carry spans fifteen decades between the planted gold
and the measured value.

Every horizontal reference is an analytic anchor from PROTOCOL.md section 20.1,
not a measurement of these runs:

- planted gold ``1.9e-15``   -- the optimum exists inside the hypothesis class;
- ordinary least squares     -- the same training split recovers the operator;
- pre-registered threshold ``1e-3``.

What the legend must say, and must not: the measured values sit **110x to 200x
above the pre-registered 1e-3 threshold**. It must NOT say "200x further than
least squares" -- the real gap to OLS is of order 5e6x, and the caption would
contradict its own panels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sld.viz.ingest import load_d3_gold
from sld.viz.style import (
    FAMILY_COLORS,
    INK_MUTED,
    INK_PRIMARY,
    apply_style,
    reference_line,
    save_fig,
)

DEFAULT_ROOT = "results_d3_oscillator"
DEFAULT_OUT = "assets/figA_d3_exactness.png"

# PROTOCOL.md section 20.1, the four independent checks. Values are analytic or
# closed-form references, never measurements of the D3 runs themselves.
REFERENCES: dict[str, dict[str, float]] = {
    "state_h100": {"gold": 1.9e-15, "ols": 7.98e-14, "threshold": 1e-3},
    "spectral": {"ols": 2.11e-08, "threshold": 1e-3},
    "action": {"ols": 1.58e-07, "threshold": 1e-3},
}

PANELS: list[tuple[str, str, str]] = [
    ("m_low", "Relative gain vs trivial baselines", r"$m_{\mathrm{low}}$  (lower is better)"),
    ("state_h100", "State error at $H=100$", "normalised state error"),
    ("spectral", "Spectral error on $J_x$", "relative spectral error"),
    ("action", "Action error on $J_u$", "relative action error"),
]


def make_figA(root: str | Path = DEFAULT_ROOT, out: str | Path = DEFAULT_OUT, *, dpi: int | None = None) -> Path:
    """Render Fig A from ``d3_gold.json`` artifacts under ``root``."""
    import matplotlib.pyplot as plt

    df = load_d3_gold(root)
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.9), constrained_layout=True)

    families = sorted(df["family"].unique())
    xpos = {f: i for i, f in enumerate(families)}

    for ax, (col, title, ylabel) in zip(axes, PANELS, strict=True):
        for fam in families:
            sub = df[df["family"] == fam]
            jitter = np.linspace(-0.11, 0.11, len(sub))
            ax.scatter(
                xpos[fam] + jitter,
                sub[col],
                s=34,
                color=FAMILY_COLORS[fam],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                label=fam,
            )
            med = float(np.median(sub[col]))
            ax.plot(
                [xpos[fam] - 0.20, xpos[fam] + 0.20],
                [med, med],
                color=FAMILY_COLORS[fam],
                linewidth=2.0,
                solid_capstyle="round",
                zorder=4,
            )
        if col == "m_low":
            reference_line(ax, 1.0, "pass: $m_{\\mathrm{low}} < 1$", style="--")
            ax.set_ylim(0, 1.15)
        else:
            refs = REFERENCES[col]
            ax.set_yscale("log")
            reference_line(ax, refs["threshold"], "threshold $10^{-3}$", style="--")
            reference_line(ax, refs["ols"], "least squares", style=":")
            if "gold" in refs:
                reference_line(ax, refs["gold"], "planted gold", style=":")
            lo = min(refs.values()) / 30
            ax.set_ylim(lo, max(float(df[col].max()) * 6, refs["threshold"] * 30))
        ax.set_title(title, color=INK_PRIMARY)
        ax.set_ylabel(ylabel, color=INK_MUTED)
        ax.set_xticks(list(xpos.values()))
        ax.set_xticklabels(families, color=INK_PRIMARY)
        ax.set_xlim(-0.5, len(families) - 0.5)

    n = len(df)
    fig.suptitle(
        "D3 control, CoupledOscillators -- the structured predictors learn (10/10) "
        f"and do not identify (0/10 on all three exactness bounds).  n = {n} runs, 5 seeds x 2 arms.",
        fontsize=9.5,
        color=INK_PRIMARY,
        y=1.04,
    )
    fig.text(
        0.5,
        -0.10,
        "Measured values sit 110x-200x above the pre-registered 1e-3 threshold. The optimum is inside the "
        "hypothesis class (planted gold 1.9e-15) and is recoverable from the same training split by ordinary "
        "least squares; the declared training regime does not reach it. Reference lines are analytic anchors "
        "(PROTOCOL.md 20.1), never measurements of these runs." +
        " Scope (25.6): measured UNDER THE DECLARED REGIME (h_fr = 25); root 1 of amendment 16 measured the sign of the structure effect unchanged at h_fr = 5 (25.8).",
        ha="center",
        fontsize=7.5,
        color=INK_MUTED,
        wrap=True,
    )
    return _save(fig, out, dpi)


def _save(fig, out: str | Path, dpi: int | None = None) -> Path:
    out = Path(out)
    save_fig(fig, out, dpi=dpi)
    return out


def main() -> None:
    path = make_figA()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
