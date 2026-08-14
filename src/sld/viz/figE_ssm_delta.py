"""Figure E — campaign C's verdict: the ssm family's divergence is a property of
`delta_max`, not of the family (PROTOCOL.md sections 23.6 and 24).

Two panels, both read from `results_ssm_delta/` and from NOTHING ELSE. Section
23.0 requires these roots to be self-sufficient -- "no result is read across
roots" -- and this figure is where that rule bites hardest: the numbers a
reader most wants beside these curves (18.0's `2.4e18` first-step gradient norm
and its `mlp` reference of `28.2`) were measured on `results_probe_d/`, on a
different system, and plotting them here as data would be exactly the merge
14.6 forbids. They appear in the caption, attributed to their root, and are not
drawn. What the panels show, this root measured itself.

- **Left** — `train_loss` against the update count, log-x so both regimes fit:
  the `delta_max = 10.0` arms stop between epoch 28 and 56, the `4*dt` arms run
  all 8000. `min_ssm` at 10.0 has NO curve at all, and that absence is the
  datum -- it goes non-finite at epoch 0, before a single row is written. It is
  annotated rather than dropped, because a missing series a reader has to
  notice is a missing series a reader will miss.
- **Right** — the first optimizer step's gradient norm, per seed, log scale.
  This panel is the one that corrected section 24: for `ssm_k` the two
  `delta_max` levels give the SAME first step, seed by seed, to three decimals.
  The clamp does not act at initialization; it acts cumulatively. The panel is
  drawn seed-paired for exactly that reason -- a group summary would hide the
  pairing, which is the whole finding.

Torch-free, like every module under `sld.viz` (REGLES section 2).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sld.viz.style import INK_FAINT, INK_MUTED, apply_style, save_fig

DEFAULT_ROOT = "results_ssm_delta"
DEFAULT_OUT = "assets/figE_ssm_delta.png"

# The declared factor's two levels (23.6), and the colour that carries them.
# NOT STRUCTURE_COLORS: the factor on this figure is `delta_max`, not the
# structure axis -- reusing the structure palette here would tell a reader
# these panels vary structure, which they do not.
LEVEL_COLOR = {"10.0": "#D55E00", "0.2": "#0072B2"}
_SMOOTH = 50   # running-mean window on the completed runs, named in the caption
LEVEL_LABEL = {"10.0": r"$\delta_{max}=10.0$  (500$\times$ dt)", "0.2": r"$\delta_{max}=0.20$  (4$\times$ dt)"}


def _flat_config(run_dir: Path) -> dict[str, str]:
    """The run's own flattened config -- the one hashed into its identity."""
    out: dict[str, str] = {}
    for line in (run_dir / "config.yaml").read_text().splitlines():
        if ":" in line and not line.startswith((" ", "\t")):
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip()
    return out


def load_cells(root: str | Path) -> list[dict]:
    """One record per run under ``root``: identity, loss trajectory, first-step
    gradient norm, and whether it diverged.

    A diverged cell is kept, with whatever it managed to write. Dropping it
    would turn "10 of 20 cells died" into "10 cells", which is the one reading
    this figure exists to prevent (PROTOCOL 15.3: a divergence is published
    with its numbers, never replaced).
    """
    cells = []
    for run_dir in sorted(Path(root).rglob("config.yaml")):
        run_dir = run_dir.parent
        cfg = _flat_config(run_dir)
        if "delta_max" not in cfg:
            continue
        rows = []
        metrics = run_dir / "metrics.csv"
        if metrics.exists():
            rows = list(csv.DictReader(metrics.open()))
        failure = run_dir / "failure.json"
        record = {
            "family": cfg["family"],
            "delta_max": cfg["delta_max"],
            "seed": int(cfg["seed"]),
            "diverged": failure.exists(),
            "loss": [float(r["train_loss"]) for r in rows if r.get("train_loss")],
            "grad0": next((float(r["grad_norm"]) for r in rows if r.get("grad_norm")), None),
        }
        if failure.exists():
            record["failed_at"] = json.loads(failure.read_text()).get("epoch")
        cells.append(record)
    return cells


def make_figE(root: str | Path = DEFAULT_ROOT, out: str | Path = DEFAULT_OUT) -> Path:
    apply_style()
    cells = load_cells(root)
    if not cells:
        raise SystemExit(f"aucune cellule sous {root}")

    fig, (ax_loss, ax_grad) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    families = ("min_ssm", "ssm_k")

    # --- left: the loss trajectory ---------------------------------------
    silent: list[str] = []
    for cell in cells:
        level = cell["delta_max"]
        curve = cell["loss"]
        if not curve:
            silent.append(f"{cell['family']} {LEVEL_LABEL[level].split('  ')[0]}")
            continue
        x = np.arange(1, len(curve) + 1)
        y = np.asarray(curve)
        style = "-" if cell["family"] == "ssm_k" else "--"
        if len(y) > 4 * _SMOOTH:
            # The per-update loss is minibatch noise an order of magnitude wider
            # than the trend; raw, twenty of these curves are one solid block.
            # Raw kept faint underneath so the smoothing hides nothing, and the
            # window is named in the caption.
            ax_loss.plot(x, y, color=LEVEL_COLOR[level], linewidth=0.4, alpha=0.12, zorder=1)
            k = np.ones(_SMOOTH) / _SMOOTH
            y = np.convolve(y, k, mode="valid")
            x = x[_SMOOTH - 1:]
        ax_loss.plot(
            x, y,
            color=LEVEL_COLOR[level],
            linewidth=1.0,
            alpha=0.9,
            linestyle=style,
            zorder=3 if cell["diverged"] else 2,
        )
        if cell["diverged"]:
            ax_loss.plot(len(curve), curve[-1], marker="x", markersize=8, markeredgewidth=1.6,
                         color=LEVEL_COLOR[level], zorder=4)

    ax_loss.set_xscale("log")
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("optimizer update (log)")
    ax_loss.set_ylabel("train_loss")
    ax_loss.set_title("Every cell, to completion or to death", loc="left")
    ax_loss.set_xlim(1, 8000)

    if silent:
        ax_loss.annotate(
            f"min_ssm @ 10.0: {len(silent)} cells, NO curve\n"
            "non-finite at epoch 0, before any row",
            xy=(0.03, 0.06), xycoords="axes fraction", ha="left", va="bottom",
            fontsize=7.5, color=INK_MUTED,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "edgecolor": INK_FAINT, "linewidth": 0.7},
        )


    # --- right: first-step gradient norm, per seed ------------------------
    positions, labels = [], []
    for i, (family, level) in enumerate([(f, lv) for f in families for lv in ("10.0", "0.2")]):
        group = sorted(
            (c for c in cells if c["family"] == family and c["delta_max"] == level),
            key=lambda c: c["seed"],
        )
        values = [(c["seed"], c["grad0"]) for c in group if c["grad0"] is not None]
        positions.append(i)
        labels.append(f"{family}\n{'10.0' if level == '10.0' else '0.20'}")
        if values:
            ax_grad.scatter([i] * len(values), [v for _, v in values], s=36,
                            color=LEVEL_COLOR[level], edgecolor="white", linewidth=0.6, zorder=3)
        else:
            ax_grad.annotate("no step\ncompleted", xy=(i, 14.0), ha="center", va="center",
                             fontsize=8, color=INK_MUTED, style="italic")

    # the finding: for ssm_k the two levels are the SAME first step, seed by seed
    pair = {}
    for lv, x in (("10.0", 2), ("0.2", 3)):
        for c in cells:
            if c["family"] == "ssm_k" and c["delta_max"] == lv and c["grad0"] is not None:
                pair.setdefault(c["seed"], {})[x] = c["grad0"]
    for seed, ends in sorted(pair.items()):
        if len(ends) == 2:
            ax_grad.plot([2, 3], [ends[2], ends[3]], color=INK_FAINT, linewidth=0.8,
                         zorder=2, solid_capstyle="round")
    ax_grad.annotate(
        "ssm_k: the two levels take the SAME\nfirst step, seed by seed (3 decimals)",
        xy=(2.5, 5.0), ha="center", va="center", fontsize=7.5, color=INK_MUTED,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
              "edgecolor": INK_FAINT, "linewidth": 0.7},
    )

    ax_grad.set_yscale("log")
    ax_grad.set_xticks(positions)
    ax_grad.set_xticklabels(labels, fontsize=8)
    ax_grad.set_xlim(-0.6, len(positions) - 0.4)
    ax_grad.set_ylabel(r"$\|\nabla\|$ at the first optimizer step")
    ax_grad.set_title("First-step gradient norm, per seed", loc="left")
    ax_grad.set_ylim(1.4, 45)

    handles = [plt.Line2D([], [], color=LEVEL_COLOR[lv], linewidth=1.6, label=LEVEL_LABEL[lv])
               for lv in ("10.0", "0.2")]
    handles += [
        plt.Line2D([], [], color=INK_MUTED, linewidth=1.1, linestyle="-", label="ssm_k (structure on)"),
        plt.Line2D([], [], color=INK_MUTED, linewidth=1.1, linestyle="--", label="min_ssm (structure off)"),
        plt.Line2D([], [], color=INK_MUTED, marker="x", linestyle="none", label="divergence (last logged update)"),
    ]
    ax_loss.legend(handles=handles, fontsize=7.5, loc="upper right", framealpha=0.95)

    fig.text(
        0.0, -0.09,
        "PROTOCOL 23.6 / 24. Root: results_ssm_delta/, 20 cells, CoupledOscillators (c=0.10), 8000 updates, "
        "lr=3e-4 and grad_clip=100 unchanged from 16.2/17.1.  CONFOUNDS: delta_max is the only factor that "
        "moves; the two families differ in structure (off/on), so the family contrast is NOT controlled here "
        "and no comparison between the dashed and solid curves is claimed.  These roots have NO gate (23.4): "
        "no verdict is issued.  Left panel: running mean over 50 updates on the runs that completed, raw trace faint underneath; the diverged runs are drawn raw (28-56 updates long).  NOT PLOTTED, and deliberately: "
        "18.0's 2.4e18 first-step norm and its mlp reference of 28.2 were measured on results_probe_d/, on "
        "DampedPendulum -- 23.0 requires this root to be self-sufficient, so they are cited here and not drawn.  Scope (25.6): measured UNDER THE DECLARED REGIME (h_fr = 25); the sign of the structure effect is unchanged at h_fr = 5 (25.8).",
        fontsize=6.6, color=INK_MUTED, va="top", ha="left", wrap=True,
    )

    out = Path(out)
    save_fig(fig, out)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    make_figE(args.root, args.out)


if __name__ == "__main__":
    main()
