"""Fig. 1 -- free-running rollout error vs. horizon, PROTOCOL.md section 27.4.

``make_fig1`` was retired (this module's earlier docstring, kept below in
full) because it decoded the rollout prediction into true-state coordinates
via a linear readout fit by OLS INDEPENDENTLY PER RUN -- a free parameter per
arm, not an at-arms-length cross-arm comparison. It is reactivated here on
section 27.1's grid: ``representation.kind = identity`` means the SAME frozen
readout is shared and identical for every one of the 30 cells
(``e_recon = 2.60e-08``), so no per-run OLS fit exists to distort the
comparison -- the retirement's stated condition ("the replacement is a
shared, frozen encoder/readout") is met, and this is the first grid where it
is. ``e_h`` is read straight off ``gate.json`` (``sld.viz.results``, which is
this module's own numeric lock reused, see ``_aggregate_curves`` below), one
endpoint, one set of bytes (section 27.3).

Retirement docstring, unchanged, for the historical record of why the OLD
method was wrong (NOT what makes the NEW one right -- the new one is right
because of the shared frozen readout above, not because the old defect is
merely un-named):

    This figure used to decode the rollout prediction into true-state
    coordinates via a linear readout ``W`` fit by OLS INDEPENDENTLY PER RUN
    (val split, encoder embedding -> dataset ``z_true``). That per-run fit is
    a free parameter PER ARM: a bad encoder can be rescued by a
    better-fitting readout, so the resulting cross-arm comparison is not at
    arms' length -- a defect of comparison METHOD, not a numerical bug (it is
    not what B3's F1 gauge-invariance fix addressed; that fix concerned how a
    single run's OWN readout is measured, not whether different arms may use
    different readouts at all).

``_aggregate_curves`` (seed-mean +/-1 SE) and ``_expected_grid``
(pre-registered-seed completeness, D1-pendulum-era helper, unrelated to
section 27's grid and left as-is for its own tests) are kept unchanged;
``_aggregate_curves`` is reused verbatim by ``sld.viz.results`` (imported
there, not duplicated -- see that module's own docstring).

Torch-free (REGLES section 2): only ``numpy``, ``yaml``, ``matplotlib`` and
stdlib. ``results`` is imported lazily inside ``make_fig1``, not at module
scope, because ``sld.viz.results`` itself imports THIS module's
``_aggregate_curves`` at its own module scope -- a top-level import here
would be circular.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from sld.viz.style import (
    CONSTRAINT_LINESTYLES,
    FAMILY_COLORS,
    INK_MUTED,
    apply_style,
    save_fig,
    sigma_train_caption_fragment,
    write_sidecar_csv,
)

# pre-registered 5-seed law (PROTOCOL.md section 3; pendulum_core.yaml's
# `seeds: [0, 1, 2, 3, 4]`) -- the reference set a "complete" cell must
# cover in strict/final-grid mode. Not exposed as a knob (ponytail): if the
# protocol's seed count ever changes, this constant changes with it.
_REQUIRED_SEEDS: frozenset[int] = frozenset(range(5))


def _aggregate_curves(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and +/-1 SE over the seed axis (``curves``, ``(n_seeds, H)``).

    Kept standalone (module docstring: not currently called by
    ``make_fig1``, which is retired pending phase D) because phase D's
    frozen-readout figure will still need this identical seed-aggregation
    arithmetic; the numeric-lock test (M6/m12) verifies this function
    directly rather than through a render. ``se`` is
    ``std(ddof=1) / sqrt(n)``; ``0`` when ``n == 1`` (ddof=1 needs
    ``n >= 2``) -- the honest "no spread observed" value, never a silent
    NaN. Mean, never the median (REGLES section 4: "+/-1 SE" bands -- SE
    is defined around the mean, not the median).
    """
    n = curves.shape[0]
    mean_curve = curves.mean(axis=0)
    se = curves.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean_curve)
    return mean_curve, se


def _expected_grid(exp_yaml: str) -> tuple[list[tuple[str, str, str]], frozenset[int]]:
    """The FULL ``(regime, family, constraint)`` grid + seeds ``exp_yaml`` promises.

    Derived, not hardcoded (M4 + m7, round 3): regimes come from
    ``exp_yaml``'s ``datasets`` block; cells and seeds come from the yaml's
    OWN ``arms``/``seeds`` declarations (the factorial schema) when
    present -- so ``smoke.yaml``'s declared 2 arms x 1 seed are what strict
    mode requires there, not an invented 8 cells x 5 seeds. Fallback when
    the yaml declares no ``arms`` (viz-test fixtures): the style modules'
    ``FAMILY_COLORS`` x ``CONSTRAINT_LINESTYLES`` with the pre-registered
    seeds 0-4 -- declared, not silent. For ``pendulum_core.yaml`` (which
    declares all 8 arms and seeds 0-4) both paths agree: 16 cells, 5 seeds.

    Kept standalone (module docstring): not currently called by
    ``make_fig1``, which is retired pending phase D; exercised directly by
    its own test.
    """
    with open(exp_yaml) as f:
        exp_config = yaml.safe_load(f)
    regimes = sorted(exp_config.get("datasets", {}))
    arms = exp_config.get("arms")
    if arms:
        cells = [
            (regime, arm["family"], arm["constraint"])
            for regime in regimes
            for arm in arms
        ]
        seeds = frozenset(int(s) for s in exp_config.get("seeds", sorted(_REQUIRED_SEEDS)))
    else:
        cells = [
            (regime, family, constraint)
            for regime in regimes
            for family in sorted(FAMILY_COLORS)
            for constraint in sorted(CONSTRAINT_LINESTYLES)
        ]
        seeds = _REQUIRED_SEEDS
    return cells, seeds


# Section 27.1's six cells, in the order (backbone, then off before on) the
# section itself lists the design in: mlp/mlp_k, transformer/transformer_k,
# min_ssm/ssm_k. Not derived from results.py (forbidden to import at module
# scope, see module docstring) -- this is section 27.1's own declared order,
# restated here as a plain tuple.
_FAMILY_ORDER: tuple[str, ...] = ("mlp", "mlp_k", "transformer", "transformer_k", "min_ssm", "ssm_k")
_PAIR_BY_FAMILY: dict[str, tuple[str, str]] = {
    "mlp": ("mlp", "off"), "mlp_k": ("mlp", "on"),
    "transformer": ("transformer", "off"), "transformer_k": ("transformer", "on"),
    "min_ssm": ("ssm", "off"), "ssm_k": ("ssm", "on"),
}

_CSV_FIELDS = ["family", "backbone", "structure", "eligible", "step", "mean_e_h", "se_e_h", "seed", "status"]


def _sidecar_rows(agg: dict) -> list[dict]:
    rows: list[dict] = []
    for family in _FAMILY_ORDER:
        backbone, structure = _PAIR_BY_FAMILY[family]
        cell = agg[family]
        base = {"family": family, "backbone": backbone, "structure": structure, "eligible": cell["eligible"]}
        if cell["eligible"]:
            for step, (m, se) in enumerate(zip(cell["mean_e_h"], cell["se_e_h"], strict=True)):
                rows.append({**base, "step": step, "mean_e_h": m, "se_e_h": se, "seed": "", "status": ""})
        else:
            for seed, status in sorted(cell["statuses"].items()):
                rows.append({**base, "step": "", "mean_e_h": "", "se_e_h": "", "seed": seed, "status": status})
    return rows


def make_fig1(
    results_root: str = "results_primary_factorial",
    exp_name: str = "primary_factorial",
    out_path: str = "assets/fig1_rollout.png",
    sidecar_path: str | None = None,
    state_scale_path: str | None = None,
) -> dict:
    """PROTOCOL.md section 27.4, Fig. 1: ``e_h`` in true-state coordinates,
    mean +/-1 SE over five seeds, six cells (3 backbones x 2 structures). A
    non-eligible cell (section 27.3: fewer than 5/5 seeds terminal and
    ``'ok'``) is never drawn as a curve -- named in the annotation with its
    five per-seed statuses instead.

    Returns a dict with the plotted data (``agg``, section 27's own
    per-cell aggregate), the image path, and the sidecar CSV path + hash --
    section 27.4's "each figure writes a sidecar CSV and a hash" and "also
    returns the data it plotted".
    """
    from sld.viz import results as results_mod

    rows = results_mod.build_primary_results_table(results_root, exp_name)
    if all(r["status"] == "missing" for r in rows):
        raise FileNotFoundError(
            f"fig1: no gate.json under {Path(results_root) / exp_name} -- nothing to plot yet"
        )
    agg = results_mod.aggregate_rollout_curves(rows)

    apply_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    non_eligible: list[tuple[str, dict]] = []
    horizon = 0
    for family in _FAMILY_ORDER:
        backbone, structure = _PAIR_BY_FAMILY[family]
        cell = agg[family]
        if not cell["eligible"]:
            non_eligible.append((family, cell["statuses"]))
            continue
        mean = np.asarray(cell["mean_e_h"])
        se = np.asarray(cell["se_e_h"])
        horizon = max(horizon, len(mean))
        x = np.arange(1, len(mean) + 1)
        ax.plot(
            x, mean, color=FAMILY_COLORS[family], linestyle=CONSTRAINT_LINESTYLES[structure],
            marker="o" if structure == "on" else "^", markersize=3.5, linewidth=1.6, label=family, zorder=4,
        )
        ax.fill_between(x, mean - se, mean + se, color=FAMILY_COLORS[family], alpha=0.15, zorder=2)

    ax.set_xlabel("horizon step $h$")
    ax.set_ylabel(r"$e_h$   (normalized state error, true-state coordinates)")
    ax.set_title("Free rollout error vs. horizon -- 3 backbones x 2 structures", loc="left")
    if ax.lines:
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.95, ncol=2)

    if non_eligible:
        # Readability fix (coordinator review): the old per-cell status box
        # was drawn INSIDE the axes and covered the low end of both curves
        # plus clipped the y-label. The full per-seed detail already lives
        # in the caption's own "Non-eligible:" clause below -- this stays a
        # single compact line, INSIDE the axes but in the one corner no
        # curve reaches: both eligible curves RISE monotonically from ~0 at
        # h=1 to their max at h=25 (top-right), so the empty corner is
        # bottom-right, not top-right or either left corner (legend sits
        # top-left).
        ax.annotate(
            f"{len(non_eligible)} cellules non eligibles -- voir legende",
            xy=(0.985, 0.03), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=7.0, color=INK_MUTED, style="italic", zorder=6,
        )

    sigma_clause = sigma_train_caption_fragment(
        state_scale_path, rows[0]["stratum_system"], rows[0]["stratum_regime"]
    )
    caption = (
        "PROTOCOL 27.1/27.4/27.7. System: CoupledOscillators. Regime: OURegime(theta_ou=1.0). "
        f"H={horizon or 25}. {sigma_clause} (sld.audit.scan_state_scale, identical across all six cells "
        "since one stratum, one dataset build (27.1)). tau=0.5 (VPT's threshold, not this figure's statistic -- "
        "27.7 makes e_h_gate = e_h at H=25 primary, read off these same curves). Censoring convention: none "
        "on e_h itself (censoring is VPT's, see Fig. 2); a non-eligible CELL (fewer than 5/5 seeds terminal "
        "and 'ok', 27.3) is never averaged and never drawn as a curve, named below instead. n=5 seeds per "
        "eligible cell." + (
            "  Non-eligible: " + "; ".join(f"{f} ({', '.join(f'{s}:{st}' for s, st in sorted(sts.items()))})"
                                            for f, sts in non_eligible)
            if non_eligible else "  Non-eligible: none."
        )
    )
    fig.text(0.0, -0.10, caption, fontsize=6.6, color=INK_MUTED, va="top", wrap=True)

    out_path = Path(out_path)
    save_fig(fig, out_path)
    plt.close(fig)

    sidecar_path = Path(sidecar_path) if sidecar_path else out_path.with_suffix(".csv")
    digest = write_sidecar_csv(_sidecar_rows(agg), _CSV_FIELDS, sidecar_path)

    print(f"wrote {out_path}")
    return {
        "image_path": out_path,
        "sidecar_csv": sidecar_path,
        "sidecar_hash": digest,
        "data": agg,
        "caption": caption,
        "non_eligible": dict(non_eligible),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig. 1 -- PROTOCOL.md section 27.4")
    ap.add_argument("--results-root", default="results_primary_factorial")
    ap.add_argument("--exp-name", default="primary_factorial")
    ap.add_argument("--out", default="assets/fig1_rollout.png")
    ap.add_argument("--state-scale", default=None, help="path to a state_scale/1 JSON (sld.audit.scan_state_scale)")
    args = ap.parse_args()
    try:
        make_fig1(
            results_root=args.results_root, exp_name=args.exp_name, out_path=args.out,
            state_scale_path=args.state_scale,
        )
    except FileNotFoundError as exc:
        print(f"fig1: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
