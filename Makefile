.PHONY: setup test smoke grid-core figA figB figC figD figE figF figG fig1 fig2 fig3 figH figures reproduce-figures clean-nb

CONFIGS := src/sld/experiments/configs
# Declared configs are written configs. `oscillators`, `lorenz`, `cartpole`,
# `encoder_axis` and `koopman_ablations` were declared here and never written,
# so `grid-core` could not run. Lorenz in particular is not wired: it is absent
# from experiments.factorial._SYSTEMS and is autonomous (action_dim = 0), which
# the action-conditioned stack does not carry. It is the declared NEXT system,
# not a silently missing one.
CORE_CONFIGS := pendulum_core_partial

# --- Bootstrap ------------------------------------------------------------

setup:
	uv sync
	uv pip install -e ".[dev]"
	uv run pre-commit install

test:
	uv run pytest -q

# --- Guard: fail with a clear message instead of a raw traceback when the ---
# --- module/config a target depends on has not been implemented yet.     ---
define require_module
	@uv run python -c "import importlib; importlib.import_module('sld.$(1)')" >/dev/null 2>&1 \
		|| (echo "sld.$(1) not implemented yet — nothing to run." && exit 1)
endef

# --- Experiments -----------------------------------------------------------

smoke:
	$(call require_module,experiments.factorial)
	@test -f $(CONFIGS)/smoke.yaml \
		|| (echo "$(CONFIGS)/smoke.yaml not written yet — nothing to smoke-test." && exit 1)
	uv run sld-factorial --config $(CONFIGS)/smoke.yaml

grid-core:
	$(call require_module,experiments.factorial)
	@for cfg in $(CORE_CONFIGS); do \
		if [ ! -f $(CONFIGS)/$$cfg.yaml ]; then \
			echo "$(CONFIGS)/$$cfg.yaml not written yet — grid-core is incomplete."; \
			exit 1; \
		fi; \
	done
	@for cfg in $(CORE_CONFIGS); do \
		uv run sld-factorial --config $(CONFIGS)/$$cfg.yaml || exit 1; \
	done

# --- Figures (each regenerates from results*/, never retrains -- REGLES 2) ---
#
# DECLARED FIGURES ARE DELIVERED FIGURES. The previous block declared fig1..fig5
# and a `reproduce-all` that depended on all five; none of the five modules
# existed, so every one of those targets failed on `require_module` and the
# publication gate could not be passed with a figure in the repo at all. That
# was the repository's real lock -- not the runs. The list below grows as
# figures ship, and never before.
#
# Each target reads verdict artifacts (gate.json / d3_gold.json) through
# sld.viz.ingest, which imports no torch: a figure module that could import a
# predictor could, one refactor later, retrain one.

FIGURES := figA figB figC figD figE figF figG

figA:                     ## D3 control: they learn, they do not identify
	uv run python -m sld.viz.figA_d3_exactness

figC:                     ## campagnes A: structure contre taille d'echantillon, dispersion explicite
	uv run python -m sld.viz.figC_sample_size --which C

figD:                     ## campagne B: absolus + baselines, puis le ratio
	uv run python -m sld.viz.figD_misspecification

figF:                     ## conditionnement de la jacobienne d'action a H=25
	uv run python -m sld.viz.figF_action_conditioning

figG:                     ## decomposition: retirer psi/pi degrade, et pourquoi
	uv run python -m sld.viz.figC_sample_size --which G

figE:                     ## campagne C: la divergence ssm appartient a delta_max, pas a la famille
	uv run python -m sld.viz.figE_ssm_delta

figB:                     ## D4 horizon-resolved: where the structured arms pay and where they cost
	uv run python -m sld.viz.figB_horizon

# fig1/fig2/fig3 -- PROTOCOL.md section 27.4, the primary factorial
# (results_primary_factorial/). The campaign is CLOSED (30/30 cells) and all
# three already ship in assets/ with CSV sidecars + .sha256. NOT added to
# FIGURES: regenerating in this same environment reproduces every sidecar
# CSV bit-for-bit (verified in a tmpdir against the committed .sha256) but
# NOT the PNG -- the tight-bbox canvas width itself differs run to run
# (e.g. fig1: 1194px vs 1211px at identical dpi, same matplotlib version),
# which is a text-layout nondeterminism, not a data difference. Folding
# them into FIGURES would make every `make figures` dirty the tree for no
# data reason. `make fig1` / `fig2` / `fig3` still work standalone.

fig1:                     ## e_h rollout curves, mean +-1 SE, 3 backbones x 2 structures
	uv run python -m sld.viz.fig1_rollout

fig2:                     ## VPT, five raw per-seed values per cell, censoring shown
	uv run python -m sld.viz.fig2_vpt

fig3:                     ## the three paired ON-OFF contrasts of e_h_gate
	uv run python -m sld.viz.fig3_factorial_effects

# figH -- PROTOCOL.md section 28, the basis-oracle root (results_basis/).
# Same reason fig1/fig2/fig3 stay out of FIGURES: results_basis/ is
# gitignored, so `make figures` on a fresh clone would fail on a root that
# was never meant to be checked in. `make figH` still works standalone once
# the root exists on disk.

figH:                     ## basis root: off / on-learned / on-oracle, P1-P3 paired contrasts
	uv run python -m sld.viz.figH_basis_oracle

figures: $(FIGURES)

# Regenerate every shipped figure from the artifacts already on disk. This
# target trains nothing, by construction.
reproduce-figures: figures

# --- Notebook hygiene (relies on nb-clean, already a dev dependency and   --
# --- pre-commit hook — no bespoke check script, cf. _publication's own    --
# --- "Notebooks cleaned" validator criterion) -----------------------------

clean-nb:
	uv run nb-clean clean --remove-empty-cells notebooks/*.ipynb
