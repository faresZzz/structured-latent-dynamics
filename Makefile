.PHONY: setup test smoke grid-core fig1 fig2 fig3 fig4 fig5 calibrate reproduce-all clean-nb

CONFIGS := src/sld/experiments/configs
CORE_CONFIGS := pendulum_core oscillators lorenz cartpole encoder_axis koopman_ablations

# --- Bootstrap ------------------------------------------------------------

setup:
	uv sync
	uv pip install -e ".[dev]"
	uv run pre-commit install

test:
	uv run pytest -q

# --- Guard: fail with a clear message instead of a raw traceback when the ---
# --- module/config a target depends on has not been implemented yet       ---
# --- (see RESUME_SESSION.md for the current wave).                       ---
define require_module
	@uv run python -c "import importlib; importlib.import_module('sld.$(1)')" >/dev/null 2>&1 \
		|| (echo "sld.$(1) not implemented yet — nothing to run. See RESUME_SESSION.md for the current wave." && exit 1)
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

calibrate:
	$(call require_module,experiments.factorial)
	@test -f $(CONFIGS)/calibration.yaml \
		|| (echo "$(CONFIGS)/calibration.yaml not written yet — nothing to calibrate." && exit 1)
	uv run sld-factorial --config $(CONFIGS)/calibration.yaml

# --- Figures (each regenerates from results/, never retrains — REGLES §2) --

fig1:
	$(call require_module,viz.fig1_rollout)
	uv run python -m sld.viz.fig1_rollout

fig2:
	$(call require_module,viz.fig2_spectrum)
	uv run python -m sld.viz.fig2_spectrum

fig3:
	$(call require_module,viz.fig3_panel)
	uv run python -m sld.viz.fig3_panel

fig4:
	$(call require_module,viz.fig4_bill)
	uv run python -m sld.viz.fig4_bill

fig5:
	$(call require_module,viz.fig5_instrument)
	uv run python -m sld.viz.fig5_instrument

reproduce-all: grid-core fig1 fig2 fig3 fig4 fig5 calibrate

# --- Notebook hygiene (relies on nb-clean, already a dev dependency and   --
# --- pre-commit hook — no bespoke check script, cf. _publication's own    --
# --- "Notebooks cleaned" validator criterion) -----------------------------

clean-nb:
	uv run nb-clean clean --remove-empty-cells notebooks/*.ipynb
