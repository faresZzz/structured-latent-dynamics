#!/usr/bin/env bash
# Campagnes A et B, debloquees apres la decision du 2026-08-11 sur `on-stable`
# (PROTOCOL.md 23.1(bis)) : les trois niveaux de structure partent.
#
#   nohup bash tools/run_campaigns_ab.sh > logs/run_campaigns_ab.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs .runstate
CFG=src/sld/experiments/configs

# `sld-gate` sort non-zero des qu'une cellule est "missing" -- et une cellule
# DIVERGEE est structurellement missing pour lui: elle n'a pas de
# run_complete.json, donc rien a scorer. C'est le comportement correct (on ne
# fabrique pas un verdict pour un run qui n'a pas fini), mais ce n'est pas une
# panne. Le controle honnete est la COUVERTURE du scoring: autant de gate.json
# que de run_complete.json.
score() {
  local name="$1" cfg="$2" root="$3"
  if [ -f ".runstate/$name" ]; then echo "== [$(date +%H:%M:%S)] $name : deja fait, saute"; return 0; fi
  echo "== [$(date +%H:%M:%S)] $name : demarre"
  local log="logs/${name}-$(date +%Y%m%d-%H%M%S).log"
  ./.venv/bin/sld-gate --config "$cfg" > "$log" 2>&1 || true
  local complete scored
  complete=$(find "$root" -name 'run_complete.json' 2>/dev/null | wc -l | tr -d ' ')
  scored=$(find "$root" -name 'gate.json' 2>/dev/null | wc -l | tr -d ' ')
  if [ "$scored" -ge "$complete" ] && [ "$complete" -gt 0 ]; then
    touch ".runstate/$name"
    echo "== [$(date +%H:%M:%S)] $name : OK -- $scored/$complete runs complets scores  ($log)"
  else
    echo "== [$(date +%H:%M:%S)] $name : ECHEC -- $scored scores pour $complete complets. Voir $log"
    tail -20 "$log"; exit 1
  fi
}

step() {
  local name="$1"; shift
  if [ -f ".runstate/$name" ]; then echo "== [$(date +%H:%M:%S)] $name : deja fait, saute"; return 0; fi
  echo "== [$(date +%H:%M:%S)] $name : demarre"
  local log="logs/${name}-$(date +%Y%m%d-%H%M%S).log"
  if "$@" > "$log" 2>&1; then
    touch ".runstate/$name"; echo "== [$(date +%H:%M:%S)] $name : OK  ($log)"
  else
    echo "== [$(date +%H:%M:%S)] $name : ECHEC. Voir $log"; tail -30 "$log"; exit 1
  fi
}

# Campagne C a fini ses 20 cellules; son scoring est independant de A/B.
score score-C "$CFG/ssm_delta.yaml" results_ssm_delta

echo "############ etape 3 : campagne A -- taille d'echantillon, 45 runs"
bash tools/run_sharded.sh campagne-A "$CFG/osc_sample_size.yaml" 5 45 results_osc_n
score score-A "$CFG/osc_sample_size.yaml" results_osc_n

echo "############ etape 4 : campagne B -- mauvaise specification, 60 runs"
bash tools/run_sharded.sh campagne-B "$CFG/osc_misspecification.yaml" 5 60 results_osc_delta
score score-B "$CFG/osc_misspecification.yaml" results_osc_delta

echo "############ TERMINE a $(date). Racines produites :"
for r in results_sweep_osc results_ssm_delta results_osc_n results_osc_delta; do
  n=$(find "$r" -name 'run_complete.json' 2>/dev/null | wc -l | tr -d ' ')
  f=$(find "$r" -name 'failure.json' 2>/dev/null | wc -l | tr -d ' ')
  echo "   $r : $n complets, $f divergences"
done
