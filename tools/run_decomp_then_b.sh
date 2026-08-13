#!/usr/bin/env bash
# PROTOCOL 25.3 puis 25.5, dans cet ordre et sans chevauchement.
#   nohup caffeinate -i bash tools/run_decomp_then_b.sh > logs/run_decomp_then_b.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs .runstate
CFG=src/sld/experiments/configs

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
    touch ".runstate/$name"; echo "== [$(date +%H:%M:%S)] $name : OK -- $scored/$complete scores  ($log)"
  else
    echo "== [$(date +%H:%M:%S)] $name : ECHEC -- $scored pour $complete complets"; tail -20 "$log"; exit 1
  fi
}

echo "############ 25.3 : decomposition, 30 runs, h_fr=25"
bash tools/run_sharded.sh decomp "$CFG/osc_decomp.yaml" 5 30 results_osc_decomp
score score-decomp "$CFG/osc_decomp.yaml" results_osc_decomp

echo "############ 25.5 : la campagne B reprend (consequence pre-enregistree)"
bash tools/run_sharded.sh campagne-B "$CFG/osc_misspecification.yaml" 5 60 results_osc_delta
score score-B "$CFG/osc_misspecification.yaml" results_osc_delta

echo "############ TERMINE a $(date)"
for r in results_osc_decomp results_osc_delta; do
  echo "   $r : $(find "$r" -name 'run_complete.json' 2>/dev/null | wc -l | tr -d ' ') complets, $(find "$r" -name 'failure.json' 2>/dev/null | wc -l | tr -d ' ') divergences"
done
