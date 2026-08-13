#!/usr/bin/env bash
# Le sous-ensemble de `run_all.sh` qui ne depend PAS du bras `on-stable`.
#
# Le bras `on-stable` est en attente d'une decision de l'auteur
# (_private/AGENT_JOURNAL.md, entree du 2026-08-11 18h10 : le rayon n'excede
# jamais 1, mais sigmoid(rho) sature a exactement 1.0 en float32 des
# rho > 17.33, donc l'assertion stricte `< 1` du test ne peut pas etre
# honoree). Les campagnes A et B contiennent 15 et 20 cellules de ce niveau et
# sont TENUES. Ce qui suit n'en contient aucune, ou n'en depend pas.
#
#   nohup bash tools/run_authorized.sh > logs/run_authorized.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs .runstate
CFG=src/sld/experiments/configs

step() {
  local name="$1"; shift
  if [ -f ".runstate/$name" ]; then echo "== [$(date +%H:%M:%S)] $name : deja fait, saute"; return 0; fi
  echo "== [$(date +%H:%M:%S)] $name : demarre"
  local log="logs/${name}-$(date +%Y%m%d-%H%M%S).log"
  if "$@" > "$log" 2>&1; then
    touch ".runstate/$name"; echo "== [$(date +%H:%M:%S)] $name : OK  ($log)"
  else
    echo "== [$(date +%H:%M:%S)] $name : ECHEC -- chaine arretee. Voir $log"; tail -30 "$log"; exit 1
  fi
}

echo "############ etape 1 : sweep de taux (section 23.3), 9 runs"
bash tools/run_sharded.sh sweep "$CFG/sweep_osc_lr.yaml" 5

# La quantite que 23.3 minimise -- e_25 sur le split VAL -- n'est ecrite par
# aucun run d'entrainement : c'est un rollout libre. `--split val` la produit
# dans gate_val.json, jamais dans gate.json (15.5 : la gate score sur test,
# toute SELECTION lit val).
step sweep-score-val  uv run sld-gate --config "$CFG/sweep_osc_lr.yaml" --split val
step sweep-argmin     uv run python -m sld.viz.read_sweep results_sweep_osc

# `select_rates.py` ecrit les taux dans les configs A et B : TENU jusqu'a la
# decision sur `on-stable`, pour que la selection soit appliquee une seule
# fois, sur la liste de bras definitive (23.3 : une selection, pas de seconde
# passe).

echo "############ etape 2 : campagne C -- resurrection ssm (section 23.6), 20 runs"
# Ses quatre bras sont `off` et `on-free` uniquement (radius: free), et ses
# taux sont ceux de 16.2, inchanges : aucune dependance ni au sweep ni au bras
# bloque.
bash tools/run_sharded.sh campagne-C "$CFG/ssm_delta.yaml" 5
step campagne-C-score  uv run sld-gate --config "$CFG/ssm_delta.yaml"

echo "############ TERMINE a $(date). Racines produites :"
for r in results_sweep_osc results_ssm_delta; do
  n=$(find "$r" -name 'run_complete.json' 2>/dev/null | wc -l | tr -d ' ')
  echo "   $r : $n runs complets"
done
echo "TENU en attente de decision : campagne A (45), campagne B (60), select_rates."
