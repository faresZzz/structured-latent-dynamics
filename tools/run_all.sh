#!/usr/bin/env bash
# Le cycle complet, sans surveillance. UNE commande, cinq etapes, un journal.
#
#   nohup bash tools/run_all.sh > logs/run_all.log 2>&1 &
#   tail -f logs/run_all.log
#
# Chaque etape ecrit un marqueur sous .runstate/ des qu'elle reussit, donc un
# relancement reprend ou il s'est arrete au lieu de tout refaire. Une etape qui
# echoue arrete la chaine: aucune campagne ne part sur un etat non verifie.
#
# Ce script ne DECIDE rien. Les seuils, les criteres et les regles de lecture
# sont dans PROTOCOL.md sections 23.3 et 23.5; ici il n'y a que l'ordre.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs .runstate

step() {                      # step <nom> <commande...>
  local name="$1"; shift
  if [ -f ".runstate/$name" ]; then
    echo "== [$(date +%H:%M:%S)] $name : deja fait, saute"
    return 0
  fi
  echo "== [$(date +%H:%M:%S)] $name : demarre"
  local log="logs/${name}-$(date +%Y%m%d-%H%M%S).log"
  if "$@" > "$log" 2>&1; then
    touch ".runstate/$name"
    echo "== [$(date +%H:%M:%S)] $name : OK  ($log)"
  else
    echo "== [$(date +%H:%M:%S)] $name : ECHEC -- chaine arretee. Voir $log"
    tail -30 "$log"
    exit 1
  fi
}

CFG=src/sld/experiments/configs

echo "############ etape 0 : environnement, tests, figures deja livrees"
step env-sync      uv sync
step env-install   uv pip install -e ".[dev]"
step tests         uv run pytest -q
step figures-ab    make figA figB
step gate          python3 ../_publication/validate_publication.py .

echo "############ etape 1 : sweep de taux (section 23.3), 9 runs"
step sweep         uv run sld-factorial --config "$CFG/sweep_osc_lr.yaml"
# Le critere de 23.3 -- e_25 sur le split VAL -- n'est ecrit par aucun run
# d'entrainement: c'est un rollout libre, pas une colonne de metrics.csv.
# `--split val` le produit dans gate_val.json (16.4(bis)).
step sweep-score   uv run sld-gate --config "$CFG/sweep_osc_lr.yaml" --split val
step select-rates  uv run python tools/select_rates.py results_sweep_osc

echo "############ etape 2 : campagne C -- resurrection ssm (section 23.6), 20 runs"
step campagne-C    uv run sld-factorial --config "$CFG/ssm_delta.yaml"
# scoring sur TEST (15.5): ecrit m_low/m_high/e_h dans gate.json, la
# source de toutes les figures. Sans lui la racine n'a que metrics.csv.
step score-C       uv run sld-gate --config "$CFG/ssm_delta.yaml"

echo "############ etape 3 : campagne A -- taille d'echantillon, 45 runs"
step campagne-A    uv run sld-factorial --config "$CFG/osc_sample_size.yaml"
# scoring sur TEST (15.5): ecrit m_low/m_high/e_h dans gate.json, la
# source de toutes les figures. Sans lui la racine n'a que metrics.csv.
step score-A       uv run sld-gate --config "$CFG/osc_sample_size.yaml"

echo "############ etape 4 : campagne B -- mauvaise specification, 60 runs"
step campagne-B    uv run sld-factorial --config "$CFG/osc_misspecification.yaml"
# scoring sur TEST (15.5): ecrit m_low/m_high/e_h dans gate.json, la
# source de toutes les figures. Sans lui la racine n'a que metrics.csv.
step score-B       uv run sld-gate --config "$CFG/osc_misspecification.yaml"

echo "############ TERMINE a $(date). Racines produites :"
for r in results_sweep_osc results_ssm_delta results_osc_n results_osc_delta; do
  n=$(find "$r" -name 'run_complete.json' 2>/dev/null | wc -l | tr -d ' ')
  echo "   $r : $n runs complets"
done
