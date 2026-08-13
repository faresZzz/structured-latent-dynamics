#!/usr/bin/env bash
# Un pas de la chaine, execute en 5 shards mono-thread.
#
#   bash tools/run_sharded.sh <nom-marqueur> <config.yaml> [K]
#
# Pourquoi ce script existe. `tools/run_all.sh` lance `sld-factorial --config`
# sans `--shard`, donc en UN processus : c'est la colonne "mono-thread" du
# recapitulatif du RUNBOOK (112 h au total), pas la colonne "sur 5 shards"
# (23 h) que le meme tableau annonce. Le decoupage est celui que PROTOCOL 14.7
# declare et que l'aide de `--shard` prescrit mot pour mot ("set
# OMP_NUM_THREADS per shard process") : l'expansion et le preflight de collision
# restent GLOBAUX dans chaque processus, seule la boucle d'evaluation se
# retrecit, donc la disjonction des shards est garantie par construction.
#
# Le shard 1 part seul et prend de l'avance : c'est lui qui materialise la
# representation gelee du stratum si elle est absente. Les autres la trouvent
# ecrite. Ceinture et bretelles -- `_resolve_stratum_representations` se declare
# deja sur pour des shards concurrents.
#
# Trois durcissements, apres que la campagne A a ete tuee a 20h32 apres 5 runs
# (aucun processus survivant, les 5 logs a 0 octet -- donc SIGKILL avec la
# sortie encore en tampon, jamais une sortie propre ; aucun jetsam au journal
# systeme, cause non etablie) :
#
#   1. `caffeinate -i` : empeche la mise en veille par inactivite pendant le
#      run. C'est un portable ; une campagne de 4 h sur un M2 Air qui s'endort
#      est le suspect le plus banal, et le moins cher a exclure. (`setsid`
#      n'existe pas sur macOS -- essaye, et le lancement echoue net.)
#   2. `PYTHONUNBUFFERED=1` : la prochaine mort laisse des logs. Un log a 0
#      octet apres 5 runs reussis ne dit rien, et c'est ce qui a coute le
#      diagnostic ici.
#   3. `.venv/bin/sld-factorial` au lieu de `uv run` : `uv run` resynchronise
#      l'environnement a chaque invocation et peut reinstaller le paquet
#      editable sous les pieds d'un run en cours. Appeler le binaire du venv
#      directement retire cette interaction, qui reste le suspect le plus
#      plausible -- une suite `uv run pytest` tournait au moment de la mort.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="$1"; CFG="$2"; K="${3:-5}"; EXPECT="${4:-0}"; ROOT="${5:-}"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p logs .runstate

if [ -f ".runstate/$NAME" ]; then
  echo "== $NAME : deja fait, saute"
  exit 0
fi

# GARDE-FOU (2026-08-11). Une generation de shards deja en vol sur cette config
# doit interdire la suivante : deux processus qui entrainent la MEME cellule
# dans le MEME run_dir se marchent dessus, et le symptome observe est un
# metrics.csv rempli d'octets NUL que la reprise ne sait pas lire.
#
# C'est arrive : `ps ... | grep sld-factorial` a rendu VIDE alors que dix
# processus tournaient (`pgrep -fl` les voyait tous), j'ai conclu que la
# campagne etait morte, et j'ai relance -- deux fois. Trois generations en
# parallele sur results_osc_n. `pgrep -f` est la commande qui dit la verite
# ici ; `ps | grep` ne la disait pas.
if pgrep -f "sld-factorial --config $CFG" > /dev/null 2>&1; then
  echo "== $NAME : REFUS -- des shards tournent DEJA sur $CFG :"
  pgrep -fl "sld-factorial --config $CFG" | sed 's/^/     /'
  echo "   Tuer la generation en cours avant de relancer, ou attendre qu'elle finisse."
  exit 1
fi

echo "== [$(date +%H:%M:%S)] $NAME : $K shards mono-thread sur $CFG"
pids=()
for i in $(seq 1 "$K"); do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    caffeinate -i ./.venv/bin/sld-factorial --config "$CFG" --shard "$i/$K" \
    > "logs/${NAME}-shard${i}-${STAMP}.log" 2>&1 &
  pids+=("$!")
  # le premier prend de l'avance pour ecrire la representation
  [ "$i" = "1" ] && sleep 90
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "== [$(date +%H:%M:%S)] $NAME shard $((i+1))/$K : OK"
  else
    echo "== [$(date +%H:%M:%S)] $NAME shard $((i+1))/$K : ECHEC"
    tail -20 "logs/${NAME}-shard$((i+1))-${STAMP}.log"
    fail=1
  fi
done

# Un shard sort non-zero des qu'UNE de ses cellules echoue -- et une divergence
# EST une cellule echouee pour sld-factorial. Or PROTOCOL 15.3 et 23.6 disent
# qu'une divergence est un RESULTAT publiable, pas une panne de chaine. Le
# discriminant honnete n'est donc pas le code de sortie mais la COUVERTURE :
# chaque cellule attendue doit porter soit run_complete.json soit failure.json.
# Si la couverture est complete, la campagne a mesure ce qu'elle devait mesurer.
if [ "$fail" != "0" ] && [ "$EXPECT" != "0" ] && [ -n "$ROOT" ]; then
  done_n=$(find "$ROOT" -name 'run_complete.json' 2>/dev/null | wc -l | tr -d ' ')
  div_n=$(find "$ROOT" -name 'failure.json' 2>/dev/null | wc -l | tr -d ' ')
  total=$((done_n + div_n))
  echo "== $NAME : couverture $total/$EXPECT ($done_n complets, $div_n divergences)"
  if [ "$total" -ge "$EXPECT" ]; then
    echo "== $NAME : couverture COMPLETE -- les divergences sont le resultat (15.3), pas une panne"
    fail=0
  fi
fi

[ "$fail" = "0" ] || { echo "== $NAME : ECHEC -- chaine arretee"; exit 1; }
touch ".runstate/$NAME"
echo "== [$(date +%H:%M:%S)] $NAME : OK (logs/${NAME}-shard*-${STAMP}.log)"
