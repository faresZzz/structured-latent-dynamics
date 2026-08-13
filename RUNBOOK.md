# RUNBOOK — ce que la machine doit exécuter

*Tout ce qui est ici demande **torch**, donc ta machine : ni le conteneur cloud ni la VM Linux
du pont ne l'exécutent. Le reste du cycle ne t'attend pas.*

Ordre strict. Chaque étape a une **vérification** à lire avant la suivante. Une étape qui échoue
n'est jamais relancée avec une autre valeur sans passer par un amendement.

---

## Étape 0 — vérifier l'environnement et le code neuf (5 min)

```bash
cd ~/Documents/Projects/structured-latent-dynamics
uv sync && uv pip install -e ".[dev]"

uv run pytest -q
make figA figB
python3 ~/Documents/Projects/_publication/validate_publication.py .
```

**Vérification.** `pytest` vert, dont `tests/test_viz_ingest.py` (réconciliation avec
`PROTOCOL §21.0` sur 8 cellules et `§20.0` sur D3) et `tests/test_structured_radius.py`
(rayon borné sans condition de régime, non-régression du mode `free`, test mutant).
`make figA figB` réécrit `assets/*.png`. Le validateur sort **0 FAIL**.

> **Si `test_structured_radius.py` échoue**, ne pas contourner : le bras `on-stable` a un critère
> de mort écrit — on tombe à deux niveaux de structure et les campagnes partent sans lui.

---

## Étape 1 — le sweep de taux, §23.3 (9 runs, ≈ 1 h)

```bash
uv run sld-factorial --config src/sld/experiments/configs/sweep_osc_lr.yaml
uv run python -m sld.viz.read_sweep results_sweep_osc
```

**Vérification.** 9/9 runs complets sous `results_sweep_osc/`. `read_sweep` imprime, par bras,
le taux qui minimise **son propre** critère sur le split **val** (jamais test), et refuse
bruyamment si la colonne déclarée n'existe pas plutôt que de deviner.

**Reporter les trois taux** dans `osc_sample_size.yaml` et `osc_misspecification.yaml`, à la
place des `lr: 0.001` marqués PLACEHOLDER. **Ne toucher à aucun autre champ.** Un bras dont
aucun taux ne s'entraîne est publié comme tel — aucun taux hors liste n'est essayé (§16.2(4)).

---

## Étape 2 — campagne C, la résurrection ssm (20 runs, ≈ 3 h)

Aucune dépendance au sweep : les taux de la famille `ssm` sont ceux de §16.2, inchangés.

```bash
uv run sld-factorial --config src/sld/experiments/configs/ssm_delta.yaml
```

**Vérification.** 20/20 sous `results_ssm_delta/`, ou divergences journalisées comme résultat
(§15.3). Les deux issues sont publiables et décrites en `PROTOCOL §23.6`.

---

## Étape 3 — campagne A, taille d'échantillon (45 runs, ≈ 8 h, une nuit)

**Ne pas lancer avant que l'étape 1 ait reporté les taux.**

```bash
uv run sld-factorial --config src/sld/experiments/configs/osc_sample_size.yaml
```

**Vérification.** 45/45 sous `results_osc_n/`. Si plus de 2 cellules sur 3 divergent à un niveau
de `n`, ce niveau est publié « non entraînable au régime déclaré » et retiré de la lecture —
**jamais réparé par un troisième sweep** (§23.5).

---

## Étape 4 — campagne B, mauvaise spécification (60 runs, ≈ 10 h, une nuit)

```bash
uv run sld-factorial --config src/sld/experiments/configs/osc_misspecification.yaml
```

**Vérification.** 60/60 sous `results_osc_delta/`. La racine est auto-suffisante : elle re-tourne
sa propre cellule de référence `c = 0.10`, donc rien n'est lu à travers deux racines (§14.6).

---

## Étape 5 — le re-run §21.3, quand tu veux le fermer

Déjà autorisé et **liant** (§21.3(5) : *one selection, one re-run, published whatever it
returns*). Indépendant des campagnes ci-dessus.

```bash
uv run sld-factorial --config src/sld/experiments/configs/lr_reselection_structured.yaml
```

---

## Récapitulatif du calcul — **mesuré, 2026-08-11** (l'estimation initiale est corrigée)

| étape | runs | mesuré, sur 5 shards | par run (médiane) |
|---|---|---|---|
| 1 — sweep (n=32, 4000 upd) | 9 | **19 min** | `mlp` 88 s · `mlp_k` 637 s |
| 2 — campagne C (n=256, 8000 upd) | 20 | **40 min** | `min_ssm` 397 s · `ssm_k` 1905 s |
| 3 — campagne A (n ∈ {8,32,256}, 8000 upd) | 45 | ≈ 3.8 h *(projeté)* | — |
| 4 — campagne B (n=32, 8000 upd) | 60 | ≈ 3.0 h *(projeté)* | — |
| **total** | **134** | **≈ 8 h** au lieu des 23 h annoncées | — |

**Ce qui était faux dans l'estimation initiale.** Elle disait : *« le budget d'updates étant
fixe, réduire `n_episodes` ne change quasiment pas le coût par run »*. La mesure le réfute,
sur la même famille et à budget d'updates normalisé :

| | `mlp_k`, n=256, 8000 upd (D3) | `mlp_k`, n=32, 4000 upd (ce sweep) |
|---|---|---|
| horloge | 4335 – 4427 s | 637 s (médiane de 6 runs) |
| par update | 0.548 s | **0.159 s** |

Soit **3.4× moins cher par update** pour 8× moins d'épisodes — sous-linéaire, mais très loin
de « quasiment pas ». Le coût par epoch n'est donc pas seulement celui du pas d'optimiseur :
une part croît avec la taille du split. Conséquence directe : la campagne B est **entièrement**
à n=32 et A n'a qu'un tiers de ses cellules à n=256, donc le total réel est très en dessous
des 23 h.

Les deux lignes mesurées le sont **sous 5 shards concurrents sur 8 cœurs**, donc avec
contention : le vrai écart est au moins celui-là, jamais moins. Les lignes A et B sont des
projections à partir des coûts par run mesurés, pas des mesures.

**Le sharding n'est pas optionnel pour tenir ces chiffres.** `tools/run_all.sh` lance
`sld-factorial` sans `--shard`, ce qui est la colonne mono-thread (≈ 112 h). La colonne
ci-dessus est celle de `tools/run_sharded.sh` : 5 processus `OMP_NUM_THREADS=1`, expansion et
preflight de collision globaux dans chacun, donc disjonction garantie par construction
(PROTOCOL §14.7).

En tâche de fond, avec journal :

```bash
mkdir -p logs
nohup uv run sld-factorial --config src/sld/experiments/configs/<cfg>.yaml \
      > logs/<cfg>-$(date +%Y%m%d-%H%M).log 2>&1 &
tail -f logs/<cfg>-*.log
```

---

## Ce qui n'attend pas la machine

Lecteurs d'artefacts, figures A et B, amendements 13 et 14, les quatre configs, les notebooks,
`docs/`, le README : faits ou en cours côté agent. Les figures C à F sont écrites contre les
schémas d'artefacts déjà connus et se rempliront dès que les racines existeront.
