# Trace historique — campagne μ_J (jalon B0)

## Ce que ce paquet est, et n'est pas

Ce paquet préserve, **avant retrait du code de la campagne μ_J**, la trace intégrale des
runs, configs, métriques et verdicts de quatre arbres de résultats : la sélection de
famille (`pendulum_core_selection`), le grid `pendulum_core` pré-amendement, sa version
réparée (`results_amended/pendulum_core`), et le harnais de fumée (`smoke`).

Ce qu'il contient : **922 fichiers, ~4,8 Mo** — `config.yaml`, `metrics.csv`,
`audit.json`, `index.csv`, `selection_report.csv`/`.json`, `failure.json` — copiés
octet pour octet depuis `results/` et `results_amended/` (traces gitignorées, jamais
publiées telles quelles).

Ce qu'il ne contient **pas** : les **228 checkpoints `.pt` (~19 Mo)** ont été exclus
délibérément. **Les poids ne sont donc pas reproductibles depuis ce seul paquet** — on
peut relire ce qui a été mesuré et décidé, pas recharger un modèle entraîné. Pour cela il
faudrait les `results/`/`results_amended/` complets, non versionnés.

Chaque fichier copié porte dans `inventory.json` sa provenance (`source`), sa
destination dans le paquet, sa taille, son sha256, et un classement
`category`/`verdict` à deux champs. `manifest.sha256` permet de vérifier l'intégrité
de la copie à tout moment.

## Campagnes

### `mu-j-calibration` — `results/pendulum_core_selection/` → `traces/pendulum_core_selection/`

**578 fichiers.** 144 répertoires de run sur disque, sous 4 familles
(`mlp`, `koopman_ssm`, `min_ssm`, `transformer`). Verdict au niveau **campagne** :
`band-not-reached-3-of-4-families` — trois familles sur quatre n'ont pas atteint la
bande cible de calibration ; `mlp` est la famille retenue (`selected`). Le verdict
est appliqué mécaniquement à chaque fichier de l'arbre, y compris ceux des runs
`mlp`, car le classement porte sur la campagne entière, pas sur un sous-ensemble
mesuré séparément par ce paquet.

`selection_report.csv` porte **49 lignes de données pour 48 `run_dir` distincts** —
`1cc12ab2-cf06208a` y figure deux fois, une fois en phase `mu_j_calibration`, une
fois en phase `lr_sweep` (vérifié : `csv.DictReader` sur le fichier, comptage des
`run_dir` uniques). **96 des 144 runs présents sur disque ne sont cités par aucune
ligne du rapport** (144 répertoires − 48 cités = 96, vérifié par comparaison
ensembliste entre les noms de répertoires et la colonne `run_dir`). Le rapport de
sélection est donc partiel par rapport à ce qui a réellement tourné.

### `pre-amendment` — 40 runs de `results/pendulum_core/` + `index.csv` → `traces/pendulum_core/`

**159 fichiers.** `index.csv` liste **40 lignes de données, toutes `constraint=off`**
(vérifié par lecture CSV). Ce grid a tourné le 2026-07-31 à 17:14, **avant**
l'amendement qui autorise son budget de 200 updates (commité à 18:48 le même jour) —
violation d'ordre que `PROTOCOL.md` §11 refuse de couvrir rétroactivement. Verdict :
`superseded-by-amended` — remplacé par le grid réparé de `results_amended/`, rejoué
dans une racine de résultats séparée pour qu'aucun sentinel ne puisse reprendre les
artefacts pré-amendement.

### `protocol-violation` — 4 runs ON de `results/pendulum_core/` → `traces/pendulum_core/`

**16 fichiers**, répertoires `b0f18976-26608bd2`, `b0f18976-cf06208a`,
`e0f6b9fc-26608bd2`, `e0f6b9fc-cf06208a`. Voir section dédiée ci-dessous.

### `post-hoc-repair` — `results_amended/pendulum_core/` sauf `e8a316a0-26608bd2` → `traces/pendulum_core_amended/`

**157 fichiers.** Grid réparé, rejoué dans une racine séparée après l'amendement qui
autorise son budget. Verdict : `primary-off-grid` — mesure post-hoc valide en tant
que diagnostic, mais hors de la grille primaire pré-enregistrée (`PROTOCOL.md` §11.2 :
« an earlier draft called it a complete and valid measurement ; that was false »).
39 des 40 cellules ont terminé ; voir la campagne suivante pour la 40e.

### `divergence` — `results_amended/pendulum_core/e8a316a0-26608bd2/` → `traces/pendulum_core_amended/e8a316a0-26608bd2/`

**3 fichiers** (`config.yaml`, `failure.json`, `metrics.csv` — pas d'`audit.json`,
pas de checkpoint). Cellule `min_ssm`, seed 4, perte non-finie à l'epoch 46
(`failure.json` : `"reason": "non-finite loss"`, `"train_loss": "inf"`,
`"val_loss": "nan"`, `RuntimeError`). Verdict : `cell-level-only` — un échec sur 40
cellules ne soutient qu'un fait au niveau cellule, rien de plus (`PROTOCOL.md` §11.5).

### `smoke` — `results/smoke/` → `traces/smoke/`

**9 fichiers**, 2 runs de harnais. Verdict : `harness-check-not-evidence` — vérifie
que le pipeline tourne, ne constitue aucune preuve scientifique.

## ⚠️ Les quatre runs ON (`protocol-violation`)

Quatre répertoires de `results/pendulum_core/` — `b0f18976-26608bd2`,
`b0f18976-cf06208a`, `e0f6b9fc-26608bd2`, `e0f6b9fc-cf06208a` — correspondent à
`mlp` μ_J = 0,01 et `koopman_ssm` μ_J = 0,1, seed 0, dans les deux régimes.
D'après `PROTOCOL.md` lignes 1010-1024 :

> The four core ON runs already on disk (`mlp` μ_J = 0.01 and `koopman_ssm`
> μ_J = 0.1, seed 0, both regimes) → **protocol-violation traces, never core
> evidence**. Two independent reasons: they were trained against the **raw** penalty
> because the factorial runner never supplied the frozen gauge, and `koopman_ssm`'s
> μ_J = 0.1 was written into the manifest from a **terminal "band not reached"**
> verdict by a patcher that ignored the verdict. Both defects are fixed in the same
> commit as this amendment; the corrupted manifest value itself is cleared by hand,
> because the repaired patcher refuses to write but does not rewrite what it already
> wrote. The runs are kept and marked, not deleted.

Deux défauts indépendants et non corrélés :

1. **(a)** les runs ont été entraînés contre la pénalité **brute** (non normalisée),
   parce que le runner factoriel n'a jamais fourni la jauge gelée au moment de
   l'entraînement.
2. **(b)** la valeur μ_J = 0,1 retenue pour `koopman_ssm` a été écrite dans le
   manifeste à partir d'un verdict **terminal « band not reached »**, par un
   patcher qui ignorait ce verdict et l'a traitée comme une valeur calibrée valide.

Les deux défauts sont corrigés dans le code (même commit que l'amendement qui les
documente). **Les quatre runs sont conservés et marqués, pas supprimés** — c'est
tout l'objet de ce paquet.

**Fait mesuré** : ces quatre runs n'apparaissent dans **aucune ligne** de
`results/pendulum_core/index.csv` (l'index a 40 lignes de données, toutes
`constraint=off` — vérifié par lecture CSV, aucune des 4 UUID de run n'y figure).
Ils n'existent que comme répertoires sur disque, sans entrée d'index qui les
signale. C'est précisément pourquoi ils doivent être classés explicitement dans ce
paquet plutôt que retrouvés au hasard d'un futur grep.

## ⚠️ Limites de cette trace

Deux faits ont été annoncés par le superviseur comme contexte de cadrage et
**revérifiés indépendamment ici** ; les nombres ci-dessous sont ceux mesurés dans
cette exécution, pas une reprise du chiffre annoncé.

1. **Les 226 `audit.json` du paquet sont vides** — littéralement `{}`, 2 octets,
   sans exception. Répartition mesurée : **144** dans `traces/pendulum_core_selection/`,
   **43** dans `traces/pendulum_core/` (44 répertoires de run moins
   `e8a316a0-26608bd2`, qui n'a pas d'`audit.json`), **39** dans
   `traces/pendulum_core_amended/` (40 répertoires moins ce même run, absent pour
   la même raison). 144 + 43 + 39 = 226 — confirmé par lecture octet-à-octet de
   chaque fichier (`open(p,'rb').read() == b'{}'`, taille 2 octets, aucune
   exception trouvée). `traces/smoke/` porte 2 `audit.json` supplémentaires,
   également vides, non comptés dans les 226 car hors des trois campagnes μ_J
   proprement dites.
   **Le compte « 144 audits » (ou 226) est un compte de fichiers présents, pas un
   compte d'audits passés.** Aucun de ces fichiers ne contient de verdict — ils
   sont vides. Rien dans ce README ne doit être lu comme « ces runs ont passé un
   audit ».
2. **`selection_report.csv` porte 49 lignes de données pour 48 `run_dir`
   distincts** : `1cc12ab2-cf06208a` y figure deux fois (phases
   `mu_j_calibration` et `lr_sweep`). **96 des 144 runs présents sur disque dans
   `pendulum_core_selection/` ne sont cités par aucune ligne du rapport** —
   vérifié par comparaison entre les 144 noms de répertoires et l'ensemble des
   `run_dir` du CSV (144 − 48 = 96).

## Vérification

Depuis la racine du repo :

```
shasum -a 256 -c reports/historical/mu_j/manifest.sha256
```

Chaque ligne doit sortir `OK`. Cette commande prouve que chaque fichier sous
`traces/` a, au moment de la vérification, le même contenu octet pour octet que
lorsque le manifeste a été généré — donc que le paquet n'a pas été altéré depuis
sa création. Elle ne prouve rien sur les sources `results/`/`results_amended/`
elles-mêmes (gitignorées, non versionnées) : c'est le paquet qui fait foi une fois
committé.
