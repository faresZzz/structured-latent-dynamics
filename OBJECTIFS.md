# structured-latent-dynamics — Objectifs, taxonomie et architecture

Charte du repo flagship. Compagnon de `REGLES.md` (conventions) et de
`PLAN_EXECUTION.md` (ordre de construction). La science du projet est fixée par
`outputs/flagship_plan_structured_latent_dynamics_v2_2026-07-11.md` (le « plan v2 »),
lui-même gouverné par `these_principale.md` et la synthèse v4. En cas de conflit :
these_principale > synthèse v4 > plan v2 > ce document.

Statut : Wave 5 close, D1 non atteint (2026-07-21). Colonnes « Statut » vérifiées
fichier par fichier contre le repo réel (existence, absence de stub, tests
dédiés passants, usage réel) le 2026-07-21 — pas de statut déclaratif non
vérifié. ✅ signifie « code + tests passants », pas « intégré dans le pipeline
scientifique final » ni « figure/résultat produit » : voir notes en bas de
chaque tableau pour les nuances.

---

## 1. Vision du projet

Un seul repo qui répond à une seule question :

> Quand une structure dynamique explicite dans le predictor latent d'un JEPA world
> model améliore-t-elle la stabilité des rollouts, la suffisance d'état et la
> planifiabilité — à quel coût par décision réussie — et comment le *mesure*-t-on ?

Ce repo est l'Année 1 de la thèse en miniature, commencée avant la thèse. Il est
la pièce centrale de la candidature : un chercheur d'AMI doit pouvoir le lire en
90 secondes (figure en haut du README, commande de repro, « What didn't work »)
et y reconnaître exactement le programme du research statement.

### Objectifs, dans l'ordre de présentation (hiérarchie these_principale §6)

1. **Le résultat Pareto en miniature** : compute par décision réussie à seuils
   absolus, CEM vs planning par gradient, 4 familles de predictors à budgets
   strictement appariés. Une taille core — pas de claim de pente (≥ 3 tailles =
   Année 2, dit dans le README).
2. **Le protocole factorial qui tourne** : backbone × structure (l'axe
   primaire depuis la phase B1 ; `structure` est architectural, jamais une
   pénalité numérique) × encodeur (2 axes), multi-seed, pré-enregistré.
   L'ancien axe architecture × contrainte commune μ_J a échoué et est
   retiré au rang d'étude secondaire (`README.md`, section Statut ; trace
   complète `reports/historical/mu_j/`).
3. **L'instrument calibré** : audit en 5 familles, formes invariantes, stress
   test de reparamétrisation, calibration fault-injection timeboxée avec
   précision de triage mesurée contre une baseline val-loss.
4. **L'honnêteté structurelle** : prédictions P1-P5 et barres de robustesse
   committées et taggées avant les runs ; barres d'erreur partout ; « What
   didn't work » ; limites dites.

### Principes de sélection

- Tout ce qui ne sert aucune des 5 figures cibles (plan v2 §2) est hors scope.
- Petit et lisible bat gros et générique : le lecteur suit le forward pass au
  débugger.
- Compute frugal par design : grille core ~550 runs ≈ un week-end de Mac,
  arithmétique affichée dans le README.
- Le vocabulaire s'arrête où la théorie s'arrête (échelle Koopman (i)/(ii)/(iii),
  REGLES §3).

### Stratégie de backend

- **PyTorch partout** (MPS/CPU) pour les modèles, losses, planning — un seul
  backend, pas de dualité NumPy/PyTorch : l'audit a besoin d'autograd.
- **float64 sur CPU pour l'audit** (gramiens, produits de Jacobiennes,
  pseudospectres) — l'audit tourne offline sur checkpoints, la précision prime.
- NumPy uniquement pour l'intégration RK4 des systèmes (vérité terrain) et le
  stockage NPZ.

---

## 2. Livrables

| Livrable | Description | Jalon |
|---|---|---|
| Fig 1 | Rollout error vs horizon + panneau Lorenz adversarial (VPT, jamais MSE seule) | D1 (v0), D2 (final) |
| Fig 2 | Portrait spectral par régime (spectre vs champ de spectres locaux) | D1 (v0), D2 |
| Fig 3 | Panel d'audit sur le factorial (stabilité = Jacobien total, pas ρ(Ā)) | D2 |
| Fig 4 | La facture : succès vs compute par décision réussie, CEM vs gradient | D2 |
| Fig 5 | L'instrument à l'épreuve : (a) triage calibré, (b) intervention P4 | D2 |
| Commit `pre-registration` | P1-P5 + barres de robustesse + seuils absolus, taggé avant la grille | D1 |
| Grille core | ~550 runs, 5 seeds, artefacts régénérables | D2 |
| Notebooks 00-09 | Le cours : dérivations M2 + preuves de ce que fait le package | D3 |
| Writeup 6-8 pages | Format papier : abstract, méthode, 5 figures, limites | D3 |
| Email Terver | Une cartouche : l'artefact qui tourne + une question précise | D3 |

Jalons : **D1** fin S3 (repo public, Fig 1-2 v0, pré-enregistrement taggé) ;
**D2** fin S8 (science complète, 5 figures, calibration, What didn't work) ;
**D3** fin S12 (notebooks, writeup, email). Détail : plan v2 §8.

---

## 3. Structure du dépôt

```text
structured-latent-dynamics/
  README.md                # Fig 1 en haut, tagline, repro, prédictions, What didn't work
  PROTOCOL.md              # l'équité opposable : budgets, tuning, seuils, marges μ_J
  OBJECTIFS.md             # ce fichier
  REGLES.md                # la charte des conventions
  FUTURE.md                # parking à idées (une ligne chacune)
  pyproject.toml           # uv ; torch, numpy, matplotlib, pyyaml ; dev : pytest, ruff
  Makefile                 # setup, test, smoke, grid-core, fig1..fig5, calibrate, reproduce-all
  assets/                  # figures finales committées (README)
  scripts/check_notebooks.py  # hygiène notebooks (outputs vides, execution_count null)
  src/sld/
    systems/               # §4.1 — vérités terrain RK4
    data/                  # §4.2 — régimes, mixing, datasets
    encoders/              # §4.3
    predictors/            # §4.4 — les 4 familles + blocs 2×2
    objectives/            # §4.5 — JEPA, VICReg, μ_J, R_spec
    training/              # §4.6 — trainer, budgets, seeds
    audit/                 # §4.7 — l'instrument
    calibration/           # §4.8 — fault injection + triage
    planning/              # §4.9 — CEM, gradient, MPC, facture
    experiments/           # §4.10 — runner factorial + configs YAML
    viz/                   # §4.11 — une fonction par figure cible
  notebooks/               # 00-09, §5
  tests/                   # miroir de src/, §6
  results/                 # <exp>/<config_hash>/{config.yaml, metrics.csv, audit.json,
                           #   ckpt.pt, figs/} — régénérables, gitignorés sauf assets/
```

---

## 4. Taxonomie détaillée du package

Statuts : ⬜ à faire · 🟨 en cours · ✅ fait · 🔁 révisé user · 🔴 retirée
(code + tests existent mais l'entrée publique refuse délibérément de
tourner, ex. `NotImplementedError`).

### 4.1 `systems/` — les vérités terrain

| Fichier | Exports | Responsabilité | Statut |
|---|---|---|---|
| `base.py` | `DynamicalSystem` (protocol) | Contrat : `f(x,u)`, `integrate(x0,u_seq,dt)` (RK4), `sample_ic(regime)`, `observe(x,mode)`, `linearize(x*,u*)` ; dims, bornes d'action | ✅ |
| `damped_pendulum.py` | `DampedPendulum` | $\ddot\theta = -\omega^2\sin\theta - 2\zeta\omega\dot\theta + u$ ; eigvals analytiques de la linéarisation ; énergie | ✅ |
| `coupled_oscillators.py` | `CoupledOscillators` | 2 masses-ressorts amorties, force sur la masse 1 ; modes analytiques ; $W_c$ exact (système linéaire) | ✅ |
| `lorenz63.py` | `Lorenz63` | σ=10, ρ=28, β=8/3 ; autonome ; $\lambda_{\max} \approx 0.906$ en référence | ✅ |
| `cartpole.py` | `CartPole` | équations complètes, force continue ; linéarisation à l'équilibre haut | ✅ |

Vérifié 2026-07-21 : `pytest` — pendulum 6/6, oscillators 3/3, lorenz63 3/3,
cartpole 4/4, tous verts. **Nuance** : `coupled_oscillators.py`, `lorenz63.py`,
`cartpole.py` sont construits et testés isolément mais **ne sont référencés
nulle part ailleurs dans `src/`** — aucune config `experiments/` ne les
instancie encore (seul `damped_pendulum.py` est câblé dans `factorial.py`).
Le ✅ couvre « le système existe et est correct », pas « le système est
utilisé dans un run ». Câblage prévu Wave 6.

### 4.2 `data/` — régimes et observations

| Fichier | Exports | Responsabilité | Statut |
|---|---|---|---|
| `regimes.py` | `OURegime`, `NarrowPolicyRegime`, `AnisotropicRegime`, `ActionConditionedRegime` | Échantillonneurs (IC, u_seq) des 4 régimes LeJEPA (plan v2 §3.3) | ✅ |
| `mixing.py` | `SmoothMixing`, `PartialObservation` | Mixing non linéaire bijectif $x = g(z)$ (rotation fixe ∘ tanh-affine ∘ rotation, inverse en forme fermée, Lipschitz déclaré) ; variantes partielles | ✅ |
| `generate.py` | `build_dataset` | (système × régime × obs × seed) → NPZ train/val/test + meta JSON | ✅ |

### 4.3 `encoders/`

| Fichier | Exports | Responsabilité | Statut |
|---|---|---|---|
| `frame_mlp.py` | `FrameMLPEncoder` | une observation → z | ✅ |
| `window_mlp.py` | `WindowMLPEncoder` | fenêtre $x_{t-k:t}$ aplatie (k=8) → z | ✅ |
| `pretrain.py` | `pretrain_frozen_encoder` | encodeur gelé par régime (JEPA+VICReg puis freeze) — stand-in LeJEPA déclaré | ⬜ |

Vérifié 2026-07-21 : `frame_mlp.py` 5/5, `window_mlp.py` 6/6, tests verts.
`pretrain.py` confirmé absent (aucun fichier, aucun stub).

### 4.4 `predictors/` — les 4 familles

| Fichier | Exports | Responsabilité | Statut |
|---|---|---|---|
| `base.py` | `Predictor` (protocol), `PredictorState`, `PredictorModule` | Contrat : `step(h,u,state)`, `rollout(h0,u_seq)`, `local_operator(h,u,state) → (A,B) \| None`. Docstring de sémantique : rollout itère dans H, π read-only, `local_operator` = LECTURE seulement, jamais l'objet de stabilité | ✅ |
| `blocks2x2.py` | `RotationScalingBlocks`, `discretize_zoh` | Paramétrisation $(\alpha_i, \omega_i)$ → blocs réels 2×2 ; ZOH exacte en forme fermée ; cellule ON : $\alpha_i = -e^{\nu_i}$ ; garde $(\alpha,\omega) \neq (0,0)$ | ✅ |
| `mlp.py` | `MLPPredictor` | $\hat h_{t+1} = \mathrm{MLP}([h;u])$ + résiduel ; `local_operator` → None | ✅ |
| `transformer.py` | `CausalTransformerPredictor` | 2 blocs d'attention causale sur la fenêtre du rollout ; famille générique + baseline O(H²) de la facture | ✅ |
| `min_ssm.py` | `SelectiveSSMPredictor` | $\bar A(c_t)$ diagonal réel, $\Delta(c_t)$, $\bar B(c_t)$ ; scan séquentiel ; régime LPV, jamais « Koopman » | ✅ |
| `koopman_ssm.py` | `KoopmanSSMPredictor` | lifting ψ + transition sur l'échelle `regime='constant'\|'bilinear'\|'selective'` + projection π ; **cellule core = bilinear** ($\bar A h + \sum_j u_j \bar B_j h + \bar B_0 u$) ; `selective` = ablation LPV | ✅ |
| `registry.py` | `get_predictor` | famille + config → instance appariée au budget | ✅ |

### 4.5 `objectives/`

| Fichier | Exports | Responsabilité | Statut |
|---|---|---|---|
| `jepa.py` | `JEPALoss` | prédiction latente + EMA target + stop-grad ; consistance liftée λ_c ($\hat h \approx \mathrm{sg}[\psi(z)]$) ; horizons TF/FR déclarés | ✅ |
| `vicreg.py` | `variance_loss`, `covariance_loss` | anti-collapse sur ψ(z), appliqué à tous les bras | ✅ |
| `jac_reg.py` | `TotalJacobianPenalty` | facteur contrainte commun μ_J : pénalité sur $\lVert J^q \rVert$ (état augmenté) par sondes JVP/Hutchinson, pas sous-échantillonnés — LE point techniquement risqué (spike S1). **Jalon B2 : démoté SECONDARY, hors thèse primaire** (échec de la campagne μ_J, `README.md`) — module conservé et testé pour la régression uniquement | ✅ |
| `spectral_reg.py` | `SpectralSeparationReg` | hinge + double-puits log-module — **ablation de la famille spectrale seulement**, jamais dans le facteur commun. **Jalon B2 : démoté SECONDARY** (même statut que `jac_reg.py` ci-dessus) | ✅ |

### 4.6 `training/`

| Fichier | Exports | Responsabilité | Statut |
|---|---|---|---|
| `trainer.py` | `Trainer` | config-driven : AdamW, schedule, mix TF/FR, logging CSV/JSON, artefacts `<exp>/<hash>/` | ✅ |
| `budget.py` | `match_budget`, `count_flops_per_step`, `count_flops_per_trajectory` | appariement params ±5 % par famille ; FLOPs analytiques ; transformer en f(H) | ✅ |
| `seeding.py` | `set_seed` | reproductibilité torch/numpy | ✅ |

### 4.7 `audit/` — l'instrument

| Fichier | Exports | Responsabilité | Statut |
|---|---|---|---|
| `spectral.py` | `operator_spectrum`, `local_spectra_field`, `operator_norm`, `pseudospectrum_grid`, `edmd_fit` | ce qu'on AFFICHE : spectre (régimes i/ii) ou champ de spectres locaux (iii/génériques) ; pseudospectre sur grille SVD (n ≤ 64) ; baseline EDMD (gold multi-pas) | ✅ |
| `jacobian.py` | `total_jacobian`, `jacobian_products`, `fr_lyapunov_exponents`, `conditional_jacobian`, `lipschitz_pi`, `fr_drift_curve` | ce qu'on AUDITE : $J^q$ sur l'état augmenté (h,m), produits Φ, Lyapunov FR (méthode QR), Lip(π) reportée partout ; drift FR normalisé (Fig 1) | ✅ |
| `gramians.py` | `reachability_gramian`, `observability_gramian`, `free_evolution_displacement` | $W_c$ LTV avec $R_u$ déclaré ; $C_k := J_\pi(h_k)$ ; gramien conditionnel vs système complet ; Δh relatif à l'évolution libre | ⬜ |
| `alignment.py` | `r_ctrl`, `e_ctrl`, `pushforward_gramian`, `cost_gradient_pullback` | formes invariantes : $r_{\mathrm{ctrl}} = \frac{g^\top W_c g}{g^\top \Sigma_h g}$ ; $E_{\mathrm{ctrl}} = g^\top \Sigma_h (W_c + \lambda\Sigma_h)^{-1} \Sigma_h g$ ; $W_c^z = J_\pi W_c J_\pi^\top$ | ⬜ |
| `identifiability.py` | `linear_recovery_r2`, `procrustes_error`, `orthogonality_error`, `covariance_anisotropy` | panel LeJEPA contre l'état vrai | ⬜ |
| `invariance.py` | `GaugeCheckResult`, `diagonal_gauges`, `check_diagonal_gauge_invariance`, `assert_diagonal_gauge_invariant`, `check_shear_breaks_invariance` | stress test de reparamétrisation diagonale (REGLES section 4) : check positif (la mesure ne bouge pas sous `D` diagonale finie inversible, y compris échelles extrêmes) + check négatif déclaré (la même mesure DOIT bouger sous un shear) ; **gate avant toute lecture scientifique** | ✅ |
| `dsa.py` | `dsa_similarity` | optionnel v1.5, jamais bloquant | ⬜ |

Vérifié 2026-08-04 (jalon B4, corrige la ligne `invariance.py` ci-dessus,
mesurée fausse par la brief B4 et revérifiée ici avant correction) :
`spectral.py` 40/40, `jacobian.py` 48/48, tous verts (le seul
`NotImplementedError` de `jacobian.py` est une garde documentée sur un
chemin de spectre partiel non supporté, pas un stub). **`invariance.py`
EXISTE et est primaire** — `tests/audit/test_invariance.py`, 9/9 verts,
et il n'exporte pas `stress_test` (ce nom n'existe nulle part dans le
fichier) mais les 5 symboles listés ci-dessus, tous ré-exportés par
`audit/__init__.py`. `gramians.py`, `alignment.py`, `identifiability.py`,
`dsa.py` restent confirmés absents — `audit/__init__.py` exporte
spectral/jacobian/invariance/rollout (`vpt`, `VPTResult`), pas seulement
spectral/jacobian.

### 4.8 `calibration/`

| Fichier | Exports | Responsabilité | Statut |
|---|---|---|---|
| `defects.py` | `TruncatedStateDefect`, `RotatedActionDefect`, `OverContractionDefect`, `NegativeControl` | injecteurs à intensité déclarée, appliqués à la génération de données ou à l'entraînement | ⬜ |
| `triage.py` | `TriagePanel` | règles de décision interprétables sur les sorties d'audit + abstention calibrée ; scoring en aveugle (labels cachés) ; baseline val-loss | ⬜ |
| `protocol.py` | `build_calibration_matrix` | 3 défauts × 3 intensités × 5 seeds + contrôles négatifs ; held-out seeds/intensités | ⬜ |

Vérifié 2026-07-21 : `src/sld/calibration/` **n'existe pas du tout** (pas
seulement les fichiers — le dossier lui-même est absent).

### 4.9 `planning/`

| Fichier | Exports | Responsabilité | Statut |
|---|---|---|---|
| `cem.py` | `CEMPlanner` | population, élites, itérations ; budgets S/I/H déclarés ; plancher de variance | ⬜ |
| `grad_plan.py` | `GradientPlanner` | descente Adam sur u_seq via autograd à travers le rollout ; G, R mesurés, k compté ; actions bornées par tanh | ⬜ |
| `mpc.py` | `MPCWrapper` | receding horizon autour d'un planner | ⬜ |
| `tasks.py` | `PendulumSwingUp`, `CartPoleSwingUp` | critères de succès **absolus** fixés dans PROTOCOL.md avant les runs | ⬜ |
| `bill.py` | `DecisionBill` | FLOPs planner par décision (échecs et replans inclus) ; `compute_per_successful_decision` — l'estimand du plan v2 §3.7 | ⬜ |

Vérifié 2026-07-21 : `src/sld/planning/` **n'existe pas du tout**.

### 4.10 `experiments/`

| Fichier | Exports | Responsabilité | Statut |
|---|---|---|---|
| `factorial.py` | CLI `sld-factorial` | expansion YAML → runs (config + seed → dossier déterministe) ; resume | ✅ |
| `loading.py` | (non listé initialement, module réel) | chargement/résolution des artefacts de run | ✅ |
| `configs/pendulum_core.yaml` | config core, gelée/historique | **✅ (mais non exécutable)** |
| `configs/pendulum_core_partial.yaml` | config core active | ✅ |
| `configs/smoke.yaml` | config smoke test | ✅ |
| `configs/calibration.yaml` | config sélection μ_J, gelée/historique | **✅ (mais non exécutable)** |
| `configs/oscillators.yaml` | — | ⬜ |
| `configs/lorenz.yaml` | — | ⬜ |
| `configs/cartpole.yaml` | — | ⬜ |
| `configs/encoder_axis.yaml` | — | ⬜ |
| `configs/koopman_ablations.yaml` | — | ⬜ |

Vérifié 2026-07-21 : `factorial.py` 35/35 tests verts, non un stub. **Correction** :
cette section était marquée ⬜ en bloc alors que `factorial.py` et 3 des 8
configs existent et passent leurs tests — corrigé ici. Les 5 configs restantes
(oscillators/lorenz/cartpole/encoder_axis/koopman_ablations) n'existent pas du
tout, cohérent avec l'absence de câblage des systèmes correspondants notée en
§4.1.

Mise à jour jalon B4 (2026-08-04) : `pendulum_core.yaml` et `calibration.yaml`
portent toujours le schéma retiré `family`/`constraint` (l'ère μ_J) et sont
désormais des configs gelées/historiques — le validateur `backbone`/`structure`
les refuse explicitement (`expand(..., dry_run=False)` lève `ValueError`,
`tests/experiments/test_factorial.py::test_historical_mu_j_configs_are_no_longer_executable`).
**Aucune grille n'a jamais tourné avec le fichier `pendulum_core.yaml`
lui-même** — le ✅ historique couvrait seulement « le runner et la config
existent et sont testés », jamais « la science a été produite » avec CE
fichier. Les 40 cellules OFF publiées (`README.md`, section Résultats)
viennent du fichier alors nommé `pendulum_core_off.yaml` (renommé
`pendulum_core_partial.yaml` en B4) — qui partage `exp_name: pendulum_core`
avec le fichier ci-dessus, d'où le nom identique du dossier de résultats ;
ce n'est pas le même fichier qui a tourné. `pendulum_core_partial.yaml` est
la config core active aujourd'hui (4 des 6 paires `backbone`/`structure`
valides ; `mlp_k`/`transformer_k` de la phase C complètent les 2 restantes) ;
`experiments/selection.py` (le sélecteur μ_J) n'existe plus dans `src/` —
historique seulement, trace complète sous `reports/historical/mu_j/traces/`
(922 fichiers, 5 043 481 octets, 228 checkpoints `.pt` volontairement
exclus).

### 4.11 `viz/`

| Fichier | Responsabilité | Statut |
|---|---|---|
| `style.py` | palette, bandes ±1 SE, tailles — cohérence des 5 figures | ✅ |
| `fig1_rollout.py` | Fig 1 (rollout + Lorenz VPT + démo contraction-améliore-MSE). **Jalon B : `make_fig1` retirée, lève `NotImplementedError`** — le readout OLS par run est un paramètre libre par bras, pas une comparaison à armes égales ; remplacé en phase D par un readout partagé gelé | 🔴 retirée (phase D) |
| `fig2_spectrum.py` | Fig 2 (spectre vs champ local, eigvals analytiques superposées). **Jalon B2 : démotée SECONDARY**, hors thèse primaire (liée à la campagne μ_J retirée) | 🟨 |
| `fig3_panel.py` | Fig 3 (panel factorial : drift FR, marge $J^q$, R², r_ctrl, Lip(π)) | ⬜ |
| `fig4_bill.py` | Fig 4 (succès vs compute/décision, CEM vs gradient ; encadré FLOPs(H)) | ⬜ |
| `fig5_instrument.py` | Fig 5 (confusion de triage + intervention P4) | ⬜ |

Vérifié 2026-07-21 (figure produite : encore vrai le 2026-08-04, revérifié
avant B4) : **aucune figure réelle n'a jamais été produite sur disque** —
recherche de `*.png`/`*.svg` dans tout le repo (y compris
`results/smoke/*/figs/` et `assets/`) : zéro fichier trouvé. `assets/`
n'existe même pas encore. Distinction délibérée entre « code testé » et
« artefact produit ».

Mise à jour jalon B (2026-08-04) : `fig1_rollout.py`'s `make_fig1` now
refuses to run at all (`NotImplementedError`, 🔴 retirée ci-dessus) rather
than being merely "in progress" — this is a deliberate design decision
(equal-footing readout is a phase-D dependency), not a regression.
`fig2_spectrum.py` stays 🟨: its unit tests on synthetic data still pass
(`tests/viz/test_viz.py`), but it is now SECONDARY (jalon B2 demotion),
tied to the retired μ_J campaign, and blocks no gate.

---

## 5. Mapping notebooks ↔ package

Notebooks dans `notebooks/`, en anglais, cours + preuves (REGLES §8). Détail
cellule par cellule : PLAN_EXECUTION Wave 10.

| NB | Fichier | Enseigne | Prouve (★) | Modules consommés | Statut | Révision user |
|---|---|---|---|---|---|---|
| 00 | `00_systems_and_the_predictability_ceiling.ipynb` | les 4 systèmes, le plafond de Lyapunov, le VPT | eigvals pendule ; ★ $T^* = \frac{1}{\lambda}\ln(\Delta_{\mathrm{tol}}/\delta_0)$ | `systems/` | ⬜ | ⬜ |
| 01 | `01_jepa_in_thirty_minutes.ipynb` | prédire en latent ; sémantique du rollout | ★ le collapse minimise la loss nue ; VICReg | `objectives/jepa,vicreg`, `encoders/` | ⬜ | ⬜ |
| 02 | `02_mlp_predictor_first_audit.ipynb` | la baseline sans état ; l'audit Jacobien | ★ propagation d'erreur ; ★ borne géométrique = lemme de motivation + ce que le vrai théorème ajoute | `predictors/mlp`, `audit/jacobian` | ⬜ | ⬜ |
| 03 | `03_ssm_from_first_principles_minimal_mamba.ipynb` | ODE → SSM discret → sélectivité | ★ ZOH exacte par bloc 2×2 ; sélectivité = LPV | `predictors/blocks2x2,min_ssm` | ⬜ | ⬜ |
| 04 | `04_koopman_theory.ipynb` | l'opérateur de composition, EDMD | ★ exemple Brunton résoluble ; ★ EDMD = moindres carrés optimaux | aucun (EDMD en cellules) → valide `audit/spectral` | ⬜ | ⬜ |
| 05 | `05_koopman_ssm_predictor.ipynb` | l'assemblage, l'échelle (i)/(ii)/(iii) | ★ bilinéarité des systèmes control-affine liftés ; pourquoi (iii) n'est pas Koopman | `predictors/koopman_ssm`, `objectives/spectral_reg` | ⬜ | ⬜ |
| 06 | `06_audit_instrument_calibration.ipynb` | l'instrument et sa calibration | ★ non-normalité 2×2 ; ★ contre-exemple des gates ; ★ gramien + énergie minimale + Range ; ★ invariance des pairings ; ★ Procrustes SVD | `audit/*`, `calibration/*` | ⬜ | ⬜ |
| 07 | `07_experimental_protocol_factorial.ipynb` | le factorial, les budgets, P1-P5 | pourquoi le swap de bundles confond ; théorème LeJEPA (énoncé) | `training/budget`, `experiments/` | ⬜ | ⬜ |
| 08 | `08_planning_and_the_bill.ipynb` | CEM, gradient, la facture | ★ CEM = cross-entropy ; ★ $C(z)=(z^2-1)^2$ : contrôlabilité nécessaire pas suffisante ; arithmétique $SI$ vs $kGR$ | `planning/*` | ⬜ | ⬜ |
| 09 | `09_reading_the_results.ipynb` | lecture des 5 figures, P1-P5, la facture | aucune théorie nouvelle | `viz/*`, `results/` | ⬜ | ⬜ |

Ordre de rédaction : **après D1** (exception : théorie de 04 et 06 pendant les
temps morts de runs). Interdiction de polir 00 avant que Fig 1 existe.

Vérifié 2026-07-21 : `notebooks/` ne contient que `example.ipynb` (scaffold
Wave 0). **Aucun des 10 notebooks 00-09 n'existe** — cohérent avec 0/10 ⬜,
attendu puisque leur rédaction est explicitement postérieure à D1.

---

## 6. Tests — cartographie

Arborescence miroir : `tests/systems/`, `tests/data/`, `tests/predictors/`,
`tests/objectives/`, `tests/audit/`, `tests/calibration/`, `tests/planning/`,
`tests/test_budget.py`, `tests/test_smoke.py`. Les tests protègent des
**comportements** (gold tests, propriétés, invariances), pas des shapes seules.
Liste exacte par wave : PLAN_EXECUTION. Les gold tests non négociables :

1. EDMD retrouve les eigvals d'un système linéaire à 1e-3 (avant tout diagnostic
   sur modèle appris).
2. ZOH des blocs 2×2 vs `matrix_exp` à 1e-6 ; cellule ON stable ∀Δ>0.
3. Cible hors de l'image de $W_c$ ⇒ énergie grande, jamais 0.
4. Stress test d'invariance : r_ctrl/E_ctrl stables sous reparamétrisation
   compensée aléatoire.
5. Contre-exemple des gates reproduit : spectre affiché contractif, $J^q$ explosif.
6. Contrôle négatif de calibration : le panel sait dire « rien ».
7. Oracle CEM sur la dynamique vraie atteint les seuils de succès déclarés
   (faisabilité des tâches).

Vérifié 2026-07-21 : `uv run pytest -q` → **376 passed, 0 failed, 0 skipped**
sur l'ensemble du repo — cohérent avec le chiffre annoncé dans
`RESUME_SESSION.md`, aucun écart constaté. Les gold tests 1, 2, 5 (EDMD,
ZOH, contre-exemple des gates) sont réellement implémentés et verts ; les
gold tests 3, 4, 6, 7 dépendent de modules encore absents
(`audit/gramians.py`, `audit/invariance.py`, `calibration/`, `planning/`) et
ne peuvent donc pas encore exister.

---

## 7. Critères de réussite du repo

Le repo est réussi si :

1. `make reproduce-all` régénère les 5 figures depuis les configs, seeds inclus.
2. Le commit `pre-registration` précède la grille core dans l'historique git.
3. Chaque diagnostic reporté a passé le stress test d'invariance.
4. La calibration donne une précision de triage mesurée, contrôles négatifs
   inclus, meilleure que la baseline val-loss — ou le README dit qu'elle ne l'est
   pas.
5. Les notebooks se lisent comme un cours autonome dont chaque affirmation
   mathématique est soit prouvée en cellule, soit citée avec lien, et chaque
   objet enseigné est vérifié contre le package (cellules de parité).
6. « What didn't work » est non vide et honnête.
7. Un chercheur pressé comprend en 90 secondes : ce qu'on mesure, ce qu'on
   trouve, ce que ça coûte, ce qui n'a pas marché.
