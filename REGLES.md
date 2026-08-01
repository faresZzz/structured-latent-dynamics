# Règles — `structured-latent-dynamics`

Charte du repo. Adaptée de `machine_learning_from_scratch/REGLES.md` à la nature
différente de ce projet : ce n'est pas un curriculum cumulatif de modèles, c'est
**un instrument scientifique et son cours**. Deux couches, une règle :

> **Le package est la vérité, les notebooks sont le cours qui la prouve.**

Le package `sld` porte les expériences (systèmes, predictors, audit, planning,
runner). Les notebooks enseignent la théorie avec des dérivations complètes,
implémentent chaque objet central en version minimale, et **vérifient
numériquement que le package fait ce que le cours a prouvé**. Aucune logique
d'expérience ne vit dans un notebook ; aucun théorème ne vit dans le package
sans son chapitre.

L'objectif n'est pas un framework. C'est un repo qu'un chercheur d'AMI peut
auditer en 90 secondes et qu'un étudiant M2 peut lire comme un cours.

---

## 1. Principe directeur : la science avant le code

Ce repo implémente un protocole pré-enregistré (plan v2). La hiérarchie des
exigences, dans l'ordre :

1. **Correction** — un diagnostic faux est pire qu'un diagnostic absent.
2. **Honnêteté** — un résultat négatif propre vaut un résultat positif ; « What
   didn't work » est un livrable, pas une excuse.
3. **Lisibilité** — le forward pass se suit au débugger ; les noms suivent les
   équations.
4. **Vitesse** — en dernier, toujours.

Corollaire : on ne « répare » jamais un résultat en changeant la métrique, le
seuil ou le périmètre après coup. Toute modification du protocole après le tag
`pre-registration` est un commit visible qui dit pourquoi.

---

## 2. Frontière d'architecture

```text
Si c'est une vérité terrain (dynamique, régime de données)      -> systems/, data/
Si c'est un modèle appris (encodeur, predictor, loss)           -> encoders/, predictors/, objectives/
Si ça mesure un modèle appris sans le modifier                  -> audit/
Si ça valide l'instrument lui-même                              -> calibration/
Si ça consomme le modèle pour agir                              -> planning/
Si ça orchestre des runs ou dessine des figures                 -> experiments/, viz/
```

Règles de dépendance :

- `audit/` dépend du **contrat** `Predictor`, jamais d'une famille concrète.
  **Aucun chemin spécial par predictor dans l'audit** — le contrat EST le
  protocole. Si l'audit a besoin de connaître la famille, le design est faux.
- `calibration/` consomme `audit/` et `training/`, jamais l'inverse.
- `viz/` lit `results/`, jamais les objets vivants — `make figN` régénère chaque
  figure sans réentraîner.
- Pas de dépendance circulaire ; pas de « magic helper » qui cache une équation.

Open/closed : ajouter une famille de predictor = un fichier + une entrée de
registry + ses tests de contrat. Ni l'audit, ni le trainer, ni les configs
existantes ne changent.

---

## 3. Discipline de vocabulaire (non négociable)

Les fautes de vocabulaire de ce projet ont déjà été payées deux fois en
adversarial review. Elles sont éliminatoires devant le public cible.

- **« Koopman »** désigne un opérateur linéaire **fixe** sur les observables
  (régime (i)) ou le cadre bilinéaire à entrées (régime (ii), terme
  $\sum_j u_j \bar B_j h$ explicite). Un opérateur $\bar A(c_t)$ état-dépendant
  est **LPV/cocycle** — il ne s'appelle jamais Koopman, ni dans le code, ni dans
  les docstrings, ni dans les notebooks, ni dans le README.
- **« Bilinéaire »** exige le terme d'interaction $\sum_j u_j \bar B_j h$.
  $\bar A h + \bar B u$ est **affine**. La confusion est une faute, pas une
  approximation.
- **« Spectre »** : au régime (iii) et pour les familles génériques, on écrit
  « champ de spectres locaux » — et le spectre affiché n'est **jamais** l'objet
  de stabilité (règle §5).
- **« Localiser »** un échec est interdit à l'instrument seul : le panel fait du
  **triage calibré** ; localiser se gagne par intervention.
- Nommages honnêtes : « Mamba-style minimal selective SSM », « minimal causal
  transformer ». On ne survend pas l'échelle.

---

## 4. Règles d'intégrité scientifique

- **Pré-enregistrement** : P1-P5, barres de robustesse, seuils de succès absolus
  et budgets de tuning sont committés et taggés `pre-registration` **avant** la
  grille core. Les runs exploratoires antérieurs sont journalisés dans
  `PROTOCOL.md`.
- **Équité** : params ±5 %, FLOPs comptés, mêmes updates, même sweep de tuning
  par bras, sweeps publiés. Le temps horloge n'est jamais un critère d'équité.
- **Barres d'erreur partout** : 5 seeds minimum, bandes ±1 SE sur toute courbe.
  Une courbe sans variance est une anecdote et ne sort pas du repo.
- **MSE jamais seule sur un système chaotique** : Lorenz se lit en VPT (unités
  de Lyapunov) + fidélité statistique long terme.
- **Langage des conclusions** : les barres de robustesse déclenchent des
  décisions pré-enregistrées (« la claim est retirée »), jamais des conclusions
  causales (« c'était la capacité »). Le causal se gagne par intervention (P4)
  ou par les cellules du factorial.
- **Contrôlabilité mesurée, jamais optimisée** en pré-entraînement.
- **Invariance obligatoire** : aucun diagnostic n'entre dans une figure ou une
  conclusion sans avoir passé le stress test de reparamétrisation
  (`audit/invariance.py`). Un diagnostic qui échoue est réparé ou retiré.
- **Fermeture et Lip(π) reportées** dans toute expérience qui roll-out.

---

## 5. Stabilité numérique — la carte des pièges

Chaque piège est répété au point d'usage (docstring + test dédié). Vue
d'ensemble :

- **Blocs 2×2** : garde $(\alpha_i, \omega_i) \neq (0,0)$ avant $A_i^{-1}$ ;
  Δ > 0 par softplus + clamp ; ZOH en forme fermée (pas de `matrix_exp` en
  production, il ne sert qu'aux tests de parité).
- **Pénalité μ_J** : sondes JVP (jamais la Jacobienne dense en entraînement) ;
  pas sous-échantillonnés ; surcoût mesuré et reporté (cible < 2.5× le step).
- **Produits de Jacobiennes / Lyapunov** : méthode QR (ré-orthonormalisation à
  chaque pas), accumulation des log ; jamais le produit brut sur > 20 pas
  (overflow/underflow garanti).
- **Gramiens et énergies** : float64 CPU ; symétriser $(W + W^\top)/2$ ; résoudre
  $(W_c + \lambda\Sigma_h)x = b$ par Cholesky, **jamais** d'inverse explicite ni
  de pseudo-inverse nu (la condition de Range : cible hors image ⇒ énergie
  grande, pas 0) ; $\Sigma_h$ régularisée et symétrisée.
- **Pseudospectre** : grille SVD seulement pour n ≤ 64 (nos dims : 16-32).
- **EMA + stop-gradient** : vérifier par test que les paramètres target ne
  reçoivent aucun gradient — le bug silencieux classique de JEPA.
- **CEM** : plancher de variance ; élites ≥ 2. **Gradient planner** : actions
  bornées par tanh-reparamétrisation ; clipping de gradient ; non-convergence
  reportée, jamais maquillée.
- **Audit sur MPS** : interdit — l'audit tourne sur CPU float64, offline, sur
  checkpoints.
- **Mixing g(z)** : bijection à inverse en forme fermée, conditionnement borné,
  Lipschitz déclaré dans le meta JSON.

---

## 6. Règles code source

- Noms des équations : `A_bar`, `B_j`, `W_c`, `r_ctrl`, `lip_pi`, `Sigma_h` —
  pas de `tmp`, `stuff`, `data`.
- Une ligne de commentaire avant chaque bloc mathématique dense : le rôle, pas
  la paraphrase. Référence de section quand l'objet vient du plan (ex.
  `# gramien conditionnel vs complet — plan v2 §3.6`).
- Toute source citée en docstring inclut son lien (arXiv/DOI à côté du nom).
- Seeds explicites partout ; aucune mutation globale cachée ; métriques finies
  (NaN = bug, pas résultat).
- Configs YAML plates ; un run = config + seed → dossier `results/<exp>/<hash>/`
  déterministe et régénérable.
- Pas de `sys.path` hacks : package installé en editable via `uv`.
- Anglais pour tout ce qui est public (code, docstrings, README, notebooks) ;
  le français reste pour les documents de gouvernance internes.

**Contrat public de sortie d'un run** (condition de comparabilité du factorial —
inspiré du contrat `{agent, history, metrics, paths}` de `RL_from_scratch`).
Toute exécution `Trainer.fit(...)` ou `sld-factorial` retourne / écrit un objet
avec les mêmes clés quelle que soit la famille de predictor :

```python
{"model": model, "history": history, "metrics": summary, "paths": paths}
```

- `history` : clés stables entre familles — `train_loss`, `val_loss`,
  `epoch`/`step`, et par diagnostic quand applicable (`jac_margin`, `r_ctrl`,
  `lip_pi`, `vpt`). Un champ absent pour une famille (ex. `lip_pi` pour un
  predictor sans projection) est explicitement `None`, jamais omis en silence.
- `metrics` : résumé court et fini (dernière/meilleure valeur, seed, hash de
  config) — jamais un tenseur brut.
- `paths` : `config.yaml`, `metrics.csv`, `audit.json`, `ckpt.pt`, `figs/`
  (cf. `OBJECTIFS.md` §3). Ce contrat est ce qui permet à `viz/` de lire
  `results/` sans connaître la famille de predictor — ne pas le casser sans
  migrer toutes les familles en même temps.

---

## 7. Tests

Proportionnels au risque, comportementaux d'abord :

- **Gold tests** (OBJECTIFS §6) : ils prouvent que l'instrument mesure juste
  avant de mesurer quoi que ce soit d'appris. Aucun run de grille ne se lance si
  un gold test est rouge.
- **Tests de propriété** : stabilité de la cellule ON ∀Δ>0 ; invariance des
  pairings sous reparamétrisation ; contrôles négatifs silencieux ; énergie
  infinie hors de l'image.
- **Tests de contrat** : les 4 familles passent la même suite `Predictor`
  (shapes, threading de l'état, rollout dans H, `local_operator` cohérent).
- **Tests de contre-exemple** : le contre-exemple 1-D des gates et le cache qui
  explose à bloc contractif sont des tests — ils documentent en code pourquoi
  la règle §5 existe.
- **Smoke test CI** : grille minuscule 1 seed + 5 figures rendues, < 5 min.

Un résultat « trop beau » déclenche la même chasse au bug qu'un résultat trop
mauvais.

### 7.1 La règle des deux checks (adoptée 2026-08-01, non négociable)

**Tout objet mathématique — cœur comme instrument de mesure — porte au minimum
DEUX tests :**

1. **Le positif.** L'objet donne la bonne réponse sur un cas où la réponse est
   connue à la main : matrice évidente, polynôme, solution analytique, forme
   fermée, limite exactement calculable. Pas « une valeur plausible » : LA
   valeur, avec sa tolérance justifiée.
2. **Le négatif.** L'objet **signale** quand ça ne marche pas. On lui présente
   l'entrée dégénérée, la violation d'hypothèse ou le cas hors domaine qu'il est
   censé attraper, et le test échoue si l'objet laisse passer en silence.

Le second est celui qu'on oublie, et c'est celui qui compte. Un test positif
seul prouve que l'objet répond juste quand tout va bien ; il ne prouve pas qu'il
**agit comme on le pense**. Un audit de 2026-08-01 a trouvé une vingtaine de
tests verts qui ne pouvaient pas échouer : un compteur de FLOPs satisfait par une
formule en `2^t`, une borne de Lipschitz satisfaite par la constante `0.25`, un
test d'ordre RK4 qui compare RK4 à un RK4 plus fin, un test OU dont les seuils
acceptent une intégration d'Euler à la place du noyau exact.

**Le critère d'écriture d'un test est donc : « une implémentation fausse
échouerait-elle à ce test ? »** Si la réponse n'est pas un oui démontrable, le
test n'est pas écrit.

**Corollaire — la référence doit être indépendante.** Un objet ne se valide
jamais contre lui-même : ni contre sa propre sortie stockée, ni contre une
quantité calculée par le même chemin de code, ni contre une version plus fine de
lui-même. La référence est analytique, faite à la main, ou issue d'une seconde
dérivation.

---

## 8. Règles notebooks — le cours qui prouve le package

Les notebooks sont le point d'entrée pédagogique. Rythme imposé, hérité de
`machine_learning_from_scratch` :

```text
Concept -> Dérivation -> Implémentation minimale -> Vérification -> Parité package -> Interprétation
```

### Histoire continue et système conducteur (hérité, adapté)

Ce rythme doit former **une histoire continue**, pas une suite de dépôts
indépendants. Une cellule ne pose jamais une définition, une équation ou un
diagnostic à froid : elle répond à un problème posé dans la cellule qui
précède, et le lecteur doit sentir pourquoi l'étape suivante devient
nécessaire — pourquoi lifter avant de faire transiter, pourquoi auditer la
stabilité avant de planifier, pourquoi le stress test d'invariance vient
avant toute lecture scientifique du diagnostic.

Quand le chapitre s'y prête, choisir un **système conducteur** — le pendule
amorti pour l'intuition, Lorenz-63 pour le cas adversarial, le régime
action-conditionné pour l'identifiabilité — et le faire réapparaître à
l'intuition, dans la dérivation, dans le code, dans les figures et dans la
conclusion. Il peut être synthétique au début puis structurant pour
l'application finale, mais garde le même rôle narratif du début à la fin du
notebook plutôt qu'une suite d'exemples isolés changés à chaque cellule.

### Structure recommandée (adaptée)

1. **Problème et motivation** — quelle propriété de la dynamique ce chapitre
   traite (stabilité, identifiabilité, coût de planification), et quelle
   limite du chapitre précédent il éclaire.
2. **Intuition** — l'idée en mots avant les équations ; le rôle de chaque
   objet (lifting, projection, gramien, marge) posé avant sa formule.
3. **Système / régime de données** — le système RK4 ou le régime LeJEPA du
   chapitre : conditions initiales, actions, bruit d'observation ; **un tracé
   de trajectoire avant tout entraînement ou audit** — jamais un diagnostic
   affiché sur un système que le lecteur n'a pas encore vu.
4. **Dérivation** (niveau 1, règle des trois niveaux ci-dessous) — chaque
   terme expliqué ; une équation arrive comme la formalisation d'une question
   déjà posée, jamais en empilement.
5. **Implémentation minimale** (niveau 2) — dans l'ordre de compréhension, un
   mini-test ou un `assert` juste après chaque bloc important.
6. **Parité package** (niveau 3).
7. **Expérience / audit** — la boucle (entraînement, audit, calibration,
   planning) lisible : préparation, calcul, lecture séparés.
8. **Résultats & diagnostics** — **les objets appris montrés avant toute
   métrique finale** (le spectre, le champ de gramiens, la marge $J^q$) ;
   chaque figure suivie d'une interprétation réelle, jamais décorative.
9. **Cas limite / contre-exemple** — le cas adversarial du chapitre (Lorenz,
   le contre-exemple des gates 1-D) relié aux courbes déjà montrées, pas une
   expérience isolée surgie sans lien.
10. **Limites & pont** — ce que ce niveau de structure apporte, où il ne
    suffit pas, pourquoi le chapitre suivant devient nécessaire (l'échelle des
    trois profondeurs, statement §3). Bloc court **"quand ce niveau de
    structure paie, quand descendre au suivant"** — sans arbre de décision
    exhaustif.
11. **Lien avec `sld`** — pointeur module par module vers ce qui a été prouvé.

### La règle des trois niveaux (spécifique à ce projet)

Chaque objet mathématique central d'un notebook passe par trois niveaux,
dans cet ordre, dans le notebook :

1. **La preuve.** Dérivation complète niveau M2, hypothèses énoncées, pas de
   rigueur mesure-théorique. Les preuves marquées ★ dans OBJECTIFS §5 sont
   celles qu'un chercheur regardera — zéro raccourci. Un algorithme complet
   s'écrit en bloc LaTeX (`\boxed{\begin{aligned}...\end{aligned}}`), pas en
   liste Markdown.
2. **L'implémentation minimale en cellule.** Une version pédagogique de l'objet
   (EDMD en 10 lignes, ZOH d'un bloc, CEM nu, gramien LTV), explicitement
   marquée comme telle, qui suit la preuve ligne à ligne.
3. **La parité package.** Une cellule de vérification qui confronte la version
   cellule à la version `sld` sur le même exemple (`np.allclose`, tolérance
   affichée) — c'est elle qui fait du notebook une **preuve de ce que fait le
   package**, pas une démo parallèle.

Contrairement à `RL_from_scratch` (zéro import) : ici les notebooks **importent
`sld`** — mais uniquement après la parité, et pour les expériences, jamais pour
remplacer une dérivation. Une brique déjà enseignée dans un notebook antérieur
se réutilise avec un rappel d'une phrase + renvoi au chapitre qui l'a prouvée.

### Qualité, pas quantité (règle absolue, héritée)

Un notebook n'est jamais jugé au volume. Chaque cellule mérite sa place : une
cellule de texte apporte une idée, une cellule de code illustre exactement le
concept du moment, chaque visuel est suivi d'une interprétation réelle. Si on
peut supprimer une cellule sans rien perdre, elle ne doit pas exister. Pas de
boucle de polish : une passe décisive, puis la revue.

### Standards (hérités, adaptés)

- **Introduction canonique** : titre, question du chapitre en une phrase, ce
  qu'on va prouver, ce qu'on va construire, carte des sections, sources.
- **Conclusion canonique** : ce qui a été prouvé et vérifié, le pointeur vers
  les modules `sld` concernés, la limite principale, le pont vers le chapitre
  suivant.
- **Vérifications pédagogiques** : après chaque implémentation minimale,
  quelques `assert` qui testent l'idée du chapitre (l'énergie se conserve à
  ζ=0 ; CKA-like sanity ; l'eigval retrouvée à 1e-3) — pas seulement l'absence
  d'erreur.
- **Figures** : chaque plot montre l'idée discutée (frontière, spectre, marge,
  facture), axes nommés, et une interprétation en dessous. Diagrammes
  conceptuels en mermaid (`fill:none`, information par bordure : gris `#64748b`,
  vert `#16a34a`, bleu `#2563eb`, orange `#d97706`) ; matplotlib pour les
  données.
- **Attention aux échelles visuelles** : Lorenz-63 diverge, un spectre proche
  du cercle unité écrase les modes contractifs, un pseudospectre explose près
  d'un bloc non normal — fixer explicitement les limites d'axes pertinentes
  (souvent log ou VPT plutôt que temps brut) plutôt que laisser un autoscale
  masquer la forme que la figure enseigne, et expliquer ce choix.
- **Objets appris montrés avant la métrique finale** : avant de commenter une
  valeur de facture ou un taux de succès, montrer les objets que l'audit a
  mesurés — spectre ou champ de spectres locaux, gramien, marge $J^q$,
  Lip(π). Une bonne lecture explique ce que le predictor a appris de la
  dynamique, pas seulement le score final.
- **Équations reliées au code** : introduire la question que résout
  l'équation avant de l'écrire, expliquer le passage entre deux formes (ex.
  la ZOH exacte du bloc 2×2), puis annoncer explicitement la fonction ou
  l'attribut du code qui l'incarne (`audit/jacobian.py:total_jacobian`) —
  jamais une formule qui reste sans ancrage dans `sld`.
- **Stabilité numérique expliquée en cellule** : quand une cellule utilise QR
  au lieu du produit brut de Jacobiennes, Cholesky au lieu d'un
  pseudo-inverse, ou un garde `(α,ω)≠(0,0)`, expliquer en une phrase le
  problème numérique évité (cf. carte des pièges §5) et confirmer que la
  correction ne change pas l'objet mathématique enseigné.
- **Mini-tests lisibles** : un `assert` seul ou un `print` suffit rarement —
  quand le test valide une propriété dynamique (stabilité, conservation
  d'énergie, VPT), l'accompagner de la courbe ou du portrait de phase qui la
  rend visible.
- **Citations avec lien cliquable**, format `[Auteur (année), Titre](url)` ;
  chaque lien **vérifié par WebFetch avant insertion** (protocole ✱ du
  workspace) ; jamais de citation morte.
- **Notation alignée code ↔ équations** : `h_t`, `A_bar`, `W_c` identiques dans
  le texte, les maths et le code.
- **Hygiène** : `execution_count: null`, outputs nettoyés avant commit
  (`scripts/check_notebooks.py` en gate) ; les figures destinées au README vivent
  dans `assets/`, régénérées par `make figN`.
- **Conclusion active** : ne pas se contenter de lister ce qui a été prouvé —
  expliquer pourquoi le rung suivant de l'échelle (§3 OBJECTIFS, ladder du
  statement) devient nécessaire, et fermer sur le bloc court "quand ce niveau
  de structure paie, quand descendre au suivant" (point 10 de la structure
  recommandée ci-dessus).
- **Passe humanizer unique** sur le texte des notebooks et du README avant
  publication (une seule passe décisive, pas une boucle — cf. « Qualité, pas
  quantité » ci-dessus) — sans toucher aux équations, noms d'API, variables
  ni citations.

---

## 9. Workflow pour ajouter un composant

```text
lire le plan -> situer -> prouver (notebook ou note) -> implémenter (package)
    -> tester (gold/propriété) -> vérifier la parité -> documenter -> committer
```

1. Lire `PLAN_EXECUTION.md` (la wave concernée) et la section du plan v2 qui
   fixe la science.
2. Si l'objet est mathématique : écrire ou vérifier la dérivation d'abord
   (elle finira dans le notebook correspondant).
3. Implémenter dans `src/sld/`, tests dans le miroir `tests/`.
4. Si l'objet est un diagnostic : le passer au stress test d'invariance avant
   de l'exposer.
5. Mettre à jour la colonne Statut d'`OBJECTIFS.md` §4/§5.
6. Commits atomiques, identité git de l'utilisateur uniquement, jamais
   d'attribution IA (politique globale du workspace). **Aucun commit sans
   demande explicite de l'utilisateur.**

---

## 10. SOLID appliqué au projet

`§2` fixe déjà la frontière et l'open/closed sur les familles de predictor ;
ce qui suit le rend explicite composant par composant, dans le style de
`machine_learning_from_scratch`/`RL_from_scratch` :

- **Single Responsibility** : un module = un rôle (`audit/spectral.py` affiche,
  `audit/jacobian.py` audite la stabilité, `calibration/triage.py` décide) ;
  un fichier de predictor = une famille.
- **Open/Closed** : ajouter une famille de predictor, un système, ou un
  diagnostic = un fichier + une entrée de registry + ses tests de contrat.
  Ni l'audit, ni le trainer, ni les configs existantes ne changent (§2).
- **Liskov** : toute famille respecte le contrat `Predictor` (§2, `predictors/base.py`)
  — `rollout`, `local_operator` — sans exception cachée par famille ; le code
  client (`audit/`, `training/`, `viz/`) ne suppose jamais une famille précise.
- **Interface Segregation** : `audit/` ne connaît pas `planning/` ; `viz/` ne
  connaît que `results/`, jamais les objets vivants (modèle, optimiseur).
- **Dependency Inversion** : `calibration/` dépend de `audit/` et de
  `training/`, jamais l'inverse ; `experiments/` orchestre, il n'implémente
  aucune équation lui-même.

---

## 11. Vérification minimale

Dans le venv (`uv`, jamais le Python système), avant de considérer une wave ou
un composant terminé :

```bash
pytest -q
python -m compileall -q src tests
python3 /Users/captain/Documents/Projects/_publication/validate_publication.py \
    /Users/captain/Documents/Projects/structured-latent-dynamics
```

- Le validateur doit sortir **0 FAIL** — gate obligatoire, **jamais
  `--no-verify`, jamais de contournement, jamais désactivé**.
- Gold tests (§7) au vert avant tout run de grille — un gold test rouge bloque
  la grille, pas seulement le CI.
- Stress test d'invariance (§4/§7) au vert avant qu'un diagnostic entre dans
  une figure.
- Notebook : JSON valide, outputs nettoyés, `execution_count: null`
  (`scripts/check_notebooks.py`), pas de hack `sys.path`, parité package
  vérifiée (§8, niveau 3).

---

## 12. Règle finale

Quand deux choix sont possibles, choisir celui qui rend le résultat plus
**vérifiable** : plus facile à reproduire, à auditer, à réfuter. Ce repo vend
une seule chose — la fiabilité de celui qui mesure proprement ce que les autres
affirment. Chaque raccourci la dépense ; chaque vérification la capitalise.
