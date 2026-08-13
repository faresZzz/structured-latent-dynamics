# %% [markdown]
# # 02 — Les figures du dépôt, régénérées depuis les artefacts
#
# Ce notebook ne calcule rien et n'entraîne rien. Il **relit les artefacts de
# verdict déjà écrits sur disque** et rend chaque figure, puis en donne la
# lecture — ce qu'on voit, et ce qu'on ne peut pas en conclure.
#
# La règle qui gouverne ce fichier est `REGLES.md §2` : *une figure se
# régénère depuis `results*/`, jamais depuis un objet d'entraînement vivant*.
# C'est pourquoi `sld.viz.ingest` n'importe pas `torch` : un module de figure
# qui pourrait importer un predictor pourrait, un refactor plus tard, en
# réentraîner un.
#
# **Contrat de ce notebook** : si le recalcul depuis les artefacts ne reproduit
# pas la table que la figure cite, **aucune figure ne sort et l'écart devient
# le résultat**. Ce n'est pas une politesse : c'est le traitement que §19.8 a
# déjà appliqué au défaut de câblage de `grad_clip`. La vérification est
# exécutée ci-dessous, avant le premier rendu.

# %%
from pathlib import Path

import pandas as pd

from sld.viz.ingest import arm_label, load_d3_gold, load_gate

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
pd.set_option("display.width", 120)

# %% [markdown]
# ## 1. La vérification, avant toute figure
#
# `PROTOCOL.md §21.0` publie, par cellule, les médianes sur cinq seeds de
# $e_1$, $m_\text{low}$ et $m_\text{high}$, **et** le nombre de seeds qui
# franchissent chaque borne individuellement (§12.4 : pas de réducteur). On les
# recalcule depuis les 40 `gate.json`.

# %%
PUBLISHED = {
    ("OU", "mlp"): (0.0106, 0.505, 0.540, 5, 5),
    ("OU", "transformer"): (0.0229, 0.723, 1.172, 5, 1),
    ("OU", "mlp_k"): (0.0520, 1.148, 2.656, 1, 0),
    ("OU", "transformer_k"): (0.0492, 1.112, 2.516, 1, 0),
    ("AC", "mlp"): (0.0119, 0.966, 0.777, 3, 5),
    ("AC", "transformer"): (0.0297, 0.854, 1.934, 5, 0),
    ("AC", "mlp_k"): (0.0438, 1.690, 2.857, 0, 0),
    ("AC", "transformer_k"): (0.0436, 1.502, 2.841, 0, 0),
}

scalars, curves = load_gate(REPO / "results_gate_d")
for frame in (scalars, curves):
    frame["arm"] = [arm_label(b, s) for b, s in zip(frame.backbone, frame.structure, strict=True)]

rows = []
for (stratum, arm), sub in scalars.groupby(["stratum", "arm"]):
    e1, mlow, mhigh, n_lo, n_up = PUBLISHED[(stratum, arm)]
    rows.append(
        {
            "strate": stratum,
            "bras": arm,
            "e_1": sub.e_1.median(),
            "e_1 publié": e1,
            "m_low": sub.m_low.median(),
            "m_low publié": mlow,
            "m_high": sub.m_high.median(),
            "m_high publié": mhigh,
            "accord": (
                abs(sub.e_1.median() - e1) < 5e-5
                and abs(sub.m_low.median() - mlow) < 5e-4
                and abs(sub.m_high.median() - mhigh) < 5e-4
                and int(sub.lower_bound.sum()) == n_lo
                and int(sub.upper_bound.sum()) == n_up
            ),
        }
    )
reconciliation = pd.DataFrame(rows)
assert reconciliation["accord"].all(), "ÉCART avec §21.0 — aucune figure ne sort, l'écart est le résultat"
reconciliation

# %% [markdown]
# **Lecture.** Les huit cellules se reproduisent, médianes **et** comptes de
# seeds. La chaîne de mesure qui alimente les figures est donc la même que
# celle qui a écrit le protocole. Le `assert` ci-dessus est le verrou : si un
# refactor déplaçait un chiffre, ce notebook s'arrêterait ici.
#
# Le même contrôle vaut pour D3 (`§20.0` : 10 runs, 10/10 sur le gain,
# 0/10 sur les trois bornes d'exactitude).

# %%
gold = load_d3_gold(REPO / "results_d3_oscillator")
print(f"D3 : {len(gold)} runs, gain {int(gold.gain_passes.sum())}/{len(gold)}")
for col in ("state_h100", "spectral", "action"):
    n = int((gold[col] < gold[f"threshold_{col.split('_')[0]}"]).sum())
    print(f"  {col:11s} médiane {gold[col].median():.4f}  seuil {gold[f'threshold_{col.split(chr(95))[0]}'].iloc[0]:g}  -> {n}/{len(gold)} passent")

# %% [markdown]
# ## 2. Figure A — le contrôle D3 : ils apprennent, ils n'identifient pas
#
# Quatre panneaux, quatre axes séparés, délibérément : `§19.2` grave que *la
# borne spectrale ne doit jamais être rapportée comme si elle résumait les
# autres*. Les mettre sur un axe commun ferait exactement cela.
#
# Les lignes horizontales sont des **ancres analytiques** (`§20.1`), jamais des
# mesures de ces runs : le gold planté à $1.9\times10^{-15}$ (l'optimum existe
# dans la classe d'hypothèses), les moindres carrés sur le même split, et le
# seuil pré-enregistré $10^{-3}$.

# %%
from sld.viz.figA_d3_exactness import make_figA

print(make_figA(REPO / "results_d3_oscillator", REPO / "assets/figA_d3_exactness.png"))

# %% [markdown]
# **Lecture.** Panneau 1 : les dix cellules battent les baselines triviales
# d'environ $32\times$ — elles apprennent, et la porte relative est franchie
# 10/10. Panneaux 2 à 4 : les trois bornes d'exactitude absolue sont manquées
# de **110 à 200 fois** le seuil pré-enregistré.
#
# Ce qu'on **ne peut pas** dire : « 200 fois plus loin que les moindres
# carrés ». L'écart réel aux MCO est de l'ordre de $5\times10^{6}$, et la
# légende se contredirait. Ce qu'on **peut** dire, et c'est le résultat : sur un
# système linéaire dont la solution exacte est dans la classe d'hypothèses *et*
# récupérable en forme close, le régime d'entraînement déclaré ne l'atteint pas.
#
# Portée, à ne jamais élargir : cet énoncé porte sur **cette** paramétrisation,
# **ce** régime (rollout fermé 25 pas), **cet** optimiseur. Il ne dit rien des
# priors structurés en général.

# %% [markdown]
# ## 3. Figure B — la grille D4, lue à travers l'horizon
#
# `§21.0` rapporte les marges de porte à un horizon unique et conclut que
# `structure: on` coûte $1.54\times$ à $2.27\times$. C'est exact, et ce n'est
# pas tout ce que portent les artefacts : les mêmes `gate.json` contiennent
# $e_h$ pour tout $h \in \{1, \dots, 25\}$.

# %%
from sld.viz.figB_horizon import make_figB

print(make_figB(REPO / "results_gate_d", REPO / "assets/figB_horizon.png"))

e1 = curves[curves.h == 1].set_index(["stratum", "arm", "seed"])["e_h"]
e25 = curves[curves.h == 25].set_index(["stratum", "arm", "seed"])["e_h"]
rho = (e25 / e1).groupby(level=[0, 1]).median().rename("rho = e_25 / e_1")
rho.to_frame()

# %% [markdown]
# **Lecture.** Le taux de composition $\rho = e_{25}/e_1$ vaut, en médiane :
#
# | strate | OFF | ON | rapport |
# |---|---|---|---|
# | OU `mlp` | 6.24 | 3.03 | **2.06×** |
# | OU `transformer` | 4.26 | 2.66 | **1.60×** |
# | AC `mlp` | 8.58 | 3.92 | **2.19×** |
# | AC `transformer` | 3.42 | 3.72 | 0.92× |
#
# Dans trois paires sur quatre, le bras structuré part d'une erreur à un pas
# plus grande et **compose plus lentement**. C'est, mot pour mot, la question de
# recherche affichée du dépôt.
#
# **Deux confondants, ni l'un ni l'autre réparable depuis ces artefacts**, et
# aucune lecture de cette figure ne doit les omettre :
#
# 1. l'asymétrie de taux de `§21.2` — chaque bras OFF a tourné à un taux choisi
#    pour lui, chaque bras ON à un taux choisi pour une *autre* architecture ;
# 2. `structure: on` est un **composite** — la transition structurée *et* le
#    lift $\psi$ *et* la projection $\pi$, qui prennent ensemble 10.3 % à
#    20.4 % du budget de paramètres (`§13.7`). Les panneaux ne séparent pas les
#    trois.
#
# C'est pourquoi ce chiffre est présenté comme un **signal à mesurer
# proprement**, pas comme un résultat : les campagnes de `§23` existent pour ça.

# %% [markdown]
# ## 4. Figures C à F — déclarées, pas encore rendues
#
# Elles attendent leurs racines de résultats, déclarées dans `PROTOCOL.md §23`
# et lançables depuis `RUNBOOK.md`. Aucune n'est rendue avant que sa racine
# existe : une figure sur des données absentes est le contraire de ce que ce
# dépôt vend.
#
# | figure | racine | ce qu'elle porte |
# |---|---|---|
# | **C** | `results_osc_n/` | gain de structure contre taille d'échantillon, trois niveaux, 5 seeds en points individuels |
# | **D** | `results_osc_delta/` | gain contre le défaut de représentabilité $\delta(c)$, abscisse **calculée en forme close** |
# | **E** | `results_ssm_delta/` | le verdict `ssm` : la famille s'entraîne-t-elle sous un $\Delta_{\max}$ physiquement borné ? |
# | **F** | `results_osc_n/` | $\kappa(\partial x_H / \partial u_{0:H-1})$ par bras — la seule quantité du dossier qui touche le chaînon planification |
#
# La règle de lecture primaire de chacune est **écrite avant les chiffres**
# (`§23.5`) : c'est le **signe** d'une interaction et sa taille d'effet, jamais
# un franchissement de seuil. Si les intervalles se recouvrent partout, le
# verdict publié est « aucune interaction détectée à 5 seeds sur ce système »,
# **et la figure sort quand même**.

# %%
for root, figure in [
    ("results_osc_n", "C et F"),
    ("results_osc_delta", "D"),
    ("results_ssm_delta", "E"),
]:
    present = (REPO / root).exists()
    print(f"  {root:22s} {'présente' if present else 'absente'}  -> figure(s) {figure}")
