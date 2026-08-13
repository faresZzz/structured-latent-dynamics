# %% [markdown]
# # Dynamique latente structurée — ce qu'un prior spectral achète, et ce qu'il ne peut pas acheter
#
# *Notebook de cours du dépôt `structured-latent-dynamics`. Farès Zaghouane.*
#
# ---
#
# ## 0. Ce que ce notebook démontre
#
# La question du dépôt, telle qu'elle est affichée dans son `README`, est une question
# **conditionnelle** :
#
# > *Quand une structure dynamique explicite dans le predictor latent d'un world model JEPA
# > améliore-t-elle la stabilité de rollout, la suffisance d'état et la planifiabilité — et à
# > quel coût par décision réussie ?*
#
# Le mot important est **quand**. Ce notebook n'essaie pas de montrer qu'un prior structurel est
# bon. Il essaie de délimiter, aussi précisément que possible, la frontière entre les régimes où
# un tel prior peut payer et ceux où il ne le peut pas — et de séparer, à l'intérieur de cette
# frontière, ce qui est **démontré**, ce qui est **mesuré**, et ce qui reste **supposé**.
#
# ### 0.1 Les trois résultats mesurés que ce notebook explique
#
# Trois faits, tirés du dépôt, organisent tout le reste.
#
# 1. **§4 — Exprimable n'est pas atteignable.** Sur un système linéaire dont la solution exacte
#    appartient à la classe d'hypothèses du modèle *et* se récupère en forme close par moindres
#    carrés, le régime d'entraînement déclaré atterrit environ deux décades au-dessus du seuil
#    d'exactitude pré-enregistré. Quatre explications concurrentes ont été testées et éliminées,
#    chacune par sa propre mesure.
# 2. **§5 — Une dégénérescence de jauge existe, et elle est exacte.** La cellule structurée du
#    dépôt possède $2r$ directions de perte exactement plates, par construction. C'est un
#    théorème, vérifié numériquement à la précision machine. Son effet sur ce que la descente de
#    gradient atteint est, lui, **non mesuré** — et le dépôt a retiré une conclusion antérieure
#    qui allait trop vite sur ce point.
# 3. **§6 — La taille d'échantillon est l'axe manquant.** À 24 480 transitions pour un LTI à 4
#    états, le problème est sur-déterminé de trois ordres de grandeur : un biais inductif, même
#    exactement correct, n'a rien à y acheter. C'est le pire régime possible pour démontrer
#    l'utilité d'un prior, et c'est celui dans lequel la mesure a été faite.

# %% [markdown]
# ### 0.2 Le plan
#
# | § | Contenu | Nature |
# |---|---|---|
# | 1 | Pourquoi un predictor générique dérive en rollout libre | démonstration + mesure du dépôt |
# | 2 | L'algèbre du bloc rotation-scaling, et son piège numérique | démonstration + vérification |
# | 3 | Quand le prior est exactement correct, et quand il ne l'est pas | démonstration + calcul |
# | 4 | Exprimable n'est pas atteignable | mesure du dépôt + miniature illustrative |
# | 5 | Une dégénérescence de jauge dans la cellule structurée | démonstration + vérification |
# | 6 | Ce que la taille d'échantillon change | démonstration + expérience jouet |
# | 7 | Ce qu'on ne sait pas | inventaire |

# %% [markdown]
# ### 0.3 Convention de citation, appliquée sans exception
#
# Ce notebook distingue trois statuts, et le statut est écrit à chaque fois :
#
# - **[MESURÉ]** — un chiffre produit par un run du dépôt, avec sa source (`PROTOCOL.md §20.1`,
#   `§21.0`, …). Il n'est jamais recalculé ici ; il est cité.
# - **[DÉMONTRÉ]** — un énoncé mathématique établi dans le notebook, avec ses hypothèses, et
#   vérifié numériquement par la cellule qui suit.
# - **[ILLUSTRATIF]** — un chiffre produit par une simulation jouet écrite dans ce notebook. Il
#   sert à rendre un mécanisme visible. **Il ne reproduit aucun run** et n'a aucune valeur de
#   preuve sur le système réel.
#
# Aucun chiffre de ce notebook n'a d'autre statut. Là où une valeur serait nécessaire au propos
# et n'est pas disponible, c'est écrit explicitement.
#
# ### 0.4 Dépendances
#
# `numpy` et `matplotlib`, rien d'autre. Aucun import du paquet `sld`, aucun `torch` : le
# notebook doit tourner partout, y compris chez un lecteur qui n'a pas le dépôt. L'exécution
# complète prend moins d'une minute.

# %%
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True, linewidth=110)
plt.rcParams.update({
    "figure.figsize": (10.5, 3.8),
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
})

# Exponentielle matricielle par série tronquée : suffisante ici parce que toutes les matrices
# manipulées ont une norme petite (Delta = 0.05, rayons proches de 1). Elle sert de *référence
# indépendante* aux formes closes que nous allons dériver : si l'on validait une forme close
# contre elle-même, on ne validerait rien.
def expm_series(M, n_terms=100):
    """exp(M) par la série de Taylor tronquee. Reference numerique independante."""
    S = np.eye(M.shape[0])
    T = np.eye(M.shape[0])
    for k in range(1, n_terms + 1):
        T = T @ M / k
        S = S + T
    return S


def rel_err(a, b):
    """Erreur relative en norme infinie, avec garde contre le zero."""
    den = max(np.abs(b).max(), 1e-300)
    return float(np.abs(a - b).max() / den)


print("numpy", np.__version__)
print("verification de expm_series contre un cas analytique connu :")
_G = np.array([[0.0, -1.0], [1.0, 0.0]])           # generateur de rotation
_ana = np.array([[np.cos(0.7), -np.sin(0.7)], [np.sin(0.7), np.cos(0.7)]])
print("  |expm_series(0.7*J) - R(0.7)|_inf =", np.abs(expm_series(0.7 * _G) - _ana).max())

# %% [markdown]
# ## 1. Pourquoi un predictor générique dérive
#
# ### 1.1 Deux objectifs qu'on confond, et qui ne sont pas le même
#
# Un predictor latent est une application $f$ qui avance l'état latent d'un pas :
# $\hat z_{t+1} = f(\hat z_t)$. On l'entraîne presque toujours sur une **erreur à un pas**
#
# $$\mathcal{L}_1(f) \;=\; \mathbb{E}\big[\;\|f(z_t) - z_{t+1}\|^2\;\big],$$
#
# où $z_t$ est un état **vrai**, fourni par l'encodeur. On s'en sert ensuite pour faire du
# **rollout libre** : on part de $\hat z_0 = z_0$, et on itère $f$ sur sa propre sortie, sans
# jamais réinjecter d'observation. L'erreur qui compte pour la planification est alors
#
# $$\mathcal{L}_H(f) \;=\; \mathbb{E}\big[\;\|\underbrace{f\circ\cdots\circ f}_{H\ \text{fois}}(z_0) - z_H\|^2\;\big].$$
#
# Ces deux quantités ne mesurent pas la même chose, et le lien entre elles n'est pas une
# constante : c'est une **composition**. Établissons-la.

# %% [markdown]
# ### 1.2 La récurrence de l'erreur
#
# **Hypothèses.** (H1) La dynamique vraie est $z_{t+1} = g(z_t)$. (H2) Le predictor $f$ est
# différentiable. (H3) On travaille dans un voisinage où le développement au premier ordre est
# valide — c'est-à-dire tant que l'erreur reste petite devant la courbure de $f$. C'est
# l'hypothèse qui tombe en premier, et nous y reviendrons.
#
# Posons l'erreur $e_t = \hat z_t - z_t$. Alors
#
# $$e_{t+1} \;=\; f(\hat z_t) - z_{t+1} \;=\; \underbrace{f(\hat z_t) - f(z_t)}_{\text{propagation}} \;+\; \underbrace{f(z_t) - z_{t+1}}_{=:\ \varepsilon_t\ \text{(erreur locale)}}.$$
#
# Le second terme $\varepsilon_t = f(z_t) - g(z_t)$ est exactement ce que $\mathcal{L}_1$
# mesure : l'erreur du predictor **partant d'un état vrai**. Le premier terme, sous (H3), vaut
# $f(\hat z_t) - f(z_t) = J_t\,e_t + o(\|e_t\|)$ avec $J_t = \partial f/\partial z\,(z_t)$.
# D'où la récurrence
#
# $$\boxed{\;e_{t+1} \;=\; J_t\,e_t \;+\; \varepsilon_t\;}$$
#
# C'est une **récurrence linéaire affine**, et elle se déroule exactement. Avec $e_0 = 0$
# (on part d'un état vrai) :
#
# $$e_1 = \varepsilon_0,\quad e_2 = J_1\varepsilon_0 + \varepsilon_1,\quad
# e_3 = J_2J_1\varepsilon_0 + J_2\varepsilon_1 + \varepsilon_2,\quad\ldots$$
#
# et par récurrence immédiate
#
# $$e_H \;=\; \sum_{k=0}^{H-1}\Big(\prod_{j=k+1}^{H-1} J_j\Big)\,\varepsilon_k .$$

# %% [markdown]
# ### 1.3 La borne géométrique
#
# Prenons une norme sous-multiplicative $\|\cdot\|$ (norme d'opérateur induite, par exemple),
# supposons (H4) $\|J_t\| \le \eta$ pour tout $t$ et (H5) $\|\varepsilon_t\| \le \varepsilon$
# pour tout $t$. L'inégalité triangulaire puis la sous-multiplicativité donnent
#
# $$\|e_H\| \;\le\; \sum_{k=0}^{H-1} \Big\|\prod_{j=k+1}^{H-1} J_j\Big\|\,\|\varepsilon_k\|
# \;\le\; \sum_{k=0}^{H-1} \eta^{\,H-1-k}\,\varepsilon
# \;=\; \varepsilon \sum_{m=0}^{H-1} \eta^{\,m}.$$
#
# La somme géométrique se ferme :
#
# $$\boxed{\;\|e_H\| \;\le\; \varepsilon \cdot \frac{\eta^{H}-1}{\eta-1}\quad (\eta \neq 1),
# \qquad \|e_H\| \;\le\; \varepsilon H \quad (\eta = 1).\;}$$
#
# Trois régimes, et ce sont eux qui décident de tout :
#
# | régime | comportement de $\|e_H\|$ | conséquence |
# |---|---|---|
# | $\eta < 1$ | borné par $\varepsilon/(1-\eta)$ | l'erreur **sature** : le rollout long est utilisable |
# | $\eta = 1$ | croît en $\varepsilon H$ | dérive **linéaire** |
# | $\eta > 1$ | croît en $\varepsilon\,\eta^{H}/(\eta-1)$ | dérive **géométrique** : rollout inutilisable |
#
# **Ce que cela dit du problème.** Deux predictors avec la même $\mathcal{L}_1$ — la même
# $\varepsilon$ — peuvent avoir des $\mathcal{L}_H$ séparées par des ordres de grandeur, si
# leurs $\eta$ diffèrent. Minimiser $\mathcal{L}_1$ ne contrôle **pas** $\eta$. C'est le trou
# exact dans lequel un prior structurel prétend s'insérer : une paramétrisation où $\eta$ est
# lisible, voire bornée par construction.

# %% [markdown]
# ### 1.4 Quand la borne est serrée, quand elle est lâche
#
# La borne devient une **égalité** quand les deux inégalités utilisées le deviennent :
#
# 1. l'inégalité triangulaire est saturée si tous les termes $\big(\prod J\big)\varepsilon_k$
#    sont **colinéaires et de même sens** ;
# 2. $\|J^m v\| = \eta^m\|v\|$ demande que $v$ soit aligné sur la direction de gain maximal de
#    $J$, et que cette direction soit **invariante** par $J$.
#
# Les deux tiennent simultanément si $J = \eta\,\mathrm{Id}$ et si les $\varepsilon_k$ sont tous
# égaux. C'est le pire cas, et il n'est pas pathologique : une erreur de modèle **systématique**
# (un biais, pas un bruit) est exactement un $\varepsilon_k$ constant.
#
# Elle est **très lâche** dans deux cas fréquents :
#
# - $\varepsilon_k$ de moyenne nulle et indépendants : les termes s'additionnent alors en
#   variance, pas en norme, et $\mathbb{E}\|e_H\|^2 = \varepsilon^2\sum_m \|J^m\|_F^2$ — la
#   racine carrée d'une somme au lieu d'une somme, soit un facteur $\sqrt{H}$ gagné dans le cas
#   $\eta = 1$ ;
# - $J$ non normale : $\|J\|$ peut valoir beaucoup plus que le rayon spectral $\rho(J)$. La
#   borne utilise $\|J\|$, mais le comportement asymptotique est piloté par $\rho(J)$. Entre les
#   deux vit la **croissance transitoire** — l'erreur monte puis redescend, et une borne en
#   $\|J\|^H$ ne voit jamais la redescente.
#
# Vérifions les trois régimes, et l'écart entre borne au pire cas et réalisation stochastique.

# %%
# --- [ILLUSTRATIF] Simulation de la recurrence e_{t+1} = J e_t + eps_t --------------------
# Deux questions distinctes : (a) la borne de §1.3 est-elle ATTEINTE dans le cas ou §1.4 dit
# qu'elle doit l'etre ? (b) de combien est-elle lache dans les deux cas ou §1.4 dit qu'elle
# doit l'etre ? On ne trace pas "une courbe d'erreur" : on teste deux predictions.

def rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def bound(eta, H):
    hs = np.arange(1, H + 1)
    return EPS * (hs if np.isclose(eta, 1.0) else (eta ** hs - 1) / (eta - 1))


def simulate(J, H, mode, n_traj=2000, seed=0):
    """mode='syst' : eps constant et aligne ; mode='alea' : eps de moyenne nulle, meme norme."""
    r = np.random.default_rng(seed)
    E = np.zeros((n_traj, 2))
    v = np.array([1.0, 0.0]) * EPS
    out = []
    for _ in range(H):
        if mode == "syst":
            d = np.broadcast_to(v, E.shape)
        else:
            d = r.standard_normal(E.shape)
            d = EPS * d / np.linalg.norm(d, axis=1, keepdims=True)
        E = E @ J.T + d
        out.append(np.linalg.norm(E, axis=1).mean())
    return np.array(out)


H, EPS = 40, 1e-3
etas = [0.90, 1.00, 1.05, 1.20]
cols = plt.cm.viridis(np.linspace(0.12, 0.82, len(etas)))
hs = np.arange(1, H + 1)

fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))

# (a) le cas ou la borne DOIT etre atteinte : J = eta * Id, erreur locale constante et alignee
print("(a) cas de saturation predit par §1.4 : J = eta*Id, eps constant")
for eta, col in zip(etas, cols):
    sim = simulate(eta * np.eye(2), H, "syst", n_traj=1)
    bd = bound(eta, H)
    axes[0].semilogy(hs, bd, "-", color=col, lw=4.0, alpha=0.30)
    axes[0].semilogy(hs, sim, "-", color=col, lw=1.3, label=rf"$\eta={eta:.2f}$")
    print(f"    eta = {eta:.2f} : ecart relatif max simulation/borne = "
          f"{np.abs(sim - bd).max() / bd.max():.2e}")
axes[0].set(xlabel="horizon $H$", ylabel=r"$\|e_H\|$",
            title="(a) trait epais : borne de §1.3 ; trait fin : simulation")
axes[0].legend(ncol=2)

# (b) les deux cas ou §1.4 predit que la borne est LACHE, a eta fixe
ETA_B = 1.05
bd = bound(ETA_B, H)
variantes = [
    ("borne §1.3", r"borne §1.3", bd, "k", "-", 3.0, 0.35),
    ("J = eta*Id, eps systematique", r"$J=\eta\,$Id, $\varepsilon$ systematique",
     simulate(ETA_B * np.eye(2), H, "syst", 1), "tab:blue", "-", 1.6, 1.0),
    ("J = eta*R(0.35), eps systematique", r"$J=\eta R(0.35)$, $\varepsilon$ systematique",
     simulate(ETA_B * rot(0.35), H, "syst", 1), "tab:orange", "-", 1.6, 1.0),
    ("J = eta*Id, eps de moyenne nulle", r"$J=\eta\,$Id, $\varepsilon$ de moyenne nulle",
     simulate(ETA_B * np.eye(2), H, "alea"), "tab:red", "--", 1.6, 1.0)]
for _plain, lab, y, c, ls, lw, al in variantes:
    axes[1].semilogy(hs, y, ls, color=c, lw=lw, alpha=al, label=lab)
axes[1].set(xlabel="horizon $H$", ylabel=r"$\|e_H\|$",
            title=rf"(b) a $\eta={ETA_B}$ fixe : de combien la borne est-elle lache ?")
axes[1].legend(fontsize=7.5)
plt.tight_layout(); plt.show()

print(f"\n(b) rapport borne / erreur realisee a H = 25, pour eta = {ETA_B} :")
for plain, _lab, y, *_ in variantes[1:]:
    print(f"    {plain:>36} : {bd[24] / y[24]:6.2f} x")

print("\ntaux de composition rho_25 = ||e_25|| / eps (cas de saturation) :")
for eta in etas:
    sim = simulate(eta * np.eye(2), 25, "syst", n_traj=1)
    print(f"    eta = {eta:.2f} : rho_25 = {sim[-1] / EPS:9.2f}   borne = {bound(eta, 25)[-1] / EPS:9.2f}")

# %% [markdown]
# **Lecture.** Le panneau (a) teste la prédiction de §1.4 : *la borne est atteinte si
# $J = \eta\,\mathrm{Id}$ et si l'erreur locale est un biais constant*. Elle l'est — l'écart
# relatif entre simulation et borne est de l'ordre de $10^{-16}$ pour les quatre valeurs de
# $\eta$, c'est-à-dire de l'arrondi. La borne de §1.3 n'est donc pas une majoration grossière :
# c'est la **solution exacte** de la récurrence dans ce cas. On lit aussi les trois régimes
# annoncés : saturation pour $\eta = 0.90$, croissance linéaire pour $\eta = 1.00$, croissance
# géométrique au-dessus.
#
# Le panneau (b) teste l'autre moitié de §1.4, à $\eta$ fixé : *la borne est lâche si les
# contributions ne s'alignent pas*. Les deux mécanismes prévus sont visibles et **quantifiés**
# dans la sortie. Faire tourner $J$ d'un angle de $0.35$ rad par pas suffit à décoller la
# courbe de la borne : les termes $J^m\varepsilon$ pointent dans des directions différentes et
# s'annulent partiellement. Une erreur locale de moyenne nulle décolle davantage encore :
# les contributions s'additionnent alors en variance, pas en norme.
#
# **Ce qu'on ne peut pas conclure de cette figure.** (i) Rien sur un predictor appris : $J$ y
# dépend de l'état, et l'hypothèse (H3) — le régime linéaire — tombe dès que l'erreur devient
# grande devant la courbure. (ii) Rien sur le classement de deux architectures : $\eta$ et
# $\varepsilon$ y varient ensemble, et le produit peut aller dans les deux sens. (iii) Rien sur
# le fait qu'un prior structurel *diminue* $\eta$ — c'est une propriété de paramétrisation,
# pas une conséquence de cette figure. Ce que la figure établit est plus modeste et plus utile :
# **le taux de composition est piloté par un seul scalaire**, et ce scalaire n'apparaît nulle
# part dans $\mathcal{L}_1$.
# %% [markdown]
# ### 1.5 Le fait mesuré : le taux de composition des bras structurés
#
# Le dépôt a couru, en phase D, 40 runs sur le pendule amorti, dans un plan factoriel
# backbone $\times$ structure $\times$ strate de données. La grandeur de §1.4 — le taux de
# composition $\rho = e_{25}/e_1$ — s'y lit directement, puisque les deux erreurs sont
# journalisées par run.
#
# **[MESURÉ — `PROTOCOL.md §21.0`, 40 runs, médiane sur 5 seeds]**
#
# | strate | backbone | structure OFF | structure ON | rapport OFF/ON |
# |---|---|---|---|---|
# | OU | `mlp` / `mlp_k` | 6.24 | 3.03 | (calculé ci-dessous) |
# | OU | `transformer` / `transformer_k` | 4.26 | 2.66 | |
# | AC | `mlp` / `mlp_k` | 8.58 | 3.92 | |
# | AC | `transformer` / `transformer_k` | 3.42 | 3.72 | |
#
# Le suffixe `_k` désigne le bras à structure ON. Les rapports sont recalculés dans la cellule
# suivante plutôt que recopiés, pour qu'aucune arithmétique ne soit prise sur parole.

# %%
# --- [MESURE citee, arithmetique derivee ici] ---------------------------------------------
# Source : PROTOCOL.md §21.0 (medianes sur 5 seeds). Les valeurs de rho sont CITEES ;
# seuls les rapports sont calcules.
rho_mesure = {
    ("OU", "mlp"):         (6.24, 3.03),
    ("OU", "transformer"): (4.26, 2.66),
    ("AC", "mlp"):         (8.58, 3.92),
    ("AC", "transformer"): (3.42, 3.72),
}

print(f"{'strate':>7} {'backbone':>12} {'rho OFF':>9} {'rho ON':>8} {'OFF/ON':>8}   verdict")
print("-" * 66)
gains = []
for (strate, bb), (off, on) in rho_mesure.items():
    r = off / on
    gains.append(r)
    verdict = "ON compose plus lentement" if r > 1 else "ON compose PLUS VITE"
    print(f"{strate:>7} {bb:>12} {off:9.2f} {on:8.2f} {r:8.2f}   {verdict}")

favorables = [g for g in gains if g > 1]
print("-" * 66)
print(f"paires ou le bras structure compose plus lentement : {len(favorables)}/4")
print(f"amplitude sur ces paires : {min(favorables):.2f}x a {max(favorables):.2f}x")
print(f"paire defavorable : rapport {min(gains):.2f} (le bras structure compose PLUS VITE)")

# %% [markdown]
# **Lecture.** Dans 3 paires sur 4, le bras structuré compose son erreur **1.6× à 2.2× plus
# lentement** que son homologue non structuré. C'est le signe attendu si la structure agit sur
# $\eta$ — mais le lien n'est pas établi : personne n'a mesuré $\eta$ dans ces runs, seulement
# ses conséquences. La quatrième paire (`AC`, `transformer`) va dans l'autre sens, et elle est
# publiée telle quelle.
#
# **Deux confondants, qui doivent être nommés avant toute lecture causale.**
#
# 1. **Asymétrie de taux d'apprentissage** (`PROTOCOL.md §21.2`). Les bras OFF ont reçu un taux
#    sélectionné pour eux ; les bras ON ont hérité du taux d'une autre architecture. Un bras
#    peut donc apparaître meilleur *ou* pire pour une raison qui n'a rien à voir avec la
#    structure. Le dépôt a ouvert un re-run (§21.3) précisément pour réparer ce point ; ce
#    notebook ne dispose pas de son verdict.
# 2. **« Structure ON » est un composite.** Le bras ON ne change pas un seul facteur : il
#    remplace la transition **et** ajoute un lift $\psi$ **et** ajoute une projection $\pi$. Un
#    effet observé n'est attribuable à aucun des trois isolément. C'est exactement le défaut qui
#    a invalidé un axe antérieur du dépôt (l'axe $\mu_J$, conservé en matériel historique).
#
# **Statut de l'énoncé, en une phrase.** *Sur ce système, dans ces cellules, avec ces deux
# confondants non contrôlés, le signe du taux de composition favorise la structure 3 fois sur
# 4.* Ce n'est pas « la structure stabilise le rollout ». La différence entre les deux phrases
# est tout l'objet de ce notebook.
#
# ---

# %% [markdown]
# ## 2. L'algèbre du bloc rotation-scaling
#
# Le prior utilisé par le dépôt n'est pas une pénalité ajoutée à la perte : c'est une
# **paramétrisation**. La transition latente y est bloc-diagonale, chaque bloc étant $2\times 2$
# et engendré par
#
# $$G \;=\; \alpha\,I \;+\; \omega\,J,
# \qquad I=\begin{pmatrix}1&0\\0&1\end{pmatrix},\quad
# J=\begin{pmatrix}0&-1\\1&0\end{pmatrix}.$$
#
# Deux nombres réels par bloc, $(\alpha, \omega)$. Cette section établit tout ce qu'on peut dire
# de ce bloc en forme close, parce que c'est ce qui rend le prior **lisible** : après
# entraînement, on lit un taux et une pulsation, pas 4 coefficients opaques.
#
# ### 2.1 $(I, J)$ est isomorphe à $(1, i)$
#
# **Observation de départ.** $J^2 = \begin{pmatrix}0&-1\\1&0\end{pmatrix}^2
# = \begin{pmatrix}-1&0\\0&-1\end{pmatrix} = -I$.
#
# **Proposition.** L'application $\Phi : \alpha + i\omega \mapsto \alpha I + \omega J$ est un
# isomorphisme d'algèbres de $\mathbb{C}$ vers la sous-algèbre
# $\mathcal{A} = \{\alpha I + \omega J : \alpha,\omega\in\mathbb{R}\} \subset \mathbb{R}^{2\times 2}$.
#
# *Démonstration.* $\Phi$ est clairement $\mathbb{R}$-linéaire et bijective sur son image
# ($I$ et $J$ sont linéairement indépendantes). Reste la multiplicativité. Pour
# $z = \alpha + i\omega$ et $z' = \alpha' + i\omega'$ :
#
# $$\Phi(z)\Phi(z') = (\alpha I + \omega J)(\alpha' I + \omega' J)
# = \alpha\alpha' I + (\alpha\omega' + \omega\alpha') J + \omega\omega' J^2 .$$
#
# Avec $J^2 = -I$ :
#
# $$\Phi(z)\Phi(z') = (\alpha\alpha' - \omega\omega')\,I + (\alpha\omega' + \omega\alpha')\,J
# = \Phi\big((\alpha\alpha' - \omega\omega') + i(\alpha\omega' + \omega\alpha')\big) = \Phi(zz').\;\square$$
#
# **Conséquence immédiate.** Toute identité analytique valable sur $\mathbb{C}$ et exprimable
# par une série entière à coefficients réels se transporte mot pour mot dans $\mathcal{A}$. En
# particulier l'exponentielle.

# %% [markdown]
# ### 2.2 La forme close de $\exp(\Delta G)$
#
# Écrivons la série, et séparons les puissances paires et impaires. Posons $g = \alpha + i\omega$,
# de sorte que $G = \Phi(g)$ et $G^n = \Phi(g^n)$.
#
# $$\exp(\Delta G) = \sum_{n\ge 0} \frac{\Delta^n G^n}{n!} = \sum_{n\ge 0}\frac{\Delta^n \Phi(g^n)}{n!}
# = \Phi\!\left(\sum_{n\ge 0}\frac{(\Delta g)^n}{n!}\right) = \Phi\big(e^{\Delta g}\big),$$
#
# l'échange somme/$\Phi$ étant licite par continuité et linéarité de $\Phi$ (dimension finie, la
# série converge absolument). Or $e^{\Delta g} = e^{\Delta\alpha}e^{i\Delta\omega}
# = e^{\Delta\alpha}\big(\cos(\Delta\omega) + i\sin(\Delta\omega)\big)$. Donc
#
# $$\boxed{\;\exp(\Delta G) \;=\; e^{\Delta\alpha}\Big(\cos(\Delta\omega)\,I + \sin(\Delta\omega)\,J\Big)
# \;=\; e^{\Delta\alpha}\,R(\Delta\omega),\qquad R(\varphi)=\begin{pmatrix}\cos\varphi & -\sin\varphi\\ \sin\varphi & \cos\varphi\end{pmatrix}.}$$
#
# *(Preuve alternative sans passer par $\mathbb{C}$, pour qui préfère : $G = \alpha I + \omega J$
# où $\alpha I$ commute avec tout, donc $\exp(\Delta G) = e^{\Delta\alpha}\exp(\Delta\omega J)$ ;
# puis $\exp(\varphi J) = \sum_k \frac{\varphi^{2k}J^{2k}}{(2k)!} + \sum_k\frac{\varphi^{2k+1}J^{2k+1}}{(2k+1)!}
# = I\sum_k\frac{(-1)^k\varphi^{2k}}{(2k)!} + J\sum_k\frac{(-1)^k\varphi^{2k+1}}{(2k+1)!}
# = \cos\varphi\,I + \sin\varphi\,J$, en utilisant $J^{2k}=(-1)^kI$ et $J^{2k+1}=(-1)^kJ$.)*

# %% [markdown]
# ### 2.3 Ce que les deux paramètres signifient
#
# La matrice discrète $\bar A = \exp(\Delta G) = e^{\Delta\alpha}R(\Delta\omega)$ est le produit
# d'un scalaire positif et d'une rotation. Ses valeurs propres sont
# $e^{\Delta\alpha}e^{\pm i\Delta\omega}$, donc :
#
# $$\boxed{\;r \;=\; \rho(\bar A) \;=\; \|\bar A\|_2 \;=\; e^{\Delta\alpha},
# \qquad \varphi \;=\; \Delta\omega. \;}$$
#
# Trois remarques qui font tout l'intérêt de la paramétrisation :
#
# 1. **Le rayon spectral est en forme close et exact.** Il n'est ni estimé ni borné : il est
#    *lu*. Cela relie directement §1 : le $\eta$ dont dépendait la composition de l'erreur est
#    ici $e^{\Delta\alpha}$, un scalaire par bloc, disponible à chaque pas d'entraînement.
# 2. **$\bar A$ est normale** ($\bar A^\top \bar A = e^{2\Delta\alpha}I$), donc
#    $\|\bar A\|_2 = \rho(\bar A)$ : il n'y a **pas** de croissance transitoire non normale à
#    l'intérieur d'un bloc. La borne de §1.3 y est donc exacte, pas conservatrice. C'est un
#    argument en faveur de la structure qui est souvent tu, et qui est propre.
# 3. **$\alpha$ et $\omega$ sont découplés** : $\alpha$ est un taux d'amortissement (dimension
#    $\mathrm{temps}^{-1}$), $\omega$ une pulsation (dimension $\mathrm{temps}^{-1}$ aussi, mais
#    angulaire). Le modèle continu sous-jacent $\dot h = Gh$ a pour solutions
#    $e^{\alpha t}(\cos\omega t, \sin\omega t)$ : une spirale.
#
# **Choix de design, et l'alternative écartée.** On aurait pu paramétrer directement les 4
# coefficients de $\bar A$ et *pénaliser* $\rho(\bar A)$. Cette alternative est écartée pour
# trois raisons : (i) $\rho$ n'est pas une fonction lisse des coefficients aux valeurs propres
# multiples, donc son gradient est mal défini là où il compte ; (ii) une pénalité est un
# compromis réglable, donc un hyperparamètre de plus dans un plan factoriel qui en a déjà
# beaucoup ; (iii) surtout, une pénalité ne garantit rien — la contrainte est **architecturale**
# ici, ce qui rend l'énoncé « la classe contient exactement ces dynamiques » vérifiable au lieu
# d'espéré. C'est la ligne du dépôt : `structure` est un facteur d'architecture, jamais un terme
# numérique dans la perte.
#
# Vérifions la forme close contre l'exponentielle par série, qui n'en sait rien.

# %%
# --- [DEMONTRE -> verifie] Forme close de exp(Delta G) contre la serie ---------------------
I2 = np.eye(2)
J2 = np.array([[0.0, -1.0], [1.0, 0.0]])
DELTA = 0.05                       # pas d'integration, celui du depot pour les oscillateurs

print("J @ J =\n", J2 @ J2, "\n-> J^2 = -I :", np.allclose(J2 @ J2, -I2), "\n")


def A_bar_closed(alpha, omega, dt=DELTA):
    """exp(dt * (alpha I + omega J)) en forme close."""
    return np.exp(dt * alpha) * (np.cos(dt * omega) * I2 + np.sin(dt * omega) * J2)


cases = [(-0.30, 2.10), (0.70, -1.30), (0.0, 3.0), (-5.0, 0.0), (0.0, 0.0), (2.0, 40.0)]
print(f"{'alpha':>8} {'omega':>8} | {'|close - serie|_inf':>21} | {'r = e^(D a)':>12} {'rho(Abar)':>11} {'|Abar|_2':>10}")
print("-" * 92)
for a, w in cases:
    G = a * I2 + w * J2
    Ac = A_bar_closed(a, w)
    As = expm_series(DELTA * G)
    rad = max(abs(np.linalg.eigvals(Ac)))
    print(f"{a:8.2f} {w:8.2f} | {np.abs(Ac - As).max():21.3e} | "
          f"{np.exp(DELTA*a):12.9f} {rad:11.9f} {np.linalg.svd(Ac, compute_uv=False)[0]:10.7f}")

# normalite : Abar^T Abar doit valoir e^{2 D alpha} I
a, w = -0.30, 2.10
Ac = A_bar_closed(a, w)
print("\nnormalite du bloc : |Abar^T Abar - e^(2 D alpha) I|_inf =",
      np.abs(Ac.T @ Ac - np.exp(2 * DELTA * a) * I2).max())

# %% [markdown]
# **Lecture.** La forme close et la série tronquée coïncident à $10^{-16}$ près, y compris au
# cas extrême $\omega = 40$ (soit $\Delta\omega = 2$ rad par pas, presque un tiers de tour) et
# au cas dégénéré $(\alpha,\omega)=(0,0)$ où $\bar A = I$. Les trois colonnes de droite
# confirment l'identité annoncée : rayon spectral, norme spectrale et $e^{\Delta\alpha}$ sont le
# **même nombre** à la précision machine — c'est la normalité du bloc, vérifiée en dernière
# ligne.
#
# **Ce qu'on ne peut pas conclure.** Rien sur la stabilité de l'entraînement : $r = e^{\Delta\alpha}$
# est exact mais **non borné** si $\alpha$ est laissé libre, ce qui est le choix du dépôt
# (`PROTOCOL.md §13.2` : *« no `-exp(nu)`, no `-softplus` »*). Un $r > 1$ sous un rollout fermé
# de $H$ pas amplifie le gradient par $r^H$ — à $r = 1.20$ et $H = 25$, un facteur $95$. La
# lisibilité du rayon et sa **maîtrise** sont deux choses différentes.

# %% [markdown]
# ### 2.4 L'intégrale ZOH, en forme close
#
# Un predictor latent doit absorber une action. Le modèle continu est
#
# $$\dot h(t) \;=\; G\,h(t) \;+\; B\,u(t).$$
#
# **Hypothèse de discrétisation (ZOH, *zero-order hold*).** L'action est constante sur
# l'intervalle : $u(t) = u_k$ pour $t \in [t_k, t_k+\Delta)$. C'est l'hypothèse standard en
# contrôle échantillonné, et elle est *exacte* — pas approchée — si le contrôleur tient
# effectivement sa commande entre deux pas, ce qui est le cas dans un environnement à pas de
# temps discret.
#
# La formule de variation de la constante donne alors, exactement :
#
# $$h(t_k+\Delta) \;=\; e^{\Delta G} h(t_k) \;+\; \left(\int_0^{\Delta} e^{sG}\,ds\right) B\,u_k
# \;=:\; \bar A\, h_k + \bar B\, u_k ,
# \qquad \bar A = e^{\Delta G},\quad \bar B = \Phi B,\quad \Phi := \int_0^{\Delta} e^{sG}\,ds .$$
#
# *(Dérivation : posons $y(s) = e^{-sG}h(t_k+s)$. Alors
# $y'(s) = -Ge^{-sG}h + e^{-sG}(Gh + Bu_k) = e^{-sG}Bu_k$, en utilisant que $G$ et $e^{-sG}$
# commutent. On intègre de $0$ à $\Delta$ : $y(\Delta) - y(0) = \big(\int_0^\Delta e^{-sG}ds\big)Bu_k$,
# puis on multiplie à gauche par $e^{\Delta G}$ et on change de variable $s \to \Delta - s$.)*
#
# **Calcul de $\Phi$ dans la base $(I,J)$.** Par l'isomorphisme de §2.1, $\Phi = \Phi_{\mathcal{A}}$
# où l'on calcule dans $\mathbb{C}$ avec $g = \alpha + i\omega$ :
#
# $$\int_0^{\Delta} e^{sg}\,ds \;=\; \frac{e^{\Delta g}-1}{g}\qquad (g \neq 0).$$
#
# On rationalise en multipliant par $\bar g = \alpha - i\omega$ :
#
# $$\frac{e^{\Delta g}-1}{g} \;=\; \frac{(e^{\Delta g}-1)(\alpha - i\omega)}{\alpha^2+\omega^2}.$$
#
# Avec $e^{\Delta g} - 1 = \big(e^{\Delta\alpha}\cos\Delta\omega - 1\big) + i\,e^{\Delta\alpha}\sin\Delta\omega$,
# le produit se développe et l'on identifie partie réelle (coefficient de $I$) et partie
# imaginaire (coefficient de $J$) :
#
# $$\boxed{\;\Phi \;=\; \frac{1}{\alpha^2+\omega^2}\Big[
# \underbrace{\big(\alpha(e^{\Delta\alpha}\cos\Delta\omega - 1) + \omega\,e^{\Delta\alpha}\sin\Delta\omega\big)}_{\text{coefficient de }I}\;I
# \;+\;
# \underbrace{\big(\alpha\,e^{\Delta\alpha}\sin\Delta\omega - \omega(e^{\Delta\alpha}\cos\Delta\omega - 1)\big)}_{\text{coefficient de }J}\;J\Big].}$$

# %% [markdown]
# ### 2.5 La singularité en $(0,0)$ est *removable* — et c'est un vrai piège numérique
#
# Le dénominateur $\alpha^2+\omega^2 = |g|^2$ s'annule en $(\alpha,\omega)=(0,0)$. **Ce n'est pas
# une singularité de $\Phi$**, seulement de son écriture. En effet, la série entière
#
# $$\frac{e^{\Delta g}-1}{g} \;=\; \frac{1}{g}\sum_{n\ge 1}\frac{(\Delta g)^n}{n!}
# \;=\; \sum_{n\ge 1}\frac{\Delta^n g^{\,n-1}}{n!}
# \;=\; \Delta + \frac{\Delta^2 g}{2} + \frac{\Delta^3 g^2}{6} + \cdots$$
#
# est **entière** en $g$ : elle converge partout, et vaut $\Delta$ en $g=0$ — ce qui est le
# résultat attendu, puisque $\int_0^\Delta e^{s\cdot 0}ds = \Delta$. La singularité est donc
# **effaçable** au sens de l'analyse complexe : la fonction se prolonge holomorphiquement.
#
# **Mais l'annulation numérique, elle, ne s'efface pas.** Quand $|g| \to 0$, le numérateur
# $e^{\Delta g}-1$ tend vers $0$ **et** le dénominateur $|g|^2$ aussi. On divise donc deux
# quantités qui ont perdu leurs chiffres significatifs par cancellation. Le mécanisme précis :
# $e^{\Delta\alpha}\cos(\Delta\omega) - 1$ est calculé comme une différence de deux nombres
# proches de $1$, ce qui détruit $\log_{10}(1/|\Delta g|)$ décimales avant même la division.
#
# **Correctif, et c'est celui à implémenter :** basculer sur la série tronquée quand
# $|g| = \sqrt{\alpha^2+\omega^2}$ passe sous un seuil. Le seuil se choisit en égalisant les deux
# sources d'erreur : la troncature à l'ordre $n$ laisse $O(\Delta^{n+1}|g|^{n}/(n+1)!)$, la
# cancellation coûte $O(\epsilon_{\text{mach}}/(\Delta|g|))$ en relatif. Avec 8 termes et
# $\Delta \sim 10^{-2}$, un seuil $|g| \sim 10^{-8}$ place les deux sous $10^{-15}$. Mesurons-le
# plutôt que de l'affirmer.

# %%
# --- [DEMONTRE -> verifie] Forme close de Phi, et le piege de la cancellation --------------
from math import factorial


def Phi_series(alpha, omega, dt=DELTA, n_terms=10):
    """Phi par serie entiere : sum_{n>=1} dt^n G^{n-1} / n!  -- valable partout, exacte en 0."""
    G = alpha * I2 + omega * J2
    S = dt * I2
    T = np.eye(2)
    for n in range(1, n_terms):
        T = T @ G
        S = S + dt ** (n + 1) / factorial(n + 1) * T
    return S


def Phi_naive(alpha, omega, dt=DELTA):
    """Forme close BRUTE : correcte en arithmetique exacte, instable pres de (0,0)."""
    n2 = alpha ** 2 + omega ** 2
    E, co, si = np.exp(dt * alpha), np.cos(dt * omega), np.sin(dt * omega)
    coef_I = alpha * (E * co - 1.0) + omega * E * si
    coef_J = alpha * E * si - omega * (E * co - 1.0)
    return (coef_I * I2 + coef_J * J2) / n2


def Phi_closed(alpha, omega, dt=DELTA, tol=1e-8):
    """Forme close AVEC bascule sur la serie : c'est la version a implementer."""
    if alpha ** 2 + omega ** 2 < tol ** 2:
        return Phi_series(alpha, omega, dt)
    return Phi_naive(alpha, omega, dt)


def Phi_quad(alpha, omega, dt=DELTA, n=20001, n_terms=40):
    """Reference independante : quadrature trapezoidale fine de int_0^dt exp(sG) ds.

    L'exponentielle est evaluee par serie tronquee, vectorisee sur les n points de quadrature.
    40 termes suffisent tres largement : ||s G|| <= dt * |g| reste petit devant 1.
    """
    s = np.linspace(0.0, dt, n)
    G = alpha * I2 + omega * J2
    Gs = s[:, None, None] * G                       # (n, 2, 2)
    S = np.broadcast_to(np.eye(2), (n, 2, 2)).copy()
    T = S.copy()
    for k in range(1, n_terms + 1):
        T = np.einsum("nij,njk->nik", T, Gs) / k
        S = S + T
    return np.trapezoid(S, s, axis=0)


print("Verification de la forme close contre une quadrature fine (20001 points) :")
print(f"{'alpha':>10} {'omega':>10} | {'|close - quad|_inf':>19} | {'|naive - quad|_inf':>19}")
print("-" * 68)
for a, w in [(-0.30, 2.10), (0.70, -1.30), (0.0, 3.0), (-5.0, 0.0)]:
    q = Phi_quad(a, w)
    print(f"{a:10.2f} {w:10.2f} | {np.abs(Phi_closed(a, w) - q).max():19.3e} |"
          f" {np.abs(Phi_naive(a, w) - q).max():19.3e}")

# Validation INDEPENDANTE du bras "serie", a |g| ~ 1 ou la quadrature est fiable : sans cela,
# comparer la bascule a la serie reviendrait a comparer la serie a elle-meme.
for a, w in [(0.6, 0.8), (-0.7, 0.7)]:
    print(f"  |Phi_series - Phi_quad|_inf en (alpha,omega)=({a},{w}) :"
          f" {np.abs(Phi_series(a, w) - Phi_quad(a, w)).max():.3e}")

print("\nApproche de la singularite (alpha = omega = t), erreur RELATIVE :")
print(f"{'|g|':>12} | {'naive':>12} | {'close (bascule)':>16} | {'reference':>13}")
print("-" * 62)
mags, err_naive, err_close = [], [], []
for t in np.logspace(-1, -12, 12):
    a = w = t / np.sqrt(2.0)
    ref = Phi_series(a, w, n_terms=16)          # exacte a cette echelle : |g| petit
    en = rel_err(Phi_naive(a, w), ref)
    ec = rel_err(Phi_closed(a, w), ref)
    # plancher d'affichage a eps_mach : sous le seuil de bascule la version corrigee EST
    # la serie de reference, donc son "erreur" mesuree est nulle par construction.
    mags.append(t); err_naive.append(max(en, 2.2e-16)); err_close.append(max(ec, 2.2e-16))
    print(f"{t:12.1e} | {en:12.2e} | {ec:16.2e} | {'(reference)':>13}")

fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.loglog(mags, err_naive, "o-", label=r"forme close brute $\;(e^{\Delta g}-1)\bar g/|g|^2$")
ax.loglog(mags, err_close, "s-", label=r"avec bascule serie pour $|g|<10^{-8}$ (plancher affiche : $\epsilon_{\rm mach}$)")
ax.axvline(1e-8, color="k", ls=":", lw=1, label=r"seuil de bascule")
ax.axhline(2.2e-16, color="gray", ls="--", lw=1, label=r"$\epsilon_{\rm mach}$")
ax.set(xlabel=r"$|g| = \sqrt{\alpha^2+\omega^2}$", ylabel=r"erreur relative sur $\Phi$",
       title=r"Singularite effacable en $(0,0)$ : analytiquement inoffensive, numeriquement non")
ax.invert_xaxis(); ax.legend(loc="upper left")
plt.tight_layout(); plt.show()

# %% [markdown]
# **Lecture.** La forme close brute et la quadrature coïncident à $\sim 10^{-13}$ près tant que
# $|g|$ est d'ordre 1 — et cet écart résiduel est celui de la **quadrature**, pas de la formule :
# le trapèze sur 20 001 points laisse un $O(h^2)$ de cet ordre. La courbe log-log montre ensuite
# ce que la formule brute fait quand on s'approche de $(0,0)$ : l'erreur relative croît à peu
# près comme $1/|g|$ et atteint $7\cdot 10^{-4}$ à $|g| = 10^{-12}$ — il ne reste alors que trois
# chiffres significatifs — puis la formule rend `nan` en $|g| = 0$ exactement (division $0/0$).
#
# **Une précaution de lecture sur la colonne de droite.** Sous le seuil de bascule, la version
# corrigée *est* la série, et la référence aussi : le zéro affiché est donc une tautologie, pas
# une mesure. C'est pourquoi le bras « série » est validé séparément, contre la quadrature, à
# $|g| \sim 1$ où celle-ci est fiable (lignes imprimées au-dessus du tableau, $\sim 10^{-13}$).
# Sans cette précaution, on aurait « vérifié » la série contre elle-même.
#
# La bascule sur la série tronquée ramène l'erreur au niveau de $\epsilon_{\text{mach}}$ sur
# toute la plage. C'est un correctif de trois lignes, et son absence est le genre de défaut qui
# ne se manifeste qu'après plusieurs milliers d'updates — au moment où un bloc, en s'amortissant,
# voit son $(\alpha,\omega)$ traverser le voisinage de l'origine.
#
# **Ce qu'on ne peut pas conclure.** Que ce chemin ait été traversé dans les runs du dépôt :
# la trajectoire de $(\alpha, \omega)$ par bloc n'est pas journalisée, donc l'événement n'est
# **ni observé ni exclu**. C'est un risque identifié, pas un bug mesuré.

# %% [markdown]
# ## 3. Quand le prior est exactement correct, et quand il ne l'est pas
#
# Un prior structurel n'est ni vrai ni faux dans l'absolu : il est **bien ou mal spécifié pour
# un système donné**. Cette section construit un système où l'on peut faire varier
# continûment le degré de mauvaise spécification, en tournant un seul bouton physique. C'est
# l'outil qui manque à la plupart des discussions sur les biais inductifs : sans axe de
# mauvaise spécification, on ne compare que des anecdotes.
#
# ### 3.1 Le système : deux masses couplées
#
# Deux masses $m_1 = m_2 = 1$, reliées au bâti par des ressorts $k_1 = 1$ et $k_2 = 1.5$, et
# l'une à l'autre par un ressort de couplage $k_c = 0.5$. Amortissement visqueux de coefficient
# $c$ sur chaque masse. Les équations de Newton donnent
#
# $$\ddot q_1 = -(k_1+k_c)q_1 + k_c q_2 - c\,\dot q_1, \qquad
# \ddot q_2 = k_c q_1 - (k_2+k_c)q_2 - c\,\dot q_2 .$$
#
# En posant $q = (q_1,q_2)^\top$ et $K = \begin{pmatrix}k_1+k_c & -k_c\\ -k_c & k_2+k_c\end{pmatrix}$,
# cela s'écrit $\ddot q = -Kq - c\dot q$, soit en état $x = (q, \dot q)^\top \in \mathbb{R}^4$ :
#
# $$\dot x = A x, \qquad A = \begin{pmatrix} 0 & I \\ -K & -cI\end{pmatrix}.$$

# %% [markdown]
# ### 3.2 Découplage modal : le spectre de $A$ à partir de celui de $K$
#
# **Proposition.** Soit $\kappa$ une valeur propre de $K$. Alors les valeurs propres associées
# de $A$ sont les racines de
#
# $$\boxed{\;\lambda^2 + c\,\lambda + \kappa = 0.\;}$$
#
# *Démonstration.* $K$ est symétrique réelle, donc orthogonalement diagonalisable :
# $K = V\Lambda V^\top$ avec $V^\top V = I$ et $\Lambda = \mathrm{diag}(\kappa_1,\kappa_2)$.
# Posons le changement de variables en **coordonnées modales**
# $\tilde q = V^\top q$, $\tilde v = V^\top \dot q$, c'est-à-dire
# $\tilde x = T x$ avec $T = \mathrm{diag}(V^\top, V^\top)$, matrice orthogonale de $\mathbb{R}^4$.
# Alors
#
# $$T A T^\top = \begin{pmatrix} V^\top & 0\\ 0 & V^\top\end{pmatrix}
# \begin{pmatrix} 0 & I\\ -K & -cI\end{pmatrix}
# \begin{pmatrix} V & 0\\ 0 & V\end{pmatrix}
# = \begin{pmatrix} 0 & V^\top V\\ -V^\top K V & -c\,V^\top V\end{pmatrix}
# = \begin{pmatrix} 0 & I\\ -\Lambda & -cI\end{pmatrix}.$$
#
# Comme $\Lambda$ est diagonale, ce système est **découplé** : chaque paire
# $(\tilde q_i, \tilde v_i)$ obéit à
#
# $$\begin{pmatrix}\dot{\tilde q}_i\\ \dot{\tilde v}_i\end{pmatrix}
# = \begin{pmatrix}0 & 1\\ -\kappa_i & -c\end{pmatrix}
# \begin{pmatrix}\tilde q_i\\ \tilde v_i\end{pmatrix},$$
#
# soit l'oscillateur scalaire amorti $\ddot{\tilde q}_i + c\,\dot{\tilde q}_i + \kappa_i \tilde q_i = 0$.
# Son polynôme caractéristique est $\lambda^2 + c\lambda + \kappa_i$. Le spectre de $A$ est la
# réunion des spectres des deux blocs $2\times 2$, car $A$ est semblable (par une similitude
# **orthogonale**, donc parfaitement conditionnée) à une matrice bloc-diagonale. $\square$
#
# **Corollaire — le seuil de suramortissement.** Les racines sont
# $\lambda_{\pm} = \frac{-c \pm \sqrt{c^2-4\kappa}}{2}$. Elles forment une paire complexe
# conjuguée si $c^2 < 4\kappa$ (mode **sous-amorti**), une racine réelle double si $c^2 = 4\kappa$
# (**critique**), deux réelles distinctes si $c^2 > 4\kappa$ (**suramorti**). Le seuil est donc
#
# $$c^*(\kappa) = 2\sqrt{\kappa}.$$
#
# **Pourquoi ce système et pas un autre.** Trois raisons, et une alternative écartée.
# (i) Il est **linéaire**, donc la classe rotation-scaling peut le contenir *exactement* — c'est
# la condition sine qua non pour distinguer « le prior est faux » de « l'optimisation échoue »
# (§4). (ii) La similitude modale est **orthogonale**, donc $\mathrm{cond}(V) = 1$ : aucun effet
# de mauvais conditionnement ne vient contaminer la mesure. (iii) Un seul scalaire $c$ fait
# passer continûment de « prior exact » à « prior faux ». *L'alternative écartée* : le pendule
# amorti, non linéaire, sur lequel l'axe ON/OFF du dépôt a effectivement été mesuré. Un prior
# LTI y est faux **par construction**, donc un résultat négatif n'y départage rien — et c'est
# précisément la critique que le dépôt s'adresse à lui-même.

# %%
# --- [DEMONTRE -> verifie] Decouplage modal et seuils de suramortissement ------------------
k1, kc, k2 = 1.0, 0.5, 1.5
K = np.array([[k1 + kc, -kc], [-kc, k2 + kc]])
kappas = np.sort(np.linalg.eigvalsh(K))
c_star = 2.0 * np.sqrt(kappas)

print("K =\n", K)
print("\nvaleurs propres de K   : kappa =", np.round(kappas, 6))
print("seuils c* = 2 sqrt(kappa) :", np.round(c_star, 6))


def A_state(c):
    Z, Id = np.zeros((2, 2)), np.eye(2)
    return np.block([[Z, Id], [-K, -c * Id]])


def canon_sort(ev, nd=10):
    """Tri canonique de valeurs propres complexes : partie reelle ARRONDIE, puis imaginaire.
    L'arrondi est indispensable : sans lui, deux valeurs propres de meme partie reelle
    mathematiquement sont departagees par un ecart de 1e-17 et l'ordre devient arbitraire."""
    ev = np.asarray(ev)
    return ev[np.lexsort((ev.imag, np.round(ev.real, nd)))]


print("\nVerification : le spectre de A coincide avec les racines de lambda^2 + c lambda + kappa")
print(f"{'c':>6} | {'spectre de A (numerique)':>44} | {'ecart aux racines modales':>25}")
print("-" * 84)
for c in [0.10, 1.00, 2.30, 3.00, 4.00]:
    ev = canon_sort(np.linalg.eigvals(A_state(c)))
    modal = []
    for kap in kappas:
        disc = complex(c * c - 4 * kap) ** 0.5
        modal += [(-c + disc) / 2, (-c - disc) / 2]
    modal = canon_sort(np.array(modal))
    print(f"{c:6.2f} | {np.array2string(ev, precision=4, suppress_small=True):>44} |"
          f" {np.abs(ev - modal).max():25.3e}")

# %% [markdown]
# **Lecture.** Le spectre calculé numériquement sur la matrice $4\times4$ et les racines des
# deux polynômes modaux coïncident à $10^{-15}$ près pour toutes les valeurs de $c$ testées : le
# découplage démontré est exact, et l'on peut désormais raisonner sur deux oscillateurs
# scalaires indépendants au lieu d'une matrice $4\times4$. Les seuils valent
# $c^*_1 = 2\sqrt{1.191} \approx 2.183$ et $c^*_2 = 2\sqrt{2.309} \approx 3.039$.
#
# **Ce qu'on ne peut pas conclure.** Que le prior soit approprié : le découplage dit seulement
# que la dynamique se factorise en deux sous-systèmes $2\times2$. Il ne dit rien sur la forme de
# ces blocs, qui est exactement la question suivante.

# %% [markdown]
# ### 3.3 Théorème de représentabilité
#
# **Énoncé.** Un bloc rotation-scaling de générateur $G = \alpha I + \omega J$ a pour spectre
# $\{\alpha + i\omega,\ \alpha - i\omega\}$ en temps continu (et
# $\{e^{\Delta\alpha}e^{\pm i\Delta\omega}\}$ en temps discret). Un tel bloc peut donc
# représenter :
#
# - une paire complexe conjuguée quelconque ($\omega \neq 0$) ;
# - une valeur propre **réelle double** ($\omega = 0$, spectre $\{\alpha,\alpha\}$).
#
# Il ne peut **pas** représenter deux valeurs propres réelles distinctes. De plus, si
# $\lambda_a \neq \lambda_b$ sont réelles, la distance spectrale minimale d'un bloc à la paire
# $\{\lambda_a,\lambda_b\}$, au sens du pire appariement (*bottleneck*), vaut **exactement**
#
# $$d(\lambda_a,\lambda_b) \;=\; \min_{\alpha,\omega\in\mathbb{R}}\;
# \max\big(|\lambda_a - (\alpha+i\omega)|,\ |\lambda_b - (\alpha-i\omega)|\big)
# \;=\; \frac{|\lambda_a - \lambda_b|}{2},$$
#
# atteinte en $\alpha = \frac{\lambda_a+\lambda_b}{2}$, $\omega = 0$.
#
# *Démonstration.* Posons $z = \alpha + i\omega \in \mathbb{C}$ ; quand $(\alpha,\omega)$ parcourt
# $\mathbb{R}^2$, $z$ parcourt $\mathbb{C}$ tout entier. Le second terme se réécrit
#
# $$|\lambda_b - (\alpha - i\omega)| \;=\; |\lambda_b - \bar z| \;=\; \big|\overline{\lambda_b - \bar z}\big|
# \;=\; |\bar\lambda_b - z|.$$
#
# Le problème devient donc : minimiser sur $z \in \mathbb{C}$ le maximum des distances de $z$ à
# **deux points fixes du plan**, $\lambda_a$ et $\mu := \bar\lambda_b$. Or pour tout $z$,
#
# $$\max\big(|\lambda_a - z|,\ |\mu - z|\big)
# \;\ge\; \tfrac{1}{2}\big(|\lambda_a - z| + |z - \mu|\big)
# \;\ge\; \tfrac{1}{2}|\lambda_a - \mu|,$$
#
# la première inégalité parce qu'un maximum majore la moyenne, la seconde par inégalité
# triangulaire. La borne $\frac{1}{2}|\lambda_a-\mu|$ est atteinte en $z^\star = \frac{\lambda_a+\mu}{2}$,
# où les deux distances valent exactement $\frac{1}{2}|\lambda_a-\mu|$. Donc
#
# $$d \;=\; \tfrac{1}{2}\,\big|\lambda_a - \bar\lambda_b\big|.$$
#
# Deux cas :
# - **$\lambda_b = \bar\lambda_a$** (paire conjuguée) : $\bar\lambda_b = \lambda_a$, donc $d = 0$
#   — le bloc représente la paire **exactement**, en prenant $\alpha = \Re\lambda_a$,
#   $\omega = \Im\lambda_a$ ;
# - **$\lambda_a,\lambda_b$ réelles** : $\bar\lambda_b = \lambda_b$, donc
#   $d = \frac{1}{2}|\lambda_a - \lambda_b|$, atteinte en $z^\star = \frac{\lambda_a+\lambda_b}{2}$
#   réel, c'est-à-dire $\omega = 0$. $\square$
#
# **Où l'énoncé tombe.** (i) Il mesure une distance **spectrale**, pas une distance entre
# opérateurs : deux matrices peuvent avoir le même spectre et des comportements transitoires
# très différents (non-normalité). (ii) Il suppose l'appariement pire cas ; avec une somme au
# lieu d'un max, l'optimum est le même ici, mais ce n'est pas automatique en dimension
# supérieure. (iii) Il ne dit rien sur ce qu'un **entraînement** trouve : c'est une borne
# d'approximation, pas d'optimisation. §4 est entièrement consacré à cet écart.

# %%
# --- [DEMONTRE -> verifie] La distance minimale a un bloc vaut |la - conj(lb)| / 2 ---------
# On ne fait pas confiance a la formule : on minimise numeriquement sur une grille fine + un
# raffinement local, et on compare a la forme close.

def bottleneck_obj(z, la, lb):
    return max(abs(la - z), abs(lb - np.conj(z)))


def brute_min(la, lb, n=241, rounds=8):
    """Minimisation directe de max(|la - z|, |lb - conj(z)|) par raffinement de grille.

    Le point de depart est (0, 0) et NON le minimiseur analytique : sinon la "verification"
    partirait de la reponse et ne verifierait rien.
    """
    ac, wc = 0.0, 0.0
    half = 4.0 + abs(la) + abs(lb)
    best = (np.inf, ac, wc)
    for _ in range(rounds):
        As = np.linspace(ac - half, ac + half, n)
        Ws = np.linspace(wc - half, wc + half, n)
        for a in As:
            zs = a + 1j * Ws
            vals = np.maximum(np.abs(la - zs), np.abs(lb - np.conj(zs)))
            j = int(np.argmin(vals))
            if vals[j] < best[0]:
                best = (float(vals[j]), float(a), float(Ws[j]))
        ac, wc = best[1], best[2]
        half /= 8.0
    return best


tests = [
    ("2 reelles distinctes", -0.4598 + 0j, -2.5902 + 0j),
    ("reelle double",        -1.5000 + 0j, -1.5000 + 0j),
    ("paire conjuguee",      -1.1500 + 0.9932j, -1.1500 - 0.9932j),
    ("paire conj. quelconque", 0.3 + 2.7j, 0.3 - 2.7j),
    ("reelle + complexe",    -0.5936 + 0j, -1.3000 + 0.7868j),
]
print(f"{'cas':>23} | {'forme close':>12} | {'min numerique':>13} | {'ecart':>10} | {'omega* trouve':>13}")
print("-" * 84)
for name, la, lb in tests:
    closed = abs(la - np.conj(lb)) / 2.0
    val, a_star, w_star = brute_min(la, lb)
    print(f"{name:>23} | {closed:12.8f} | {val:13.8f} | {abs(val-closed):10.2e} | {w_star:13.6f}")

# %% [markdown]
# **Lecture.** La forme close et la minimisation numérique aveugle — qui part de $(0,0)$ et non
# de la réponse — coïncident à $\sim 10^{-8}$ près. L'écart résiduel est la **résolution de la
# grille** au dernier tour, pas une erreur de la formule : il décroît si l'on ajoute des tours.
# Les trois lignes qui portent le théorème :
#
# - **paire conjuguée** $\Rightarrow$ distance $0$ : le prior est *exactement* correct ;
# - **réelle double** $\Rightarrow$ distance $0$, avec $\omega^\star = 0$ : le cas critique est
#   dans la classe, à la frontière ;
# - **deux réelles distinctes** $\Rightarrow$ distance $|\lambda_a-\lambda_b|/2 > 0$, avec
#   $\omega^\star = 0$ : le bloc « fait la moyenne » des deux taux et ne peut pas mieux.
#
# La dernière ligne (réelle + complexe) est là pour rappeler que la formule
# $\frac{1}{2}|\lambda_a - \bar\lambda_b|$ est générale et ne suppose pas que la paire soit
# conjuguée ou réelle. Elle servira au §3.4.
#
# ### 3.4 L'axe de mauvaise spécification $\delta(c)$
#
# On dispose maintenant de tout : le spectre de $A(c)$ (§3.2) et la distance d'une paire à un
# bloc (§3.3). Il reste à décider **comment apparier** les 4 valeurs propres de $A$ aux 2 blocs
# du prior. C'est un choix de convention, il change les chiffres, et il doit donc être déclaré
# avant de les regarder.
#
# **Convention déclarée.** *Une paire complexe conjuguée n'est jamais scindée entre deux blocs.*
# Autrement dit, $\delta(c)$ est le minimum, sur les appariements des 4 valeurs propres en 2
# paires **qui respectent la conjugaison**, du maximum des distances par paire :
#
# $$\delta(c) \;=\; \min_{\substack{\text{appariements}\\ \text{respectant la conjugaison}}}\;\;
# \max_{\{\lambda_a,\lambda_b\}\ \in\ \text{appariement}} \;\frac{|\lambda_a - \bar\lambda_b|}{2}.$$
#
# **Justification, et le prix de la convention.** Une paire conjuguée est représentable à coût
# nul ; la scinder revient à dépenser deux blocs pour une paire déjà gratuite. La convention
# garde donc $\delta$ interprétable comme *« combien de modes physiques sortent de la classe »*.
# Elle n'est **pas** la borne inférieure absolue : l'appariement libre, qui autorise un bloc à
# chevaucher une paire conjuguée, peut donner strictement moins. Nous calculons **les deux**
# courbes et nous mesurons l'écart, plutôt que de le passer sous silence.

# %%
# --- [DEMONTRE -> calcule] delta(c) sous les deux conventions ------------------------------
PAIRINGS = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]


def pair_dist(la, lb):
    """Distance minimale d'un bloc rotation-scaling a la paire {la, lb}  (theoreme 3.3)."""
    return abs(la - np.conj(lb)) / 2.0


def respects_conjugation(ev, p, q, tol=1e-9):
    """Vrai si aucune paire conjuguee n'est scindee par l'appariement (p, q)."""
    for (i, j) in (p, q):
        if abs(ev[i].imag) > tol or abs(ev[j].imag) > tol:
            if abs(ev[i] - np.conj(ev[j])) > tol:
                return False
    return True


def delta_of_c(c, free=False):
    ev = np.linalg.eigvals(A_state(c))
    best = np.inf
    for (p, q) in PAIRINGS:
        if (not free) and (not respects_conjugation(ev, p, q)):
            continue
        best = min(best, max(pair_dist(ev[p[0]], ev[p[1]]),
                             pair_dist(ev[q[0]], ev[q[1]])))
    return best


cs = np.linspace(0.01, 4.50, 1400)
d_decl = np.array([delta_of_c(c) for c in cs])
d_free = np.array([delta_of_c(c, free=True) for c in cs])

fig, ax = plt.subplots(figsize=(8.2, 4.0))
ax.plot(cs, d_decl, lw=2.0, label=r"$\delta(c)$ — convention declaree (paires conjuguees non scindees)")
ax.plot(cs, d_free, lw=1.4, ls="--", color="tab:orange", label=r"appariement libre (borne inferieure)")
for cst, lab, ytxt in zip(c_star, [r"$c^*_1=2\sqrt{\kappa_1}$", r"$c^*_2=2\sqrt{\kappa_2}$"],
                          [0.60, 1.15]):
    ax.axvline(cst, color="k", ls=":", lw=1.1)
    ax.text(cst + 0.05, ytxt, lab + f"\n{cst:.3f}", fontsize=8,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
ax.axvspan(0.0, c_star[0], color="tab:green", alpha=0.08)
ax.text(1.05, 0.10, "prior EXACT\n(2 paires conjuguees)", ha="center", fontsize=8.5, color="darkgreen")
ax.axvspan(c_star[1], 4.5, color="tab:red", alpha=0.06)
ax.text(3.95, 0.70, "4 valeurs propres reelles :\n$\\delta$ REDESCEND", ha="center", fontsize=8.5,
        color="darkred")
tab_c = [0.10, 2.30, 2.60, 3.00, 3.05, 4.00]
ax.plot(tab_c, [delta_of_c(c) for c in tab_c], "ko", ms=5, zorder=5)
ax.set(xlabel="amortissement visqueux $c$", ylabel=r"defaut spectral $\delta$",
       title=r"$\delta(c)$ : de combien le spectre de $A(c)$ sort de la classe rotation-scaling")
ax.legend(loc="upper left"); ax.set_ylim(-0.05, 1.30)
plt.tight_layout(); plt.show()

print(f"{'c':>7} | {'delta declare':>13} | {'appariement libre':>17} | {'nature du spectre':>34}")
print("-" * 82)
for c in tab_c:
    ev = np.linalg.eigvals(A_state(c))
    n_real = int(np.sum(np.abs(ev.imag) < 1e-9))
    nature = {0: "2 paires conjuguees", 2: "1 paire conj. + 2 reelles", 4: "4 reelles"}[n_real]
    print(f"{c:7.2f} | {delta_of_c(c):13.3f} | {delta_of_c(c, True):17.3f} | {nature:>34}")

mono = np.all(np.diff(d_decl[cs <= c_star[1]]) >= -1e-12)
print(f"\ndelta declare monotone croissant sur [0, c*_2] : {bool(mono)}")
print(f"delta declare juste avant c*_2 = {delta_of_c(c_star[1] - 1e-4):.3f}"
      f"  ->  juste apres = {delta_of_c(c_star[1] + 1e-2):.3f}  (discontinuite de la convention)")

# %% [markdown]
# **Lecture.** La courbe $\delta(c)$ se lit en trois zones.
#
# 1. **$c < c^*_1 = 2.183$ (zone verte).** Les deux modes sont sous-amortis, le spectre de $A$
#    est constitué de deux paires conjuguées, et $\delta = 0$ : **le prior rotation-scaling est
#    exactement correct**. À $c = 0.10$, la classe d'hypothèses contient la solution exacte.
#    C'est le régime de la campagne D3 du dépôt, et c'est ce qui rend son résultat interprétable
#    (§4).
# 2. **$c^*_1 < c < c^*_2 = 3.039$.** Le premier mode devient suramorti : deux valeurs propres
#    réelles distinctes, non représentables. $\delta$ croît continûment et **monotonement**
#    (vérifié dans la sortie) de $0$ à $\approx 1.06$. C'est l'axe utilisable : un seul scalaire
#    physique règle la mauvaise spécification.
# 3. **$c > c^*_2$ (zone rouge).** Le second mode bascule à son tour. Les 4 valeurs propres sont
#    réelles, et le prior peut alors **réapparier** : au lieu de dépenser un bloc sur l'énorme
#    écart intra-mode, il apparie les deux réelles proches entre elles. $\delta$ **redescend**
#    ($0.468$ à $c = 3.05$, $0.188$ à $c = 4.00$). L'axe cesse d'être monotone, donc il cesse
#    d'être un axe : deux valeurs de $c$ y correspondent au même $\delta$ avec des dynamiques
#    physiques opposées. **C'est la raison pour laquelle le balayage s'arrête à $c = 3.039$**,
#    et cette raison est structurelle, pas esthétique.
#
# **La table des valeurs, arrondie au millième :** $c = 0.10 \to \delta = 0.000$ ;
# $c = 2.30 \to 0.363$ ; $c = 2.60 \to 0.706$ ; $c = 3.00 \to 1.029$ ; puis, au-delà du second
# seuil, $c = 3.05 \to 0.468$ et $c = 4.00 \to 0.188$.
#
# **Le prix de la convention, mesuré.** La courbe orange (appariement libre) décroche de la
# courbe déclarée entre $c^*_1$ et $c^*_2$ : à $c = 2.60$, $0.529$ contre $0.706$ ; à
# $c = 3.00$, $0.529$ contre $1.029$. Autrement dit, un bloc a **le droit** de chevaucher une
# paire conjuguée et d'y gagner. La convention déclarée est donc une **sur-estimation** du
# défaut réel dans cette zone, d'un facteur allant jusqu'à $1.9$. On la garde parce qu'elle est
# monotone et interprétable, mais on ne peut pas la présenter comme la distance minimale. La
# sortie mesure aussi sa conséquence : $\delta$ déclaré est **discontinu** en $c^*_2$ (saut de
# $1.057$ à $0.471$), la version libre non.
#
# **Ce qu'on ne peut pas conclure.** (i) Que $\delta$ prédise une erreur d'apprentissage : c'est
# une distance **spectrale**, et le lien entre défaut spectral et erreur de prédiction dépend
# du conditionnement de la base propre — ici $\mathrm{cond}(V) = 1$, ce qui est le cas le plus
# favorable, et donc le moins généralisable. (ii) Que $\delta = 0$ garantisse quoi que ce soit
# sur l'entraînement. C'est exactement le sujet de la section suivante.

# %% [markdown]
# ## 4. Exprimable n'est pas atteignable
#
# ### 4.1 Le contrôle D3, et ce qu'il mesure
#
# Le dépôt a couru une campagne — le contrôle **D3** — sur exactement ce système, dans le régime
# où $\delta = 0$ : oscillateurs couplés sous-amortis, prior **exactement correct**. 10 cellules
# (2 bras structurés $\times$ 5 seeds), 8 000 updates chacune, rollout fermé de 25 pas.
#
# **[MESURÉ — `PROTOCOL.md §21.0`, 10/10 runs complets]**
#
# | critère | résultat | lecture |
# |---|---|---|
# | gain relatif contre baselines triviales | **10/10 réussissent**, $m_{\text{low}}$ médian $0.156$ | les modèles battent la persistance et l'extrapolation linéaire d'environ $32\times$ |
# | exactitude absolue, erreur d'état à $H=100$ | $0.189$ | seuil pré-enregistré : $10^{-3}$ |
# | exactitude absolue, erreur spectrale | $0.112$ | seuil pré-enregistré : $10^{-3}$ |
# | exactitude absolue, erreur d'action | $0.198$ | seuil pré-enregistré : $10^{-3}$ |
# | verdict d'exactitude | **0/10** sur les trois bornes | environ $110$ à $200\times$ au-dessus du seuil |
#
# Les deux lignes de ce tableau ne se contredisent pas, elles mesurent deux choses différentes :
# le modèle est **utile** (il bat largement des baselines triviales) et il est **inexact** (il
# reste deux décades au-dessus d'un seuil qui avait été fixé, et publié, avant le run).

# %% [markdown]
# ### 4.2 Quatre explications concurrentes, chacune éliminée par sa propre mesure
#
# C'est la partie méthodologiquement importante. Un échec d'exactitude admet a priori quatre
# causes ; le dépôt les a testées **une par une, avec pour chacune une mesure dédiée**, et non
# par argument.
#
# **[MESURÉ — `PROTOCOL.md §20.1`]**
#
# 1. **Ce ne sont pas les données.** Les moindres carrés ordinaires, sur *le même split*,
#    rendent une erreur spectrale de $2.11\times 10^{-8}$, une erreur d'action de
#    $1.58\times 10^{-7}$, et une erreur de rollout **libre** à 100 pas de
#    $e_{100} = 7.98\times 10^{-14}$. L'information nécessaire est donc entièrement présente
#    dans le jeu d'entraînement, et récupérable en forme close.
# 2. **Ce n'est pas l'architecture.** Avec des paramètres *plantés à la main* (le « gold »), la
#    même architecture reproduit la solution ZOH exacte à $1.9\times 10^{-15}$. La classe
#    d'hypothèses **contient** la solution, à la précision machine.
# 3. **Ce n'est pas le budget.** La décroissance de la perte de validation passe de
#    $-1.66$ décade pour 1 000 updates (début) à $-0.023$ (fin). En extrapolant ce taux
#    terminal, il faudrait $50\,000$ à $100\,000$ updates de plus, soit 7 à 13 fois le budget
#    déclaré. Le run n'est pas « interrompu trop tôt » : il est sur un plateau.
# 4. **Ce n'est pas la sur-paramétrisation de la tête contextuelle.** L'hypothèse était que la
#    tête, en produisant $(\alpha,\omega)$ dépendants du contexte, ajoutait de la liberté
#    inutile. Le test : forcer $W_\theta = 0$, donc rendre la tête constante. Résultat, l'erreur
#    d'action **se dégrade** de $0.138$ à $0.947$. L'hypothèse est **réfutée par son propre
#    test**, et enregistrée comme telle : la dépendance au contexte porte du signal.

# %% [markdown]
# ### 4.3 L'énoncé, et sa portée
#
# > **[MESURÉ]** Sur un système linéaire dont la solution exacte appartient à la classe
# > d'hypothèses **et** se récupère en forme close par moindres carrés à $10^{-8}$, le régime
# > d'entraînement déclaré atterrit environ $200\times$ au-dessus du seuil d'exactitude
# > pré-enregistré. Les quatre explications alternatives testées ont été éliminées, chacune par
# > une mesure dédiée.
#
# **Ce que cela dit du paysage d'optimisation.** L'écart entre « la solution est dans la classe »
# et « la descente de gradient l'atteint » n'est pas un détail d'ingénierie. Il existe une
# littérature qui l'établit dans un cadre voisin : [Hardt, Ma & Recht, *Gradient descent learns
# linear dynamical systems*, JMLR 19(29), 2018](https://jmlr.org/papers/v19/16-465.html)
# démontrent la convergence de la descente de gradient pour l'identification de systèmes
# linéaires, mais **sous des conditions** — notamment une condition de quasi-convexité qui
# demande que le système reste dans une région de stabilité contrôlée — et leur analyse porte
# sur une paramétrisation et un objectif donnés. Sortir de ces conditions ne rend pas le
# problème insoluble ; cela retire la garantie. Le résultat D3 est une instance mesurée de
# cette sortie de garantie.
#
# **Bornage explicite de la portée.** Cet énoncé porte sur :
# **(a)** cette paramétrisation — blocs rotation-scaling avec $\alpha$ libre, plus $\psi$ et
# $\pi$ non contraints (§5) ; **(b)** ce régime — rollout **fermé** de 25 pas depuis
# l'initialisation, sans curriculum, un régime dont `PROTOCOL.md §17.0` reconnaît par écrit
# qu'*« aucun travail publié n'entraîne la loss de dynamique d'un SSM sur un rollout
# auto-consommé de 25 pas depuis l'initialisation sans garde »* ; **(c)** cet optimiseur et ce
# budget. Il **ne porte pas** sur les priors structurels en général, ni sur l'entraînabilité des
# modèles latents linéaires en général. Confondre les deux serait exactement l'erreur que la
# section 3.2 reproche au site expérimental du dépôt.

# %% [markdown]
# ### 4.4 Une miniature — et ce qu'elle peut prouver
#
# Ce qui suit est **[ILLUSTRATIF]**. Ce n'est pas une reproduction du run D3 : pas le même
# encodeur (il n'y en a pas), pas la même tête contextuelle, pas la même taille, pas le même
# optimiseur exact, pas les mêmes données. C'est une **maquette du mécanisme**, construite pour
# répondre à une question précise et falsifiable :
#
# > *Quel est l'ingrédient minimal qui fait passer un problème exactement résoluble en forme
# > close à un problème que la descente de gradient ne résout pas ?*
#
# On procède par élimination, comme §4.2, sur une maquette où l'on peut allumer les ingrédients
# un par un. On commence par le cas de référence : les moindres carrés.

# %%
# --- [ILLUSTRATIF] Le systeme, ses donnees, et la solution en forme close ------------------
C_DAMP = 0.10                                   # regime ou delta(c) = 0 : prior EXACT
A_cont = A_state(C_DAMP)
A_true = expm_series(DELTA * A_cont)            # operateur discret ZOH exact, 4 x 4

ev_disc = np.linalg.eigvals(A_true)
print("rayon spectral discret r =", max(abs(ev_disc)).round(8))
print("spectre discret :", np.round(canon_sort(ev_disc), 5))
print("delta(c) a c =", C_DAMP, ":", round(delta_of_c(C_DAMP), 12), " -> le prior est EXACT\n")

rng = np.random.default_rng(7)
N_EP, T_EP = 60, 200
X_list, Y_list = [], []
for _ in range(N_EP):
    x = 0.5 * rng.standard_normal(4)
    for _ in range(T_EP):
        xn = A_true @ x
        X_list.append(x); Y_list.append(xn)
        x = xn
X = np.array(X_list); Y = np.array(Y_list)
print(f"donnees : {len(X)} transitions, dimension d'etat 4 (donnees NON bruitees)")

# --- reference 1 : moindres carres ordinaires (forme close) --------------------------------
A_ls = np.linalg.lstsq(X, Y, rcond=None)[0].T
print(f"\n[reference] moindres carres : |A_ls - A_true|_F = {np.linalg.norm(A_ls - A_true):.3e}")

# rollout libre a 100 pas, comme le fait le depot
x0 = 0.5 * np.random.default_rng(123).standard_normal((256, 4))
xs_true, xs_ls = x0.copy(), x0.copy()
for _ in range(100):
    xs_true = xs_true @ A_true.T
    xs_ls = xs_ls @ A_ls.T
print(f"[reference] moindres carres : e_100 (rollout libre) = "
      f"{np.linalg.norm(xs_ls - xs_true, axis=1).mean():.3e}")
print(f"[reference] conditionnement de la matrice de Gram : "
      f"{np.linalg.cond(X.T @ X):.3f}")

# %% [markdown]
# **Lecture.** Les moindres carrés récupèrent l'opérateur exact à $1.3\times 10^{-15}$ et le
# rollout libre à 100 pas dérive de $2.9\times 10^{-14}$ : sur ce jeu, la solution n'est pas
# seulement *dans* la classe, elle est **calculable en une ligne**. Le conditionnement de la
# matrice de Gram vaut $2.4$ : le problème des moindres carrés est **très bien conditionné**.
# Retenons ce chiffre, il va servir à éliminer une explication.
#
# *(L'ordre de grandeur de notre $e_{100}$ est voisin du $7.98\times10^{-14}$ mesuré en §20.1.
# C'est une coïncidence de deux problèmes bien conditionnés à 4 états ; ce n'est pas une
# reproduction, et il ne faut rien en tirer.)*

# %% [markdown]
# ### 4.5 Hypothèse A — « la descente de gradient est trop faible »
#
# C'est l'explication réflexe. Testons-la sur la **même** paramétrisation structurée que le
# dépôt : deux blocs rotation-scaling $\bar A(\alpha,\omega)$, un lift linéaire $P$ et une
# projection linéaire $Q$, prédiction $\hat z_{t+1} = Q\,\bar A\,P\,z_t$. Quarante paramètres
# libres pour un problème qui en a 16. Objectif : erreur à **un pas**, comme
# $\mathcal{L}_1$ de §1.1.
#
# Les gradients sont analytiques (pas de différences finies) ; les deux dérivées non triviales
# sont celles du bloc, obtenues par dérivation de la forme close de §2.2 :
#
# $$\frac{\partial \bar A_i}{\partial \alpha_i} = \Delta\,\bar A_i,
# \qquad
# \frac{\partial \bar A_i}{\partial \omega_i} = \Delta\,\bar A_i J,$$
#
# la seconde parce que $\frac{d}{d\varphi}R(\varphi) = R(\varphi)J$ — une rotation dérivée est
# la même rotation composée avec un quart de tour. Ces deux identités seront **vérifiées contre
# des différences finies** avant tout entraînement : un gradient faux produit un joli plateau
# qu'on interpréterait comme un résultat.

# %%
# --- [ILLUSTRATIF] Cellule structuree lineaire + Adam, gradients analytiques ---------------
R_BLOCKS = 2
DZ = 4


def A_blocks(alpha, omega, dt=DELTA):
    """Transition bloc-diagonale : r blocs rotation-scaling exp(dt (alpha_i I + omega_i J))."""
    M = np.zeros((2 * R_BLOCKS, 2 * R_BLOCKS))
    for i in range(R_BLOCKS):
        M[2 * i:2 * i + 2, 2 * i:2 * i + 2] = A_bar_closed(alpha[i], omega[i], dt)
    return M


def block_grads(dA, alpha, omega, dt=DELTA):
    """d(loss)/d(alpha_i), d(loss)/d(omega_i) a partir de d(loss)/d(Abar)."""
    ga, gw = np.zeros(R_BLOCKS), np.zeros(R_BLOCKS)
    for i in range(R_BLOCKS):
        g = dA[2 * i:2 * i + 2, 2 * i:2 * i + 2]
        B = A_bar_closed(alpha[i], omega[i], dt)
        ga[i] = dt * np.sum(g * B)              # dAbar/dalpha = dt * Abar
        gw[i] = dt * np.sum(g * (B @ J2))       # dAbar/domega = dt * Abar J
    return ga, gw


# --- controle de gradient : les deux identites analytiques contre differences finies -------
_a = np.array([-0.21, 0.13]); _w = np.array([1.7, -0.9]); _h = 1e-6
_W = np.random.default_rng(3).standard_normal((2 * R_BLOCKS, 2 * R_BLOCKS))
_f = lambda a, w: float(np.sum(_W * A_blocks(a, w)))
_ga, _gw = block_grads(_W, _a, _w)
_fd_a = np.array([(_f(_a + _h * np.eye(2)[i], _w) - _f(_a - _h * np.eye(2)[i], _w)) / (2 * _h)
                  for i in range(2)])
_fd_w = np.array([(_f(_a, _w + _h * np.eye(2)[i]) - _f(_a, _w - _h * np.eye(2)[i])) / (2 * _h)
                  for i in range(2)])
print("controle de gradient (analytique vs differences finies centrees) :")
print(f"  d/d alpha : ecart relatif max = {np.abs(_ga - _fd_a).max() / np.abs(_fd_a).max():.2e}")
print(f"  d/d omega : ecart relatif max = {np.abs(_gw - _fd_w).max() / np.abs(_fd_w).max():.2e}")


class Adam:
    def __init__(self, params, lr):
        self.p, self.lr = params, lr
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for j, (x, g) in enumerate(zip(self.p, grads)):
            self.m[j] = 0.9 * self.m[j] + 0.1 * g
            self.v[j] = 0.999 * self.v[j] + 0.001 * g ** 2
            mh = self.m[j] / (1 - 0.9 ** self.t)
            vh = self.v[j] / (1 - 0.999 ** self.t)
            x -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def init_params(seed):
    r = np.random.default_rng(seed)
    return [0.1 * r.standard_normal(R_BLOCKS),      # alpha
            r.standard_normal(R_BLOCKS),            # omega
            r.standard_normal((2 * R_BLOCKS, DZ)) / 2,   # P (lift)
            r.standard_normal((DZ, 2 * R_BLOCKS)) / 2]   # Q (projection)


def train_one_step(seed, iters=2000, lr=1e-2, batch=None):
    """Objectif a UN pas : || Q Abar P z_t - z_{t+1} ||^2."""
    alpha, omega, P, Q = params = init_params(seed)
    opt = Adam(params, lr)
    r = np.random.default_rng(10_000 + seed)
    n = len(X)
    for _ in range(iters):
        if batch is None:
            Xb, Yb = X, Y
        else:
            idx = r.integers(0, n, batch); Xb, Yb = X[idx], Y[idx]
        Ab = A_blocks(alpha, omega)
        M = Q @ Ab @ P
        E = Xb @ M.T - Yb
        dM = 2.0 / len(Xb) * (E.T @ Xb)
        dQ = dM @ (Ab @ P).T
        dA = Q.T @ dM @ P.T
        dP = (Q @ Ab).T @ dM
        ga, gw = block_grads(dA, alpha, omega)
        opt.step([ga, gw, dP, dQ])
    return Q @ A_blocks(alpha, omega) @ P


print("\n[ILLUSTRATIF] Hypothese A : Adam sur la loss a UN pas, parametrisation structuree")
print(f"{'variante':>26} | {'|M - A_true|_F, par seed':>40}")
print("-" * 72)
for label, kw in [("plein lot", dict(batch=None)), ("mini-lot 256", dict(batch=256)),
                  ("mini-lot 32", dict(batch=32))]:
    errs = [np.linalg.norm(train_one_step(s, **kw) - A_true) for s in range(3)]
    print(f"{label:>26} | {'  '.join(f'{e:.2e}' for e in errs):>40}")

# %% [markdown]
# **Lecture — l'hypothèse A est réfutée par notre propre maquette.** Sur l'objectif à un pas,
# Adam atteint $\sim 10^{-16}$, c'est-à-dire la **précision machine**, et cela quel que soit le
# lot : plein, 256, ou 32. La descente de gradient n'a donc aucune difficulté intrinsèque avec
# cette paramétrisation, ni avec sa non-convexité (le produit $Q\bar A P$ l'est pourtant), ni
# avec le bruit de mini-lot.
#
# **Pourquoi le bruit de mini-lot ne coûte rien ici, et ce que ça implique.** Les données sont
# **sans résidu** : il existe des paramètres qui annulent la perte exactement. Au minimum, le
# gradient de *chaque* mini-lot est nul, pas seulement celui du lot complet. C'est le régime
# dit d'**interpolation**, où SGD converge au même taux que la descente à plein lot. Un bruit
# d'observation le casserait — mais §20.1 mesure que les données du dépôt sont, elles aussi,
# quasi sans résidu (les MCO y descendent à $2.11\times 10^{-8}$). L'hypothèse « c'est le bruit
# de gradient » est donc, elle aussi, écartée sur cette maquette.
#
# **Ce qu'on ne peut pas conclure.** Que le run réel n'ait pas eu de problème d'optimiseur : nous
# n'avons pas la même échelle, ni les MLP $\psi/\pi$ non linéaires, ni la tête contextuelle. Ce
# que la maquette établit, c'est que **ces trois ingrédients-là ne suffisent pas** à créer
# l'écart. Il faut chercher ailleurs.

# %% [markdown]
# ### 4.6 Hypothèse B — c'est le régime d'entraînement, pas l'optimiseur
#
# Le dépôt n'entraîne pas sur une erreur à un pas. Depuis l'amendement 6, `h_fr = 25` : la perte
# est calculée sur un **rollout fermé de 25 pas depuis l'initialisation**, c'est-à-dire que le
# modèle consomme ses propres prédictions, sans réinjection d'observation, dès le premier pas
# d'entraînement.
#
# §1.3 dit ce que cela change : le gradient traverse $H$ applications de $\bar A$, donc il est
# multiplié par $\sim r^{H}$ avec $r = e^{\Delta\alpha}$. À l'initialisation, $\alpha$ est
# aléatoire et **non borné** (`PROTOCOL.md §13.2` laisse `alpha` libre, sans reparamétrisation
# $-\exp(\nu)$ ni softplus). Un $r = 1.20$ donne $r^{25} \approx 95$ ; un $r = 1.50$ donne
# $2.5\times10^{4}$.
#
# Changeons **uniquement** cet ingrédient — même modèle, mêmes gradients analytiques, même
# optimiseur, même budget — et mesurons.

# %%
# --- [ILLUSTRATIF] Meme modele, meme optimiseur : objectif rollout FERME a H pas ------------
H_ROLL = 25
N_TRAJ_TR = 400

rng_tr = np.random.default_rng(11)
Z0 = 0.5 * rng_tr.standard_normal((N_TRAJ_TR, DZ))
TRAJ = np.zeros((N_TRAJ_TR, H_ROLL + 1, DZ))
TRAJ[:, 0] = Z0
for t in range(H_ROLL):
    TRAJ[:, t + 1] = TRAJ[:, t] @ A_true.T


def train_rollout(seed, iters=2000, lr=1e-2, batch=128, H=H_ROLL):
    """Objectif rollout FERME : h_0 = P z_0, h_{t+1} = Abar h_t, perte sur t = 1..H."""
    alpha, omega, P, Q = params = init_params(seed)
    opt = Adam(params, lr)
    r = np.random.default_rng(20_000 + seed)
    g0 = None
    for _ in range(iters):
        idx = r.integers(0, N_TRAJ_TR, batch)
        tj = TRAJ[idx]
        Ab = A_blocks(alpha, omega)
        Hs = [tj[:, 0] @ P.T]
        for _t in range(H):
            Hs.append(Hs[-1] @ Ab.T)
        dQ = np.zeros_like(Q); dAb = np.zeros((2 * R_BLOCKS, 2 * R_BLOCKS))
        gH = [np.zeros((batch, 2 * R_BLOCKS)) for _ in range(H + 1)]
        scale = 2.0 / (batch * H)
        for t in range(1, H + 1):                        # passe avant : pertes par pas
            E = scale * (Hs[t] @ Q.T - tj[:, t])
            dQ += E.T @ Hs[t]
            gH[t] += E @ Q
        for t in range(H, 0, -1):                        # BPTT exacte (modele lineaire)
            dAb += gH[t].T @ Hs[t - 1]
            gH[t - 1] += gH[t] @ Ab
        dP = gH[0].T @ tj[:, 0]
        ga, gw = block_grads(dAb, alpha, omega)
        grads = [ga, gw, dP, dQ]
        if g0 is None:
            g0 = np.sqrt(sum(float(np.sum(g ** 2)) for g in grads))
        opt.step(grads)
        if not np.all(np.isfinite(np.concatenate([np.ravel(x) for x in params]))):
            return np.nan, g0, np.nan
    Ab = A_blocks(alpha, omega)
    return np.linalg.norm(Q @ Ab @ P - A_true), g0, float(np.exp(DELTA * alpha).max())


# On balaie aussi le taux d'apprentissage : le depot nomme l'asymetrie de taux comme un
# confondant (PROTOCOL.md §21.2), il serait malhonnete de conclure sur un seul taux.
print("[ILLUSTRATIF] Hypothese B : seul l'objectif change. Mediane sur 3 seeds de |M - A_true|_F")
print(f"{'lr':>8} | {'objectif UN pas':>17} | {'objectif ROLLOUT 25':>20} | {'rapport':>10}")
print("-" * 66)
summary = {}
for lr in [1e-3, 3e-3, 1e-2]:
    e1 = np.median([np.linalg.norm(train_one_step(s, iters=2000, lr=lr, batch=256) - A_true)
                    for s in range(3)])
    eH = np.median([train_rollout(s, iters=2000, lr=lr)[0] for s in range(3)])
    summary[lr] = (e1, eH)
    print(f"{lr:8.0e} | {e1:17.2e} | {eH:20.2e} | {eH / e1:10.1e}")

print("\nDiagnostic du regime rollout au taux le plus favorable (lr = 1e-2), par seed :")
print(f"{'seed':>5} | {'|M - A_true|_F':>15} | {'|grad| au 1er pas':>18} | {'r max final':>12}")
print("-" * 62)
for sd in range(4):
    e, g0, rmax = train_rollout(sd, iters=2000, lr=1e-2)
    print(f"{sd:5d} | {e:15.3e} | {g0:18.3e} | {rmax:12.5f}")

# Chaque objectif est credite de SON meilleur taux : c'est la comparaison la moins favorable
# a notre these, donc la seule qu'on ait le droit de publier.
best_1 = min(v[0] for v in summary.values())
best_H = min(v[1] for v in summary.values())
print("\nrecapitulatif [ILLUSTRATIF] — chaque objectif a son meilleur taux parmi les 3 testes :")
print(f"  moindres carres (forme close)          : {np.linalg.norm(A_ls - A_true):.2e}")
print(f"  Adam, objectif a UN pas                : {best_1:.2e}")
print(f"  Adam, objectif ROLLOUT ferme 25 pas    : {best_H:.2e}")
print(f"  ecart entre les deux objectifs         : {best_H / best_1:.1e} x")

# %% [markdown]
# **Lecture.** Un seul ingrédient a changé — l'objectif passe de « un pas » à « rollout fermé de
# 25 pas » — et l'écart au minimiseur exact passe de la précision machine à $\sim 10^{-3}$, soit
# **treize ordres de grandeur**. Le modèle, les gradients, l'optimiseur, le budget, les données
# sont identiques. La solution exacte est toujours dans la classe, toujours accessible en une
# ligne de moindres carrés.
#
# Trois choses que le tableau montre et qu'il faut lire ensemble :
#
# - **Le taux d'apprentissage compte, et il compte différemment selon l'objectif.** À
#   $\text{lr} = 10^{-3}$, l'objectif à un pas n'a pas non plus convergé en 2 000 updates : il
#   est simplement lent. Aux deux taux supérieurs il atteint la précision machine. L'objectif
#   rollout, lui, ne l'atteint à **aucun** des trois taux. C'est exactement pourquoi le
#   confondant nommé par `PROTOCOL.md §21.2` — les bras qui n'ont pas reçu le même soin de
#   sélection de taux — est un vrai confondant, et pourquoi il fallait le balayer ici avant de
#   conclure quoi que ce soit.
# - **La dispersion inter-seed explose.** Au meilleur taux, les quatre seeds vont de
#   $3.5\times10^{-4}$ à $1.2\times10^{-1}$, deux ordres et demi. Une seule seed ne dirait rien.
# - **Le rayon spectral final reste $\approx 0.9975$**, la bonne valeur. Le modèle n'a donc pas
#   divergé : il s'est arrêté sur un point qui a *presque* le bon spectre et une erreur
#   d'opérateur $10^{13}$ fois trop grande. C'est une information : l'échec n'est pas une
#   instabilité, c'est un plateau.
#
# **Ce qu'on ne peut pas conclure, et il faut être net.** (i) Cette maquette **ne reproduit pas**
# D3 : elle n'a ni encodeur, ni MLP non linéaires, ni tête contextuelle, ni la même échelle ; le
# fait que son erreur terminale ($10^{-3}$ à $10^{-1}$) recouvre l'ordre de grandeur mesuré en
# §21.0 ($0.189$) est une **coïncidence numérique** et ne doit pas être présenté autrement.
# (ii) Elle ne démontre pas que le rollout fermé soit la cause dans le run réel — elle démontre
# qu'il **suffit**, sur une maquette où tout le reste est contrôlé, à produire un écart de la
# bonne nature. (iii) Elle ne dit rien sur ce qu'un curriculum (rollout de longueur croissante),
# un *teacher forcing* partiel, ou une borne sur $r$ changeraient : ces trois correctifs sont
# hors du protocole déclaré du dépôt, et donc hors mesure.
#
# **Ce que la maquette établit, et c'est déjà beaucoup.** Sur un problème où la solution exacte
# est dans la classe et récupérable en forme close, ni la non-convexité de la paramétrisation,
# ni le bruit de mini-lot, ni le mauvais conditionnement (Gram à $2.4$) ne suffisent à empêcher
# la descente de gradient d'atteindre la précision machine. Le **régime d'entraînement**, lui,
# y suffit à lui seul. C'est le candidat que la miniature désigne, et c'est un candidat
# testable : il se falsifie en rajoutant un bras à curriculum.

# %% [markdown]
# ## 5. Une dégénérescence de jauge dans la cellule structurée
#
# ### 5.1 La cellule, telle qu'elle est écrite
#
# Le bras à structure ON du dépôt (`src/sld/models/structured.py`) implémente
#
# $$h_0 = \psi(z_0), \qquad h_{t+1} = \bar A_t\,h_t + \bar B_t\,u_t, \qquad \hat z_t = \pi(h_t),$$
#
# où $\bar A_t$ est bloc-diagonale de blocs rotation-scaling (§2), $\bar B_t = \Phi_t B_t$ avec
# $\Phi_t$ l'intégrale ZOH (§2.4), et où $\psi$ et $\pi$ sont **deux MLP indépendants** :
# `psi = _make_mlp(d_z, 2r, ...)`, `pi = _make_mlp(2r, d_z, ...)`. Il n'y a
# **aucun terme de reconstruction** dans les objectifs, et **aucune contrainte d'inversibilité**
# — une recherche de `recon|invers|pinv|orthogon` dans `src/sld/objectives/` ne rend rien.
#
# Cette absence a une conséquence exacte, démontrable, et elle n'est pas un détail.

# %% [markdown]
# ### 5.2 Théorème de jauge
#
# **Énoncé.** Soit $\mathcal{G} = \big(\mathbb{R}_{>0} \times SO(2)\big)^{r}$, agissant sur les
# paramètres $(\psi, \bar A, \bar B, \pi)$ par
#
# $$g\cdot(\psi,\bar A,\bar B,\pi) \;=\; \big(\,T\psi,\ \ T\bar A T^{-1},\ \ T\bar B,\ \ \pi\circ T^{-1}\,\big),
# \qquad T \;=\; \mathrm{blockdiag}\big(c_1 R(\theta_1),\ \ldots,\ c_r R(\theta_r)\big).$$
#
# Alors, pour toute condition initiale $z_0$ et toute séquence d'actions $(u_t)$, la sortie
# $(\hat z_t)_{t\ge 0}$ est **exactement inchangée**. De plus $T\bar A T^{-1} = \bar A$ : la
# transition elle-même est un point fixe de l'action.
#
# *Démonstration, en trois pas.*
#
# **(i) $T$ commute avec $\bar A$.** Les deux matrices sont bloc-diagonales de même découpage ;
# il suffit donc de le vérifier bloc par bloc. Sur le bloc $i$,
#
# $$\big(c_i R(\theta_i)\big)\,\big(e^{\Delta\alpha_i}R(\Delta\omega_i)\big)\,\big(c_i R(\theta_i)\big)^{-1}
# = e^{\Delta\alpha_i}\,R(\theta_i)R(\Delta\omega_i)R(-\theta_i)
# = e^{\Delta\alpha_i}R(\theta_i + \Delta\omega_i - \theta_i) = \bar A_i,$$
#
# où l'on a utilisé (a) que le scalaire $c_i$ se simplifie car il commute avec tout, et (b) que
# $SO(2)$ est **abélien** : $R(a)R(b) = R(a+b)$. C'est le cœur de la démonstration, et c'est ce
# qui rend cette jauge propre à la paramétrisation rotation-scaling : elle vient de ce que le
# groupe de rotations planes commute avec lui-même. *(Le même argument vaut pour $\Phi_i$, qui
# appartient à la même algèbre $\mathcal{A} = \{aI + bJ\}$ de §2.1 — donc $T\bar B$ se réalise
# aussi bien en transformant $B$ seul, sans toucher à $\Phi$.)*
#
# **(ii) L'état caché est simplement transporté.** Posons $h'_t$ l'état de la cellule
# transformée. Par récurrence : $h'_0 = T\psi(z_0) = T h_0$ ; et si $h'_t = T h_t$, alors
#
# $$h'_{t+1} = (T\bar A T^{-1})\,h'_t + T\bar B u_t
# = \bar A\,T h_t + T\bar B u_t
# = T\big(\bar A h_t + \bar B u_t\big) = T\,h_{t+1}.$$
#
# **(iii) La sortie est invariante.** $\hat z'_t = (\pi\circ T^{-1})(h'_t) = \pi(T^{-1}Th_t) = \pi(h_t) = \hat z_t$. $\square$
#
# **Réalisation dans les paramètres d'un MLP.** L'action n'est pas abstraite : $T\psi$ s'obtient
# en remplaçant la **dernière couche linéaire** de $\psi$, $(W_L, b_L)$, par $(TW_L, Tb_L)$ ; et
# $\pi\circ T^{-1}$ en remplaçant la **première couche linéaire** de $\pi$, $W_1$, par
# $W_1T^{-1}$. C'est exact pour tout MLP dont les couches de bord sont affines — donc pour
# l'implémentation standard. Aucune approximation n'intervient, quelle que soit la
# non-linéarité interne.

# %% [markdown]
# ### 5.3 Combien de directions plates
#
# Le groupe $\mathcal{G}$ est de dimension $2r$ : un facteur d'échelle et un angle par bloc.
# L'action laisse la perte **exactement** invariante — pas approximativement, pas au voisinage
# d'un optimum, mais partout et pour toute perte qui ne dépend des paramètres qu'à travers
# $(\hat z_t)$. Donc :
#
# > **[DÉMONTRÉ]** La perte possède au moins $2r$ directions de courbure exactement nulle en
# > tout point où l'action est localement libre. Sur la configuration D3 ($r = 2$) : **4
# > directions de dégénérescence exacte**.
#
# Les générateurs infinitésimaux sont explicites. Pour le bloc $i$, en notant $E_i$ le
# projecteur sur le bloc $i$ et $L_i = E_i J$ :
#
# | | direction sur $\psi$ | sur $\bar B$ | sur $\pi$ (cas linéaire $\pi = Q$) |
# |---|---|---|---|
# | échelle $i$ | $E_i\psi$ | $E_i\bar B$ | $-QE_i$ |
# | rotation $i$ | $L_i\psi$ | $L_i\bar B$ | $-QL_i$ |
#
# **Pourquoi c'est un problème, et pourquoi ce n'en est peut-être pas un.** Côté problème : une
# variété d'optima de dimension $2r$ signifie que la norme des paramètres n'est pas identifiée,
# donc que toute régularisation en norme ($\ell_2$, weight decay) agit sur la **jauge** et non
# sur la fonction ; et que deux runs peuvent aboutir à des paramètres arbitrairement éloignés
# tout en calculant la même chose — ce qui rend illégitime toute comparaison de paramètres entre
# seeds. Côté « peut-être pas » : les réseaux profonds sont massivement invariants
# (permutations, rescaling ReLU) et s'entraînent quand même. **Une dégénérescence exacte n'est
# pas une preuve de difficulté d'optimisation.** C'est un fait de structure ; son effet est une
# question empirique distincte, sur laquelle nous reviendrons en §5.6.
#
# Vérifions les deux jauges, la commutation, et le compte de directions.

# %%
# --- [DEMONTRE -> verifie] Les deux jauges, sur une cellule complete (MLP + actions) --------
D_U = 2                                   # dimension d'action
TWO_R = 2 * R_BLOCKS


def mlp_init(dims, seed):
    r = np.random.default_rng(seed)
    Ws = [r.standard_normal((dims[i + 1], dims[i])) / np.sqrt(dims[i]) for i in range(len(dims) - 1)]
    bs = [0.1 * r.standard_normal(dims[i + 1]) for i in range(len(dims) - 1)]
    return [Ws, bs]


def mlp_apply(net, x):
    Ws, bs = net
    h = x
    for i, (W, b) in enumerate(zip(Ws, bs)):
        h = h @ W.T + b
        if i < len(Ws) - 1:
            h = np.tanh(h)                # non-linearite interne : la jauge n'en depend pas
    return h


def cell_rollout(psi, pi, Ab, Bb, z0, U):
    """h0 = psi(z0) ; h_{t+1} = Ab h_t + Bb u_t ; z_hat_t = pi(h_t)."""
    h = mlp_apply(psi, z0)
    outs = []
    for t in range(U.shape[1]):
        h = h @ Ab.T + U[:, t] @ Bb.T
        outs.append(mlp_apply(pi, h))
    return np.stack(outs, axis=1)


def gauge_T(scales, angles):
    """T = blockdiag(c_i R(theta_i))."""
    T = np.zeros((TWO_R, TWO_R))
    for i in range(R_BLOCKS):
        th = angles[i]
        T[2 * i:2 * i + 2, 2 * i:2 * i + 2] = scales[i] * np.array(
            [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return T


def apply_gauge(psi, pi, Bb, T):
    """(T psi, T Bb, pi o T^-1) realise dans les poids : derniere couche de psi, premiere de pi."""
    Ti = np.linalg.inv(T)
    psi_g = [[*psi[0][:-1], T @ psi[0][-1]], [*psi[1][:-1], T @ psi[1][-1]]]
    pi_g = [[pi[0][0] @ Ti, *pi[0][1:]], [*pi[1]]]
    return psi_g, pi_g, T @ Bb


rg = np.random.default_rng(5)
psi = mlp_init([DZ, 16, 16, TWO_R], 1)
pi = mlp_init([TWO_R, 16, 16, DZ], 2)
alpha_g = np.array([-0.20, 0.10]); omega_g = np.array([2.0, -1.3])
Ab_g = A_blocks(alpha_g, omega_g)
Phi_g = np.zeros((TWO_R, TWO_R))
for i in range(R_BLOCKS):
    Phi_g[2 * i:2 * i + 2, 2 * i:2 * i + 2] = Phi_closed(alpha_g[i], omega_g[i])
B_raw = rg.standard_normal((TWO_R, D_U)) / 2
Bb_g = Phi_g @ B_raw

z0_g = rg.standard_normal((32, DZ))
U_g = rg.standard_normal((32, 20, D_U))
ref = cell_rollout(psi, pi, Ab_g, Bb_g, z0_g, U_g)
scale_ref = np.abs(ref).max()

print("(i) commutation, bloc par bloc")
for name, T in [("echelle pure  ", gauge_T([0.1, 7.3], [0.0, 0.0])),
                ("rotation pure ", gauge_T([1.0, 1.0], [0.9, -2.4])),
                ("echelle+rot.  ", gauge_T([0.1, 7.3], [0.9, -2.4]))]:
    print(f"   {name} : |T Abar T^-1 - Abar|_inf = "
          f"{np.abs(T @ Ab_g @ np.linalg.inv(T) - Ab_g).max():.2e}"
          f"   |T Phi - Phi T|_inf = {np.abs(T @ Phi_g - Phi_g @ T).max():.2e}")

print("\n(ii)+(iii) invariance EXACTE de la sortie latente (erreur relative au max de |z_hat|)")
for name, T in [("echelle (c = 0.1, 7.3)   ", gauge_T([0.1, 7.3], [0.0, 0.0])),
                ("echelle (c = 1000, 1e-3) ", gauge_T([1000.0, 1e-3], [0.0, 0.0])),
                ("rotation (0.9, -2.4 rad) ", gauge_T([1.0, 1.0], [0.9, -2.4])),
                ("combinee                 ", gauge_T([0.1, 7.3], [0.9, -2.4]))]:
    psi_g, pi_g, Bb_gg = apply_gauge(psi, pi, Bb_g, T)
    out = cell_rollout(psi_g, pi_g, Ab_g, Bb_gg, z0_g, U_g)
    print(f"   {name} : max|dz_hat| = {np.abs(out - ref).max():.2e}"
          f"   relatif = {np.abs(out - ref).max() / scale_ref:.2e}")

# %%
# --- [DEMONTRE -> verifie] Le compte : 2r directions plates, et elles sont independantes ----
# Sur le coeur lineaire (psi = P, pi = Q), on construit les 2r generateurs infinitesimaux et
# on verifie (a) que la derivee de l'operateur predit le long de chacun est nulle,
# (b) qu'ils sont lineairement independants, donc que la variete plate est bien de dim 2r.
P_lin = rg.standard_normal((TWO_R, DZ)) / 2
Q_lin = rg.standard_normal((DZ, TWO_R)) / 2
M_ref = Q_lin @ Ab_g @ P_lin

gens = []
for i in range(R_BLOCKS):
    E = np.zeros((TWO_R, TWO_R)); E[2 * i:2 * i + 2, 2 * i:2 * i + 2] = np.eye(2)
    L = np.zeros((TWO_R, TWO_R)); L[2 * i:2 * i + 2, 2 * i:2 * i + 2] = J2
    gens.append((f"echelle bloc {i}", E))
    gens.append((f"rotation bloc {i}", L))

print("derivee de M = Q Abar P le long de chaque generateur, et invariance sur un chemin fini :")
tangents = []
for name, Xg in gens:
    dM = Q_lin @ (Ab_g @ Xg - Xg @ Ab_g) @ P_lin          # d/de [Q e^{-eX} Abar e^{eX} P]
    Texp = expm_series(0.7 * Xg)
    M_path = (Q_lin @ np.linalg.inv(Texp)) @ Ab_g @ (Texp @ P_lin)
    tangents.append(np.concatenate([(Xg @ P_lin).ravel(), (-Q_lin @ Xg).ravel()]))
    print(f"   {name:>18} : |dM/de|_inf = {np.abs(dM).max():.2e}"
          f"    |M(e=0.7) - M(0)|_inf = {np.abs(M_path - M_ref).max():.2e}")

Tg = np.array(tangents)
sv = np.linalg.svd(Tg, compute_uv=False)
print(f"\nvaleurs singulieres des {len(gens)} vecteurs tangents : {np.round(sv, 4)}")
print(f"rang numerique (seuil 1e-8) = {int(np.sum(sv > 1e-8))}   attendu 2r = {2 * R_BLOCKS}")
print(f"dimension de l'espace de parametres du coeur lineaire = {P_lin.size + Q_lin.size}")

# %% [markdown]
# **Lecture.** Les trois blocs de sortie disent trois choses distinctes, et il faut les
# distinguer.
#
# 1. **La commutation est exacte** : $T\bar A T^{-1} = \bar A$ à $10^{-16}$, pour l'échelle
#    seule, la rotation seule et la combinaison. Idem pour $\Phi$, ce qui garantit que la jauge
#    se réalise sur $B$ sans toucher à l'intégrale ZOH — donc sans toucher à la tête.
# 2. **La sortie latente est invariante à la précision machine**, sur un rollout de 20 pas avec
#    actions, à travers deux MLP à tanh, y compris pour des facteurs d'échelle extrêmes
#    ($10^{3}$ et $10^{-3}$). L'erreur résiduelle relative est de l'ordre de $10^{-16}$ : c'est
#    de l'arrondi, pas une approximation.
# 3. **Le compte est confirmé** : les 4 générateurs annulent exactement la dérivée de
#    l'opérateur prédit, l'invariance tient aussi sur un chemin **fini** ($\varepsilon = 0.7$,
#    donc loin de l'infinitésimal), et les 4 vecteurs tangents sont linéairement indépendants
#    (4 valeurs singulières franchement non nulles). La variété plate est bien de dimension
#    $2r = 4$ dans un espace de paramètres de dimension 32 pour le seul cœur linéaire.
#
# **Ce qu'on ne peut pas conclure.** (i) Que ces $2r$ directions soient les **seules** : le
# théorème donne une borne inférieure. (ii) Que la perte réelle soit invariante — elle l'est
# *si et seulement si* elle ne dépend des paramètres qu'à travers $(\hat z_t)$ ; un terme de
# régularisation en norme des poids briserait l'invariance, et c'est précisément pour cela
# qu'un weight decay agirait ici sur la jauge plutôt que sur la fonction. (iii) Que cela
# ralentisse l'entraînement. Rien dans cette cellule ne le mesure.

# %% [markdown]
# ### 5.4 $\pi\circ\psi \neq \mathrm{Id}$, et le correctif analytique
#
# La jauge n'est pas le seul effet de l'absence de contrainte entre $\psi$ et $\pi$. Il y a plus
# élémentaire : **rien n'oblige $\pi$ à défaire $\psi$**. Deux MLP initialisés indépendamment
# donnent $\pi(\psi(z)) \neq z$, et la perte ne contient aucun terme qui pousse vers l'égalité.
#
# Pourquoi cela devrait déranger. Le rôle prêté à $\psi$ dans le récit Koopman est celui d'un
# **plongement d'observables** : on monte dans un espace où la dynamique devient linéaire, on y
# avance, on redescend. Si la descente n'est pas l'inverse de la montée, l'objet appris n'est
# pas « la dynamique dans les coordonnées de Koopman » — c'est un réseau à trois couches dont
# celle du milieu se trouve être bloc-diagonale. La lisibilité que le prior promettait
# (§2.3 : *on lit un taux et une pulsation*) n'est alors garantie qu'à une conjugaison inconnue
# près.
#
# **Le correctif, dans le cas linéaire.** Si $\psi$ est linéaire de matrice $P \in \mathbb{R}^{2r\times d_z}$,
# poser $\pi = \psi^{+}$, c'est-à-dire $Q = P^{+}$ (pseudo-inverse de Moore–Penrose), donne
#
# $$Q P = P^{+}P = \mathrm{Id}_{d_z} \qquad \Longleftrightarrow \qquad \mathrm{rang}(P) = d_z
# \qquad \Longleftarrow \qquad 2r \ge d_z \ \text{et}\ P\ \text{de rang plein}.$$
#
# *Démonstration.* $P^{+}P$ est le projecteur orthogonal sur $\mathrm{ligne}(P) \subseteq \mathbb{R}^{d_z}$
# (propriété caractéristique de Moore–Penrose). Ce projecteur vaut $\mathrm{Id}_{d_z}$ si et
# seulement si $\mathrm{ligne}(P) = \mathbb{R}^{d_z}$, c'est-à-dire
# $\mathrm{rang}(P) = d_z$. Comme $\mathrm{rang}(P) \le \min(2r, d_z)$, la condition
# $2r \ge d_z$ est **nécessaire** ; elle est suffisante pour $P$ générique. $\square$
#
# Noter le sens : c'est $P^{+}P = \mathrm{Id}$ (l'aller-retour $z \to h \to z$) qui est
# récupérable, **pas** $PP^{+} = \mathrm{Id}$ — ce dernier demanderait $2r \le d_z$, donc les
# deux ensemble forceraient $2r = d_z$ et $P$ inversible. Sur D3, $2r = d_z = 4$ : le cas
# frontière.
#
# **La littérature impose cette contrainte, et le dépôt l'omet.** Dans les auto-encodeurs de
# Koopman profonds, le terme de reconstruction $\|\pi(\psi(z)) - z\|^2$ est présent **par
# construction** — voir [Lusch, Kutz & Brunton, *Deep learning for universal linear embeddings
# of nonlinear dynamics*, Nature Communications 9, 4950 (2018)](https://www.nature.com/articles/s41467-018-07210-0),
# dont la perte comporte explicitement une composante de reconstruction en plus de la
# composante de prédiction. Ce n'est pas un raffinement esthétique : c'est ce qui fait de
# $\psi$ un plongement plutôt qu'une couche.
#
# **L'alternative écartée, et pourquoi.** On pourrait imposer $\pi = \psi^\top$ avec $\psi$
# contraint orthogonal (par paramétrisation de Cayley ou de Householder). C'est plus fort
# — l'aller-retour est exact **et** les normes sont préservées, donc la jauge d'échelle
# disparaît aussi — mais cela coûte une contrainte de variété à chaque pas, et cela change ce
# que le facteur `structure` désigne : on ajouterait de l'orthogonalité au composite déjà
# constitué de (transition + $\psi$ + $\pi$). Or §1.5 rappelle que ce composite est déjà le
# principal obstacle à l'interprétation causale. Ajouter un quatrième ingrédient au facteur,
# c'est aggraver le défaut qu'on cherche à corriger.

# %%
# --- [DEMONTRE -> verifie] pi o psi n'est pas l'identite ; le correctif pseudo-inverse ------
zz = rg.standard_normal((256, DZ))

print("(a) deux MLP independants : pi(psi(z)) vs z")
err_mlp = np.abs(mlp_apply(pi, mlp_apply(psi, zz)) - zz).max()
print(f"    max |pi(psi(z)) - z| = {err_mlp:.3e}   (a comparer a max|z| = {np.abs(zz).max():.2f})")

print("\n(b) coeur lineaire, deux matrices independantes")
print(f"    max |Q P - I| = {np.abs(Q_lin @ P_lin - np.eye(DZ)).max():.3e}")

print("\n(c) correctif pi = psi^+ (pseudo-inverse), selon la relation entre 2r et d_z")
print(f"{'2r':>5} {'d_z':>5} | {'rang(P)':>8} | {'|P^+ P - I_dz|':>16} | {'|P P^+ - I_2r|':>16}")
print("-" * 62)
for two_r in [2, 3, 4, 6, 8]:
    Pm = np.random.default_rng(40 + two_r).standard_normal((two_r, DZ))
    Qm = np.linalg.pinv(Pm)
    e_left = np.abs(Qm @ Pm - np.eye(DZ)).max()
    e_right = np.abs(Pm @ Qm - np.eye(two_r)).max()
    print(f"{two_r:>5} {DZ:>5} | {np.linalg.matrix_rank(Pm):>8} | {e_left:16.3e} | {e_right:16.3e}")

print("\n(d) cas degenere : P de rang deficient bien que 2r >= d_z")
Pd = np.random.default_rng(9).standard_normal((6, DZ))
Pd[:, 3] = Pd[:, 0] + Pd[:, 1]                     # colonne dependante -> rang 3 < d_z = 4
print(f"    rang(P) = {np.linalg.matrix_rank(Pd)} < d_z = {DZ}"
      f"  ->  |P^+ P - I| = {np.abs(np.linalg.pinv(Pd) @ Pd - np.eye(DZ)).max():.3e}")
print("    la condition 2r >= d_z est NECESSAIRE, elle n'est pas SUFFISANTE.")

print("\n(e) le correctif elimine-t-il la jauge ? (verification, pas affirmation)")
for c in [0.5, 2.0]:
    Pc = c * P_lin
    print(f"    P -> {c}*P  :  |pinv(cP)(cP) - I| = "
          f"{np.abs(np.linalg.pinv(Pc) @ Pc - np.eye(DZ)).max():.2e}"
          f"   |pinv(cP) - pinv(P)/{c}| = {np.abs(np.linalg.pinv(Pc) - np.linalg.pinv(P_lin) / c).max():.2e}")

# %% [markdown]
# **Lecture.** (a) et (b) : sans contrainte, l'aller-retour $z \to h \to z$ est faux d'un ordre
# de grandeur comparable au signal lui-même ; ce n'est pas une petite erreur, c'est une
# transformation arbitraire. (c) : le correctif $\pi = \psi^{+}$ donne $P^{+}P = \mathrm{Id}$ à
# $10^{-15}$ **dès que $2r \ge d_z$**, et échoue proprement quand $2r < d_z$ — la ligne
# $2r = 2$ affiche une erreur d'ordre 1, ce qui est le comportement attendu d'une projection sur
# un sous-espace strict. La colonne de droite montre la dissymétrie : $PP^{+} = \mathrm{Id}_{2r}$
# ne tient que dans l'autre sens, et les deux ne coïncident qu'à $2r = d_z$.
#
# (d) est le garde-fou honnête : la condition $2r \ge d_z$ est **nécessaire et non suffisante**.
# Un $P$ de rang déficient la satisfait et échoue quand même. Une implémentation qui se
# contenterait de vérifier $2r \ge d_z$ passerait à côté.
#
# (e) est une **réfutation partielle de notre propre correctif** : la jauge d'échelle **survit**
# à $\pi = \psi^{+}$, puisque $(cP)^{+} = P^{+}/c$ exactement, donc le couple $(cP, (cP)^{+})$
# est encore dans l'orbite. Le pseudo-inverse fixe l'aller-retour, il ne fixe pas la jauge. Pour
# supprimer la jauge d'échelle il faudrait normaliser $P$ (par exemple $\|P\|_F = 1$), et pour
# supprimer la jauge de rotation il faudrait fixer une phase par bloc. **Le correctif ne fait
# donc pas ce qu'un lecteur pressé lui attribuerait.** Cela devait être écrit.

# %% [markdown]
# ### 5.5 Statut épistémique, noir sur blanc
#
# Cette section a établi deux choses de statuts très différents, et la distinction est le point
# le plus important du notebook.
#
# | énoncé | statut | preuve |
# |---|---|---|
# | Il existe une orbite de dimension $2r$ le long de laquelle la prédiction latente est exactement constante | **DÉMONTRÉ**, vérifié à $10^{-16}$ | §5.2, §5.3 |
# | $\pi\circ\psi \neq \mathrm{Id}$ sans terme de reconstruction | **DÉMONTRÉ**, vérifié | §5.4 |
# | $\pi = \psi^{+}$ restaure l'aller-retour dès $2r \ge d_z$ (et ne supprime pas la jauge) | **DÉMONTRÉ**, vérifié | §5.4 |
# | Cette jauge **explique** l'écart d'exactitude mesuré en §4 | **HYPOTHÈSE NON MESURÉE** | aucune |
# | Ajouter le terme de reconstruction **améliorerait** les runs | **HYPOTHÈSE NON MESURÉE** | aucune |
#
# Les deux dernières lignes ne sont pas des litotes. Le dépôt a **retiré** (amendement 13,
# `PROTOCOL.md §22`) une conclusion antérieure qui reliait trop vite la structure $\psi/\pi$ à
# l'échec d'exactitude : la conclusion de §21.1 est retirée **purement**, sans réécriture, avec
# la mention explicite que le contrefactuel envisagé — « insérer l'opérateur exact dans des
# $\psi/\pi$ entraînés » — est **hors distribution, donc mal posé**. Il aurait été plus flatteur
# d'écrire que $\psi$ et $\pi$ « compensent » ; c'est exactement le raisonnement non mesuré qui
# avait déjà invalidé un axe antérieur du dépôt.
#
# La règle appliquée ici : *une dégénérescence démontrée n'autorise pas une conclusion sur
# l'optimisation.* Le lien est testable — il suffit d'un bras avec terme de reconstruction, à
# budget et backbone appariés — et il n'a pas été testé. Tant qu'il ne l'est pas, la seule chose
# publiable est le théorème.

# %% [markdown]
# ## 6. Ce que la taille d'échantillon change
#
# ### 6.1 L'argument, et pourquoi il est décisif
#
# Un biais inductif n'achète pas de la précision : il achète de l'**efficacité d'échantillonnage**.
# C'est une distinction élémentaire et elle décide de la lecture de tout le §4.
#
# **[MESURÉ — `PROTOCOL.md §20.1`]** Le régime de données de la campagne D3 est
# `n_episodes = 256`, soit **24 480 transitions**, pour un système linéaire à **4 états**. Sur ce
# split, les moindres carrés atteignent une erreur spectrale de $2.11\times10^{-8}$.
#
# Comptons ce que cela signifie. Un LTI $x_{t+1} = Ax_t + Bu_t$ avec $d = 4$ états et une action
# scalaire a
#
# $$p_{\text{libre}} \;=\; \underbrace{d^2}_{A\ :\ 16} + \underbrace{d\cdot d_u}_{B\ :\ 4} \;=\; 20
# \quad\text{paramètres réels},$$
#
# et chaque transition observée fournit $d = 4$ équations scalaires. Le seuil d'identifiabilité
# est donc $4n \ge 20$, soit $n \ge 5$ transitions. Avec la contrainte rotation-scaling, $A$ n'a
# plus que $2r = 4$ paramètres (deux blocs, $(\alpha_i,\omega_i)$ chacun), donc
# $p_{\text{contraint}} = 4 + 4 = 8$ et le seuil tombe à $n \ge 2$.
#
# À $n = 24\,480$, on dispose de $97\,920$ équations pour $20$ inconnues : un facteur
# $\approx 4\,900$, **trois ordres de grandeur de sur-détermination**. Dans ce régime, un prior
# correct ne peut rien acheter, parce qu'il n'y a rien à acheter : le problème est résolu par
# comptage.

# %% [markdown]
# ### 6.2 Pourquoi le gain doit décroître — la forme de la loi
#
# **Hypothèses.** (H1) modèle bien spécifié, $x_{t+1} = Ax_t + w_t$, $w_t$ centré, de covariance
# $\sigma^2 I$, indépendant du design ; (H2) design $X$ de $n$ lignes avec
# $X^\top X \approx n\,I_d$ (états blanchis) ; (H3) estimateur des moindres carrés dans une
# classe **linéaire** de $p$ paramètres.
#
# Pour l'estimateur libre, $\hat A - A = \big(X^\top X\big)^{-1}X^\top W$ transposé, d'où
#
# $$\mathbb{E}\,\|\hat A - A\|_F^2 \;=\; \sigma^2\, d\;\mathrm{tr}\big((X^\top X)^{-1}\big)
# \;\approx\; \sigma^2\,\frac{d^2}{n} \;=\; \sigma^2\,\frac{p_{\text{libre}}^{(A)}}{n}.$$
#
# Le même calcul dans une sous-classe linéaire de dimension $p$ (les blocs rotation-scaling en
# sont une : l'application $(a,b)\mapsto \begin{pmatrix} a & -b\\ b & a\end{pmatrix}$
# est **linéaire** en $(a,b)$, donc l'estimateur contraint est lui aussi un moindre carré
# ordinaire) donne $\mathbb{E}\|\hat A - A\|_F^2 \approx \sigma^2 p/n$. Donc
#
# $$\boxed{\;\|\hat A - A\|_F \;\sim\; \sigma\sqrt{\frac{p}{n}},
# \qquad \frac{\text{erreur libre}}{\text{erreur contrainte}} \;\xrightarrow[n\to\infty]{}\;
# \sqrt{\frac{p_{\text{libre}}^{(A)}}{p_{\text{contraint}}^{(A)}}} = \sqrt{\frac{16}{4}} = 2.\;}$$
#
# **Trois conséquences, et la troisième est celle qu'on lit mal.**
#
# 1. Sous le seuil d'identifiabilité du modèle libre ($n < 5$), le prior n'apporte pas un gain
#    de facteur 2 : il apporte la différence entre **estimable** et **non estimable**. Le gain
#    y est arbitrairement grand.
# 2. En régime asymptotique, le **rapport** des erreurs tend vers une constante ($2$ ici), il ne
#    tend pas vers $1$. Écrire « l'écart se referme » sans préciser en quel sens est faux.
# 3. En revanche l'écart **absolu** $\sigma(\sqrt{p_1}-\sqrt{p_2})/\sqrt{n}$ s'effondre en
#    $1/\sqrt{n}$, et c'est lui qui décide face à un seuil de tâche fixé. À $n = 24\,480$, les
#    deux estimateurs sont sous n'importe quel seuil raisonnable : le prior est correct, gratuit,
#    et **sans effet mesurable**. C'est exactement le régime de D3.
#
# **La prédiction pré-enregistrée du dépôt.** La campagne déclarée en `PROTOCOL.md §23`
# (amendement 14, écrit avant tout run) fixe sa règle de lecture primaire **avant** les chiffres :
# ce qui est lu est le **signe de l'interaction** $\text{gain(structure)} \times \log n$, jamais
# un franchissement de seuil. L'analyse ci-dessus prédit ce signe : **négatif** — le gain de la
# structure décroît quand $n$ croît. Si le signe mesuré était positif, l'analyse de §6.2 serait
# fausse et devrait être retirée.

# %% [markdown]
# ### 6.3 L'expérience jouet, et ce qu'on concède au prior
#
# **[ILLUSTRATIF]** Deux estimateurs, deux systèmes, un balayage en $n$.
#
# - **Estimateur libre** : moindres carrés sur les 16 coefficients de $A$, sans contrainte.
# - **Estimateur contraint** : moindres carrés sur 4 coefficients — deux blocs rotation-scaling
#   $\begin{pmatrix} a_i & -b_i\\ b_i & a_i\end{pmatrix}$ — **dans une base modale $V$
#   qu'on lui donne**.
#
# Ce dernier point est une **concession explicite au prior**, et il faut la nommer : dans un
# modèle appris, la base est apprise par $\psi$ et $\pi$, elle n'est pas donnée. En la donnant,
# on retire au prior son coût principal et on isole l'effet du comptage de paramètres. Le
# résultat est donc une **borne optimiste** sur ce qu'un prior structurel peut acheter.
# On choisit de plus $V$ **orthogonale** ($\mathrm{cond}(V) = 1$), pour que le design reste
# isotrope dans la base modale — sinon le biais de mauvaise spécification dépendrait aussi de
# la covariance, et on mélangerait deux effets.
#
# Système **A** (bien spécifié) : $V\,\mathrm{blockdiag}\big(0.95R(0.7),\,0.88R(-1.9)\big)V^\top$
# — le prior est exact, $\delta = 0$.
# Système **B** (mal spécifié) : le premier bloc est remplacé par
# $\mathrm{diag}(\lambda_a,\lambda_b)$ avec $\lambda_a = 0.95$, $\lambda_b = 0.40$, deux réelles
# distinctes. Par le théorème 3.3, $\delta = |\lambda_a-\lambda_b|/2 = 0.275$.

# %%
# --- [ILLUSTRATIF] Efficacite d'echantillonnage : prior exact vs prior faux -----------------
rng6 = np.random.default_rng(2024)
D6 = 4


def blkdiag2(B1, B2):
    M = np.zeros((4, 4)); M[:2, :2] = B1; M[2:, 2:] = B2
    return M


V6, _ = np.linalg.qr(rng6.standard_normal((D6, D6)))      # base modale ORTHOGONALE, donnee au prior
M_ok = blkdiag2(0.95 * rot(0.7), 0.88 * rot(-1.9))        # 2 paires conjuguees : delta = 0
LA, LB = 0.95, 0.40
M_bad = blkdiag2(np.diag([LA, LB]), 0.88 * rot(-1.9))     # 2 reelles distinctes : delta > 0
A_ok, A_bad = V6 @ M_ok @ V6.T, V6 @ M_bad @ V6.T
SIGMA = 0.05
delta_bad = abs(LA - LB) / 2

print(f"systeme A : spectre {np.round(canon_sort(np.linalg.eigvals(A_ok)), 4)}")
print(f"            delta = {0.0:.4f}  (2 paires conjuguees : prior EXACT)")
print(f"systeme B : spectre {np.round(canon_sort(np.linalg.eigvals(A_bad)), 4)}")
print(f"            delta = {delta_bad:.4f}  (theoreme 3.3 : |la - lb| / 2)")


def fit_free(Xs, Ys):
    """Moindres carres sur les 16 coefficients de A."""
    return np.linalg.lstsq(Xs, Ys, rcond=None)[0].T


def fit_constrained(Xs, Ys, V):
    """Moindres carres sur 4 coefficients : deux blocs [[a,-b],[b,a]] dans la base modale V.

    La parametrisation (a, b) -> [[a,-b],[b,a]] est LINEAIRE, donc c'est encore un moindre
    carre ordinaire -- pas une optimisation non convexe. C'est ce qui rend la comparaison
    propre : deux estimateurs de meme nature, seul p change.
    """
    Xm, Ym = Xs @ V, Ys @ V                        # V orthogonale : V^{-1} = V^T
    M = np.zeros((4, 4))
    for b in range(2):
        i = 2 * b
        x1, x2 = Xm[:, i], Xm[:, i + 1]
        y1, y2 = Ym[:, i], Ym[:, i + 1]
        Dsg = np.stack([np.concatenate([x1, x2]), np.concatenate([-x2, x1])], axis=1)
        a_b = np.linalg.lstsq(Dsg, np.concatenate([y1, y2]), rcond=None)[0]
        M[i:i + 2, i:i + 2] = np.array([[a_b[0], -a_b[1]], [a_b[1], a_b[0]]])
    return V @ M @ V.T


def sweep(A_true_, ns, reps=24):
    out = {"free": [], "cons": [], "free_q": [], "cons_q": []}
    for n in ns:
        ef, ec = [], []
        for rep in range(reps):
            rr = np.random.default_rng(97_000 + 131 * rep + n)
            Xs = rr.standard_normal((n, D6))
            Ys = Xs @ A_true_.T + SIGMA * rr.standard_normal((n, D6))
            ef.append(np.linalg.norm(fit_free(Xs, Ys) - A_true_))
            ec.append(np.linalg.norm(fit_constrained(Xs, Ys, V6) - A_true_))
        out["free"].append(np.median(ef)); out["cons"].append(np.median(ec))
        out["free_q"].append((np.quantile(ef, .25), np.quantile(ef, .75)))
        out["cons_q"].append((np.quantile(ec, .25), np.quantile(ec, .75)))
    return {k: np.array(v) for k, v in out.items()}


NS = np.array([1, 2, 3, 4, 5, 7, 10, 15, 25, 50, 100, 250, 600, 1500, 4000, 10000])
res_ok, res_bad = sweep(A_ok, NS), sweep(A_bad, NS)
print("\nbalayage termine :", len(NS), "valeurs de n, 24 repetitions chacune")

# %%
# --- [ILLUSTRATIF] La figure : erreur contre n, et le plafond de biais ---------------------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))

for ax, res, A_t, titre, exact in [
    (axes[0], res_ok, A_ok, "(a) systeme A : prior EXACT ($\\delta = 0$)", True),
    (axes[1], res_bad, A_bad, f"(b) systeme B : prior FAUX ($\\delta = {delta_bad:.3f}$)", False),
]:
    for key, lab, col in [("free", "libre (16 parametres)", "tab:blue"),
                          ("cons", "contraint (4 parametres)", "tab:red")]:
        ax.loglog(NS, res[key], "o-", ms=4, color=col, label=lab)
        ax.fill_between(NS, res[key + "_q"][:, 0], res[key + "_q"][:, 1], color=col, alpha=0.15)
    # lois theoriques sigma sqrt(p / n), sans constante ajustee
    ax.loglog(NS, SIGMA * np.sqrt(16 / NS), "--", color="tab:blue", lw=1, alpha=0.8,
              label=r"$\sigma\sqrt{16/n}$")
    ax.loglog(NS, SIGMA * np.sqrt(4 / NS), "--", color="tab:red", lw=1, alpha=0.8,
              label=r"$\sigma\sqrt{4/n}$")
    ax.axvline(5, color="k", ls=":", lw=1)
    ax.text(5.6, 4.0, "seuil d'identifiabilite\ndu modele libre ($n=5$)", fontsize=7.5,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax.set_ylim(3e-4, 12.0)
    if not exact:
        ax.axhline(np.sqrt(2) * delta_bad, color="tab:red", ls="-.", lw=1.2)
        ax.text(1.6, np.sqrt(2) * delta_bad * 0.33,
                r"plafond de biais $\sqrt{2}\,\delta$", fontsize=8, color="darkred")
    ax.set(xlabel="$n$ (transitions)", ylabel=r"$\|\hat A - A\|_F$ (mediane, 24 rep.)", title=titre)
    ax.legend(fontsize=7.5, loc="lower left")

ax = axes[2]
ax.semilogx(NS, res_ok["free"] / res_ok["cons"], "o-", ms=4, color="tab:green",
            label="systeme A (prior exact)")
ax.semilogx(NS, res_bad["free"] / res_bad["cons"], "s-", ms=4, color="tab:purple",
            label="systeme B (prior faux)")
ax.axhline(1.0, color="k", lw=1)
ax.axhline(2.0, color="tab:green", ls="--", lw=1, label=r"asymptote $\sqrt{p_1/p_2}=2$")
ax.set(xlabel="$n$ (transitions)", ylabel="erreur libre / erreur contrainte",
       title="(c) gain du prior ; $>1$ = le prior aide", yscale="log")
ax.legend(fontsize=8, loc="lower left")
plt.tight_layout(); plt.show()

print(f"{'n':>7} | {'A: libre':>10} {'A: contr.':>10} {'gain':>7} | "
      f"{'B: libre':>10} {'B: contr.':>10} {'gain':>7}")
print("-" * 76)
for i, n in enumerate(NS):
    print(f"{n:7d} | {res_ok['free'][i]:10.3e} {res_ok['cons'][i]:10.3e} "
          f"{res_ok['free'][i]/res_ok['cons'][i]:7.2f} | "
          f"{res_bad['free'][i]:10.3e} {res_bad['cons'][i]:10.3e} "
          f"{res_bad['free'][i]/res_bad['cons'][i]:7.2f}")

print(f"\nplafond mesure du contraint sur le systeme B (n = {NS[-1]}) : {res_bad['cons'][-1]:.6f}")
print(f"prediction analytique sqrt(2) * delta                      : {np.sqrt(2)*delta_bad:.6f}")
print(f"ecart relatif                                              : "
      f"{abs(res_bad['cons'][-1] - np.sqrt(2)*delta_bad)/(np.sqrt(2)*delta_bad):.2e}")
cross = NS[np.argmax(res_bad["free"] / res_bad["cons"] < 1.0)]
print(f"croisement sur le systeme B (le libre repasse devant)       : n = {cross}")

# %% [markdown]
# **Lecture.** Trois régimes, et ils sont tous visibles.
#
# **Panneau (a) — prior exact.** Sous $n = 5$, l'estimateur libre est **sous-déterminé** : les
# moindres carrés rendent la solution de norme minimale, et l'erreur est d'ordre 1. L'estimateur
# contraint, lui, est déjà identifié à $n = 2$ et se trompe dix à seize fois moins. C'est le
# vrai gain d'un prior : il déplace le seuil d'identifiabilité, pas la constante. Passé
# $n = 10$, les deux courbes deviennent parallèles en log-log, de pente $-1/2$, et se collent aux
# lois $\sigma\sqrt{p/n}$ tracées **sans aucune constante ajustée**. Le rapport se stabilise
# autour de $2 = \sqrt{16/4}$, exactement la prédiction de §6.2.
#
# **Panneau (b) — prior faux.** Le contraint part devant (il reste identifiable là où le libre ne
# l'est pas), puis **plafonne**. Le plafond est mesuré à $0.38896$, contre une prédiction
# analytique $\sqrt{2}\,\delta = 0.38891$ : $1.4\times10^{-4}$ d'écart relatif. La prédiction se
# dérive en une ligne : sous design isotrope dans la base modale, le moindre carré contraint
# réalise la **projection orthogonale de Frobenius** de $\mathrm{diag}(\lambda_a,\lambda_b)$ sur
# le sous-espace $\{aI + bJ\}$ ; or
# $\mathrm{diag}(\lambda_a,\lambda_b) = \frac{\lambda_a+\lambda_b}{2}I + \frac{\lambda_a-\lambda_b}{2}\mathrm{diag}(1,-1)$
# et $\mathrm{diag}(1,-1)$ est orthogonale à la fois à $I$ et à $J$. Le résidu est donc
# exactement $\frac{\lambda_a-\lambda_b}{2}\mathrm{diag}(1,-1)$, de norme de Frobenius
# $\frac{|\lambda_a-\lambda_b|}{2}\sqrt{2} = \sqrt{2}\,\delta$. **Le défaut spectral $\delta$ de
# §3 n'est donc pas une abstraction : c'est le plancher d'erreur, à un facteur $\sqrt2$ près
# qui vient de la norme choisie.**
#
# **Panneau (c) — le croisement.** Le gain du prior est $>1$ à petit $n$ dans **les deux**
# systèmes, y compris quand le prior est faux : sous le seuil d'identifiabilité, un mauvais
# modèle identifiable bat un bon modèle non identifiable. Puis les deux courbes divergent :
# gain $\to 2$ quand le prior est exact, gain $\to 0$ quand il est faux, avec un croisement de
# la ligne $1$ à $n = 5$ — c'est-à-dire **exactement** au seuil d'identifiabilité du modèle
# libre. Un prior faux n'est pas neutre : il devient nuisible dès que les données suffisent à
# s'en passer.
#
# **Ce qu'on ne peut pas conclure, et il y a beaucoup à dire.**
#
# 1. **La base modale a été donnée au prior.** Dans un modèle appris, $\psi$ et $\pi$ doivent
#    la trouver, et le §5 montre qu'ils ne sont contraints par rien pour le faire. Le gain
#    mesuré ici est donc une **borne supérieure optimiste**.
# 2. **Les deux estimateurs sont des moindres carrés**, donc convexes et résolus en forme close.
#    Le §4 a mesuré que le régime d'entraînement réel n'atteint pas cet optimum. Le gain
#    d'échantillonnage démontré ici est un gain **statistique**, pas un gain **atteignable** :
#    les deux se composent, et rien ne dit que la composition soit favorable.
# 3. **Le design est i.i.d. gaussien**, pas une trajectoire. Sur une trajectoire, les états sont
#    corrélés, le nombre effectif d'échantillons est plus petit que $n$, et le seuil
#    d'identifiabilité se déplace vers la droite. La figure est donc optimiste sur l'axe $n$
#    aussi.
# 4. **Le rapport ne dit rien de l'utilité.** À $n = 10\,000$, le prior exact divise l'erreur
#    par 2 — mais les deux valent $\sim 10^{-3}$, et aucune tâche en aval ne distingue
#    $10^{-3}$ de $2\times10^{-3}$. C'est l'argument central de §6.1, ici sous forme graphique.

# %% [markdown]
# ### 6.4 Synthèse : ce que l'axe manquant aurait mesuré
#
# En superposant §4 et §6, la lecture du dossier change de forme.
#
# | | régime $n$ petit | régime $n$ grand (D3 : 24 480) |
# |---|---|---|
# | prior exact ($\delta = 0$) | gain d'identifiabilité, potentiellement grand | gain $\to$ facteur constant, **sans effet sur une tâche** |
# | prior faux ($\delta > 0$) | gain possible malgré tout | **nuisible**, plafond $\propto \delta$ |
#
# La campagne D3 s'est placée dans la colonne de droite, ligne du haut : le seul régime où un
# prior correct ne peut **rien** démontrer. `PLAN_REFOCUS §7` appelle ce réglage « le meilleur
# cas possible » ; l'analyse ci-dessus dit que c'est le **pire cas possible** pour faire
# apparaître un biais inductif. Ce n'est pas une erreur de mesure — les 10 runs sont propres et
# leur verdict tient. C'est une erreur de **placement**, et elle est réparable : c'est
# exactement ce que déclare l'amendement 14 (`PROTOCOL.md §23`), en ajoutant l'axe
# `n_episodes` $\in \{8, 32, 256\}$ à backbone gelé.
#
# **Ce que ce notebook ne peut pas faire.** Prédire le résultat de cette campagne. L'analyse de
# §6.2 est valable pour des **estimateurs des moindres carrés** ; le dépôt entraînera des
# réseaux par descente de gradient sur un rollout fermé, régime dont §4 mesure qu'il n'atteint
# pas l'optimum des moindres carrés. Le signe de l'interaction reste donc une **prédiction
# falsifiable**, et c'est bien ainsi qu'elle est enregistrée.

# %% [markdown]
# ## 7. Ce qu'on ne sait pas
#
# Cette section n'est pas une clause de style. Elle liste ce qui manque, avec la même précision
# que ce qui précède, parce qu'un dossier qui se vend sur l'honnêteté méthodologique se juge
# ici.
#
# ### 7.1 Les quatre trous du dépôt
#
# 1. **Le contraste ON/OFF n'a jamais été mesuré là où le prior est exact.** L'axe structure
#    ON/OFF n'existe que sur le pendule amorti, un système **non linéaire** où un prior LTI est
#    faux par construction (§3, panneau b : le régime $\delta > 0$). Sur le seul système où
#    $\delta = 0$ — les oscillateurs couplés — `PROTOCOL.md §19.1` a redéfini D3 de
#    « comparaison entre bras » en « contrôle de capacité », ce qui a **supprimé les bras OFF**.
#    Le texte du protocole le dit littéralement : *« D3 compares no arms. »* Les 10 cellules
#    sont `mlp_k` et `transformer_k`, zéro `mlp`, zéro `transformer`. **Le contraste central de
#    la thèse a donc exactement zéro mesure dans le régime où il est décidable.**
# 2. **L'axe taille d'échantillon n'a jamais été couru.** Tout le §6 est une analyse et une
#    simulation jouet. La campagne est déclarée (`PROTOCOL.md §23`, amendement 14 : racine
#    `results_osc_n/`, backbone gelé, `n_episodes` $\in \{8, 32, 256\}$, 5 seeds, lecture
#    primaire = le **signe** de l'interaction) et n'a pas tourné. Aucun chiffre de §6 n'est un
#    chiffre du dépôt.
# 3. **L'effet de la jauge n'est pas mesuré.** §5 démontre $2r$ directions exactement plates et
#    les vérifie à $10^{-16}$. Ce que cela coûte à la descente de gradient est **inconnu** : il
#    n'existe ni bras avec terme de reconstruction, ni bras avec $\pi = \psi^{+}$, ni mesure du
#    spectre du hessien. Le dépôt a d'ailleurs retiré (amendement 13) une conclusion antérieure
#    qui reliait trop vite $\psi/\pi$ à l'échec d'exactitude.
# 4. **Le régime d'entraînement n'a pas d'équivalent publié.** Rollout **fermé** de 25 pas
#    depuis l'initialisation, sans curriculum, sans garde sur le rayon spectral. `§17.0` du
#    protocole le reconnaît par écrit. Le §4 de ce notebook mesure sur une maquette que **ce
#    seul ingrédient** suffit à produire un écart de treize ordres de grandeur à l'optimum des
#    moindres carrés. Conclusion opérationnelle : les verdicts d'entraînabilité rendus dans ce
#    régime (« la famille SSM ne s'entraîne pas ») portent sur **le régime**, pas sur les
#    familles.

# %% [markdown]
# ### 7.2 Ce que ce notebook lui-même n'établit pas
#
# - Les sections 4 et 6 contiennent des simulations **[ILLUSTRATIVES]**. Elles montrent des
#   mécanismes sur des maquettes où tout est contrôlé. Elles ne reproduisent aucun run, et
#   aucune de leurs valeurs ne doit être citée comme un résultat du dépôt.
# - $\delta(c)$ dépend d'une **convention d'appariement déclarée** (§3.4). Sous appariement
#   libre, les valeurs changent jusqu'à un facteur $1.9$ dans la zone intermédiaire. La
#   convention a été fixée avant de regarder les chiffres, et son coût est mesuré et publié.
# - Le lien entre défaut spectral $\delta$ et erreur d'apprentissage n'est établi que dans un
#   cas très favorable : design isotrope, base modale **orthogonale** ($\mathrm{cond}(V) = 1$),
#   estimateur des moindres carrés. Hors de ce cas, le plancher $\sqrt{2}\,\delta$ n'a aucune
#   raison de tenir.
# - Le taux de composition $\rho$ de §1.5 est cité, pas recalculé. Il porte les **deux
#   confondants** nommés (asymétrie de taux, composite structure + $\psi$ + $\pi$), et l'un des
#   deux fait l'objet d'un re-run dont ce notebook n'a pas le verdict.

# %% [markdown]
# ### 7.3 Récapitulatif des vérifications numériques de ce notebook
#
# | § | énoncé | vérifié contre | résultat |
# |---|---|---|---|
# | 1.3 | $\|e_H\| \le \varepsilon(\eta^H-1)/(\eta-1)$, atteinte au pire cas | simulation directe | superposition exacte |
# | 2.2 | $\exp(\Delta G) = e^{\Delta\alpha}R(\Delta\omega)$ | série de Taylor | $\le 3\times10^{-16}$ |
# | 2.3 | $r = \rho(\bar A) = \|\bar A\|_2 = e^{\Delta\alpha}$ | valeurs propres, SVD | égalité machine |
# | 2.4 | forme close de $\Phi$ | quadrature fine | $\sim 10^{-13}$ (limite de la quadrature) |
# | 2.5 | singularité effaçable, cancellation réelle | série vs forme brute | $7\times10^{-4}$ à $\|g\|=10^{-12}$ |
# | 3.2 | $\lambda^2 + c\lambda + \kappa = 0$ | spectre de $A$ $4\times4$ | $\le 3\times10^{-15}$ |
# | 3.3 | distance minimale $= \frac12\|\lambda_a - \bar\lambda_b\|$ | minimisation aveugle depuis $(0,0)$ | $\sim 10^{-8}$ (pas de grille) |
# | 4.5 | Adam atteint la précision machine sur la loss à un pas | comparaison aux MCO | $\sim 10^{-16}$ |
# | 4.6 | le rollout fermé, seul, crée l'écart | même code, objectif changé | $\sim 10^{13}\times$ |
# | 5.2 | invariance de jauge exacte (échelle et rotation) | rollout MLP complet avec actions | $\sim 5\times10^{-16}$ relatif |
# | 5.3 | $2r$ directions plates, indépendantes | dérivées + chemin fini + SVD | rang $= 4$, dérivées nulles |
# | 5.4 | $\psi^{+}\psi = \mathrm{Id}$ ssi $\mathrm{rang}(\psi) = d_z$ | pseudo-inverse, cas dégénéré inclus | $\le 1.3\times10^{-15}$ |
# | 6.3 | plancher $= \sqrt{2}\,\delta$ pour un prior faux | balayage en $n$, 24 répétitions | $1.4\times10^{-4}$ relatif |
#
# ---

# %% [markdown]
# ## Références
#
# - **Hardt, Ma & Recht (2018).** *Gradient descent learns linear dynamical systems.* JMLR
#   19(29). — Conditions sous lesquelles la descente de gradient identifie un système linéaire ;
#   le §4 est une instance mesurée de sortie de ces conditions.
#   <https://jmlr.org/papers/v19/16-465.html>
# - **Lusch, Kutz & Brunton (2018).** *Deep learning for universal linear embeddings of
#   nonlinear dynamics.* Nature Communications 9, 4950. — L'auto-encodeur de Koopman, dont le
#   terme de reconstruction est exactement la contrainte que §5.4 montre manquante.
#   <https://www.nature.com/articles/s41467-018-07210-0>
# - **`PROTOCOL.md`** du dépôt `structured-latent-dynamics` : §13.2 (rayon libre), §17.0
#   (régime de rollout sans équivalent publié), §19.1 (D3 ne compare aucun bras), §20.1
#   (élimination des quatre explications), §21.0 (table $e_1$/$e_{25}$), §21.2 (asymétrie de
#   taux), §22 (amendement 13, retrait), §23 (amendement 14, campagne taille d'échantillon).
#
# ---
#
# *Toutes les cellules de code de ce notebook s'exécutent avec `numpy` et `matplotlib`
# uniquement, en moins d'une minute au total, sans accès au dépôt.*
