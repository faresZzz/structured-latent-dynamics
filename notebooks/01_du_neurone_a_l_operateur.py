# %% [markdown]
# # Du neurone à l'opérateur : ce que le LIF sous-seuil dit du bloc 2×2 de `structured-latent-dynamics`
#
# **Notebook de cours — `structured-latent-dynamics`**
#
# ---
#
# ## Le problème que ce notebook résout
#
# Le dépôt `structured-latent-dynamics` teste une question expérimentale :
# *quand une structure dynamique explicite dans le predictor latent d'un JEPA world model
# améliore-t-elle la stabilité de rollout, la suffisance d'état et la planifiabilité ?*
# Le niveau `on` du facteur `structure` est une transition latente bloc-diagonale dont
# chaque bloc est un opérateur 2×2 de type **rotation-scaling** :
#
# $$\bar{A} = \exp(\Delta G), \qquad G = \alpha I + \omega J, \qquad J = \begin{pmatrix} 0 & -1 \\ 1 & 0\end{pmatrix}.$$
#
# Ce notebook établit un fait, puis en tire une décision de conception.
#
# **Le fait.** Cet opérateur n'est pas *inspiré* d'un neurone. Il **est** un neurone — plus
# précisément, il est la discrétisation exacte d'un neurone adaptatif (adLIF) en régime
# sous-seuil. La correspondance est une identité algébrique, pas une analogie : on la
# démontre, puis on la vérifie numériquement à la précision machine.
#
# **La décision.** Une fois l'identité posée, la littérature neurone devient utilisable
# comme littérature **sur la discrétisation**. Or c'est exactement là que le dépôt a un
# problème mesuré : le rayon spectral $r = e^{\Delta\alpha}$ est libre, donc non borné,
# donc l'amplification du gradient sur un rollout fermé de $H$ pas vaut $r^H$. Ce notebook
# dérive la reparamétrisation qui borne $r$ sans rien changer d'autre, montre pourquoi la
# solution évidente (Euler symplectique) ne tient pas sa promesse, et énonce pourquoi le
# rayon borné doit être un **troisième niveau de facteur mesuré** et jamais un correctif.
#
# ---
#
# ## Ce qu'il faut savoir avant de lire
#
# Rien de spécifique aux neurones. On utilise : équations différentielles linéaires du
# premier ordre, exponentielle de matrice, valeurs propres d'une matrice 2×2, et la règle
# de la chaîne. Tout le reste est dérivé ici.
#
# ## Périmètre, annoncé d'entrée
#
# On reste en **régime sous-seuil** : pas de spike, pas de reset. La section 4 démontre
# *pourquoi* — le reset dur annule le chemin de gradient exactement là où la thèse du
# dossier en a besoin. Ce n'est pas une omission, c'est un résultat.
#
# ## Convention sur les chiffres
#
# Deux statuts, jamais confondus :
#
# - **mesuré (dépôt)** — un chiffre produit par une campagne de `structured-latent-dynamics`
#   ou consigné dans son annexe de vérifications ; la source est nommée à chaque fois.
# - **reproduit ici** — un chiffre que la cellule de code qui suit vient de calculer.
#   Il est recalculé à chaque exécution du notebook.

# ---
#
# ## Sommaire
#
# 1. **Le LIF sous-seuil EST un SSM diagonal de dimension 1.** Résolution exacte sous ZOH,
#    vérification contre RK4 fin, et ce que l'Euler explicite change.
# 2. **Monter en dimension 2 : adLIF et resonate-and-fire.** Le système 2×2, sa conjugaison
#    au bloc rotation-scaling, et la limite honnête de ce que l'identité implique.
# 3. **La discrétisation décide du rayon** (le cœur). Pourquoi $r$ est non borné, le fait
#    mesuré de la mort de la famille `ssm`, la carte symplectique et son piège, et la
#    paramétrisation retenue $\alpha = \log \sigma(\rho)/\Delta$.
# 4. **Ce que le spike coûterait ici**, et pourquoi on s'arrête au sous-seuil.
# 5. **Le pont, en une page.** Les quatre références, et ce qui reste ouvert.

# %%
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=False, linewidth=110)

# Style commun a toutes les figures du notebook : lisible en niveaux de gris,
# grille discrete, pas de dependance a un theme externe.
plt.rcParams.update(
    {
        "figure.figsize": (10.0, 3.6),
        "figure.dpi": 110,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "legend.frameon": False,
    }
)

J2 = np.array([[0.0, -1.0], [1.0, 0.0]])  # generateur des rotations planes, J2 @ J2 == -I

print("numpy", np.__version__)
print("J2 @ J2 =\n", J2 @ J2)

# %% [markdown]
# ---
#
# # 1. Le LIF sous-seuil EST un SSM diagonal de dimension 1
#
# ## 1.1 L'équation membranaire, terme par terme
#
# Le modèle *leaky integrate-and-fire* (LIF) décrit la tension de membrane $V(t)$ d'un
# neurone par une seule équation différentielle :
#
# $$\tau \frac{dV}{dt} = -\bigl(V(t) - V_{\text{rest}}\bigr) + R\,I(t).$$
#
# Chaque terme a un sens physique, et il faut le tenir pour lire la suite :
#
# | terme | nom | rôle |
# |---|---|---|
# | $V(t)$ | tension de membrane | l'**état** du système, en mV |
# | $V_{\text{rest}}$ | potentiel de repos | l'état d'équilibre en l'absence d'entrée |
# | $\tau = RC$ | constante de temps membranaire | fixe la **vitesse d'oubli** ; ordre 10–30 ms |
# | $R$ | résistance membranaire | convertit un courant en tension |
# | $I(t)$ | courant d'entrée | l'**entrée** du système |
#
# Lecture dynamique : le terme $-(V - V_{\text{rest}})$ est un **rappel vers l'équilibre**
# proportionnel à l'écart — c'est la « fuite » (*leak*). Le terme $R I(t)$ est un forçage.
# Si l'on coupe l'entrée, $V$ relaxe vers $V_{\text{rest}}$ exponentiellement avec la
# constante de temps $\tau$. Rien d'autre n'est postulé : cette équation est celle d'un
# circuit RC, et c'est tout ce qu'un LIF est **sous le seuil**.
#
# ## 1.2 Pourquoi on veut une solution exacte plutôt qu'un pas d'intégration
#
# Un réseau de neurones simulé avance par pas de temps $\Delta$ fixe. Deux voies s'offrent :
#
# 1. **approximer** la dérivée (Euler, RK4, ...) et itérer ;
# 2. **résoudre exactement** l'équation sur $[t_k, t_k + \Delta]$ sous une hypothèse sur
#    l'entrée, et itérer la solution exacte.
#
# La voie 2 est disponible ici parce que l'équation est **linéaire** en $V$. L'hypothèse
# requise porte uniquement sur l'entrée : on suppose $I(t)$ **constante sur chaque
# intervalle**, égale à $I[k]$ sur $[t_k, t_k+\Delta)$. C'est l'hypothèse
# **ZOH** (*zero-order hold*, bloqueur d'ordre zéro). Elle n'est pas une approximation de la
# dynamique — elle est une **définition du signal d'entrée** : dans un système discret,
# l'entrée *est* constante entre deux échantillons. C'est exactement l'hypothèse que fait
# la discrétisation des state-space models (S4, S4D, Mamba), et c'est pour cette raison
# qu'on la choisit ici plutôt qu'une quadrature d'ordre élevé : elle rend la récurrence
# **exacte** au lieu de la rendre précise.
#
# *Alternative écartée.* On pourrait supposer $I$ affine par morceaux (FOH, *first-order
# hold*). C'est plus fidèle pour un signal physique échantillonné, mais cela introduit un
# second terme d'entrée $\bar{B}_1 (I[k+1] - I[k])$, donc une dépendance au futur immédiat,
# donc un décalage causal dans un rollout auto-consommé. On refuse pour cette raison, pas
# pour une raison de précision.

# %% [markdown]
# ## 1.3 Résolution exacte sur un pas — la dérivation complète
#
# **Hypothèses.** $\tau > 0$, $R$, $V_{\text{rest}}$ constants ; $I(t) = I[k]$ constante sur
# $[t_k,\, t_k+\Delta)$ ; $V$ continue.
#
# **Étape 1 — mettre sous forme normalisée.** On divise par $\tau$ et on regroupe les
# termes en $V$ à gauche :
#
# $$\frac{dV}{dt} + \frac{1}{\tau} V = \frac{V_{\text{rest}} + R I[k]}{\tau} \;=:\; \frac{V_\infty}{\tau},$$
#
# où $V_\infty := V_{\text{rest}} + R I[k]$ est la valeur d'équilibre **sous cette entrée**
# (on l'obtient en posant $dV/dt = 0$). C'est la forme $y' + p\,y = q$ avec $p = 1/\tau$
# constant.
#
# **Étape 2 — facteur intégrant.** On cherche $\mu(t)$ tel que $\mu(V' + V/\tau)$ soit la
# dérivée d'un produit. En imposant $\mu' = \mu/\tau$ on obtient $\mu(t) = e^{t/\tau}$.
# Alors
#
# $$\frac{d}{dt}\Bigl(e^{t/\tau} V(t)\Bigr) = e^{t/\tau} V'(t) + \frac{1}{\tau} e^{t/\tau} V(t) = e^{t/\tau}\,\frac{V_\infty}{\tau}.$$
#
# La première égalité est la règle du produit ; la seconde est l'équation de l'étape 1
# multipliée par $e^{t/\tau}$. **Aucune approximation n'a été faite.**
#
# **Étape 3 — intégrer sur le pas.** On intègre de $t_k$ à $t_k + \Delta$ :
#
# $$e^{(t_k+\Delta)/\tau} V(t_k+\Delta) - e^{t_k/\tau} V(t_k) = \frac{V_\infty}{\tau}\int_{t_k}^{t_k+\Delta} e^{s/\tau}\,ds = \frac{V_\infty}{\tau}\cdot \tau\Bigl(e^{(t_k+\Delta)/\tau} - e^{t_k/\tau}\Bigr).$$
#
# Le $\tau$ de l'intégrale annule le $1/\tau$ du préfacteur — c'est ici que $V_\infty$
# apparaît sans coefficient.
#
# **Étape 4 — multiplier par $e^{-(t_k+\Delta)/\tau}$** pour isoler $V(t_k+\Delta)$ :
#
# $$V(t_k+\Delta) = e^{-\Delta/\tau} V(t_k) + V_\infty\Bigl(1 - e^{-\Delta/\tau}\Bigr).$$
#
# **Conclusion.** En posant $a := e^{-\Delta/\tau}$ et $V[k] := V(t_k)$ :
#
# $$\boxed{\;V[k+1] = a\,V[k] + (1-a)\bigl(V_{\text{rest}} + R\,I[k]\bigr), \qquad a = e^{-\Delta/\tau}.\;}$$
#
# **Cas de rupture.** La dérivation utilise $\tau > 0$ (sans quoi $\mu$ n'est pas un facteur
# intégrant utile et l'équilibre est répulsif) et la constance de $I$ sur le pas. Elle ne
# suppose **rien** sur l'amplitude de $\Delta$ : la formule est exacte pour $\Delta = 0.001\,$ms
# comme pour $\Delta = 1000\,$ms. C'est la propriété qui distingue une discrétisation exacte
# d'un schéma numérique, et c'est le point de toute la section 3.
#
# **Sanity check de la formule.** Deux limites doivent tomber juste :
# $\Delta \to 0 \Rightarrow a \to 1 \Rightarrow V[k+1] \to V[k]$ (rien ne bouge) ;
# $\Delta \to \infty \Rightarrow a \to 0 \Rightarrow V[k+1] \to V_\infty$ (l'état oublie
# tout et saute à l'équilibre). Les deux sont physiquement correctes.

# %%
# --- Parametres physiologiques standard, fixes une fois pour tout le notebook ---
TAU = 20.0        # ms   : constante de temps membranaire
DELTA = 1.0       # ms   : pas de simulation (valeur usuelle des bibliotheques SNN)
R_MEM = 1.0       # conversion courant -> tension
V_REST = -65.0    # mV   : potentiel de repos

A_ZOH = np.exp(-DELTA / TAU)          # coefficient exact
A_EULER = 1.0 - DELTA / TAU           # coefficient d'Euler explicite

print(f"a_ZOH   = exp(-D/tau) = {A_ZOH:.12f}")
print(f"a_Euler = 1 - D/tau   = {A_EULER:.12f}")
print(f"ecart relatif sur a   = {abs(A_EULER - A_ZOH) / A_ZOH:.6e}")

# Verification des deux limites annoncees.
print("\nlimites :")
for d in (1e-6, 1e6):
    print(f"  Delta={d:>8.0e} ms -> a = {np.exp(-d / TAU):.12f}")


def lif_zoh(V0, I, a=A_ZOH, v_rest=V_REST, R=R_MEM):
    """Recurrence exacte du LIF sous-seuil sous ZOH. Un SSM diagonal 1-D."""
    V = np.empty(len(I) + 1)
    V[0] = V0
    for k, Ik in enumerate(I):
        V[k + 1] = a * V[k] + (1.0 - a) * (v_rest + R * Ik)
    return V


def lif_rk4(V0, I, delta=DELTA, sub=100, tau=TAU, v_rest=V_REST, R=R_MEM):
    """Reference : RK4 a `sub` sous-pas par pas, entree tenue constante (meme ZOH).

    Le role de cette fonction est d'etre un integrateur *different* : si les deux
    coincident a la precision machine, la recurrence n'est pas une approximation.
    """
    h = delta / sub
    V = np.empty(len(I) + 1)
    V[0] = V0
    v = V0
    for k, Ik in enumerate(I):
        for _ in range(sub):
            f = lambda x: (-(x - v_rest) + R * Ik) / tau
            k1 = f(v)
            k2 = f(v + 0.5 * h * k1)
            k3 = f(v + 0.5 * h * k2)
            k4 = f(v + h * k3)
            v = v + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        V[k + 1] = v
    return V

# %% [markdown]
# ## 1.4 La vérification qui décide : récurrence exacte contre RK4 fin
#
# Un accord à $10^{-3}$ prouverait seulement que la récurrence est une bonne approximation.
# Ce qu'on veut établir est plus fort : elle est **la solution**. Le test doit donc être
# lu en unités de précision machine, pas en unités d'erreur numérique.
#
# On travaille en variable **adimensionnée** $v = (V - V_{\text{rest}})/V_{\text{scale}}$
# d'ordre 1, pour que l'erreur absolue soit directement comparable à
# $\varepsilon_{\text{machine}} = 2.22\times10^{-16}$. C'est la convention de l'annexe de
# vérifications du dépôt, qui consigne **5.83e-16** (mesuré, `PLAN_V3.md`, annexe #1 ;
# même chiffre dans `REVIEW_FLAGSHIPS.md` §5). On mesure aussi l'erreur en **ulp**
# (*units in the last place*) : c'est la seule unité dans laquelle « égalité machine » a
# un sens indépendant de l'échelle.

# %%
rng = np.random.default_rng(0)
K_STEPS = 400
I_seq = rng.normal(0.0, 1.0, size=K_STEPS)   # entree adimensionnee, O(1)

# Version adimensionnee : v_rest = 0, etat initial 1.0, amplitudes O(1).
V_exact = lif_zoh(1.0, I_seq, a=A_ZOH, v_rest=0.0)
V_ref = lif_rk4(1.0, I_seq, v_rest=0.0)

err_abs = np.max(np.abs(V_exact - V_ref))
scale = np.max(np.abs(V_exact))
ulp = np.spacing(scale)

print(f"pas simules                      : {K_STEPS}")
print(f"sous-pas RK4 par pas             : 100  (soit {K_STEPS * 100} evaluations fines)")
print(f"erreur absolue max               : {err_abs:.4e}     [reproduit ici]")
print(f"echelle max |v|                  : {scale:.4f}")
print(f"erreur relative                  : {err_abs / scale:.4e}")
print(f"erreur en ulp                    : {err_abs / ulp:.2f} ulp")
print(f"epsilon machine (float64)        : {np.finfo(float).eps:.4e}")
print()
print("depot (mesure, PLAN_V3 annexe #1) : 5.83e-16")
print("verdict : meme ordre, quelques ulp. Les derniers chiffres sont du bruit")
print("          d'arrondi, pas du signal : la recurrence n'approxime pas, elle resout.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))

t = np.arange(K_STEPS + 1) * DELTA
axes[0].plot(t, V_ref, lw=2.4, color="0.75", label="RK4 fin (100 sous-pas / pas)")
axes[0].plot(t, V_exact, lw=1.0, color="black", ls="--", label="recurrence ZOH exacte")
axes[0].set_xlabel("temps (ms)")
axes[0].set_ylabel("v (adimensionne)")
axes[0].set_title("Les deux trajectoires sont superposees")
axes[0].legend(loc="upper right")

axes[1].semilogy(t, np.maximum(np.abs(V_exact - V_ref), 1e-20), lw=0.9, color="black")
axes[1].axhline(np.finfo(float).eps, color="crimson", ls=":", lw=1.2,
                label=r"$\varepsilon_{machine} = 2.22\times 10^{-16}$")
axes[1].axhline(1e-3, color="tab:blue", ls="-.", lw=1.2, label="1e-3 (seuil 'bonne approximation')")
axes[1].set_xlabel("temps (ms)")
axes[1].set_ylabel("|erreur|")
axes[1].set_ylim(1e-20, 1e-2)
axes[1].set_title("Ecart ZOH vs RK4, echelle logarithmique")
axes[1].legend(loc="lower right")

fig.suptitle("Figure 1 — la recurrence LIF n'approxime pas la dynamique, elle la resout", y=1.03)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Lecture.** Panneau gauche : les deux courbes ne sont pas *proches*, elles sont
# indiscernables — on ne voit qu'un seul trait parce que le second est dessous. Panneau
# droit : l'écart reste sous la ligne rouge $\varepsilon_{\text{machine}}$ presque partout,
# et **treize ordres de grandeur** sous la ligne bleue qu'on utiliserait pour déclarer une
# « bonne approximation ». La quantité tracée n'est pas une erreur de méthode : c'est
# l'accumulation d'arrondis flottants de deux chemins de calcul différents vers le même
# nombre réel. Conséquence pour la suite : tout ce qu'on démontrera sur la récurrence
# $V[k+1] = aV[k] + \dots$ vaut sans réserve pour l'EDO continue, et réciproquement. Les
# deux objets sont interchangeables.

# %% [markdown]
# ## 1.5 Énoncé : un LIF sous-seuil est un SSM diagonal de dimension 1
#
# Rappelons la forme canonique d'un **state-space model** linéaire à temps continu :
#
# $$\dot{x}(t) = A\,x(t) + B\,u(t), \qquad y(t) = C\,x(t),$$
#
# discrétisé sous ZOH en
#
# $$x[k+1] = \bar{A}\,x[k] + \bar{B}\,u[k], \qquad \bar{A} = e^{\Delta A}, \quad \bar{B} = \left(\int_0^{\Delta} e^{sA}\,ds\right) B.$$
#
# La famille **S4D** (Gu et al.) impose $A$ **diagonale**, ce qui découple les états : chaque
# coordonnée suit sa propre récurrence scalaire $x_i[k+1] = e^{\Delta A_{ii}} x_i[k] + \bar{B}_i u[k]$.
#
# Confrontons terme à terme avec ce qu'on vient de dériver :
#
# | S4D, une voie | LIF sous-seuil |
# |---|---|
# | $A_{ii}$ (appris, complexe) | $-1/\tau$ (**fixé** par la physiologie, réel négatif) |
# | $\bar{A}_{ii} = e^{\Delta A_{ii}}$ | $a = e^{-\Delta/\tau}$ |
# | $B_i$ (appris) | $R/\tau$, donc $\bar{B} = R(1-a)$ (**fixé**) |
# | $\Delta$ (appris, par voie) | $\Delta$ (**fixé**, le pas de simulation) |
# | terme affine : aucun | $(1-a)V_{\text{rest}}$ |
#
# **Le LIF sous-seuil est donc exactement une voie de S4D dont tous les paramètres sont
# gelés**, plus un décalage affine constant. Vérifions que $\bar{B} = R(1-a)$ tombe bien de
# la formule générale et non d'une réécriture heureuse :
#
# $$\int_0^{\Delta} e^{-s/\tau}\,ds \cdot \frac{R}{\tau} = \tau\bigl(1 - e^{-\Delta/\tau}\bigr)\cdot\frac{R}{\tau} = R\,(1-a). \qquad \checkmark$$
#
# ### Ce que Mamba ajoute, précisément
#
# **Mamba** (Gu & Dao, 2023) part du même $\bar{A} = \exp(\Delta A)$ diagonal et rend
# $\Delta$, $B$ et $C$ **fonctions de l'entrée courante** : $\Delta_t = \text{softplus}(W_\Delta u_t)$,
# d'où $\bar{A}_t = \exp(\Delta_t A)$. La récurrence devient linéaire **variant dans le
# temps**, et le coefficient d'oubli devient un mécanisme de **sélectivité** : le modèle
# peut décider, entrée par entrée, s'il retient ou s'il oublie.
#
# Le LIF n'a **ni l'un ni l'autre** :
#
# 1. **pas de sélectivité** — $a = e^{-\Delta/\tau}$ ne dépend pas de $I[k]$. La membrane
#    oublie au même taux quoi qu'il arrive ;
# 2. **pas de $\Delta$ appris** — $\Delta$ est le pas de simulation, choisi par l'ingénieur,
#    pas optimisé ;
# 3. **pas de $\tau$ appris non plus**, dans la formulation classique — $\tau$ est une
#    constante physiologique. (Les variantes à $\tau$ entraînable existent et sont
#    précisément le premier pas vers un SSM.)
#
# **Ordre de puissance expressive, donc :** LIF sous-seuil $\subsetneq$ S4D (coefficients
# fixes $\subsetneq$ coefficients appris) $\subsetneq$ Mamba (coefficients appris fixes
# $\subsetneq$ coefficients appris dépendant de l'entrée). Le neurone est le membre **le
# plus contraint** de la famille. C'est ce qui rend l'énoncé intéressant : ce n'est pas
# « le neurone est puissant », c'est « le neurone est *déjà* dans la famille, tout en bas ».

# %%
# Verification directe : une "voie S4D" ecrite dans le vocabulaire des SSM,
# et le LIF, produisent la meme trajectoire au bit pres.
A_ssm = -1.0 / TAU                                  # A diagonal, 1 x 1
B_ssm = R_MEM / TAU
Abar = np.exp(DELTA * A_ssm)                        # exp(Delta * A)
Bbar = (np.expm1(DELTA * A_ssm) / A_ssm) * B_ssm    # (int_0^D exp(sA) ds) * B, forme stable

print(f"Abar (S4D)      = {Abar:.16f}")
print(f"a    (LIF)      = {A_ZOH:.16f}")
print(f"Bbar (S4D)      = {Bbar:.16f}")
print(f"R(1-a) (LIF)    = {R_MEM * (1 - A_ZOH):.16f}")
print(f"|Abar - a|      = {abs(Abar - A_ZOH):.3e}")
print(f"|Bbar - R(1-a)| = {abs(Bbar - R_MEM * (1 - A_ZOH)):.3e}")

x_ssm = np.empty(K_STEPS + 1)
x_ssm[0] = 1.0
for k in range(K_STEPS):
    x_ssm[k + 1] = Abar * x_ssm[k] + Bbar * I_seq[k]

print(f"\necart trajectoire S4D vs LIF     : {np.max(np.abs(x_ssm - V_exact)):.3e}  [reproduit ici]")
print("(v_rest = 0 dans cette variante, donc le terme affine du LIF est nul)")

# %% [markdown]
# ## 1.6 Ce que l'Euler explicite change — et pourquoi presque toutes les bibliothèques SNN l'emploient
#
# Le schéma d'Euler explicite remplace $dV/dt$ par $\bigl(V[k+1]-V[k]\bigr)/\Delta$ :
#
# $$V[k+1] = V[k] + \frac{\Delta}{\tau}\Bigl(-(V[k]-V_{\text{rest}}) + R I[k]\Bigr) = \underbrace{\Bigl(1 - \tfrac{\Delta}{\tau}\Bigr)}_{a_{\text{Euler}}} V[k] + \frac{\Delta}{\tau}\bigl(V_{\text{rest}} + R I[k]\bigr).$$
#
# **C'est le développement au premier ordre de la solution exacte.** En posant
# $h := \Delta/\tau$ :
#
# $$a = e^{-h} = 1 - h + \frac{h^2}{2} - \frac{h^3}{6} + O(h^4), \qquad a_{\text{Euler}} = 1 - h.$$
#
# L'écart est donc
#
# $$a_{\text{Euler}} - a = -\frac{h^2}{2} + O(h^3), \qquad \frac{|a_{\text{Euler}} - a|}{a} = \frac{h^2}{2}\,e^{h} + O(h^3).$$
#
# Pour $\tau = 20$ ms et $\Delta = 1$ ms, $h = 0.05$ : $h^2/2 = 1.25\times10^{-3}$, et
# l'écart relatif prédit vaut $1.25\times10^{-3}\cdot e^{0.05} \approx 1.31\times10^{-3}$ —
# à comparer au **1.29e-03** mesuré (dépôt, `PLAN_V3.md` annexe #2). Les deux coïncident
# à l'ordre suivant près : le développement explique le chiffre.
#
# **Le gain d'entrée diverge aussi, et davantage.** Le schéma exact multiplie l'entrée par
# $1-a = h - h^2/2 + \dots$, Euler par $h$. L'écart relatif y vaut donc $\approx h/2 = 2.5\%$,
# soit **vingt fois** l'écart sur $a$. Une lecture courante ne regarde que le coefficient de
# fuite ; c'est le mauvais terme à surveiller.
#
# ### Les deux raisons pour lesquelles ce n'est pas anodin
#
# **(a) Domaine de stabilité.** Le coefficient exact vérifie $a = e^{-h} \in (0,1)$ **pour
# tout** $h > 0$ : la récurrence exacte est contractante quel que soit le pas. Le
# coefficient d'Euler vérifie $|1-h| < 1 \iff 0 < h < 2$, et $1 - h < 0$ dès $h > 1$ : au-delà
# d'un pas égal à $\tau$, la tension **change de signe à chaque pas**, ce qui n'a aucun
# sens physique, et au-delà de $2\tau$ elle diverge. La discrétisation exacte n'a pas de
# domaine de stabilité — elle est inconditionnellement stable parce qu'elle est exacte.
#
# **(b) Le biais est systématique, pas aléatoire.** $a_{\text{Euler}} < a$ toujours : le
# neurone d'Euler **oublie trop vite**, à chaque pas, dans le même sens. Sur un rollout de
# $H$ pas l'écart se compose en $(a_{\text{Euler}}/a)^H$.
#
# ### Pourquoi les bibliothèques SNN l'emploient quand même
#
# Trois raisons, toutes défendables isolément : (i) le schéma s'écrit en une ligne et se
# généralise sans effort à des non-linéarités (adaptation multiplicative, conductances
# synaptiques) où aucune forme close n'existe ; (ii) le seuil et le reset rendent la
# trajectoire non lisse de toute façon — un argument valable *au-dessus* du seuil, mais
# **pas sous le seuil**, qui est le régime qui nous occupe ; (iii) la communauté SNN vise
# des pas déjà petits devant $\tau$, où $h^2/2$ paraît négligeable.
#
# Le point de ce notebook est que (iii) cesse d'être vrai dès qu'on **entraîne** le
# coefficient : un $\tau$ appris peut descendre vers $\Delta$, et c'est exactement là que
# $1-h$ sort du domaine où il approxime $e^{-h}$. La section 3 montre que c'est cette
# différence, et non le modèle de neurone, qui décide de la stabilité de l'entraînement.

# %%
def lif_euler(V0, I, delta=DELTA, tau=TAU, v_rest=0.0, R=R_MEM):
    """Euler explicite : a = 1 - Delta/tau, gain d'entree Delta/tau."""
    h = delta / tau
    V = np.empty(len(I) + 1)
    V[0] = V0
    for k, Ik in enumerate(I):
        V[k + 1] = (1.0 - h) * V[k] + h * (v_rest + R * Ik)
    return V


h_ratio = DELTA / TAU
V_euler = lif_euler(1.0, I_seq)

err_traj_max = np.max(np.abs(V_euler - V_exact))
err_traj_rms = np.sqrt(np.mean((V_euler - V_exact) ** 2))

print(f"h = Delta/tau                         = {h_ratio:.4f}")
print(f"ecart relatif sur a  (reproduit ici)  = {abs(A_EULER - A_ZOH) / A_ZOH:.6e}")
print(f"prediction h^2/2 * e^h                = {0.5 * h_ratio**2 * np.exp(h_ratio):.6e}")
print(f"depot (mesure, PLAN_V3 annexe #2)     = 1.29e-03")
print()
print(f"ecart relatif sur le gain d'entree    = {abs(h_ratio - (1 - A_ZOH)) / (1 - A_ZOH):.6e}"
      f"   (~ h/2 = {h_ratio / 2:.4f})")
print()
print(f"erreur trajectoire, max  (400 pas)    = {err_traj_max:.4e}   [reproduit ici]")
print(f"erreur trajectoire, RMS  (400 pas)    = {err_traj_rms:.4e}   [reproduit ici]")

# Dispersion sur 30 tirages d'entree : l'erreur trajectoire est une statistique,
# pas une constante. On publie l'intervalle, pas un chiffre isole.
rms_pool = []
for s in range(30):
    Is = np.random.default_rng(s).normal(0.0, 1.0, size=K_STEPS)
    rms_pool.append(np.sqrt(np.mean((lif_euler(1.0, Is) - lif_zoh(1.0, Is, v_rest=0.0)) ** 2)))
rms_pool = np.array(rms_pool)
print(f"RMS sur 30 tirages : moyenne {rms_pool.mean():.4e}, "
      f"min {rms_pool.min():.4e}, max {rms_pool.max():.4e}   [reproduit ici]")
print("ordre attendu ~4e-03 : l'erreur d'Euler est ~1e13 fois celle du schema exact.")

# %%
fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4))

n_show = 120
axes[0].plot(t[:n_show], V_exact[:n_show], lw=1.6, color="black", label="ZOH exact")
axes[0].plot(t[:n_show], V_euler[:n_show], lw=1.2, color="tab:red", ls="--", label="Euler explicite")
axes[0].set_xlabel("temps (ms)")
axes[0].set_ylabel("v")
axes[0].set_title(f"Trajectoires, $h=\\Delta/\\tau={h_ratio:.2f}$")
axes[0].legend()

axes[1].semilogy(t, np.maximum(np.abs(V_euler - V_exact), 1e-20), lw=0.9,
                 color="tab:red", label="Euler - exact")
axes[1].semilogy(t, np.maximum(np.abs(V_ref - V_exact), 1e-20), lw=0.9,
                 color="0.4", label="RK4 - exact")
axes[1].axhline(np.finfo(float).eps, color="crimson", ls=":", lw=1.2, label=r"$\varepsilon_{machine}$")
axes[1].set_xlabel("temps (ms)")
axes[1].set_ylabel("|erreur|")
axes[1].set_ylim(1e-20, 1e-1)
axes[1].set_title("13 ordres de grandeur separent les deux")
axes[1].legend(loc="center right")

hh = np.linspace(0.0, 2.6, 400)
axes[2].plot(hh, np.exp(-hh), color="black", lw=1.6, label=r"$a = e^{-h}$ (exact)")
axes[2].plot(hh, 1 - hh, color="tab:red", lw=1.4, ls="--", label=r"$a = 1-h$ (Euler)")
axes[2].axhline(0.0, color="0.5", lw=0.8)
axes[2].axhline(-1.0, color="crimson", lw=1.0, ls=":")
axes[2].axvline(1.0, color="tab:blue", lw=1.0, ls="-.", label=r"$h=1$ : Euler change de signe")
axes[2].axvline(2.0, color="crimson", lw=1.0, ls="-.", label=r"$h=2$ : Euler diverge")
axes[2].scatter([h_ratio], [np.exp(-h_ratio)], s=28, color="black", zorder=5)
axes[2].set_xlabel(r"$h = \Delta/\tau$")
axes[2].set_ylabel("coefficient de fuite")
axes[2].set_title("Domaine de validite")
axes[2].legend(loc="lower left", fontsize=7.5)

fig.suptitle("Figure 2 — Euler explicite est le premier ordre de la solution, avec un domaine", y=1.03)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Lecture.** Panneau gauche : à $h = 0.05$ les deux trajectoires se suivent — l'erreur
# d'Euler est invisible à l'œil, ce qui explique qu'elle passe inaperçue. Panneau central :
# elle est pourtant à $\sim 10^{-2}$, soit **treize ordres de grandeur** au-dessus de
# l'erreur du schéma exact ; les deux courbes rouge et grise mesurent la même chose et ne
# vivent pas dans le même monde. Panneau droit : la raison structurelle. La courbe noire
# $e^{-h}$ reste dans $(0,1)$ pour tout $h$ ; la droite rouge $1-h$ la quitte à $h=1$
# (coefficient négatif : la tension alterne de signe à chaque pas, ce qui ne correspond à
# aucun phénomène physique) puis franchit $-1$ à $h=2$ (divergence pure). Le point noir
# marque le réglage physiologique usuel : confortablement dans le domaine — **tant que
# $\tau$ n'est pas appris**. Dès qu'on optimise $\tau$, rien n'empêche la descente d'aller
# chercher $h > 1$, et c'est le mécanisme que la section 3 mesure.

# %% [markdown]
# ---
#
# # 2. Monter en dimension 2 : adLIF et resonate-and-fire
#
# ## 2.1 Pourquoi une seconde variable, et laquelle
#
# Le LIF de la section 1 a un défaut structurel : $a = e^{-\Delta/\tau} \in (0,1)$ est
# **réel positif**. Une récurrence scalaire à coefficient réel positif ne peut produire
# qu'une relaxation monotone. Elle ne peut pas osciller, pas résonner, pas coder une phase.
# Or les neurones biologiques le font, et — c'est ce qui nous intéresse ici — les systèmes
# dynamiques que le dépôt cherche à prédire aussi (oscillateurs couplés, pendule).
#
# **Pour osciller, il faut au moins deux dimensions.** C'est une contrainte algébrique, pas
# biologique : les valeurs propres d'une matrice **réelle** vont par paires conjuguées ;
# une valeur propre complexe non réelle exige donc au minimum un bloc $2\times2$.
#
# Le neurone **adLIF** (*adaptive LIF*) fournit exactement cette seconde variable, avec un
# sens physiologique : un **courant d'adaptation** $w$ qui s'accumule quand la membrane se
# dépolarise et qui la freine en retour. C'est une boucle de rétroaction négative retardée,
# et une boucle de rétroaction négative retardée est ce qui fait osciller un système.
#
# $$\begin{cases}
# \tau_u \dfrac{du}{dt} = -\bigl(u - u_{\text{rest}}\bigr) - w + R\,I(t) & \text{(membrane, freinée par } w) \\[6pt]
# \tau_w \dfrac{dw}{dt} = -w + a\,\bigl(u - u_{\text{rest}}\bigr) & \text{(adaptation, pilotée par } u)
# \end{cases}$$
#
# Le paramètre $a > 0$ est le **couplage sous-seuil** : il dit à quel point la
# dépolarisation recrute l'adaptation. C'est le seul paramètre nouveau. Le neurone
# *resonate-and-fire* (Izhikevich) est la même structure écrite directement en coordonnées
# de résonance ; on le retrouvera au §2.4 comme une simple réécriture.
#
# ## 2.2 Forme d'état
#
# Posons $x = \begin{pmatrix} u - u_{\text{rest}} \\ w\end{pmatrix}$. En divisant chaque
# ligne par sa constante de temps :
#
# $$\dot{x} = G\,x + b\,I, \qquad
# G = \begin{pmatrix} -1/\tau_u & -1/\tau_u \\ a/\tau_w & -1/\tau_w \end{pmatrix}, \qquad
# b = \begin{pmatrix} R/\tau_u \\ 0\end{pmatrix}.$$
#
# Deux vérifications de cohérence avant d'aller plus loin. Si $a = 0$, $G$ devient
# $\begin{pmatrix}-1/\tau_u & -1/\tau_u\\ 0 & -1/\tau_w\end{pmatrix}$, triangulaire
# supérieure, de valeurs propres **réelles** $-1/\tau_u$ et $-1/\tau_w$ : sans couplage,
# l'adaptation décroît toute seule et la membrane redevient un LIF forcé, sans oscillation
# possible — conforme au §2.1. Et si $\tau_w \to \infty$, $w$ se fige et l'on retrouve un
# LIF décalé d'une constante. Les deux limites dégénèrent correctement.

# ## 2.3 Le régime sous-amorti, et la condition exacte qui le définit
#
# Pour une matrice $2\times2$, le polynôme caractéristique s'écrit avec la trace $T$ et le
# déterminant $D$ seuls :
#
# $$\lambda^2 - T\lambda + D = 0, \qquad \lambda_{\pm} = \frac{T \pm \sqrt{T^2 - 4D}}{2}.$$
#
# Ici :
#
# $$T = \operatorname{tr} G = -\Bigl(\frac{1}{\tau_u} + \frac{1}{\tau_w}\Bigr), \qquad
# D = \det G = \frac{1}{\tau_u\tau_w} + \frac{a}{\tau_u\tau_w} = \frac{1+a}{\tau_u\tau_w}.$$
#
# *(Le déterminant : $(-1/\tau_u)(-1/\tau_w) - (-1/\tau_u)(a/\tau_w) = (1+a)/(\tau_u\tau_w)$.
# Le couplage $a$ n'entre que dans $D$, jamais dans $T$ — on va s'en servir.)*
#
# Les valeurs propres sont **complexes conjuguées** ssi $T^2 < 4D$. Développons, en posant
# $p := 1/\tau_u$, $q := 1/\tau_w$ :
#
# $$(p+q)^2 < 4(1+a)pq \iff p^2 + 2pq + q^2 - 4pq < 4a\,pq \iff (p-q)^2 < 4a\,pq,$$
#
# soit, en repassant aux constantes de temps :
#
# $$\boxed{\;a > a_{\min} = \frac{(p-q)^2}{4pq} = \frac{(\tau_w - \tau_u)^2}{4\,\tau_u\tau_w}.\;}$$
#
# **Lecture de cette condition.** Elle dit exactement ceci : il faut assez de couplage pour
# vaincre l'asymétrie des échelles de temps. Si $\tau_u = \tau_w$, $a_{\min} = 0$ : tout
# couplage strictement positif suffit. Plus les deux constantes de temps sont éloignées,
# plus il faut de couplage. C'est le contenu physique de « sous-amorti » : la rétroaction
# doit être assez forte pour dépasser l'équilibre, pas seulement le rejoindre.
#
# Dans ce régime, en écrivant $\lambda_\pm = \alpha \pm i\omega$ :
#
# $$\alpha = \frac{T}{2} = -\frac{1}{2}\Bigl(\frac{1}{\tau_u} + \frac{1}{\tau_w}\Bigr), \qquad
# \omega = \frac{\sqrt{4D - T^2}}{2}.$$
#
# Deux remarques qui serviront en section 3 :
#
# - $\alpha$ ne dépend **que** des constantes de temps ; $\omega$ dépend en plus de $a$.
#   Amortissement et fréquence ne sont donc **pas** librement découplés dans la
#   paramétrisation biophysique $(\tau_u, \tau_w, a)$ — c'est précisément ce que la
#   littérature SE-adLIF cherche à corriger en reparamétrant.
# - $\alpha < 0$ **toujours**, puisque $\tau_u,\tau_w > 0$. Le neurone biophysique est
#   intrinsèquement contractant. Cette propriété est perdue dès qu'on apprend $\alpha$
#   librement — c'est le sujet du §3.1.

# ## 2.4 Conjugaison au bloc rotation-scaling : le calcul de changement de base
#
# **Énoncé.** Soit $G \in \mathbb{R}^{2\times2}$ de valeurs propres $\alpha \pm i\omega$
# avec $\omega \neq 0$. Alors il existe $P \in GL_2(\mathbb{R})$ telle que
#
# $$P^{-1} G P = \alpha I + \omega J, \qquad J = \begin{pmatrix}0 & -1\\ 1 & 0\end{pmatrix}.$$
#
# **Démonstration.** Soit $v \in \mathbb{C}^2$ un vecteur propre pour $\lambda = \alpha + i\omega$,
# et écrivons $v = p + iq$ avec $p, q \in \mathbb{R}^2$. La relation $Gv = \lambda v$ donne
#
# $$G(p + iq) = (\alpha + i\omega)(p + iq) = (\alpha p - \omega q) + i\,(\omega p + \alpha q).$$
#
# Comme $G$ est **réelle**, $Gp$ et $Gq$ sont réels ; on identifie donc parties réelle et
# imaginaire séparément :
#
# $$G p = \alpha p - \omega q, \qquad G q = \omega p + \alpha q. \tag{$\star$}$$
#
# Posons $P := \begin{pmatrix} q & p \end{pmatrix}$ (colonne 1 $= q$, colonne 2 $= p$).
# Alors $GP = \begin{pmatrix} Gq & Gp\end{pmatrix} = \begin{pmatrix} \omega p + \alpha q & \alpha p - \omega q\end{pmatrix}$
# par $(\star)$. D'autre part, pour $C = \begin{pmatrix} \alpha & -\omega \\ \omega & \alpha\end{pmatrix}$ :
#
# $$PC = \begin{pmatrix} q & p\end{pmatrix}\begin{pmatrix} \alpha & -\omega \\ \omega & \alpha\end{pmatrix}
# = \begin{pmatrix} \alpha q + \omega p & -\omega q + \alpha p \end{pmatrix}.$$
#
# Colonne par colonne, $GP = PC$. Il reste à justifier que $P$ est inversible : si $p$ et
# $q$ étaient colinéaires, $v = p+iq$ serait un multiple complexe d'un vecteur **réel**,
# donc $Gv = \lambda v$ forcerait $\lambda$ réel, contredisant $\omega \neq 0$. Donc
# $\{p,q\}$ est une base de $\mathbb{R}^2$ et $P^{-1}GP = C = \alpha I + \omega J$. $\blacksquare$
#
# **Cas de rupture.** Si $\omega = 0$ la construction s'effondre : les valeurs propres sont
# réelles et $G$ est soit diagonalisable sur $\mathbb{R}$ (deux modes de relaxation
# indépendants), soit un bloc de Jordan (relaxation avec terme en $t\,e^{\alpha t}$). Aucun
# des deux n'est de la forme $\alpha I + \omega J$. **Le bloc rotation-scaling ne couvre pas
# les paires réelles distinctes** — c'est exactement le défaut de représentabilité que le
# dépôt quantifie sous le nom $\delta(c)$, et c'est pourquoi la condition $T^2 < 4D$ n'est
# pas un détail technique mais une hypothèse de modélisation.
#
# ### Conséquence : la forme de l'opérateur discret
#
# $\alpha I$ et $\omega J$ **commutent** ($I$ commute avec tout), donc
# $\exp(\Delta G') = \exp(\Delta\alpha I)\exp(\Delta\omega J)$ avec $G' = \alpha I + \omega J$.
# Et comme $J^2 = -I$, la série exponentielle de $\theta J$ se scinde en parties paire et
# impaire, qui sont exactement les séries de $\cos\theta$ et $\sin\theta$ :
#
# $$\exp(\theta J) = \sum_{n\ge 0}\frac{\theta^n J^n}{n!} = \Bigl(\sum_{m}\frac{(-1)^m\theta^{2m}}{(2m)!}\Bigr) I + \Bigl(\sum_{m}\frac{(-1)^m\theta^{2m+1}}{(2m+1)!}\Bigr) J = \cos\theta\, I + \sin\theta\, J = R(\theta).$$
#
# D'où la forme close de l'opérateur discret :
#
# $$\boxed{\;\bar{A} = \exp(\Delta G') = e^{\Delta\alpha}\,R(\Delta\omega), \qquad \text{valeurs propres } r e^{\pm i\varphi}, \quad r = e^{\Delta\alpha},\ \varphi = \Delta\omega.\;}$$
#
# **C'est, mot pour mot, le bloc `rotation_scaling_zoh` du dépôt.** Le neurone adLIF
# sous-seuil et le bloc $2\times2$ de `structured-latent-dynamics` ne sont pas analogues :
# ce sont deux écritures du même opérateur, reliées par la matrice de passage $P$ qu'on
# vient de construire.

# %%
# --- Verification numerique de l'identite algebrique ---
TAU_U, TAU_W, A_CPL = 20.0, 100.0, 60.0     # ms, ms, couplage sous-seuil

G = np.array([[-1.0 / TAU_U, -1.0 / TAU_U],
              [A_CPL / TAU_W, -1.0 / TAU_W]])

T_tr, D_det = np.trace(G), np.linalg.det(G)
a_min = (TAU_W - TAU_U) ** 2 / (4.0 * TAU_U * TAU_W)

print(f"G =\n{G}")
print(f"\ntrace T   = {T_tr:+.6f}      det D = {D_det:.6f}")
print(f"T^2 - 4D  = {T_tr**2 - 4 * D_det:+.6f}   -> {'sous-amorti' if T_tr**2 < 4*D_det else 'sur-amorti'}")
print(f"a_min = (tau_w - tau_u)^2 / (4 tau_u tau_w) = {a_min:.4f}   (ici a = {A_CPL})")

alpha_c = T_tr / 2.0
omega_c = np.sqrt(4.0 * D_det - T_tr**2) / 2.0
print(f"\nalpha = T/2                = {alpha_c:+.9f}  (1/ms)")
print(f"omega = sqrt(4D - T^2)/2   = {omega_c:+.9f}  (rad/ms)")
print(f"periode 2*pi/omega         = {2 * np.pi / omega_c:.3f} ms")
print(f"valeurs propres de G       = {np.linalg.eigvals(G)}")

# Changement de base explicite : P = [q | p] avec v = p + i q vecteur propre de lambda_+
evals, evecs = np.linalg.eig(G)
idx = int(np.argmax(evals.imag))            # celle dont la partie imaginaire est > 0
v = evecs[:, idx]
P = np.column_stack([v.imag, v.real])       # [q | p]
C = np.linalg.solve(P, G @ P)
C_target = alpha_c * np.eye(2) + omega_c * J2

print(f"\nP =\n{P}")
print(f"\nP^-1 G P =\n{C}")
print(f"\nalpha*I + omega*J =\n{C_target}")
print(f"\nresidu max |P^-1 G P - (alpha I + omega J)| = {np.max(np.abs(C - C_target)):.3e}   [reproduit ici]")

# %%
def expm_2x2(M, terms=64):
    """exp(M) par serie de Taylor avec 'scaling and squaring' (numpy seul).

    On divise M par 2^s pour ramener sa norme sous 1/2 (la serie converge alors en
    quelques termes), puis on eleve au carre s fois : exp(M) = (exp(M/2^s))^(2^s).
    Le nombre de mises au carre est choisi au minimum : chaque mise au carre double
    l'erreur d'arrondi, on n'en fait donc aucune de trop.
    """
    nrm = np.max(np.sum(np.abs(M), axis=1))
    s = max(0, int(np.ceil(np.log2(nrm))) + 1) if nrm > 0 else 0
    Ms = M / (2.0**s)
    S = np.eye(M.shape[0])
    T = np.eye(M.shape[0])
    for k in range(1, terms):
        T = T @ Ms / k
        S = S + T
    for _ in range(s):
        S = S @ S
    return S


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


DELTA_LAT = 0.5   # pas de discretisation latent utilise dans cette section

Abar_num = expm_2x2(DELTA_LAT * G)                                   # exp(Delta * G), brut
Abar_str = P @ (np.exp(DELTA_LAT * alpha_c) * rot(DELTA_LAT * omega_c)) @ np.linalg.inv(P)

r_block = np.exp(DELTA_LAT * alpha_c)
print(f"exp(Delta*G) direct =\n{Abar_num}")
print(f"\nP [ e^(D alpha) R(D omega) ] P^-1 =\n{Abar_str}")
print(f"\nresidu max = {np.max(np.abs(Abar_num - Abar_str)):.3e}   [reproduit ici]")
print(f"\nrayon predit  r = e^(Delta*alpha) = {r_block:.9f}")
print(f"rayon mesure  |eig(exp(Delta G))|  = {np.abs(np.linalg.eigvals(Abar_num))}")
print(f"phase predite Delta*omega          = {DELTA_LAT * omega_c:.9f} rad")
print(f"phase mesuree arg(eig)             = "
      f"{abs(np.angle(np.linalg.eigvals(Abar_num))[0]):.9f} rad")

# %%
fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))

# (a) trajectoire libre du neurone adLIF sous-seuil, coordonnees biophysiques
N = 400
x = np.zeros((N + 1, 2))
x[0] = np.array([1.0, 0.0])
for k in range(N):
    x[k + 1] = Abar_num @ x[k]
axes[0].plot(x[:, 0], x[:, 1], lw=0.9, color="black")
axes[0].scatter([x[0, 0]], [x[0, 1]], s=28, color="tab:red", zorder=5, label="$x_0$")
axes[0].set_xlabel("$u - u_{rest}$")
axes[0].set_ylabel("$w$ (adaptation)")
axes[0].set_title("(a) plan de phase adLIF\ncoordonnees biophysiques")
axes[0].legend(loc="lower left", fontsize=8)
axes[0].set_aspect("equal", adjustable="datalim")

# (b) la meme trajectoire vue dans la base P : une spirale logarithmique pure
y = np.linalg.solve(P, x.T).T
axes[1].plot(y[:, 0], y[:, 1], lw=0.9, color="tab:blue")
theta = np.linspace(0, 2 * np.pi, 200)
for rad in (1.0, 0.5, 0.25):
    axes[1].plot(rad * np.cos(theta) * np.linalg.norm(y[0]),
                 rad * np.sin(theta) * np.linalg.norm(y[0]),
                 color="0.8", lw=0.7, zorder=0)
axes[1].set_xlabel("$z_1$")
axes[1].set_ylabel("$z_2$")
axes[1].set_title("(b) meme trajectoire, base $P$\nrotation + contraction pures")
axes[1].set_aspect("equal", adjustable="datalim")

# (c) le module decroit exactement en r^k
kk = np.arange(N + 1)
axes[2].semilogy(kk, np.linalg.norm(y, axis=1), lw=1.4, color="tab:blue", label=r"$\|z_k\|$ mesure")
axes[2].semilogy(kk, np.linalg.norm(y[0]) * r_block**kk, lw=1.0, color="black", ls="--",
                 label=r"$\|z_0\|\, r^k$, $r=e^{\Delta\alpha}$")
axes[2].set_xlabel("pas $k$")
axes[2].set_ylabel("module")
axes[2].set_title("(c) le module suit $r^k$ exactement")
axes[2].legend()

fig.suptitle("Figure 3 — le neurone adLIF sous-seuil EST un bloc rotation-scaling", y=1.04)
fig.tight_layout()
plt.show()

print(f"ecart max entre ||z_k|| et ||z_0|| r^k : "
      f"{np.max(np.abs(np.linalg.norm(y, axis=1) - np.linalg.norm(y[0]) * r_block**kk)):.3e}   [reproduit ici]")

# %% [markdown]
# **Lecture.** Panneau (a) : dans les coordonnées biophysiques $(u-u_{\text{rest}},\,w)$, la
# trajectoire libre du neurone est une spirale **oblique** — les axes ne sont pas les axes
# propres, la conjugaison n'est donc pas triviale. Panneau (b) : la **même** trajectoire
# écrite dans la base $P = [q\,|\,p]$ construite au §2.4 devient une spirale
# logarithmique droite, invariante par rotation autour de l'origine ; le changement de base
# a absorbé toute l'obliquité. C'est la lecture géométrique de $P^{-1}GP = \alpha I + \omega J$.
# Panneau (c) : le module décroît **exactement** comme $r^k$ avec $r = e^{\Delta\alpha}$ — la
# courbe mesurée et la loi prédite sont confondues. Ce panneau est celui qui compte pour la
# section 3 : il montre que *toute* l'information de stabilité du bloc tient dans le seul
# nombre $r$, et que $\omega$ n'y participe pas.

# %% [markdown]
# ## 2.5 Ce que cette identité implique — et ce qu'elle n'implique pas
#
# ### Ce qu'elle implique
#
# **Le bloc $2\times2$ du dépôt et le neurone adLIF sous-seuil sont le même objet.** Écrire
# $\bar{A} = \exp(\Delta G)$ avec $G = \alpha I + \omega J$, c'est écrire un neurone
# adaptatif sous-seuil dans sa base propre réelle. La conséquence opérationnelle est
# immédiate : **toute la littérature sur la discrétisation et la paramétrisation des
# neurones adaptatifs s'applique directement au predictor latent du dépôt**, sans transfert
# analogique, sans « inspiration ». Une garantie prouvée sur un adLIF discrétisé est une
# garantie sur le bloc.
#
# ### Ce qu'elle n'implique pas — la partie honnête
#
# Il faut résister à quatre glissements, tous tentants et tous faux :
#
# 1. **Le neurone n'apporte aucune capacité supplémentaire.** La classe de fonctions
#    $\{\bar{A} = e^{\Delta\alpha}R(\Delta\omega)\}$ est la même, qu'on l'appelle « bloc
#    rotation-scaling » ou « adLIF sous-seuil ». Deux paramètres réels par bloc, avant comme
#    après. Renommer n'augmente pas l'expressivité, et personne ne devrait le prétendre.
# 2. **Ce n'est pas un argument de plausibilité biologique.** Le fait qu'un opérateur
#    ressemble à un neurone ne dit rien sur ses performances. Le dossier tient la neuro pour
#    *source d'hypothèses*, jamais pour autorité — c'est une règle écrite du workspace, et
#    elle s'applique ici en premier.
# 3. **Ce n'est pas un argument d'efficacité énergétique.** Le gain énergétique invoqué pour
#    les SNN vient du **spike** (calcul événementiel, activité creuse), pas du régime
#    sous-seuil. Un bloc sous-seuil dense coûte exactement ce que coûte une matrice $2\times2$.
#    Voir la section 4.
# 4. **L'identité ne valide pas la structure.** Que le prior soit « biologique » ne le rend
#    pas approprié au système à modéliser. Si la dynamique cible a une paire propre **réelle
#    distincte**, aucun bloc rotation-scaling ne la contient (§2.4, cas de rupture), neurone
#    ou pas.
#
# ### Ce que ça apporte réellement, en une phrase
#
# **Une littérature sur la discrétisation.** Le problème ouvert du dépôt n'est pas
# « quelle structure ? » mais « comment paramétrer cette structure pour qu'elle
# s'entraîne ? ». La communauté des neurones à dynamique se pose cette question depuis
# 2024 et publie des réponses testées. C'est le seul transfert revendiqué — et il suffit,
# comme la section 3 va le montrer.

# %% [markdown]
# ---
#
# # 3. La discrétisation décide du rayon
#
# C'est la section centrale. Tout ce qui précède sert à établir qu'on a le droit
# d'utiliser la littérature neurone ici ; tout ce qui suit s'en sert.
#
# ## 3.1 Le rayon est libre, donc non borné
#
# Le dépôt discrétise en **temps continu exact** : il paramétrise le générateur $G = \alpha I + \omega J$
# et calcule $\bar{A} = \exp(\Delta G) = e^{\Delta\alpha} R(\Delta\omega)$. Le rayon spectral
# de l'opérateur discret est donc
#
# $$r = |\lambda(\bar{A})| = e^{\Delta\alpha}.$$
#
# Le protocole du dépôt (`PROTOCOL.md §13.2`) laisse $\alpha$ **libre** — explicitement :
# *« no `-exp(nu)`, no `-softplus` »*. La raison est méthodologique et bonne : on veut que
# les modes **stables et instables restent représentables**, et que la stabilité soit
# **mesurée, pas imposée**. Un opérateur qu'on a contraint à contracter ne peut plus servir
# à répondre à la question « la structure apporte-t-elle de la stabilité ? », puisque la
# réponse est cousue dans la paramétrisation.
#
# Mais laisser $\alpha \in \mathbb{R}$ libre veut dire $r = e^{\Delta\alpha} \in (0, +\infty)$.
# Il n'y a **aucune** valeur de $\alpha$ interdite, et rien dans l'initialisation ne garantit
# $\alpha < 0$.
#
# ### L'amplification en rollout fermé
#
# Un rollout **fermé** (*closed-loop*, auto-consommé) de $H$ pas compose l'opérateur $H$ fois
# sans réinjecter d'observation :
#
# $$x_H = \bar{A}^H x_0 \quad\Longrightarrow\quad \frac{\partial x_H}{\partial x_0} = \bar{A}^H, \qquad \bigl\|\bar{A}^H\bigr\| \sim r^H.$$
#
# Et par la règle de la chaîne, le gradient de la perte par rapport à $\alpha$ contient le
# même facteur : $\partial x_H/\partial\alpha = H\Delta\, x_H$, donc
# $\|\nabla_\alpha L\| \propto H\Delta\, \|x_H\|^2 \sim H\Delta\, r^{2H}$. **Le gradient est
# quadratique en l'amplification de l'état.**
#
# Le dépôt utilise $H = 25$ depuis l'amendement 6. Le tableau ci-dessous est la raison pour
# laquelle un écart de quelques pourcents sur $r$ n'est pas un détail.

# %%
H_ROLL = 25

print(f"Amplification r^H pour un rollout ferme de H = {H_ROLL} pas")
print(f"{'r':>8} | {'r^H':>14} | {'r^(2H) (~gradient)':>20}")
print("-" * 50)
for r_ in (0.90, 0.95, 0.99, 1.00, 1.05, 1.20, 1.50, 2.00):
    print(f"{r_:>8.2f} | {r_**H_ROLL:>14.4g} | {r_**(2*H_ROLL):>20.4g}")
print()
print("depot (PLAN_V3 / REVIEW_FLAGSHIPS 3.1, mesure de reference) :")
print("  r=1.05 -> 3.4 ;  r=1.20 -> 95 ;  r=1.50 -> 2.5e4")
print(f"reproduit ici           :  {1.05**25:.1f} ;  {1.2**25:.0f} ;  {1.5**25:.3e}")
print()
print("Marge en alpha : avec Delta = 1, passer de r=1 a r=1.05 demande")
print(f"  alpha = ln(1.05) = {np.log(1.05):.5f}. Un ecart de 5 centiemes sur alpha.")

# %% [markdown]
# ## 3.2 Le fait mesuré : comment la famille `ssm` du dépôt est morte
#
# Les chiffres de ce paragraphe sont **mesurés** dans `structured-latent-dynamics`
# (§17.0 / §18.0, repris dans `REVIEW_FLAGSHIPS.md` §3.1). Ils ne sont pas reproduits ici :
# ils viennent de runs réels, sur le harnais réel.
#
# | observation | famille `ssm` | famille `mlp` |
# |---|---|---|
# | norme du gradient au **premier** pas | **2.4e18** | **28.2** |
# | perte à l'époque 0 | saine | saine |
# | perte à l'époque 1, **avec** clip de gradient | **6.4413e24** | — |
# | perte à l'époque 1, **sans** clip de gradient | **6.4415e24** | — |
#
# Conclusion enregistrée dans le dépôt : la famille `ssm` a été déclarée **sans taux
# entraînable** après divergence à l'époque 2, à tous les taux d'apprentissage essayés,
# **même harnais** que le `mlp`.
#
# ### Le raisonnement, étape par étape
#
# La ligne décisive est la dernière paire. Voici pourquoi.
#
# **Hypothèse A — « le pas d'optimiseur est trop grand ».** Si la divergence venait d'un pas
# trop grand, le clip de gradient serait exactement le remède : il borne la norme du
# gradient, donc borne le déplacement des paramètres, donc borne la dégradation de la perte
# au pas suivant. Sous A, on **prédit** que la perte à l'époque 1 chute de plusieurs ordres
# de grandeur quand on active le clip.
#
# **Observation.** $6.4413\times10^{24}$ contre $6.4415\times10^{24}$ : les deux pertes
# coïncident sur **cinq chiffres significatifs**. L'écart relatif est de
# $3\times10^{-5}$. Le clip n'a rien changé.
#
# **Réfutation de A.** Une perte invariante au clip ne peut pas être causée par la taille du
# pas, puisque le clip agit précisément sur la taille du pas et n'a produit aucun effet. Ce
# n'est donc pas un problème d'optimiseur.
#
# **Hypothèse B — « la carte avant diverge ».** Si $r > 1$, alors $\|x_H\| \sim r^H \|x_0\|$
# est déjà astronomique **à paramètres gelés**, avant toute mise à jour. La perte est alors
# essentiellement une fonction de $r$, et pour la ramener à un niveau raisonnable il faut
# déplacer $\alpha$ d'une distance **finie** (traverser $\alpha = 0$), ce qu'aucun budget de
# pas bornés ne fait en une époque. Sous B, on **prédit** exactement ce qu'on observe :
# perte énorme des deux côtés, quasi identique, et gradient géant dès le premier pas parce
# que $\|\nabla L\| \sim r^{2H}$.
#
# **Ordre de grandeur, en contrôle :** $2.4\times10^{18}$ est cohérent avec $r^{2H}$ pour
# $r \approx 1.94$ à $H=25$ ($\ln(2.4\!\times\!10^{18})/50 = 0.845$, soit $r = e^{0.845}$).
# Un rayon initial autour de $2$ suffit à produire ce chiffre — on n'a pas besoin de
# supposer une pathologie exotique.
#
# **Ce que le dépôt en conclut, et ce qu'il n'en conclut pas.** §17.0 admet par écrit :
# *« aucun travail publié n'entraîne la loss de dynamique d'un SSM sur un rollout
# auto-consommé de 25 pas depuis l'initialisation sans garde »*. Le régime testé est donc
# inédit, et l'énoncé « la famille SSM ne s'entraîne pas » n'est **pas** un fait sur les
# priors structurels : c'est un fait sur **cette discrétisation à ce rayon dans ce régime**.
# La cellule suivante reproduit le mécanisme sur un modèle jouet, à l'échelle où on peut le
# regarder.

# %%
# --- Reproduction du MECANISME sur un modele jouet (pas des chiffres du depot) ---
# Modele : bloc rotation-scaling 2x2, rollout FERME de H = 25 pas, cible engendree
# par un bloc verite-terrain contractant. Deux parametres seulement, (alpha, omega),
# gradient analytique exact. Tout le reste est apparie entre les conditions.

rng_toy = np.random.default_rng(7)
X0 = rng_toy.normal(size=(64, 2))
ALPHA_TRUE, OMEGA_TRUE = -0.05, 0.9


def block(alpha, omega, t):
    """exp(t * (alpha I + omega J)) en forme close."""
    return np.exp(t * alpha) * rot(t * omega)


Y_TARGET = X0 @ block(ALPHA_TRUE, OMEGA_TRUE, H_ROLL).T


def loss_and_grad(theta):
    """L = 1/2 E ||x_H - y||^2 avec x_H = exp(H(alpha I + omega J)) x_0.

    d x_H / d alpha = H x_H   et   d x_H / d omega = H J x_H  (car alpha I et omega J
    commutent, donc la derivee de l'exponentielle est un simple produit).
    """
    alpha, omega = theta
    XH = X0 @ block(alpha, omega, H_ROLL).T
    Rres = XH - Y_TARGET
    L = 0.5 * np.mean(np.sum(Rres**2, axis=1))
    g_alpha = np.mean(np.sum(Rres * (H_ROLL * XH), axis=1))
    g_omega = np.mean(np.sum(Rres * (H_ROLL * (XH @ J2.T)), axis=1))
    return L, np.array([g_alpha, g_omega])


# Controle : le gradient analytique contre differences finies centrees.
th_chk = np.array([0.12, 0.8])
L0, g_ana = loss_and_grad(th_chk)
eps_fd = 1e-6
g_fd = np.array([
    (loss_and_grad(th_chk + eps_fd * e)[0] - loss_and_grad(th_chk - eps_fd * e)[0]) / (2 * eps_fd)
    for e in np.eye(2)
])
print(f"gradient analytique      : {g_ana}")
print(f"differences finies       : {g_fd}")
print(f"ecart relatif            : {np.max(np.abs(g_ana - g_fd) / np.abs(g_fd)):.3e}   [reproduit ici]")

# %%
def adam_epoch(theta0, n_steps=200, lr=1e-4, clip=None, b1=0.9, b2=0.999, eps=1e-8):
    """Une 'epoque' d'Adam. Renvoie la perte finale, la norme du gradient au pas 1
    et la trajectoire des parametres. clip=None -> aucun ecretage."""
    theta = theta0.copy()
    m = np.zeros(2)
    v = np.zeros(2)
    g1_norm = None
    for t in range(1, n_steps + 1):
        L, g = loss_and_grad(theta)
        if t == 1:
            g1_norm = np.linalg.norm(g)
        if clip is not None:
            n = np.linalg.norm(g)
            if n > clip:
                g = g * (clip / n)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        theta = theta - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
    return loss_and_grad(theta)[0], g1_norm, theta


THETA_DIV = np.array([0.50, 0.9])    # bras 'rayon libre' : alpha > 0 a l'init -> r = e^0.5
THETA_STA = np.array([-0.20, 0.9])   # bras apparie, seule difference : alpha < 0 -> r < 1

L_clip, g1_div, th_clip = adam_epoch(THETA_DIV, clip=1.0)
L_free, _, th_free = adam_epoch(THETA_DIV, clip=None)
L_sta, g1_sta, _ = adam_epoch(THETA_STA, clip=1.0)

print("Modele jouet, harnais strictement apparie (memes donnees, meme H, meme optimiseur).")
print(f"  bras divergent  : alpha0 = {THETA_DIV[0]:+.2f} -> r = {np.exp(THETA_DIV[0]):.3f}")
print(f"  bras contractant: alpha0 = {THETA_STA[0]:+.2f} -> r = {np.exp(THETA_STA[0]):.3f}")
print()
print(f"norme du gradient au pas 1, bras divergent   : {g1_div:.4e}")
print(f"norme du gradient au pas 1, bras contractant : {g1_sta:.4e}")
print(f"rapport                                      : {g1_div / g1_sta:.3e}   [reproduit ici]")
print(f"  (depot, mesure : 2.4e18 pour 'ssm' contre 28.2 pour 'mlp', meme harnais)")
print()
print(f"perte apres une epoque, AVEC clip            : {L_clip:.6e}")
print(f"perte apres une epoque, SANS clip            : {L_free:.6e}")
print(f"rapport clip/sans-clip                       : {L_clip / L_free:.4f}   [reproduit ici]")
print(f"  (depot, mesure : 6.4413e24 contre 6.4415e24, rapport 0.99997)")
print()
print(f"alpha apres une epoque, AVEC clip            : {th_clip[0]:+.4f}  -> r = {np.exp(th_clip[0]):.3f}")
print(f"alpha apres une epoque, SANS clip            : {th_free[0]:+.4f}  -> r = {np.exp(th_free[0]):.3f}")
print(f"budget de deplacement d'Adam en {200} pas a lr=1e-4 : {200 * 1e-4:.3f} en alpha")
print(f"distance a franchir pour atteindre r < 1     : {THETA_DIV[0]:.3f}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))

alphas = np.linspace(-0.35, 0.55, 300)
gnorms = np.array([np.linalg.norm(loss_and_grad(np.array([a_, 0.9]))[1]) for a_ in alphas])
losses = np.array([loss_and_grad(np.array([a_, 0.9]))[0] for a_ in alphas])

axes[0].semilogy(np.exp(alphas), losses, lw=1.6, color="black")
axes[0].axvline(1.0, color="crimson", ls="--", lw=1.2, label="$r = 1$")
axes[0].set_xlabel(r"rayon $r = e^{\Delta\alpha}$")
axes[0].set_ylabel("perte (rollout ferme, H=25)")
axes[0].set_title("(a) la perte est une fonction du rayon")
axes[0].legend()

axes[1].semilogy(np.exp(alphas), gnorms, lw=1.6, color="black", label=r"$\|\nabla L\|$ mesure")
ref = gnorms[np.argmin(np.abs(np.exp(alphas) - 1.0))] * np.exp(alphas) ** (2 * H_ROLL)
axes[1].semilogy(np.exp(alphas), ref, lw=1.0, ls="--", color="tab:blue", label=r"$\propto r^{2H}$")
axes[1].axvline(1.0, color="crimson", ls="--", lw=1.2)
axes[1].set_xlabel(r"rayon $r = e^{\Delta\alpha}$")
axes[1].set_ylabel(r"$\|\nabla L\|$")
axes[1].set_title("(b) le gradient suit $r^{2H}$")
axes[1].legend(loc="upper left")

bars = ["clip = 1.0", "sans clip"]
vals = [L_clip, L_free]
axes[2].bar(bars, vals, color=["0.35", "0.7"], width=0.55)
axes[2].set_yscale("log")
axes[2].set_ylabel("perte apres une epoque")
axes[2].set_title("(c) le clip ne deplace pas le niveau")
for i, vv in enumerate(vals):
    axes[2].text(i, vv * 1.25, f"{vv:.3e}", ha="center", fontsize=8)
axes[2].set_ylim(min(vals) * 0.2, max(vals) * 8)

fig.suptitle("Figure 4 — le rayon fixe le niveau de la perte et l'echelle du gradient", y=1.04)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Lecture.** Panneau (a) : la perte du rollout fermé est essentiellement une fonction du
# seul rayon, et elle traverse plusieurs ordres de grandeur en franchissant $r=1$ (ligne
# rouge). Panneau (b) : la norme du gradient mesurée suit la loi prédite $r^{2H}$ sur toute
# la plage — la droite bleue n'est pas ajustée, c'est $r^{2H}$ normalisé en $r=1$. Un rayon
# à $1.5$ suffit à produire un gradient $10^{8}$ fois plus grand qu'à $r=1$, sans aucune
# pathologie d'implémentation : c'est de l'arithmétique. Panneau (c) : les deux pertes après
# une époque restent au même niveau, écrêtage ou non. Notre modèle jouet ne reproduit pas
# les cinq chiffres du dépôt — nous obtenons un rapport de l'ordre de l'unité, le dépôt
# mesure $0.99997$ — mais il reproduit **le fait diagnostique** : borner le pas ne déplace
# pas le niveau de la perte, parce que ce niveau est fixé par $r$ et que ramener $r$ sous 1
# demande de traverser une distance finie en $\alpha$ que le budget de pas ne couvre pas.
# La conclusion opérationnelle est donc : **ce n'est pas un problème d'optimiseur, c'est un
# problème de paramétrisation du rayon.**

# %% [markdown]
# ## 3.3 La discrétisation symplectique — la solution évidente, et son piège
#
# La littérature des neurones adaptatifs a rencontré exactement ce problème, et y a répondu
# par un changement de **schéma d'intégration** plutôt que de paramétrisation. C'est
# l'approche de **SE-adLIF** (arXiv:2408.07517, Nature Communications 2025).
#
# ### Le schéma
#
# L'Euler **symplectique** (dit aussi semi-implicite) diffère de l'Euler explicite par une
# seule chose : la seconde équation utilise la valeur **déjà mise à jour** de la première.
# En notation normalisée $\alpha := 1 - \Delta/\tau_u$ et $\beta := 1 - \Delta/\tau_w$ (donc
# $\alpha,\beta \in (0,1)$ dès que $\Delta < \min(\tau_u,\tau_w)$) :
#
# $$\begin{aligned}
# u[k+1] &= \alpha\,u[k] - (1-\alpha)\,w[k] + (1-\alpha)\,I[k] & &\text{(explicite en } w)\\
# w[k+1] &= \beta\,w[k] + (1-\beta)\,a\,\underbrace{u[k+1]}_{\text{déjà mis à jour}} & &\text{(implicite en } u)
# \end{aligned}$$
#
# En substituant la première ligne dans la seconde :
#
# $$w[k+1] = (1-\beta)a\alpha\,u[k] + \Bigl(\beta - (1-\beta)a(1-\alpha)\Bigr) w[k] + (1-\beta)a(1-\alpha) I[k],$$
#
# d'où la matrice de transition, qui est exactement celle de la littérature SE-adLIF :
#
# $$M(\alpha,\beta,a) = \begin{pmatrix} \alpha & -(1-\alpha) \\[2pt] (1-\beta)a\alpha & \beta - (1-\beta)a(1-\alpha)\end{pmatrix}.$$
#
# ### La propriété remarquable : $\det M = \alpha\beta$, indépendamment de $a$
#
# **Démonstration.** Par définition du déterminant d'une matrice $2\times2$ :
#
# $$\begin{aligned}
# \det M &= \alpha\cdot\Bigl(\beta - (1-\beta)a(1-\alpha)\Bigr) \;-\; \Bigl(-(1-\alpha)\Bigr)\cdot\Bigl((1-\beta)a\alpha\Bigr)\\
# &= \alpha\beta \;-\; \alpha(1-\alpha)(1-\beta)a \;+\; \alpha(1-\alpha)(1-\beta)a\\
# &= \alpha\beta.
# \end{aligned}$$
#
# Les deux termes en $a$ sont **identiques au signe près** et s'annulent. L'identité est
# exacte pour tout $(\alpha,\beta,a) \in \mathbb{R}^3$ — ce n'est pas une approximation au
# petit pas. $\blacksquare$
#
# **Pourquoi c'est intéressant.** Pour une matrice $2\times2$ à valeurs propres complexes
# conjuguées $\lambda_\pm = \rho e^{\pm i\varphi}$, on a $\det M = \lambda_+\lambda_- = \rho^2$,
# donc
#
# $$|\lambda| = \sqrt{\det M} = \sqrt{\alpha\beta} < 1 \quad \text{pour } \alpha,\beta \in (0,1).$$
#
# **La stabilité serait garantie par construction, sans contraindre $a$**, et le couplage
# $a$ (donc la fréquence) serait libre : amortissement et fréquence **découplés**. C'est
# exactement la promesse de SE-adLIF, et c'est ce que le plan v2 de ce dossier avait
# retenu.

# ### Le piège : la borne n'est pas inconditionnelle
#
# L'égalité $|\lambda| = \sqrt{\det M}$ n'est vraie **que si les valeurs propres sont
# complexes conjuguées**. La condition est
#
# $$\operatorname{tr}(M)^2 < 4\det(M) = 4\alpha\beta.$$
#
# Or la trace, elle, **dépend de $a$** :
#
# $$\operatorname{tr} M = \alpha + \beta - (1-\beta)\,a\,(1-\alpha).$$
#
# Elle décroît linéairement en $a$ et devient arbitrairement négative. Dès que
# $\operatorname{tr}^2 \ge 4\alpha\beta$, les valeurs propres deviennent **réelles** :
#
# $$\lambda_\pm = \frac{\operatorname{tr} \pm \sqrt{\operatorname{tr}^2 - 4\det}}{2}, \qquad
# |\lambda|_{\max} = \frac{|\operatorname{tr}| + \sqrt{\operatorname{tr}^2 - 4\det}}{2}.$$
#
# Et quand $\det = \alpha\beta \to 0$ (petits $\alpha,\beta$, c'est-à-dire pas grand devant
# les constantes de temps), $|\lambda|_{\max} \to |\operatorname{tr}| \to (1-\beta)a(1-\alpha)$,
# **linéaire en $a$ et donc non bornée**.
#
# La borne $\sqrt{\alpha\beta}$ ne protège donc que dans une région, et c'est la région
# oscillatoire — celle-là même qu'on veut quitter librement pour explorer les fréquences.
#
# **Ce n'est pas une critique de SE-adLIF**, dont le domaine de paramètres physiologiques
# reste oscillatoire ; c'est une critique de son usage comme *garantie* dans un dépôt où
# $a$ serait appris sans contrainte.

# %%
# --- Verification 1 : det M = alpha*beta, exactement, independamment de a ---
def M_symp(alpha, beta, a):
    return np.array([[alpha, -(1.0 - alpha)],
                     [(1.0 - beta) * a * alpha, beta - (1.0 - beta) * a * (1.0 - alpha)]])


A_EX, B_EX = 0.9, 0.8
print(f"alpha = {A_EX}, beta = {B_EX}  ->  alpha*beta = {A_EX * B_EX:.12f}")
print(f"\n{'a':>8} | {'tr(M)':>9} | {'det(M)':>14} | {'|lambda|max':>11} | regime")
print("-" * 62)
for a_ in (0.5, 3.0, 30.0, 300.0):
    Mx = M_symp(A_EX, B_EX, a_)
    tr, dt = np.trace(Mx), np.linalg.det(Mx)
    lam = np.linalg.eigvals(Mx)
    reg = "complexe" if tr**2 < 4 * dt else "REEL - borne perdue"
    print(f"{a_:>8.1f} | {tr:>+9.4f} | {dt:>14.12f} | {np.max(np.abs(lam)):>11.4f} | {reg}")

print("\ndepot (mesure, PLAN_V3 II.1) : 0.8485 pour a in {0.5, 3, 30} et 4.1255 pour a = 300")
print("det(M) - alpha*beta sur une grille dense de (alpha, beta, a) :")
gg = np.linspace(0.01, 0.99, 40)
aa = np.exp(np.linspace(-6, 6, 40))
worst = 0.0
for al_ in gg:
    for be_ in gg:
        for a_ in aa:
            worst = max(worst, abs(np.linalg.det(M_symp(al_, be_, a_)) - al_ * be_))
print(f"  ecart maximal = {worst:.3e}   [reproduit ici]  (arrondi flottant pur)")

# %%
# --- Verification 2 : 200 000 tirages, quel est le rayon maximal reellement atteint ? ---
N_DRAW = 200_000
radii_by_seed = []
for seed in (0, 1, 2, 3, 4):
    rg = np.random.default_rng(seed)
    al = rg.uniform(0.0, 1.0, N_DRAW)
    be = rg.uniform(0.0, 1.0, N_DRAW)
    a_s = np.exp(rg.uniform(-6.0, 6.0, N_DRAW))          # a dans [e^-6, e^6]
    tr = al + be - (1.0 - be) * a_s * (1.0 - al)
    det = al * be
    disc = tr * tr - 4.0 * det
    rad = np.where(disc < 0.0, np.sqrt(det),
                   0.5 * (np.abs(tr) + np.sqrt(np.maximum(disc, 0.0))))
    radii_by_seed.append(rad)
    print(f"seed {seed} : rayon max = {rad.max():7.1f} | "
          f"fraction reelle = {np.mean(disc >= 0):.4f} | "
          f"fraction |lambda| > 1 = {np.mean(rad > 1.0):.4f}")

rad0 = radii_by_seed[0]
maxima = np.array([r.max() for r in radii_by_seed])
print(f"\nmax sur 5 replicats : min {maxima.min():.1f}, max {maxima.max():.1f}   [reproduit ici]")
print(f"depot (mesure, PLAN_V3 II.1, un tirage de 200 000) : 368")
print()
print("Borne superieure theorique : quand alpha, beta -> 0, det -> 0 et")
print("|lambda|max -> |tr| -> (1-beta) a (1-alpha) -> a, donc sup = e^6 = "
      f"{np.exp(6.0):.2f}.")
print("Le maximum echantillonne est une statistique d'extremes : il approche ce sup")
print("sans l'atteindre, et sa valeur exacte depend du tirage. C'est l'ORDRE qui est")
print("le resultat, pas le chiffre : la borne sqrt(alpha*beta) < 1 est violee de plus")
print("de deux ordres de grandeur.")

# %%
fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6))

# On rejoue exactement le tirage seed=0 pour tracer trace et determinant.
rg2 = np.random.default_rng(0)
al_p = rg2.uniform(0, 1, N_DRAW)
be_p = rg2.uniform(0, 1, N_DRAW)
a_p = np.exp(rg2.uniform(-6, 6, N_DRAW))
tr_p = al_p + be_p - (1 - be_p) * a_p * (1 - al_p)
det_p = al_p * be_p

# Sous-echantillon lisible, restreint a la fenetre de trace affichee.
m = np.abs(tr_p) < 6
idx = np.random.default_rng(1).choice(np.flatnonzero(m), 4000, replace=False)
cplx = tr_p[idx] ** 2 < 4 * det_p[idx]
axes[0].scatter(tr_p[idx][cplx], det_p[idx][cplx], s=2, color="tab:blue", label="complexe : $|\\lambda|=\\sqrt{\\alpha\\beta}<1$")
axes[0].scatter(tr_p[idx][~cplx], det_p[idx][~cplx], s=2, color="tab:red", label="reel : borne perdue")
tline = np.linspace(-6, 6, 400)
axes[0].plot(tline, tline**2 / 4, color="black", lw=1.2, label=r"$\mathrm{tr}^2 = 4\det$")
axes[0].set_xlim(-6, 3)
axes[0].set_ylim(0, 1.05)
axes[0].set_xlabel(r"$\mathrm{tr}(M)$")
axes[0].set_ylabel(r"$\det(M) = \alpha\beta$")
axes[0].set_title("(a) plan trace-determinant")
axes[0].legend(fontsize=7, loc="upper center")

a_grid = np.logspace(-1, 3, 400)
rad_ex = []
for a_ in a_grid:
    Mx = M_symp(A_EX, B_EX, a_)
    rad_ex.append(np.max(np.abs(np.linalg.eigvals(Mx))))
axes[1].loglog(a_grid, rad_ex, lw=1.6, color="black")
axes[1].axhline(np.sqrt(A_EX * B_EX), color="tab:blue", ls="--", lw=1.2,
                label=r"$\sqrt{\alpha\beta}=%.4f$" % np.sqrt(A_EX * B_EX))
axes[1].axhline(1.0, color="crimson", ls=":", lw=1.2, label=r"$|\lambda|=1$")
for a_ in (0.5, 3.0, 30.0, 300.0):
    Mx = M_symp(A_EX, B_EX, a_)
    axes[1].scatter([a_], [np.max(np.abs(np.linalg.eigvals(Mx)))], s=24, color="tab:red", zorder=5)
axes[1].set_xlabel("couplage $a$")
axes[1].set_ylabel(r"$|\lambda|_{\max}$")
axes[1].set_title(r"(b) $\alpha=0.9,\ \beta=0.8$ : la borne saute")
axes[1].legend(fontsize=7.5, loc="upper left")

axes[2].hist(np.log10(np.maximum(rad0, 1e-6)), bins=120, color="0.4")
axes[2].axvline(0.0, color="crimson", ls="--", lw=1.2, label=r"$|\lambda| = 1$")
axes[2].axvline(np.log10(rad0.max()), color="tab:red", ls="-", lw=1.2,
                label=f"max = {rad0.max():.0f}")
axes[2].axvline(np.log10(np.exp(6.0)), color="black", ls=":", lw=1.2,
                label=r"sup theorique $e^6$")
axes[2].set_yscale("log")
axes[2].set_xlabel(r"$\log_{10} |\lambda|_{\max}$")
axes[2].set_ylabel("effectif")
axes[2].set_title("(c) 200 000 tirages")
axes[2].legend(fontsize=7.5, loc="upper left")

fig.suptitle("Figure 5 — la garantie symplectique tient dans une region, et seulement la", y=1.04)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Lecture.** Panneau (a) : le plan trace-déterminant sépare les deux régimes par la
# parabole $\operatorname{tr}^2 = 4\det$. Les points bleus (sous la parabole) sont
# oscillatoires et vérifient $|\lambda| = \sqrt{\alpha\beta} < 1$ ; les points rouges,
# majoritaires, sont réels et ne sont soumis à aucune borne — **68 %** des tirages tombent
# de ce côté. Panneau (b) : à $(\alpha,\beta)$ fixés, le rayon reste exactement plat à
# $\sqrt{\alpha\beta} = 0.8485$ tant que $a$ est modéré, puis décroche brutalement au
# franchissement de la parabole et croît ensuite linéairement en $a$ ; les quatre points
# rouges sont les valeurs $a \in \{0.5, 3, 30, 300\}$ du tableau. Panneau (c) : l'histogramme
# des 200 000 rayons a un mode net juste sous 1 (le régime protégé) **et** une queue qui
# s'étend sur deux décades au-delà, jusqu'au voisinage du supremum théorique $e^6$.
#
# **Conclusion de méthode, et c'est la décision de conception de tout ce notebook.** Une
# garantie qui dépend d'une inégalité à surveiller n'est pas le bon objet pour un axe
# expérimental. Il faudrait : contraindre $a$, écrire un test qui vérifie la condition à
# chaque pas, définir un comportement au franchissement, et documenter un cas limite de plus
# — dans un dépôt qui compte déjà quatre variables confondantes non contrôlées. On garde
# donc l'algèbre (le générateur continu) et on change **uniquement** la façon dont $\alpha$
# est produit.

# %% [markdown]
# ## 3.4 La paramétrisation retenue, et sa démonstration
#
# On conserve intégralement le générateur continu $G = \alpha I + \omega J$ et la
# discrétisation exacte $\bar{A} = \exp(\Delta G)$. On remplace **la seule production de
# $\alpha$** :
#
# $$\boxed{\;\alpha := \frac{\log \sigma(\rho)}{\Delta}, \qquad \rho \in \mathbb{R} \text{ libre}, \qquad \sigma(\rho) = \frac{1}{1+e^{-\rho}}.\;}$$
#
# **Proposition.** Pour tout $\rho \in \mathbb{R}$ et tout $\Delta > 0$, le rayon spectral du
# bloc discret vaut $r = \sigma(\rho) \in (0,1)$ strictement, sans aucune condition.
#
# **Démonstration.** Trois faits, chacun élémentaire.
#
# 1. *$\sigma$ est à valeurs dans $(0,1)$, bornes exclues.* On a $e^{-\rho} > 0$ pour tout
#    $\rho$ réel, donc $1 + e^{-\rho} > 1$, donc $\sigma(\rho) < 1$ ; et $\sigma(\rho) > 0$
#    comme inverse d'un réel strictement positif fini. De plus $\sigma$ est strictement
#    croissante ($\sigma' = \sigma(1-\sigma) > 0$) et surjective sur $(0,1)$, donc
#    $\rho \mapsto \sigma(\rho)$ est une **bijection** $\mathbb{R} \to (0,1)$ : aucune partie
#    de l'intervalle n'est inaccessible.
# 2. *$\alpha$ est bien défini et strictement négatif.* Comme $\sigma(\rho) \in (0,1)$,
#    $\log\sigma(\rho)$ est fini et strictement négatif ; divisé par $\Delta > 0$, on a
#    $\alpha \in (-\infty, 0)$.
# 3. *Le rayon.* Par le §2.4, $r = e^{\Delta\alpha}$. En substituant :
#    $$r = \exp\!\Bigl(\Delta \cdot \frac{\log\sigma(\rho)}{\Delta}\Bigr) = \exp\bigl(\log\sigma(\rho)\bigr) = \sigma(\rho).$$
#    Le $\Delta$ s'annule **exactement** — c'est la raison pour laquelle on divise par
#    $\Delta$ dans la définition plutôt que d'écrire $\alpha = \log\sigma(\rho)$. $\blacksquare$
#
# **Ce qui est démontré, précisément :** $r \in (0,1)$ pour **tout** $\rho$ réel, sans
# hypothèse sur $\omega$, sur le pas $\Delta$, ni sur quoi que ce soit d'autre. Il n'y a
# aucune inégalité à surveiller, aucun test de régime, aucun cas limite. C'est exactement
# ce que la carte symplectique ne fournissait pas.
#
# **Cas de rupture — il y en a deux, et il faut les nommer.**
#
# *(i) Saturation du paramètre.* La borne devient *vide* dans les limites :
# $\rho \to +\infty \Rightarrow r \to 1^-$ (le gradient $\partial r/\partial\rho = \sigma(1-\sigma) \to 0$,
# le paramètre sature) et $\rho \to -\infty \Rightarrow r \to 0^+$ (sur-contraction, l'état
# est effacé en un pas). Aucun des deux ne casse la garantie de stabilité, mais les deux
# créent des plateaux d'optimisation. C'est le prix de la borne, et il doit être mesuré, pas
# supposé absent.
#
# *(ii) Arithmétique flottante.* L'inégalité $r < 1$ est stricte dans $\mathbb{R}$, mais
# $\sigma(\rho)$ **arrondit à $1.0$** dès que $1-\sigma(\rho) = \sigma(-\rho)$ passe sous
# $\varepsilon/2$, soit $\rho \gtrsim 37$ en `float64` (et $\rho \gtrsim 17$ en `float32`).
# La cellule de vérification le mesure. Cette dégénérescence est **bénigne et il faut
# savoir pourquoi** : elle produit $r = 1$, donc $r^H = 1$, donc **aucune amplification**.
# La borne ne peut jamais se dégrader en divergence, seulement en neutralité — ce qui n'est
# pas le cas de la carte symplectique, dont la dégénérescence produit $|\lambda| = 368$.
#
# ### Ce qui reste rigoureusement identique
#
# | objet | `on-free` | `on-stable` | identique ? |
# |---|---|---|---|
# | générateur | $G = \alpha I + \omega J$ | $G = \alpha I + \omega J$ | oui |
# | discrétisation | $\bar{A} = \exp(\Delta G)$ | $\bar{A} = \exp(\Delta G)$ | oui |
# | pulsation $\omega$ | tête linéaire, brute | tête linéaire, brute | oui |
# | intégrale ZOH $\Phi = \int_0^\Delta e^{sG}ds$ | forme close | **même** forme close | oui |
# | $\bar{B} = \Phi B$ | idem | idem | oui |
# | nombre de paramètres | $2$ par bloc : $(\alpha,\omega)$ | $2$ par bloc : $(\rho,\omega)$ | **oui** |
# | production de $\alpha$ | $\alpha$ = sortie brute | $\alpha = \log\sigma(\rho)/\Delta$ | **non — c'est la seule différence** |
#
# Le contrat d'équité du protocole (`§13.7` : *« the SAME 16 numbers in every arm »*) tient
# donc **sans amendement**. Le contraste `free` contre `bounded` est le plus propre qu'on
# puisse construire : une seule reparamétrisation, tout le reste apparié — budget, backbone,
# encodeur, perte, taux d'apprentissage, initialisation.

# %%
def logsigmoid(x):
    """log sigma(x) calcule de facon stable : -softplus(-x) = -log(1 + e^{-x}).

    np.logaddexp(0, -x) vaut log(1 + e^{-x}) sans jamais former e^{-x} explicitement.
    """
    return -np.logaddexp(0.0, -np.asarray(x, dtype=float))


def alpha_bounded(rho, delta):
    """Production bornee de alpha. C'est LA seule ligne qui change entre les deux bras."""
    return logsigmoid(rho) / delta


RHOS = np.array([-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0, 12.0])
DELTA_B = 0.5

print(f"{'rho':>6} | {'sigma(rho)':>12} | {'alpha=log s/D':>14} | {'r=e^{D alpha}':>14} | {'r^25':>12}")
print("-" * 70)
for rho in RHOS:
    s_ = 1.0 / (1.0 + np.exp(-rho))
    al_ = alpha_bounded(rho, DELTA_B)
    r_ = np.exp(DELTA_B * al_)
    print(f"{rho:>6.1f} | {s_:>12.6f} | {al_:>14.6f} | {r_:>14.6f} | {r_**25:>12.6g}")

print("\ndepot (valeurs de sigma verifiees, PLAN_V3 II.2) :")
print("  0.002473, 0.047426, 0.268941, 0.5, 0.731059, 0.952574, 0.997527, 0.999994")
print("reproduit ici                                       :")
print("  " + ", ".join(f"{1/(1+np.exp(-r)):.6f}" for r in RHOS))

# Le point qui compte : r reste dans (0,1) sur toute la plage, sans condition.
rho_wide = np.linspace(-700.0, 700.0, 200001)
r_wide = np.exp(DELTA_B * alpha_bounded(rho_wide, DELTA_B))
print(f"\nsur {rho_wide.size} valeurs de rho dans [-700, 700] :")
print(f"  r max                 = {r_wide.max():.16f}   (jamais > 1 ? {bool(np.all(r_wide <= 1.0))})")
print(f"  r min                 = {r_wide.min():.6e}   (jamais <= 0 ? {bool(np.all(r_wide > 0.0))})")
print(f"  r^25 max              = {np.max(r_wide**25):.16f}   [reproduit ici]")
print(f"  aucun NaN / inf       : {bool(np.all(np.isfinite(r_wide)))}")

# Nuance honnete : en arithmetique flottante, sigma(rho) ARRONDIT a 1.0 des que
# 1 - sigma(rho) = sigma(-rho) tombe sous eps/2. On mesure ou.
grid = np.linspace(30.0, 45.0, 15001)
first = grid[np.argmax(np.exp(DELTA_B * alpha_bounded(grid, DELTA_B)) >= 1.0)]
print(f"\n  float64 : r arrondit a 1.0 des rho ~ {first:.2f}")
print(f"  (estimation analytique -log(eps/2) = {-np.log(np.finfo(float).eps / 2):.2f})")
print("  L'inegalite r < 1 est STRICTE dans R ; en flottant elle degenere en r <= 1.")
print("  La degenerescence est benigne : r = 1 donne r^H = 1, soit AUCUNE amplification.")
print("  La borne ne peut jamais se degrader en divergence, seulement en neutralite.")

# %%
fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))

rho_plot = np.linspace(-10, 14, 600)

axes[0].plot(rho_plot, np.exp(DELTA_B * alpha_bounded(rho_plot, DELTA_B)), lw=1.8,
             color="tab:green", label=r"$r = \sigma(\rho)$  (bounded)")
axes[0].axhline(1.0, color="crimson", ls="--", lw=1.2, label=r"$r=1$")
axes[0].scatter(RHOS, [1 / (1 + np.exp(-r)) for r in RHOS], s=22, color="black", zorder=5)
axes[0].set_xlabel(r"paramètre brut $\rho$")
axes[0].set_ylabel("rayon $r$")
axes[0].set_ylim(-0.05, 1.6)
axes[0].set_title(r"(a) bounded : $r=\sigma(\rho)$ ne touche jamais 1")
axes[0].legend(loc="lower right", fontsize=8)

alpha_plot = np.linspace(-1.5, 0.8, 600)
axes[1].plot(alpha_plot, np.exp(DELTA_B * alpha_plot), lw=1.8, color="tab:orange",
             label=r"$r = e^{\Delta\alpha}$  (free)")
axes[1].axhline(1.0, color="crimson", ls="--", lw=1.2)
axes[1].fill_between(alpha_plot, 1.0, np.exp(DELTA_B * alpha_plot),
                     where=np.exp(DELTA_B * alpha_plot) > 1.0, color="crimson", alpha=0.15,
                     label="zone divergente accessible")
axes[1].set_xlabel(r"paramètre brut $\alpha$")
axes[1].set_ylabel("rayon $r$")
axes[1].set_title(r"(b) free : $\alpha>0$ est atteignable")
axes[1].legend(loc="upper left", fontsize=8)

r_axis = np.logspace(-1.2, 0.6, 500)
axes[2].loglog(r_axis, r_axis**25, lw=1.8, color="black", label=r"$r^{25}$")
axes[2].axvspan(1.0, r_axis.max(), color="crimson", alpha=0.13,
                label="atteignable par 'free' seulement")
axes[2].axvspan(r_axis.min(), 1.0, color="tab:green", alpha=0.10,
                label="atteignable par 'bounded'")
axes[2].axvline(1.0, color="crimson", ls="--", lw=1.2)
axes[2].set_ylim(1e-12, 1e10)
axes[2].set_xlabel("rayon $r$")
axes[2].set_ylabel(r"$r^{25}$")
axes[2].set_title("(c) ensembles atteignables")
axes[2].legend(loc="upper left", fontsize=7.5)

fig.suptitle(r"Figure 6 — une seule reparamétrisation sépare $r$ non borné de $r \in (0,1)$", y=1.04)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Lecture.** Panneau (a) : la courbe verte $r = \sigma(\rho)$ approche 1 par en dessous et
# ne l'atteint jamais ; les points noirs sont les valeurs tabulées $\rho \in \{-6,\dots,12\}$.
# Panneau (b) : avec $\alpha$ libre, la même quantité franchit $r=1$ dès $\alpha>0$ et la
# zone rouge — divergente — est un demi-espace entier du domaine des paramètres, atteignable
# par n'importe quelle initialisation ou n'importe quel pas de descente. Panneau (c) : la
# conséquence sur le rollout de 25 pas, lue comme un **ensemble atteignable**. La courbe
# $r \mapsto r^{25}$ est la même dans les deux bras ; ce qui change est la portion de
# l'abscisse que la paramétrisation autorise. `bounded` est confiné à la bande verte
# $r \in (0,1)$, donc $r^{25} \le 1$ ; `free` a en plus accès à la bande rouge, où $r^{25}$
# franchit dix décades pour un facteur 2.5 sur $r$. Noter que la bande verte descend aussi
# très bas : la borne supprime l'explosion, **pas** l'effacement de l'état. Les deux échecs
# ne sont pas symétriques et le protocole doit les mesurer séparément.

# %%
# --- Verification d'appariement : tout sauf la production de alpha est identique ---
def zoh_pair(alpha, omega, delta):
    """Renvoie (Abar, Phi) pour G = alpha I + omega J.

    Phi = int_0^delta exp(sG) ds = (Abar - I) G^{-1}, avec G^{-1} = (alpha I - omega J)/(alpha^2+omega^2).
    """
    G_ = alpha * np.eye(2) + omega * J2
    Abar_ = np.exp(delta * alpha) * rot(delta * omega)
    Ginv = (alpha * np.eye(2) - omega * J2) / (alpha**2 + omega**2)
    Phi_ = (Abar_ - np.eye(2)) @ Ginv
    return G_, Abar_, Phi_


OMEGA_T = 1.3
RHO_T = 0.7
alpha_from_rho = alpha_bounded(RHO_T, DELTA_B)

# Bras 'bounded' : alpha produit par la reparametrisation.
G_b, Abar_b, Phi_b = zoh_pair(alpha_from_rho, OMEGA_T, DELTA_B)
# Bras 'free' auquel on IMPOSE la meme valeur de alpha : le code aval doit etre identique.
G_f, Abar_f, Phi_f = zoh_pair(alpha_from_rho, OMEGA_T, DELTA_B)

print(f"alpha issu de rho={RHO_T} : {alpha_from_rho:.9f}  -> r = {np.exp(DELTA_B*alpha_from_rho):.9f}")
print(f"|Abar_bounded - Abar_free| = {np.max(np.abs(Abar_b - Abar_f)):.3e}")
print(f"|Phi_bounded  - Phi_free | = {np.max(np.abs(Phi_b - Phi_f)):.3e}")
print("-> le chemin aval (Abar, Phi, Bbar) est litteralement la MEME fonction.")

# Controle independant de Phi : quadrature fine de int_0^delta exp(sG) ds.
ss = np.linspace(0.0, DELTA_B, 20001)
integrand = np.array([expm_2x2(s_ * G_b) for s_ in ss])
Phi_quad = np.trapezoid(integrand, ss, axis=0)
print(f"\nPhi (forme close) =\n{Phi_b}")
print(f"Phi (quadrature)  =\n{Phi_quad}")
print(f"ecart max = {np.max(np.abs(Phi_b - Phi_quad)):.3e}   [reproduit ici]")
print("(c'est l'erreur de la quadrature trapezoidale, en O(h^2), pas une erreur de forme :")
print(" la forme close est exacte, c'est le controle numerique qui est approche.)")

print(f"\nnombre de parametres par bloc : free = 2 (alpha, omega) | bounded = 2 (rho, omega)")
print("omega n'est pas touche ; le compte de parametres est inchange ; le contrat")
print("d'equite du protocole (§13.7, 'the SAME 16 numbers in every arm') tient sans amendement.")

# %% [markdown]
# ## 3.5 La note numérique qui n'est pas un détail : `logsigmoid`, jamais `log(sigmoid(.))`
#
# On calcule $\log\sigma(\rho)$ par
#
# $$\log\sigma(\rho) = -\log\bigl(1 + e^{-\rho}\bigr) = -\operatorname{softplus}(-\rho),$$
#
# et **jamais** en composant `log(sigmoid(rho))`. La raison est un dépassement de capacité,
# et il se produit dans une région que la descente visite.
#
# **Le mécanisme.** En composant naïvement, on forme d'abord $\sigma(\rho) = 1/(1+e^{-\rho})$.
# Pour $\rho$ très négatif, $e^{-\rho}$ **déborde vers $+\infty$** ; l'inverse vaut alors
# exactement $0$ ; et $\log 0 = -\infty$. Le gradient qui remonte est `nan`, et il
# contamine tout le modèle en une mise à jour.
#
# **Où est le seuil, concrètement.** En `float64`, $e^{-\rho}$ déborde pour
# $\rho \lesssim -709$. En **`float32`** — la précision par défaut de l'entraînement — le
# débordement arrive dès $\rho \lesssim -89$. C'est une valeur que $\rho$ peut atteindre :
# $\rho = -89$ correspond à $r = \sigma(\rho) \approx 10^{-39}$, c'est-à-dire à un bloc
# complètement sur-contracté, exactement l'attracteur vers lequel la descente pousse quand
# elle cherche à annuler une erreur de rollout qu'elle n'arrive pas à réduire autrement.
# **La queue gauche n'est pas une région exotique : c'est là que le modèle va quand il
# renonce.**
#
# **Pourquoi la forme stable ne déborde pas.** $\log(1+e^{-\rho})$ se calcule par
# $\texttt{logaddexp}(0, -\rho) = \max(0,-\rho) + \log\bigl(1 + e^{-|\rho|}\bigr)$ : on ne
# forme jamais $e^{-\rho}$ pour $\rho$ négatif, seulement $e^{-|\rho|} \le 1$. Le résultat
# est asymptotiquement exact ($\log\sigma(\rho) \to \rho$ quand $\rho \to -\infty$) et
# n'excède jamais la plage des flottants.

# %%
# --- Demonstration du debordement, en float32 (la precision d'entrainement usuelle) ---
print(f"{'rho':>7} | {'log(sigmoid(rho)) naif':>24} | {'logsigmoid(rho) stable':>24}")
print("-" * 62)
with np.errstate(over="ignore", divide="ignore"):
    for rho in (-10.0, -50.0, -80.0, -88.0, -90.0, -120.0, -400.0):
        r32 = np.float32(rho)
        naive = np.log(np.float32(1.0) / (np.float32(1.0) + np.exp(-r32)))
        stable = -np.logaddexp(np.float32(0.0), -r32)
        print(f"{rho:>7.1f} | {float(naive):>24} | {float(stable):>24}")

print("\nEn float32, exp(-rho) deborde vers +inf des rho ~ -89 : sigma vaut alors")
print("exactement 0, log(0) = -inf, et le gradient remonte en nan.")
print("La forme stable rend exactement rho dans la queue gauche (log sigma(rho) -> rho).")
print()
print("Verification de l'equivalence dans la plage sure (float64) :")
rr = np.linspace(-30.0, 30.0, 60001)
naive64 = np.log(1.0 / (1.0 + np.exp(-rr)))
print(f"  ecart max |naif - stable| sur [-30, 30] = "
      f"{np.max(np.abs(naive64 - logsigmoid(rr))):.3e}   [reproduit ici]")
print("  Les deux formes calculent bien la meme fonction ; l'une survit a la queue, pas l'autre.")

# %% [markdown]
# ## 3.6 La déclaration d'honnêteté — pourquoi c'est un niveau de facteur et pas un correctif
#
# Cette sous-section n'est pas une précaution rhétorique. C'est la partie du raisonnement
# qui décide du **design expérimental**, et elle doit être écrite avant de voir un chiffre.
#
# **Le fait.** Borner le rayon **impose** la stabilité au lieu de la mesurer. Un bras dont
# $r \in (0,1)$ par construction ne peut plus servir à répondre à la question « la structure
# apporte-t-elle de la stabilité de rollout ? », puisque la réponse est cousue dans la
# paramétrisation. C'est exactement ce que `PROTOCOL.md §13.2` refusait délibérément :
# *« stable AND unstable modes stay representable and stability is measured, not imposed »*.
#
# **La tentation.** Face à un bras `on-free` qui diverge, la réaction naturelle est de dire
# « c'était un bug de paramétrisation, on le corrige, et on relit les résultats de `on` ».
# C'est doublement faux :
#
# 1. **scientifiquement** — on remplacerait une mesure par une autre en gardant l'étiquette,
#    donc on confondrait « structure » et « contraction ». Si le bras borné gagne, on ne
#    saurait pas si c'est parce qu'il est structuré ou parce qu'il contracte ;
# 2. **procéduralement** — le dépôt interdit de réécrire une mesure. L'axe $\mu_J$ antérieur
#    a échoué et est conservé tel quel en matériel historique, pour cette raison précise.
#
# **La décision.** Le rayon borné est un **troisième niveau du facteur `structure`** :
#
# $$\texttt{structure} \in \{\underbrace{\texttt{off}}_{\text{pas de structure}},\ \underbrace{\texttt{on-free}}_{\text{structure, rayon libre}},\ \underbrace{\texttt{on-stable}}_{\text{structure, rayon borné}}\}$$
#
# mesuré **à côté** de `on-free`, jamais appliqué **à** `on-free`. La question devient alors
# identifiable : *où, sur l'axe {pas de structure, structure à spectre libre, structure à
# spectre borné}, le gain vit-il ?* Écrit comme une réparation, c'est du bricolage ; écrit
# comme un niveau de facteur, c'est le résultat — et les deux issues sont publiables, ce
# qui est le test de qualité d'une prédiction pré-enregistrée.
#
# **Ce qui n'est pas revendiqué.** La borne n'est pas neuve : Koopman Dreamer utilise des
# rayons bornés, et une ancienne cellule du dépôt (`koopman_ssm(constrained=True)`) bornait
# déjà $r$ par $\alpha = -e^{\nu}$. Ce qui est neuf est le **contraste à trois niveaux avec
# tout le reste apparié**, et la **courbe** de gain contre taille d'échantillon et contre le
# défaut de représentabilité $\delta(c)$.
#
# *Note sur la paramétrisation écartée.* $\alpha = -e^{\nu}$ borne aussi le rayon
# ($r = e^{-\Delta e^\nu} \in (0,1)$) mais couvre mal la plage utile : $\alpha \to -\infty$
# quand $\nu$ croît, donc $r \to 0$ très vite, et l'essentiel de la masse de $\nu$ produit
# une sur-contraction. La forme $\sigma(\rho)$ est une bijection sur $(0,1)$ tout entier :
# à ce titre elle domine strictement, à coût de calcul identique.

# %% [markdown]
# ---
#
# # 4. Ce que le spike coûterait ici, et pourquoi on s'arrête au sous-seuil
#
# Tout ce qui précède est sous-seuil. Un neurone réel ne l'est pas : quand la tension
# franchit un seuil, il émet un **spike** et se réinitialise. Cette section démontre
# pourquoi ajouter ce mécanisme *ici, dans ce dossier, maintenant* coûterait exactement la
# propriété dont la thèse dépend. Ce n'est pas une omission : c'est un résultat de
# périmètre, et il se démontre.
#
# ## 4.1 Le modèle avec seuil et reset dur
#
# On complète le LIF de la section 1 :
#
# $$\begin{aligned}
# V_{\text{libre}}[k+1] &= a\,V[k] + (1-a)\bigl(V_{\text{rest}} + R I[k]\bigr) && \text{(intégration sous-seuil, inchangée)}\\
# s[k+1] &= \mathbf{1}\bigl[V_{\text{libre}}[k+1] \ge V_{\text{th}}\bigr] && \text{(détection de spike)}\\
# V[k+1] &= \bigl(1 - s[k+1]\bigr)\,V_{\text{libre}}[k+1] && \text{(reset dur à } 0)
# \end{aligned}$$
#
# **Conséquence structurelle.** Le système reste **linéaire par morceaux** : sur chaque
# intervalle entre deux spikes, la dynamique est exactement la récurrence affine de la
# section 1. Ce qui change, c'est que la carte $V[k] \mapsto V[k+1]$ est maintenant
# **discontinue** en $V_{\text{libre}} = V_{\text{th}}$ : elle vaut $V_{\text{libre}}$ juste
# en dessous et $0$ juste au-dessus. Une fonction discontinue n'est différentiable en aucun
# point de sa discontinuité, et le saut vaut $-V_{\text{th}}$ — il ne tend pas vers zéro.
#
# ## 4.2 L'argument décisif : le reset annule le chemin de gradient
#
# Calculons la jacobienne d'un pas. Presque partout (c'est-à-dire hors de l'ensemble de
# mesure nulle où $V_{\text{libre}} = V_{\text{th}}$ exactement), l'indicatrice $s$ est
# localement constante, donc $\partial s/\partial V = 0$, et :
#
# $$\frac{\partial V[k+1]}{\partial V[k]} = \bigl(1-s[k+1]\bigr)\,\frac{\partial V_{\text{libre}}[k+1]}{\partial V[k]} = \bigl(1-s[k+1]\bigr)\,a =
# \begin{cases} a & \text{si pas de spike au pas } k+1,\\[3pt] \mathbf{0} & \text{si spike au pas } k+1. \end{cases}$$
#
# **Le produit le long d'une trajectoire est donc**
#
# $$\frac{\partial V[T]}{\partial V[0]} = \prod_{k=1}^{T} \bigl(1 - s[k]\bigr)\,a = \begin{cases} a^{T} & \text{si aucun spike,}\\[3pt] 0 & \text{dès qu'il y a **un seul** spike.}\end{cases}$$
#
# Ce n'est pas une atténuation, c'est une **annulation**. Un seul spike suffit à couper
# complètement le chemin de gradient entre l'état initial et l'état final.
#
# ## 4.3 Pourquoi c'est fatal *pour cette thèse en particulier*
#
# Le chaînon central du dossier est le suivant : au forward simple l'encodeur domine le
# coût, mais en **planification** (de l'ordre de $3\times10^5$ appels au predictor par
# décision dans un CEM réaliste) le predictor devient le poste dominant — et **si ses
# gradients d'action sont bien conditionnés à travers le rollout, on planifie par descente
# de gradient au lieu d'échantillonner**, ce qui vaut un à deux ordres de grandeur. L'audit
# de contrôlabilité $\partial x_H/\partial u$ n'est donc pas un thermomètre posé à côté :
# c'est la **condition d'accès** au régime efficace.
#
# Or la quantité $\partial x_H / \partial u$ se factorise par la règle de la chaîne en un
# produit de jacobiennes d'état :
#
# $$\frac{\partial x_H}{\partial u_t} = \Bigl(\prod_{k=t+1}^{H-1}\frac{\partial x_{k+1}}{\partial x_k}\Bigr)\frac{\partial x_{t+1}}{\partial u_t}.$$
#
# Le reset dur met un facteur exactement nul dans ce produit à chaque spike. **Le spike coupe
# le gradient précisément là où la thèse a besoin qu'il soit bien conditionné.** L'argument
# n'est pas « c'est difficile », il est « la quantité que le dossier mesure devient
# identiquement nulle ».
#
# ## 4.4 Ce qu'il faudrait ajouter, et ce que ça coûterait
#
# La littérature SNN répond par le **surrogate gradient** : on garde l'indicatrice au
# forward et on la remplace, **au backward seulement**, par la dérivée d'une fonction lisse
# (sigmoïde raide, fenêtre triangulaire, arctan). Cela restaure un chemin de gradient non
# nul, mais au prix explicite d'un **estimateur biaisé** — le gradient calculé n'est le
# gradient d'aucune fonction que le forward implémente.
#
# Pour ce dossier, cela introduirait :
#
# 1. une **variable confondante supplémentaire** (la forme et la largeur du surrogate) dans
#    un projet qui en compte déjà quatre non contrôlées ;
# 2. une **ambiguïté d'attribution** irréductible : si le rollout se dégrade, est-ce la
#    dynamique ou le biais de l'estimateur ? Cette question n'a pas de réponse
#    expérimentale bon marché ;
# 3. une **incompatibilité avec l'instrument principal** : le conditionnement
#    $\kappa(\partial x_H/\partial u)$ n'a plus de sens comme audit si la jacobienne mesurée
#    n'est pas celle du modèle.
#
# ## 4.5 Le périmètre, énoncé positivement
#
# Le régime **sous-seuil** est ce qui sert la thèse aujourd'hui : c'est un opérateur
# linéaire, différentiable partout, dont on peut auditer le spectre et le conditionnement en
# forme close. Le spike deviendra pertinent le jour où **le gain énergétique sera l'objet
# mesuré** — question légitime, et elle a sa propre forme : *« une unité à dynamique native
# réduit-elle le coût par décision réussie en planification ? »*. Elle demande un banc
# énergétique, un corpus, et un protocole ; c'est un projet, pas une variante de bras.
#
# **Hodgkin-Huxley, et pourquoi c'est non.** On pourrait vouloir remonter plus haut en
# fidélité biophysique : le modèle de Hodgkin-Huxley décrit les courants ioniques par
# quatre EDO couplées et **raides** par neurone ($V$ et trois variables de porte $m,h,n$),
# avec des constantes de temps séparées de deux ordres de grandeur. Trois raisons de
# l'écarter, et aucune n'est un jugement de valeur sur le modèle :
#
# 1. **la raideur** impose des pas très petits ou un solveur implicite — il n'existe pas de
#    chemin BPTT qui passe à l'échelle sur des rollouts de dizaines de pas ;
# 2. **le coût** : quatre états et plusieurs exponentielles par neurone et par pas, contre
#    deux nombres pour le bloc $2\times2$, pour une classe d'opérateurs dont on ne sait pas
#    caractériser le spectre en forme close ;
# 3. **l'absence de lien** au world model prédictif : rien dans HH ne répond à la question
#    du dépôt, qui porte sur la structure de l'opérateur de transition latente.
#
# HH est de la neuroscience computationnelle et c'en est du bon ; l'inclure dans un dossier
# JEPA affaiblirait le dossier au lieu de le renforcer.

# %%
# --- Simulation : LIF avec seuil et reset dur ---
V_TH = 1.0          # seuil (unites adimensionnees, v_rest = 0)
DRIVE = 1.5         # entree constante : il FAUT que v_ss = DRIVE > V_TH, sinon zero spike
T_SPK = 200

assert DRIVE > V_TH, "sans v_ss > V_th le neurone n'emet aucun spike"

V_sp = np.zeros(T_SPK + 1)
spikes = np.zeros(T_SPK + 1)
jac_reset = np.zeros(T_SPK)      # d V[k+1] / d V[k]  AVEC reset
jac_free = np.full(T_SPK, A_ZOH)  # la meme quantite SANS reset (sous-seuil pur)

for k in range(T_SPK):
    V_free_k = A_ZOH * V_sp[k] + (1.0 - A_ZOH) * DRIVE
    s = 1.0 if V_free_k >= V_TH else 0.0
    V_sp[k + 1] = V_free_k * (1.0 - s)
    spikes[k + 1] = s
    jac_reset[k] = A_ZOH * (1.0 - s)

n_spikes = int(spikes.sum())
isi = np.diff(np.flatnonzero(spikes > 0))
print(f"nombre de spikes sur {T_SPK} pas          : {n_spikes}   [reproduit ici]")
assert n_spikes > 0, "le regime choisi ne produit aucun spike : la demonstration serait vide"
print(f"premier spike au pas                   : {int(np.argmax(spikes > 0))}")
print(f"intervalle inter-spike (pas)           : {np.unique(isi)}")
print(f"prediction analytique de l'ISI         : ceil(ln(DRIVE/(DRIVE-V_th)) / (Delta/tau))"
      f" = {int(np.ceil(np.log(DRIVE / (DRIVE - V_TH)) / (DELTA / TAU)))}")
print()
cum_reset = np.cumprod(jac_reset)
cum_free = np.cumprod(jac_free)
print(f"produit des jacobiennes SANS reset (a^T) : {cum_free[-1]:.6e}")
print(f"produit des jacobiennes AVEC reset       : {cum_reset[-1]:.6e}   <- exactement zero")
print(f"nombre de pas avant annulation           : {int(np.argmax(cum_reset == 0.0)) + 1}")

# %%
# --- Ce qu'un surrogate gradient produirait a la place, et pourquoi c'est un autre objet ---
BETA_SURR = 10.0


def surrogate_bump(x, beta=BETA_SURR):
    """Derivee de substitution 'fast sigmoid' : 1 / (1 + beta |x|)^2. Vaut 1 au seuil."""
    return 1.0 / (1.0 + beta * np.abs(x)) ** 2


jac_surr = np.zeros(T_SPK)
V_s2 = np.zeros(T_SPK + 1)
for k in range(T_SPK):
    V_free_k = A_ZOH * V_s2[k] + (1.0 - A_ZOH) * DRIVE
    s = 1.0 if V_free_k >= V_TH else 0.0
    V_s2[k + 1] = V_free_k * (1.0 - s)
    # d/dV[k] de (1-s) V_free, en remplacant H'(.) par le bump au backward SEULEMENT
    jac_surr[k] = A_ZOH * (1.0 - s) - V_free_k * surrogate_bump(V_free_k - V_TH) * A_ZOH

cum_surr = np.cumprod(jac_surr)
print(f"jacobienne au pas de spike, vraie (p.p.)     : {jac_reset[21]:.6f}")
print(f"jacobienne au pas de spike, surrogate        : {jac_surr[21]:.6f}")
print(f"produit cumule a T={T_SPK}, surrogate            : {cum_surr[-1]:.6e}")
print(f"produit cumule a T={T_SPK}, vrai                 : {cum_reset[-1]:.6e}")
print()
print("Le surrogate ne corrige pas le zero : il le REMPLACE par un nombre qui n'est")
print("la derivee d'aucune fonction implementee au forward. C'est un estimateur biaise,")
print("et son biais depend de beta -- un hyperparametre de plus, confondu avec la dynamique.")

# %%
fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6))

tt = np.arange(T_SPK + 1)
axes[0].plot(tt, V_sp, lw=1.2, color="black", label="$V(t)$ avec seuil + reset")
axes[0].axhline(V_TH, color="crimson", ls="--", lw=1.2, label=r"$V_{th}$")
axes[0].axhline(DRIVE, color="tab:blue", ls=":", lw=1.0, label=r"$V_\infty = RI$")
sp_t = np.flatnonzero(spikes > 0)
axes[0].scatter(sp_t, np.full(sp_t.size, V_TH), marker="|", s=180, color="crimson",
                label=f"{n_spikes} spikes")
axes[0].set_xlabel("pas $k$")
axes[0].set_ylabel("$V$")
axes[0].set_ylim(-0.05, 2.35)
axes[0].set_title("(a) le neurone emet reellement")
axes[0].legend(fontsize=7.5, loc="upper center", ncol=2)

FLOOR = 1e-30
axes[1].semilogy(np.arange(1, T_SPK + 1), np.maximum(cum_free, FLOOR), lw=1.6,
                 color="black", label=r"sans reset : $a^T$")
axes[1].semilogy(np.arange(1, T_SPK + 1), np.maximum(cum_reset, FLOOR), lw=1.6,
                 color="crimson", label="avec reset dur : 0")
for s_ in sp_t:
    axes[1].axvline(s_, color="0.85", lw=0.8, zorder=0)
axes[1].set_ylim(FLOOR / 10, 10)
axes[1].set_xlabel("horizon $T$")
axes[1].set_ylabel(r"$\left|\partial V[T]/\partial V[0]\right|$")
axes[1].set_title("(b) le premier spike coupe le chemin")
axes[1].legend(fontsize=8, loc="lower left")

axes[2].semilogy(np.arange(1, T_SPK + 1), np.maximum(np.abs(cum_surr), FLOOR), lw=1.6,
                 color="tab:purple", label="surrogate (biaise)")
axes[2].semilogy(np.arange(1, T_SPK + 1), np.maximum(cum_free, FLOOR), lw=1.2,
                 color="black", ls="--", label=r"sous-seuil : $a^T$")
axes[2].semilogy(np.arange(1, T_SPK + 1), np.maximum(cum_reset, FLOOR), lw=1.6,
                 color="crimson", label="vrai (p.p.) : 0")
axes[2].set_ylim(FLOOR / 10, 1e3)
axes[2].set_xlabel("horizon $T$")
axes[2].set_ylabel("produit des jacobiennes")
axes[2].set_title("(c) le surrogate remplace, il ne repare pas")
axes[2].legend(fontsize=8, loc="lower left")

fig.suptitle("Figure 7 — le reset dur annule le chemin de gradient a travers le rollout", y=1.04)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Lecture.** Panneau (a) : le régime choisi produit bien **9 spikes** sur 200 pas — la
# démonstration porte sur un neurone qui décharge réellement, pas sur un cas dégénéré ; on
# a vérifié par assertion que le nombre de spikes est non nul, et l'intervalle inter-spike
# mesuré (22 pas) coïncide avec la prédiction analytique
# $\lceil \ln\bigl(V_\infty/(V_\infty - V_{\text{th}})\bigr)/(\Delta/\tau)\rceil$. Panneau
# (b) : le produit des jacobiennes. Sans reset, il décroît régulièrement comme $a^T$ — un
# gradient qui s'atténue, donc encore exploitable sur des horizons de quelques dizaines de
# pas. Avec reset dur, il tombe à **exactement zéro** au pas 22, au premier spike, et n'en
# repart jamais : c'est le trait rouge qui plonge sous le plancher du graphe. La
# différence entre les deux courbes n'est pas quantitative — l'une est un petit nombre,
# l'autre est le nombre zéro. Panneau (c) : le surrogate gradient restaure une valeur non
# nulle (courbe violette), mais cette valeur n'est la dérivée d'aucune fonction que le
# forward calcule. Elle dépend de la largeur $\beta$ choisie ; changer $\beta$ change la
# courbe sans changer le modèle. C'est précisément la définition d'une variable confondante,
# et c'est pourquoi ce notebook s'arrête au sous-seuil.

# %% [markdown]
# ---
#
# # 5. Le pont, en une page
#
# ## 5.1 L'énoncé
#
# **La contrainte spectrale que la communauté world-model redécouvre en juillet 2026 est
# celle que la communauté neurone a prouvée stable en 2024-2025.**
#
# Les sections 1 à 3 en donnent la raison technique, et elle tient en trois lignes :
#
# 1. le bloc de transition latente $\bar{A} = e^{\Delta\alpha}R(\Delta\omega)$ **est** un
#    neurone adaptatif sous-seuil (§2.4, identité algébrique, résidu $\sim10^{-17}$) ;
# 2. son unique mode d'échec en rollout fermé est $r > 1$, avec amplification $r^H$
#    (§3.1-3.2, et le mécanisme reproduit en Figure 4) ;
# 3. la communauté des neurones à dynamique travaille cette question depuis 2024 sous le
#    nom de « frontière de divergence » et publie des paramétrisations testées (§3.3-3.4).
#
# Les deux communautés résolvent le même problème d'algèbre, sur le même opérateur, sans se
# citer. C'est ce pont que le dépôt revendique — et il ne coûte pas une ligne de code
# supplémentaire, seulement la reparamétrisation du §3.4.
#
# ## 5.2 Les quatre références, et ce que chacune établit exactement
#
# **[Koopman Dreamer: Spectrally Constrained Latent Dynamics for Stable World-Model
# Imagination](https://arxiv.org/abs/2607.19719)** — arXiv:2607.19719, 22 juillet 2026.
# Remplace la transition latente d'un Dreamer par des blocs 2-D rotation-scaling **à rayon
# borné** — la paramétrisation du §3.4, dans un world model **reconstructif**. Résultats
# rapportés : meilleur que DreamerV3 sur **8 tâches DMC proprioceptives sur 9**, et
# navigation UAV-LiDAR portée de **53.8 % à 73.8 %**. C'est la validation externe que la
# contrainte spectrale aide en imagination long-horizon.
#
# **[SE-adLIF — Advancing spatio-temporal processing through adaptation in spiking neural
# networks](https://arxiv.org/abs/2408.07517)** — arXiv:2408.07517, publié dans *Nature
# Communications* 2025. Discrétisation d'**Euler symplectique** du neurone adLIF, avec
# amortissement et fréquence **découplés**. C'est la source de la carte $M(\alpha,\beta,a)$
# du §3.3 — et de son analyse, y compris la condition de régime que la §3.3 mesure.
#
# **[Balanced Resonate-and-Fire Neurons](https://arxiv.org/abs/2402.14603)** —
# arXiv:2402.14603, ICML 2024. Optimisation **consciente de la frontière de divergence**
# pour un neurone resonate-and-fire, avec convergence BPTT stable sur des centaines de pas.
# C'est la formulation la plus proche du problème du §3.1 : le rayon comme objet à gérer
# explicitement pendant l'entraînement, et non comme conséquence subie.
#
# **[Oscillatory State-Space Models (LinOSS)](https://arxiv.org/abs/2410.03943)** —
# arXiv:2410.03943, ICLR 2025 (oral). Établit l'universalité des SSM **oscillatoires**,
# c'est-à-dire de la classe d'opérateurs du §2.4. C'est la référence qui empêche de lire la
# contrainte comme une perte d'expressivité : la classe reste dense dans les opérateurs
# causaux réguliers.
#
# *(Références de contexte citées en passant dans le notebook, sans chiffre repris : S4D
# — Gu, Gupta, Goel, Ré — pour la forme diagonale ; Mamba — Gu & Dao — pour la sélectivité ;
# Izhikevich pour le neurone resonate-and-fire.)*

# ## 5.3 Ce qui reste ouvert — et que le dépôt va mesurer
#
# Les quatre références ci-dessus laissent trois trous, vérifiés :
#
# **1. Personne n'a comparé rayon libre contre rayon borné à encodeur, décodeur et backbone
# appariés.** L'ablation de Koopman Dreamer (§VI-D) ne teste que leurs propres composants ;
# elle ne dispose pas d'un bras structuré **non contraint** évalué dans le même harnais.
# Le facteur à trois niveaux du §3.6 est exactement ce contraste, et le §3.4 démontre qu'il
# est réalisable à **une seule ligne de différence**.
#
# **2. Personne ne publie la courbe de gain de structure contre taille d'échantillon.** Un
# biais inductif achète de l'**efficacité d'échantillonnage** ; il n'a rien à acheter dans
# un régime sur-déterminé. Le dépôt mesure ce point directement : sur ses oscillateurs
# couplés, $n = 256$ épisodes valent 24 480 transitions pour un système LTI à 4 états, et
# les moindres carrés y résolvent le système à **2.11e-08** (mesuré, §20.1). Un prior n'a
# **rien** à apporter dans ce régime : c'est le pire cas possible pour démontrer un biais
# inductif, pas le meilleur. La courbe contre $n$ est l'axe manquant.
#
# **3. Personne n'a fait ce pont dans un world model prédictif-latent (JEPA).** Koopman
# Dreamer est **reconstructif** : sa perte contient un décodeur, donc la contrainte
# spectrale y est évaluée contre un objectif qui pénalise la perte d'information de pixel.
# Dans un JEPA, la perte vit **entièrement dans l'espace latent** et l'effondrement de
# représentation est un mode d'échec de premier ordre. Rien ne garantit a priori qu'une
# contrainte de contraction, bénéfique quand un décodeur force l'état à rester informatif,
# le reste quand rien ne l'y force. **C'est une question ouverte, et son signe n'est pas
# prédit** — les deux issues sont le résultat.
#
# ### Les deux prédictions à enregistrer avant de voir un chiffre
#
# | contraste | prédiction | ce qui la tue |
# |---|---|---|
# | gain(structure) × $\log n$ | le gain de `on-stable` sur `off` **décroît** quand $n$ croît | aucune interaction détectable à 5 graines : publié tel quel |
# | `on-free` contre `on-stable` | **non prédit** — c'est la question ouverte | rien : les deux issues sont le résultat |
#
# Et la barre de robustesse, pré-enregistrée elle aussi : **si les gains disparaissent sous
# appariement strict, la revendication est retirée** ; on ne conclut pas « effet de
# capacité ».

# %% [markdown]
# ---
#
# # Récapitulatif des vérifications
#
# Ce notebook a produit deux catégories de chiffres. La table ci-dessous les sépare
# explicitement ; la cellule de code qui suit **recalcule** la colonne « reproduit ici » à
# chaque exécution, de sorte qu'un écart devient visible plutôt que silencieux.
#
# | # | énoncé | statut |
# |---|---|---|
# | 1 | LIF sous-seuil ZOH $=$ récurrence diagonale, contre RK4 fin | dépôt : **5.83e-16** · reproduit ici |
# | 2 | Euler explicite $a = 1-\Delta/\tau$, écart relatif sur $a$ | dépôt : **1.29e-03** · reproduit ici |
# | 3 | adLIF $\to$ bloc rotation-scaling, résidu de conjugaison | reproduit ici |
# | 4 | amplification $r^{H}$, $H=25$ | dépôt : **3.4 / 95 / 2.5e4** · reproduit ici |
# | 5 | norme de gradient au premier pas, `ssm` contre `mlp` | dépôt (mesuré) : **2.4e18** contre **28.2** |
# | 6 | perte époque 1, clippée contre non clippée | dépôt (mesuré) : **6.4413e24** contre **6.4415e24** |
# | 7 | carte symplectique : $\det M = \alpha\beta$ indépendamment de $a$ | reproduit ici |
# | 8 | $\lvert\lambda\rvert$ hors régime oscillatoire, 200 000 tirages | dépôt : **368** · reproduit ici · sup théorique $e^6$ |
# | 9 | $\alpha = \log\sigma(\rho)/\Delta \Rightarrow r = \sigma(\rho) \in (0,1)$ | reproduit ici, sans condition |
# | 10 | reset dur : $\partial V[T]/\partial V[0] = 0$ dès un spike | reproduit ici |

# %%
print("=" * 74)
print("RECAPITULATIF - colonne 'reproduit ici', recalculee a l'instant")
print("=" * 74)
rows = [
    ("1  LIF ZOH vs RK4 fin (400 pas, 100 sous-pas)", f"{err_abs:.3e}", "depot 5.83e-16"),
    ("2  Euler : ecart relatif sur a", f"{abs(A_EULER - A_ZOH) / A_ZOH:.3e}", "depot 1.29e-03"),
    ("2b Euler : erreur trajectoire RMS (30 tirages)", f"{rms_pool.mean():.3e}", "ordre 4e-03"),
    ("3  residu de conjugaison P^-1 G P", f"{np.max(np.abs(C - C_target)):.3e}", "identite exacte"),
    ("4  r^25 a r=1.05 / 1.20 / 1.50",
     f"{1.05**25:.1f}/{1.2**25:.0f}/{1.5**25:.2e}", "depot 3.4/95/2.5e4"),
    ("5  rapport des normes de gradient (jouet)", f"{g1_div / g1_sta:.2e}", "depot 2.4e18 vs 28.2"),
    ("6  rapport perte clip / sans clip (jouet)", f"{L_clip / L_free:.4f}", "depot 0.99997"),
    ("7  max |det M - alpha*beta| sur grille", f"{worst:.3e}", "exact, arrondi seul"),
    ("8  rayon max, 200 000 tirages (5 replicats)",
     f"{maxima.min():.0f}-{maxima.max():.0f}", "depot 368, sup e^6=403"),
    ("9  max de r=sigma(rho) sur rho dans [-700,700]", f"{r_wide.max():.6f}", "borne par 1"),
    ("10 produit des jacobiennes avec reset dur", f"{cum_reset[-1]:.1e}", "exactement 0"),
]
for label, got, ref in rows:
    print(f"{label:<48} {got:>14}   ({ref})")
print("=" * 74)
print("Aucun chiffre de ce tableau n'est ecrit en dur : tous sont des variables")
print("calculees par les cellules ci-dessus. Les valeurs 'depot' sont, elles, des")
print("constantes citees avec leur source dans le texte.")

# %% [markdown]
# ---
#
# # Ce qu'il faut retenir
#
# 1. **Un LIF sous-seuil est exactement un SSM diagonal de dimension 1 à coefficient fixe.**
#    Vérifié contre RK4 fin à quelques ulp. Ce n'est pas une approximation : $a = e^{-\Delta/\tau}$
#    est la solution. La discrétisation d'Euler explicite qu'emploie la majorité des
#    bibliothèques SNN en est le développement au premier ordre, et son domaine de validité
#    disparaît dès qu'on apprend la constante de temps.
#
# 2. **Un neurone adaptatif sous-seuil est exactement le bloc $2\times2$ rotation-scaling
#    du dépôt.** La conjugaison est explicite ($P = [q\,|\,p]$ construite à partir du vecteur
#    propre complexe) et le résidu est de l'ordre de $10^{-17}$. Identité, pas analogie —
#    mais identité **sans capacité supplémentaire** : le transfert porte sur la
#    discrétisation, pas sur l'expressivité.
#
# 3. **La discrétisation décide du rayon, et le rayon décide de tout le reste.** Avec
#    $\alpha$ libre, $r = e^{\Delta\alpha}$ est non borné et l'amplification en rollout fermé
#    vaut $r^H$ : c'est la signature exacte de la mort mesurée de la famille `ssm` du dépôt,
#    et l'invariance de la perte au clip de gradient est ce qui distingue une carte avant
#    divergente d'un pas d'optimiseur trop grand.
#
# 4. **La solution symplectique ne tient pas sa promesse hors du régime oscillatoire.**
#    $\det M = \alpha\beta$ est exact, mais $|\lambda| = \sqrt{\alpha\beta}$ ne vaut que sous
#    $\operatorname{tr}^2 < 4\det$ ; au-delà on mesure des rayons de l'ordre de 368, et le
#    supremum théorique est $e^6 \approx 403$. Une garantie conditionnelle n'est pas le bon
#    objet pour un axe expérimental.
#
# 5. **La reparamétrisation $\alpha = \log\sigma(\rho)/\Delta$ donne $r = \sigma(\rho) \in (0,1)$
#    inconditionnellement**, en changeant une seule ligne : $\omega$, $\Phi$, $\bar{B}$ et le
#    nombre de paramètres sont inchangés. C'est le contraste le plus propre possible entre
#    rayon libre et rayon borné.
#
# 6. **Borner le rayon impose la stabilité au lieu de la mesurer** — c'est pourquoi c'est un
#    troisième niveau de facteur, mesuré à côté du libre, et jamais un correctif appliqué au
#    libre.
#
# 7. **Le spike est hors périmètre pour une raison démontrée** : le reset dur annule
#    $\partial V[k+1]/\partial V[k]$, donc coupe le chemin de gradient à travers le rollout,
#    exactement la quantité dont dépend la thèse. Le surrogate le remplace par un estimateur
#    biaisé, donc par une variable confondante. Le spike redeviendra pertinent quand le gain
#    énergétique sera l'objet mesuré.
#
# ---
#
# **Suite.** La version publique anglaise des sections 1 à 4 est dans
# `docs/neuron_as_operator.md`.
