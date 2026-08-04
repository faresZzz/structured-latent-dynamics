"""Regle des deux checks (REGLES.md sec.7.1) pour la demotion mu_J/Fig.2
(jalon B2). Ne verifie pas le CONTENU des objets, seulement que chaque
fichier demote porte sa bannere de statut -- le plan exige explicitement
qu'aucune presentation de mu_J ou de Fig. 2 comme gate primaire ne passe
inapercue.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MARKER = "SECONDARY -- not primary thesis"

# fichiers demotes jalon B2 -- liste exacte, pas de decouverte automatique
# (une liste + une boucle suffit, REGLES: pas de framework/fixture generique)
DEMOTED_FILES = [
    REPO_ROOT / "src/sld/objectives/jac_reg.py",
    REPO_ROOT / "src/sld/objectives/spectral_reg.py",
    REPO_ROOT / "src/sld/viz/fig2_spectrum.py",
    REPO_ROOT / "src/sld/audit/spectral.py",
    REPO_ROOT / "src/sld/audit/jacobian.py",
    REPO_ROOT / "src/sld/training/trainer.py",
    REPO_ROOT / "reports/spike_jacreg.md",
    REPO_ROOT / "scripts/spike_jacreg.py",
]


def test_every_demoted_file_carries_its_banner():
    """Check positif : le marqueur est present dans chaque fichier demote."""
    missing = [str(p) for p in DEMOTED_FILES if MARKER not in p.read_text(encoding="utf-8")]
    assert not missing, f"missing demotion banner in: {missing}"


def test_banner_disappearing_from_one_file_is_caught():
    """Check negatif : si la bannere disparait d'UN fichier, le test doit
    echouer -- pas passer silencieusement (un test de bannere qui passerait
    sur un fichier sans bannere est un test mort)."""
    target = DEMOTED_FILES[0]
    original = target.read_text(encoding="utf-8")
    assert MARKER in original  # precondition: la bannere est bien la avant l'alteration
    stripped = original.replace(MARKER, "nothing to see here")
    assert MARKER not in stripped

    missing = [str(p) for p in DEMOTED_FILES if MARKER not in (stripped if p == target else p.read_text(encoding="utf-8"))]
    assert missing == [str(target)]
