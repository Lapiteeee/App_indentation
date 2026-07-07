from pathlib import Path

RACINE = Path(__file__).parent

CONNECTIVITE = RACINE / "connectiviteNoeud.dat"
COORDONNEE   = RACINE / "coordonnee.dat"

CHEMINS_MAILLAGE = {
    "connectivite": str(CONNECTIVITE),
    "coordonnee":   str(COORDONNEE),
}
