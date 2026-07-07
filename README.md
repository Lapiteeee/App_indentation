# Identification de Loi d'Écrouissage de Hollomon par Indentation

Interface graphique Flet pour identifier les paramètres de la loi d'écrouissage de Hollomon
(σ_y, n) à partir de courbes d'indentation force-déplacement (F-h).

## Principe

```
Courbe F-h expérimentale
        ↓
Optimisation par coïncidence de courbes (Nelder-Mead)
sur base de données éléments finis
        ↓
Paramètres matériau : σ_y (MPa), n (coefficient d'écrouissage)
        ↓
Loi de Hollomon : σ = σ_y^(1-n) · E^n · ε^n
```

La méthode Fmax aligne les courbes au niveau de la force maximale expérimentale,
puis affine l'identification sur des portions décroissantes de la courbe (de 100 % à 5 %).
La déformation représentative est extraite par analyse de la matrice Hessienne du résidu.

## Architecture

```
claude/
├── app.py              Point d'entrée — lance l'interface Flet
├── config.py           Chemins vers les fichiers de maillage
├── requirements.txt    Dépendances Python
├── core/
│   ├── maths.py        Fonctions pures (Hollomon, dérivées, intégrale)
│   ├── geometrie.py    Maillage EF (quadrangles, interpolation de Hencky)
│   ├── io_donnees.py   Lecture courbe F-h, maillage, base de données
│   └── fmax.py         Moteur d'optimisation Fmax (physique pure, sans GUI)
└── ui/
    └── interface.py    Interface Flet (graphes live + résultats)
```

**Règle d'or** : `core/` ne contient aucun appel à `matplotlib`, `print`, ni I/O
d'affichage. Le moteur retourne des données brutes ; l'interface dessine.

## Installation

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate.bat       # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Lancement

### Terminal
```bash
python app.py
```

### VS Code

1. Ouvrir le dossier du projet : **Fichier → Ouvrir le dossier…**
2. Sélectionner l'interpréteur Python : `Cmd+Shift+P` → `Python: Select Interpreter` → choisir le `venv` créé ci-dessus (`./venv/bin/python`)
3. Ouvrir `app.py`
4. Lancer avec `F5` ou le bouton ▷ en haut à droite

Pour configurer le lancement en un clic, créer `.vscode/launch.json` :
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Indentation",
            "type": "debugpy",
            "request": "launch",
            "program": "app.py",
            "console": "integratedTerminal"
        }
    ]
}
```

L'interface s'ouvre automatiquement.

## Utilisation

1. **Charger la BDD** — fichier Python (`.py`) exposant `base_donnee_modifiee_3`
2. **Charger la courbe F-h** — fichier `.dat` au format `h;F` (voir §Format ci-dessous)
3. **Régler les paramètres** :
   - Module d'Young E (par défaut 210 000 MPa pour l'acier)
   - Valeurs initiales σ_y et n (guess de départ pour Nelder-Mead)
   - Rayon expérimental a (µm) — optionnel, pour calcul d'écart
4. **Cliquer « Lancer l'optimisation »**
   - Le graphe de gauche montre la convergence en temps réel (résidu + trajectoire)
   - À la fin, le graphe de droite affiche la loi de Hollomon identifiée
5. **Exporter Excel** — bouton disponible après convergence (colonnes : ε, σ, F_max, σ_y, n, résidu)

Le bouton **Stopper** interrompt proprement le calcul à tout moment.

## Format des fichiers

### Courbe F-h (`.dat`)
```
h;F
1.00E-02;1.23E-14
2.00E-02;4.56E-13
...
```
Séparateur `;`, notation scientifique acceptée. Seule la partie montante (jusqu'au maximum de F) est utilisée.

### Base de données (`BDD_xxx.py`)
```python
base_donnee_modifiee_3 = {
    0.0: {          # coefficient d'écrouissage n
        525.0: {    # limite élastique σ_y (MPa)
            'h':     [...],  # profondeurs d'indentation (µm)
            'Force': [...],  # forces correspondantes (N)
            'a':     [...],  # rayons d'empreinte (µm)
        },
        ...
    },
    ...
}
```

### Fichiers de maillage (à la racine, fournis)
- `connectiviteNoeud.dat` — `id  n1  n2  n3  n4` (90 éléments quadrangles)
- `coordonnee.dat` — `id  x  y` avec x = σ_y/E, y = n (110 nœuds)

## Cas de test

Le fichier `F-h_sphere_rig_n0.2_sigma132.3.dat` est fourni comme cas de validation :

| Paramètre | Vérité terrain | Résultat attendu |
|-----------|---------------|-----------------|
| n         | 0.200         | ≈ 0.20          |
| σ_y       | 132.3 MPa     | ≈ 132 MPa       |

Utiliser : E = 210 000 MPa, σ_y guess = 100 MPa, n guess = 0.1

## Feuille de route

| Méthode       | État      | Description |
|---------------|-----------|-------------|
| **Fmax**      | ✅ Livré  | Alignement au maximum de force |
| Frelatif      | 🔜 À venir | Pivot à une fraction relative de F_max |
| Libre         | 🔜 À venir | Décalage et paramètres libres |
| Zero          | 🔜 À venir | Ancrage au point d'origine |
| Faible        | 🔜 À venir | Pivot en zone de faible chargement |

Chaque nouveau moteur respectera la même signature :
```python
def executer_optimisation_xxx(E, sig_y_guess, n_guess, chemin_courbe,
                               bdd_module, portions, chemins_maillage,
                               callback=None, stop_flag=None)
    -> tuple[pd.DataFrame, dict]
```
