import importlib.util
import numpy as np


def lire_courbe_fh(chemin: str):
    h_exp, F_exp = [], []
    with open(chemin, 'r') as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            parts = ligne.split(';')
            if len(parts) != 2:
                continue
            try:
                h_exp.append(float(parts[0]))
                F_exp.append(float(parts[1]))
            except ValueError:
                pass

    h = np.array(h_exp)
    F = np.array(F_exp)

    # Portage fidèle du thésard :
    # 1. extraire_avant_extremum : EXCLUT le point max ([:max_index])
    max_index = max(enumerate(F), key=lambda item: item[1])[0]
    h_avant = h[:max_index]
    F_avant = F[:max_index]

    # 2. ordonner_par_x est appelé mais le résultat est écrasé — données NON triées
    # (comportement exact du code brut lignes 145-147)
    return h_avant, F_avant


def lire_maillage(chemins: dict, E: float):
    element_dict = {}
    with open(chemins["connectivite"], 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = [int(v) for v in line.split()]
            number = values[0]
            nodes = values[1:]
            element_dict[number] = {'number': number, 'nodes': nodes}

    nodes_dict = {}
    with open(chemins["coordonnee"], 'r') as f:
        for line in f:
            values = line.split()
            if len(values) < 3:
                continue
            nodes_dict[int(values[0])] = [float(values[1]) * E, float(values[2])]

    return element_dict, nodes_dict


def charger_bdd_module(chemin_bdd: str):
    spec = importlib.util.spec_from_file_location("bdd_user", chemin_bdd)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def construire_bdd_dimensionnelle(bdd_module, E: float) -> dict:
    base_donnee_modifiee_3 = getattr(bdd_module, 'base_donnee_modifiee_3')
    bdd = {}
    for n_coef in sorted(base_donnee_modifiee_3.keys()):
        bdd[n_coef] = {}
        for sig_lim in sorted(base_donnee_modifiee_3[n_coef].keys()):
            bdd[n_coef][sig_lim] = {
                'h':     np.array(base_donnee_modifiee_3[n_coef][sig_lim]['h']),
                'Force': np.array(base_donnee_modifiee_3[n_coef][sig_lim]['Force']),
                'a':     np.array(base_donnee_modifiee_3[n_coef][sig_lim]['a']),
            }
    return bdd
