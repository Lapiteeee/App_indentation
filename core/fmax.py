import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from lmfit import Parameters, minimize

from .maths import second_derivative, calculate_K, calculate_sigma_y, loi_hollomon, calculer_integrale
from .geometrie import numero_element, interpolation_Hencky
from .io_donnees import lire_courbe_fh, lire_maillage, construire_bdd_dimensionnelle


class ArretUtilisateur(Exception):
    pass

_ArretUtilisateur = ArretUtilisateur


def executer_optimisation_fmax(
    E, sig_y_guess, n_guess,
    chemin_courbe, bdd_module, portions,
    chemins_maillage,
    h_max_override=None,   # µm — si fourni, remplace la recherche automatique
    callback=None,
    stop_flag=None,
    log_callback=None,
):
    def _log(msg):
        if log_callback is not None:
            log_callback(msg)

    _log("Lecture courbe F-h...")
    h_exp, F_exp = lire_courbe_fh(chemin_courbe)
    _log(f"  {len(h_exp)} points chargés.")
    _log("Lecture maillage...")
    element_dict, nodes_dict = lire_maillage(chemins_maillage, E)
    _log("Construction BDD dimensionnelle...")
    base_donnee_dimensionelle = construire_bdd_dimensionnelle(bdd_module, E)
    _log("  BDD prête.")

    # État partagé entre les closures (remplace les globals du code brut)
    decalage_list = []
    rayon_list = []

    # -------------------------------------------------------------------------
    # Phase 1 — rosenbrock global (100 % de F_exp_max)
    # -------------------------------------------------------------------------
    def rosenbrock(params):
        x = params['x'].value
        y = params['y'].value
        point = (x, y)

        Flag_element, num_element = numero_element(point, element_dict, nodes_dict)
        noeud_1 = element_dict[num_element]['nodes'][0]
        noeud_2 = element_dict[num_element]['nodes'][1]
        noeud_3 = element_dict[num_element]['nodes'][2]
        noeud_4 = element_dict[num_element]['nodes'][3]

        x1 = nodes_dict[noeud_1][0]; y1 = nodes_dict[noeud_1][1]
        x2 = nodes_dict[noeud_2][0]; y2 = nodes_dict[noeud_2][1]
        x3 = nodes_dict[noeud_3][0]; y3 = nodes_dict[noeud_3][1]
        x4 = nodes_dict[noeud_4][0]; y4 = nodes_dict[noeud_4][1]

        h1 = base_donnee_dimensionelle[y1][x1]['h']
        F1 = base_donnee_dimensionelle[y1][x1]['Force']
        a1 = base_donnee_dimensionelle[y1][x1]['a']
        h2 = base_donnee_dimensionelle[y2][x2]['h']
        F2 = base_donnee_dimensionelle[y2][x2]['Force']
        a2 = base_donnee_dimensionelle[y2][x2]['a']
        h3 = base_donnee_dimensionelle[y3][x3]['h']
        F3 = base_donnee_dimensionelle[y3][x3]['Force']
        a3 = base_donnee_dimensionelle[y3][x3]['a']
        h4 = base_donnee_dimensionelle[y4][x4]['h']
        F4 = base_donnee_dimensionelle[y4][x4]['Force']
        a4 = base_donnee_dimensionelle[y4][x4]['a']

        F1_inter = interp1d(h1, F1, kind='linear', fill_value="extrapolate")
        F2_inter = interp1d(h2, F2, kind='linear', fill_value="extrapolate")
        F3_inter = interp1d(h3, F3, kind='linear', fill_value="extrapolate")
        F4_inter = interp1d(h4, F4, kind='linear', fill_value="extrapolate")
        a1_inter = interp1d(h1, a1, kind='linear', fill_value="extrapolate")
        a2_inter = interp1d(h2, a2, kind='linear', fill_value="extrapolate")
        a3_inter = interp1d(h3, a3, kind='linear', fill_value="extrapolate")
        a4_inter = interp1d(h4, a4, kind='linear', fill_value="extrapolate")

        delta_h = np.min([np.max(h1), np.max(h2), np.max(h3), np.max(h4)])
        h_num_tempo = np.linspace(0, delta_h, 100)
        F_num_tempo = []
        for i in range(len(h_num_tempo)):
            z1 = F1_inter(h_num_tempo[i])
            z2 = F2_inter(h_num_tempo[i])
            z3 = F3_inter(h_num_tempo[i])
            z4 = F4_inter(h_num_tempo[i])
            F_num_tempo.append(interpolation_Hencky(point, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4))
        F_num_tempo = np.asarray(F_num_tempo)

        F_exp_max = np.max(F_exp)
        indiceF_scale = np.where(F_exp <= F_exp_max)[0][-1]
        F_exp_scaled = F_exp[:indiceF_scale + 1]
        h_exp_scaled = h_exp[:indiceF_scale + 1]

        if h_max_override is not None:
            h_max = float(h_max_override)
        else:
            h_max = delta_h
            crossings = np.where((F_num_tempo[:-1] <= F_exp_max) & (F_num_tempo[1:] >= F_exp_max))[0]
            if len(crossings) > 0:
                i = crossings[0]
                h_max = float(np.interp(F_exp_max,
                                        [F_num_tempo[i], F_num_tempo[i + 1]],
                                        [h_num_tempo[i], h_num_tempo[i + 1]]))

        decalage = -h_exp_scaled[-1] + h_max
        h_exp_decale = h_exp_scaled + decalage
        decalage_list.append(decalage)

        F_num = []
        for i in range(len(h_exp_decale)):
            if h_exp_decale[i] < 0.0:
                F_num.append(0.0)
            else:
                z1 = F1_inter(h_exp_decale[i])
                z2 = F2_inter(h_exp_decale[i])
                z3 = F3_inter(h_exp_decale[i])
                z4 = F4_inter(h_exp_decale[i])
                F_num.append(interpolation_Hencky(point, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4))

        rayon_test_1 = a1_inter(h_exp_decale[-1])
        rayon_test_2 = a2_inter(h_exp_decale[-1])
        rayon_test_3 = a3_inter(h_exp_decale[-1])
        rayon_test_4 = a4_inter(h_exp_decale[-1])
        rayon_num = interpolation_Hencky(
            point,
            x1, y1, rayon_test_1,
            x2, y2, rayon_test_2,
            x3, y3, rayon_test_3,
            x4, y4, rayon_test_4,
        )
        rayon_list.append(rayon_num)

        return calculer_integrale(h_exp_decale, F_exp_scaled, np.asarray(F_num))

    def iter_cb_phase1(params, iteration, resid):
        if stop_flag is not None and stop_flag():
            raise _ArretUtilisateur()
        if callback is not None:
            callback(params, iteration, resid)

    # -------------------------------------------------------------------------
    # Phase 1 — optimisation globale (100 % F_max), départ unique = guess utilisateur
    # -------------------------------------------------------------------------
    _log("─── Phase 1 : optimisation globale (100% F_max) ───")
    _log(f"  · départ : σ_y₀ = {sig_y_guess:.1f}  n₀ = {n_guess:.4f}")

    params = Parameters()
    params.add('x', value=sig_y_guess, vary=True, min=33.6, max=3360.0)
    params.add('y', value=n_guess, vary=True, min=0.0, max=0.45)
    try:
        result = minimize(
            rosenbrock, params,
            method='Nelder-Mead',
            iter_cb=iter_cb_phase1,
            options={'xatol': 1e-10, 'fatol': 1e-10},
        )
    except _ArretUtilisateur:
        raise

    decalage_list.clear()
    rosenbrock(result.params)  # recalcule le décalage pour le résultat retenu

    sigma_y_optimale = result.params['x'].value
    n_optimale = result.params['y'].value
    decalage_global = decalage_list[-1]
    _log(f"  → retenu : σ_y = {sigma_y_optimale:.3f} MPa  |  n = {n_optimale:.4f}  |  décalage = {decalage_global:.4f}")

    # -------------------------------------------------------------------------
    # Phase 2 — rosenbrock_1 par portion (décalage global figé)
    # -------------------------------------------------------------------------
    def rosenbrock_1(params, portion):
        x = params['x'].value
        y = params['y'].value
        point = (x, y)

        Flag_element, num_element = numero_element(point, element_dict, nodes_dict)
        noeud_1 = element_dict[num_element]['nodes'][0]
        noeud_2 = element_dict[num_element]['nodes'][1]
        noeud_3 = element_dict[num_element]['nodes'][2]
        noeud_4 = element_dict[num_element]['nodes'][3]

        x1 = nodes_dict[noeud_1][0]; y1 = nodes_dict[noeud_1][1]
        x2 = nodes_dict[noeud_2][0]; y2 = nodes_dict[noeud_2][1]
        x3 = nodes_dict[noeud_3][0]; y3 = nodes_dict[noeud_3][1]
        x4 = nodes_dict[noeud_4][0]; y4 = nodes_dict[noeud_4][1]

        h1 = base_donnee_dimensionelle[y1][x1]['h']
        F1 = base_donnee_dimensionelle[y1][x1]['Force']
        a1 = base_donnee_dimensionelle[y1][x1]['a']
        h2 = base_donnee_dimensionelle[y2][x2]['h']
        F2 = base_donnee_dimensionelle[y2][x2]['Force']
        a2 = base_donnee_dimensionelle[y2][x2]['a']
        h3 = base_donnee_dimensionelle[y3][x3]['h']
        F3 = base_donnee_dimensionelle[y3][x3]['Force']
        a3 = base_donnee_dimensionelle[y3][x3]['a']
        h4 = base_donnee_dimensionelle[y4][x4]['h']
        F4 = base_donnee_dimensionelle[y4][x4]['Force']
        a4 = base_donnee_dimensionelle[y4][x4]['a']

        F1_inter = interp1d(h1, F1, kind='linear', fill_value="extrapolate")
        F2_inter = interp1d(h2, F2, kind='linear', fill_value="extrapolate")
        F3_inter = interp1d(h3, F3, kind='linear', fill_value="extrapolate")
        F4_inter = interp1d(h4, F4, kind='linear', fill_value="extrapolate")

        delta_h = np.min([np.max(h1), np.max(h2), np.max(h3), np.max(h4)])
        h_num_tempo = np.linspace(0, delta_h, 100)
        F_num_tempo = []
        for i in range(len(h_num_tempo)):
            z1 = F1_inter(h_num_tempo[i])
            z2 = F2_inter(h_num_tempo[i])
            z3 = F3_inter(h_num_tempo[i])
            z4 = F4_inter(h_num_tempo[i])
            F_num_tempo.append(interpolation_Hencky(point, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4))
        F_num_tempo = np.asarray(F_num_tempo)

        F_exp_max = np.max(F_exp)
        F_exp_portion = portion * F_exp_max
        indiceF_scale = np.where(F_exp <= F_exp_portion)[0][-1]
        F_exp_scaled = F_exp[:indiceF_scale + 1]
        h_exp_scaled = h_exp[:indiceF_scale + 1]

        h_max = delta_h
        crossings = np.where((F_num_tempo[:-1] <= F_exp_portion) & (F_num_tempo[1:] >= F_exp_portion))[0]
        if len(crossings) > 0:
            i = crossings[0]
            h_max = float(np.interp(F_exp_portion,
                                    [F_num_tempo[i], F_num_tempo[i + 1]],
                                    [h_num_tempo[i], h_num_tempo[i + 1]]))

        h_exp_decale = h_exp_scaled + decalage_global

        F_num = []
        for i in range(len(h_exp_decale)):
            if h_exp_decale[i] < 0.0:
                F_num.append(0.0)
            else:
                z1 = F1_inter(h_exp_decale[i])
                z2 = F2_inter(h_exp_decale[i])
                z3 = F3_inter(h_exp_decale[i])
                z4 = F4_inter(h_exp_decale[i])
                F_num.append(interpolation_Hencky(point, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4))

        return calculer_integrale(h_exp_decale, F_exp_scaled, np.asarray(F_num))

    _log(f"─── Phase 2 : optimisation par portion ({len(portions)} portions) ───")
    resultats_optimisation = {}
    x_opt = sigma_y_optimale
    y_opt = n_optimale
    sigma_y_resultats = []
    n_resultats = []

    for k, portion in enumerate(portions, 1):
        _log(f"  Portion {k}/{len(portions)} — {portion*100:.0f}% F_max")
        params = Parameters()
        params.add('x', value=x_opt, vary=True, min=33.6, max=3360.0)
        params.add('y', value=y_opt, vary=True, min=0.0, max=0.45)

        if stop_flag is not None and stop_flag():
            raise _ArretUtilisateur()

        result = minimize(
            rosenbrock_1, params,
            args=(portion,),
            method='Nelder-Mead',
            options={'xatol': 1e-10, 'fatol': 1e-10},
        )

        x_final = result.params['x'].value
        y_final = result.params['y'].value
        sigma_y_resultats.append(x_final)
        n_resultats.append(y_final)
        resultats_optimisation[portion] = {'x': x_final, 'y': y_final}
        _log(f"    σ_y = {x_final:.3f}  n = {y_final:.4f}")

        x_opt = x_final
        y_opt = y_final

    # -------------------------------------------------------------------------
    # Phase 3 — Hessienne par différences finies → déformation représentative
    # -------------------------------------------------------------------------
    _log(f"─── Phase 3 : matrice Hessienne ({len(resultats_optimisation)} portions) ───")
    delta_k = 0.01
    derivatives_results = {}
    deformations = []
    contraintes = []

    for k, (portion, values) in enumerate(resultats_optimisation.items(), 1):
        _log(f"  Hessienne {k}/{len(resultats_optimisation)} — portion {portion*100:.0f}%")
        if stop_flag is not None and stop_flag():
            raise _ArretUtilisateur()

        sigma_y = values['x']
        n_p = values['y']
        K_final = calculate_K(sigma_y, n_p, E)
        _log(f"    dérivée d²/dK²...")


        K_values = [
            K_final - 2.0 * delta_k,
            K_final - delta_k,
            K_final,
            K_final + delta_k,
            K_final + 2.0 * delta_k,
        ]
        resid_valeur_k = []
        for K in K_values:
            params['x'].value = calculate_sigma_y(K, n_p, E)
            params['y'].value = n_p
            resid_valeur_k.append(rosenbrock_1(params, portion))

        n_values = [
            n_p - 2.0 * delta_k,
            n_p - delta_k,
            n_p,
            n_p + delta_k,
            n_p + 2.0 * delta_k,
        ]
        resid_valeur_n = []
        for n in n_values:
            params['x'].value = calculate_sigma_y(K_final, n, E)
            params['y'].value = n
            resid_valeur_n.append(rosenbrock_1(params, portion))

        derivative_k = second_derivative(
            resid_valeur_k[4], resid_valeur_k[3], resid_valeur_k[2],
            resid_valeur_k[1], resid_valeur_k[0], delta_k,
        )
        _log(f"    dérivée d²/dn²...")
        derivative_n = second_derivative(
            resid_valeur_n[4], resid_valeur_n[3], resid_valeur_n[2],
            resid_valeur_n[1], resid_valeur_n[0], delta_k,
        )
        derivatives_results[portion] = {'derivative_k': derivative_k, 'derivative_n': derivative_n}
        _log(f"    dérivée mixte d²/dKdn (16 évaluations)...")

        # Dérivée mixte — formule exacte du code brut (lignes 1127-1129)
        params['x'].value = calculate_sigma_y(K_final - delta_k, n_p - delta_k, E)
        params['y'].value = n_p - delta_k
        f_1_1 = rosenbrock_1(params, portion)

        params['x'].value = calculate_sigma_y(K_final + delta_k, n_p - delta_k, E)
        params['y'].value = n_p - delta_k
        f1_1 = rosenbrock_1(params, portion)

        params['x'].value = calculate_sigma_y(K_final + delta_k, n_p + delta_k, E)
        params['y'].value = n_p + delta_k
        f11 = rosenbrock_1(params, portion)

        params['x'].value = calculate_sigma_y(K_final - delta_k, n_p + delta_k, E)
        params['y'].value = n_p + delta_k
        f_11 = rosenbrock_1(params, portion)

        params['x'].value = calculate_sigma_y(K_final - 2.0 * delta_k, n_p - 2.0 * delta_k, E)
        params['y'].value = n_p - 2.0 * delta_k
        f_2_2 = rosenbrock_1(params, portion)

        params['x'].value = calculate_sigma_y(K_final - delta_k, n_p - 2.0 * delta_k, E)
        params['y'].value = n_p - 2.0 * delta_k
        f_1_2 = rosenbrock_1(params, portion)

        params['x'].value = calculate_sigma_y(K_final + delta_k, n_p - 2.0 * delta_k, E)
        params['y'].value = n_p - 2.0 * delta_k
        f1_2 = rosenbrock_1(params, portion)

        params['x'].value = calculate_sigma_y(K_final + 2.0 * delta_k, n_p - 2.0 * delta_k, E)
        params['y'].value = n_p - 2.0 * delta_k
        f2_2 = rosenbrock_1(params, portion)

        params['x'].value = calculate_sigma_y(K_final + 2.0 * delta_k, n_p + delta_k, E)
        params['y'].value = n_p + delta_k
        f21 = rosenbrock_1(params, portion)

        params['x'].value = calculate_sigma_y(K_final + 2.0 * delta_k, n_p + 2.0 * delta_k, E)
        params['y'].value = n_p + 2.0 * delta_k
        f22 = rosenbrock_1(params, portion)

        params['x'].value = calculate_sigma_y(K_final + 2.0 * delta_k, n_p - delta_k, E)
        params['y'].value = n_p - delta_k
        f2_1 = rosenbrock_1(params, portion)

        params['x'].value = calculate_sigma_y(K_final - 2.0 * delta_k, n_p + 2.0 * delta_k, E)
        params['y'].value = n_p + 2.0 * delta_k
        f_22 = rosenbrock_1(params, portion)

        params['x'].value = calculate_sigma_y(K_final - delta_k, n_p + 2.0 * delta_k, E)
        params['y'].value = n_p + 2.0 * delta_k
        f_12 = rosenbrock_1(params, portion)

        params['x'].value = calculate_sigma_y(K_final + delta_k, n_p + 2.0 * delta_k, E)
        params['y'].value = n_p + 2.0 * delta_k
        f12 = rosenbrock_1(params, portion)

        params['x'].value = calculate_sigma_y(K_final - 2.0 * delta_k, n_p - 1.0 * delta_k, E)
        params['y'].value = n_p - 1.0 * delta_k
        f_2_1 = rosenbrock_1(params, portion)

        params['x'].value = calculate_sigma_y(K_final - 2.0 * delta_k, n_p + 1.0 * delta_k, E)
        params['y'].value = n_p + 1.0 * delta_k
        f_21 = rosenbrock_1(params, portion)

        d2resKdnbis_2 = (
            f_2_2 - 8.0 * f_1_2 + 8.0 * f1_2 - f2_2
            - 8.0 * f_2_1 + 64.0 * f_1_1 - 64.0 * f1_1 + 8.0 * f2_1
            + 8.0 * f_21 - 64.0 * f_11 + 64.0 * f11 - 8.0 * f21
            - f_22 + 8.0 * f_12 - 8.0 * f12 + f22
        ) / (144.0 * delta_k ** 2)

        derivatives_results[portion]['d2resKdnbis_2'] = d2resKdnbis_2

        matrice = np.array([[derivative_k, d2resKdnbis_2], [d2resKdnbis_2, derivative_n]])
        valeurs_propres, vecteurs_propres = np.linalg.eig(matrice)
        v10 = float(np.real(vecteurs_propres[1, 0]))
        v00 = float(np.real(vecteurs_propres[0, 0]))
        if abs(v10) < 1e-300:
            deformation = 0.0
        else:
            deformation = float(np.real(np.exp(-v00 / v10)))

        deformations.append(deformation)
        contrainte = loi_hollomon(sigma_y, n_p, deformation, E)
        contraintes.append(contrainte)
        _log(f"    → ε = {deformation:.4f}  σ = {contrainte:.2f} MPa")

    _log("─── Calcul terminé ───")
    # Résidu final par portion
    residu_par_portion = []
    for portion, values in resultats_optimisation.items():
        params['x'].value = values['x']
        params['y'].value = values['y']
        residu_par_portion.append(rosenbrock_1(params, portion))

    df_resultats = pd.DataFrame({
        'epsilon': deformations,
        'sigma': contraintes,
        'F_max': list(resultats_optimisation.keys()),
        'Sigma_y': sigma_y_resultats,
        'n': n_resultats,
        'residu': residu_par_portion,
    })

    global_optimal = {
        'sigma_y': x_opt,   # dernière portion Phase 2
        'n': y_opt,
        'rayon_num': rayon_list[-1] if rayon_list else None,
    }

    return df_resultats, global_optimal
