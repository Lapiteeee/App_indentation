import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from lmfit import Parameters, minimize

from .maths import second_derivative, calculate_K, calculate_sigma_y, loi_hollomon, calculer_integrale
from .geometrie import numero_element, interpolation_Hencky
from .io_donnees import lire_courbe_fh, lire_maillage, construire_bdd_dimensionnelle


class ArretUtilisateur(Exception):
    pass


def executer_optimisation_zero(
    E, sig_y_guess, n_guess,
    chemin_courbe, bdd_module, portions,
    chemins_maillage,
    h_pivot,
    callback=None,
    stop_flag=None,
    log_callback=None,
):
    """
    Pivot Zéro : l'utilisateur choisit le point d'origine (h_pivot).
    h_exp est tronqué avant ce point et décalé pour démarrer à 0.
    Aucun paramètre décalage dans l'optimisation.
    """
    def _log(msg):
        if log_callback is not None:
            log_callback(msg)

    _log("Lecture courbe F-h...")
    h_raw, F_raw = lire_courbe_fh(chemin_courbe)
    _log(f"  {len(h_raw)} points chargés.")
    _log("Lecture maillage...")
    element_dict, nodes_dict = lire_maillage(chemins_maillage, E)
    _log("Construction BDD dimensionnelle...")
    base_donnee_dimensionelle = construire_bdd_dimensionnelle(bdd_module, E)
    _log("  BDD prête.")

    # --- Pré-traitement : recalage à l'origine sélectionnée par l'utilisateur ---
    h_arr = np.array(h_raw)
    F_arr = np.array(F_raw)
    mask_pivot = h_arr >= h_pivot
    h_arr = h_arr[mask_pivot]
    F_arr = F_arr[mask_pivot]
    h_arr = h_arr - h_arr[0]   # le point sélectionné devient h = 0
    h_exp = h_arr
    F_exp = F_arr
    _log(f"  Origine recalée à h_pivot = {h_pivot:.4f} µm → {len(h_exp)} points conservés.")

    rayon_list = []

    # -------------------------------------------------------------------------
    # Phase 1 — rosenbrock global (courbe complète, h_exp déjà recalé)
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

        F_num = []
        for i in range(len(h_exp)):
            z1 = F1_inter(h_exp[i])
            z2 = F2_inter(h_exp[i])
            z3 = F3_inter(h_exp[i])
            z4 = F4_inter(h_exp[i])
            F = interpolation_Hencky(point, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)
            F_num.append(F)

        rayon_test_1 = a1_inter(h_exp[-1])
        rayon_test_2 = a2_inter(h_exp[-1])
        rayon_test_3 = a3_inter(h_exp[-1])
        rayon_test_4 = a4_inter(h_exp[-1])
        rayon_num = interpolation_Hencky(
            point,
            x1, y1, rayon_test_1,
            x2, y2, rayon_test_2,
            x3, y3, rayon_test_3,
            x4, y4, rayon_test_4,
        )
        rayon_list.append(rayon_num)
        F_num = np.asarray(F_num)
        return calculer_integrale(h_exp, F_exp, F_num)

    def iter_cb_phase1(params, iteration, resid):
        if stop_flag is not None and stop_flag():
            raise ArretUtilisateur()
        if iteration % 50 == 0:
            _log(f"  iter {iteration} — résidu = {resid:.4e}")
        if callback is not None:
            callback(params, iteration, resid)

    # -------------------------------------------------------------------------
    # Phase 0 — scan sur les nœuds BDD avec la fonction de coût réelle
    # (même coût que Phase 1) pour garantir la cohérence du point de départ.
    # -------------------------------------------------------------------------
    _log("─── Phase 0 : scan des nœuds BDD ───")
    best_sig_y = sig_y_guess
    best_n = n_guess
    best_cost_node = float('inf')
    for n_noeud in base_donnee_dimensionelle:
        for sig_noeud in base_donnee_dimensionelle[n_noeud]:
            params_scan = Parameters()
            params_scan.add('x', value=sig_noeud, vary=False)
            params_scan.add('y', value=n_noeud, vary=False)
            try:
                c = rosenbrock(params_scan)
                if c < best_cost_node:
                    best_cost_node = c
                    best_sig_y = sig_noeud
                    best_n = n_noeud
            except Exception:
                pass
    _log(f"  → meilleur nœud : σ_y = {best_sig_y:.1f} MPa  |  n = {best_n:.4f}  |  coût = {best_cost_node:.4e}")

    _log("─── Phase 1 : optimisation globale (courbe complète, origine fixée) ───")
    params = Parameters()
    params.add('x', value=best_sig_y, vary=True, min=33.6, max=3360.0)
    params.add('y', value=best_n,     vary=True, min=0.0,  max=0.45)

    result = minimize(
        rosenbrock, params,
        method='Nelder-Mead',
        iter_cb=iter_cb_phase1,
        options={'xatol': 1e-10, 'fatol': 1e-10},
    )

    sigma_y_optimale = result.params['x'].value
    n_optimale       = result.params['y'].value
    _log(f"  → σ_y = {sigma_y_optimale:.3f} MPa  |  n = {n_optimale:.4f}")

    # -------------------------------------------------------------------------
    # Phase 2 — rosenbrock_1 par portion
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
        h2 = base_donnee_dimensionelle[y2][x2]['h']
        F2 = base_donnee_dimensionelle[y2][x2]['Force']
        h3 = base_donnee_dimensionelle[y3][x3]['h']
        F3 = base_donnee_dimensionelle[y3][x3]['Force']
        h4 = base_donnee_dimensionelle[y4][x4]['h']
        F4 = base_donnee_dimensionelle[y4][x4]['Force']

        F1_inter = interp1d(h1, F1, kind='linear', fill_value="extrapolate")
        F2_inter = interp1d(h2, F2, kind='linear', fill_value="extrapolate")
        F3_inter = interp1d(h3, F3, kind='linear', fill_value="extrapolate")
        F4_inter = interp1d(h4, F4, kind='linear', fill_value="extrapolate")

        F_num = []
        for i in range(len(h_exp)):
            z1 = F1_inter(h_exp[i])
            z2 = F2_inter(h_exp[i])
            z3 = F3_inter(h_exp[i])
            z4 = F4_inter(h_exp[i])
            F = interpolation_Hencky(point, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)
            F_num.append(F)

        F_num = np.asarray(F_num)

        F_max   = np.max(F_exp)
        F_scale = portion * F_max
        h_interp_from_F = interp1d(
            F_exp, h_exp, kind='linear', fill_value='extrapolate', assume_sorted=True
        )
        h_limit      = h_interp_from_F(F_scale)
        mask         = h_exp <= h_limit
        h_exp_scaled = h_exp[mask]
        F_exp_scaled = F_exp[mask]
        F_num_scaled = F_num[mask]

        return calculer_integrale(h_exp_scaled, F_exp_scaled, F_num_scaled)

    _log(f"─── Phase 2 : optimisation par portion ({len(portions)} portions) ───")
    resultats_optimisation = {}
    x_opt = sigma_y_optimale
    y_opt = n_optimale
    sigma_y_resultats = []
    n_resultats       = []

    for k, portion in enumerate(portions, 1):
        _log(f"  Portion {k}/{len(portions)} — seuil {portion*100:.0f}% de F_exp")
        params = Parameters()
        params.add('x', value=x_opt, vary=True, min=33.6, max=3360.0)
        params.add('y', value=y_opt, vary=True, min=0.0,  max=0.45)

        if stop_flag is not None and stop_flag():
            raise ArretUtilisateur()

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
    delta_k = 0.02
    derivatives_results = {}
    deformations = []
    contraintes  = []

    for k, (portion, values) in enumerate(resultats_optimisation.items(), 1):
        _log(f"  Hessienne {k}/{len(resultats_optimisation)} — seuil {portion*100:.0f}%")
        if stop_flag is not None and stop_flag():
            raise ArretUtilisateur()

        sigma_y = values['x']
        n_p     = values['y']
        K_final = calculate_K(sigma_y, n_p, E)

        _log(f"    dérivée d²/dK²...")
        K_vals = [K_final - 2*delta_k, K_final - delta_k, K_final, K_final + delta_k, K_final + 2*delta_k]
        resid_k = []
        for K in K_vals:
            params['x'].value = calculate_sigma_y(K, n_p, E)
            params['y'].value = n_p
            resid_k.append(rosenbrock_1(params, portion))

        _log(f"    dérivée d²/dn²...")
        n_vals = [n_p - 2*delta_k, n_p - delta_k, n_p, n_p + delta_k, n_p + 2*delta_k]
        resid_n = []
        for n in n_vals:
            params['x'].value = calculate_sigma_y(K_final, n, E)
            params['y'].value = n
            resid_n.append(rosenbrock_1(params, portion))

        derivative_k = second_derivative(resid_k[4], resid_k[3], resid_k[2], resid_k[1], resid_k[0], delta_k)
        derivative_n = second_derivative(resid_n[4], resid_n[3], resid_n[2], resid_n[1], resid_n[0], delta_k)
        derivatives_results[portion] = {'derivative_k': derivative_k, 'derivative_n': derivative_n}

        _log(f"    dérivée mixte d²/dKdn (16 évaluations)...")

        def _r1(K, n, _portion=portion):
            params['x'].value = calculate_sigma_y(K, n, E)
            params['y'].value = n
            return rosenbrock_1(params, _portion)

        dk = delta_k
        f_1_1 = _r1(K_final - dk,   n_p - dk)
        f1_1  = _r1(K_final + dk,   n_p - dk)
        f11   = _r1(K_final + dk,   n_p + dk)
        f_11  = _r1(K_final - dk,   n_p + dk)
        f_2_2 = _r1(K_final - 2*dk, n_p - 2*dk)
        f_1_2 = _r1(K_final - dk,   n_p - 2*dk)
        f1_2  = _r1(K_final + dk,   n_p - 2*dk)
        f2_2  = _r1(K_final + 2*dk, n_p - 2*dk)
        f21   = _r1(K_final + 2*dk, n_p + dk)
        f22   = _r1(K_final + 2*dk, n_p + 2*dk)
        f2_1  = _r1(K_final + 2*dk, n_p - dk)
        f_22  = _r1(K_final - 2*dk, n_p + 2*dk)
        f_12  = _r1(K_final - dk,   n_p + 2*dk)
        f12   = _r1(K_final + dk,   n_p + 2*dk)
        f_2_1 = _r1(K_final - 2*dk, n_p - dk)
        f_21  = _r1(K_final - 2*dk, n_p + dk)

        d2resKdnbis_2 = (
            f_2_2 - 8.0*f_1_2 + 8.0*f1_2 - f2_2
            - 8.0*f_2_1 + 64.0*f_1_1 - 64.0*f1_1 + 8.0*f2_1
            + 8.0*f_21 - 64.0*f_11 + 64.0*f11 - 8.0*f21
            - f_22 + 8.0*f_12 - 8.0*f12 + f22
        ) / (144.0 * delta_k**2)

        derivatives_results[portion]['d2resKdnbis_2'] = d2resKdnbis_2

        matrice = np.array([[derivative_k, d2resKdnbis_2], [d2resKdnbis_2, derivative_n]])
        _, vecteurs_propres = np.linalg.eig(matrice)
        v10 = float(np.real(vecteurs_propres[1, 0]))
        v00 = float(np.real(vecteurs_propres[0, 0]))
        deformation = 0.0 if abs(v10) < 1e-300 else float(np.real(np.exp(-v00 / v10)))

        deformations.append(deformation)
        contrainte = loi_hollomon(sigma_y, n_p, deformation, E)
        contraintes.append(contrainte)
        _log(f"    → ε = {deformation:.4f}  σ = {contrainte:.2f} MPa")

    _log("─── Calcul terminé ───")

    residu_par_portion = []
    for portion, values in resultats_optimisation.items():
        params['x'].value = values['x']
        params['y'].value = values['y']
        residu_par_portion.append(rosenbrock_1(params, portion))

    df_resultats = pd.DataFrame({
        'epsilon': deformations,
        'sigma':   contraintes,
        'F_max':   list(resultats_optimisation.keys()),
        'Sigma_y': sigma_y_resultats,
        'n':       n_resultats,
        'residu':  residu_par_portion,
    })

    global_optimal = {
        'sigma_y':   x_opt,
        'n':         y_opt,
        'rayon_num': rayon_list[-1] if rayon_list else None,
    }

    return df_resultats, global_optimal
