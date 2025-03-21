import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# =============================================================================
# Paramètres globaux et données
# =============================================================================
L = 0.75  # Longueur du tube (m)
T = 300   # Température (K) approximative
k_B = 1.38e-23  # Constante de Boltzmann (J/K)

# Diamètres moléculaires (m)
d_mol = {
    "Air": 3.7e-10,
    "Ar": 3.4e-10,
    "CO2": 4.5e-10
}
experiences = [
    {"nom": "Air1",   "gaz": "Air",  "r0": 1.55e-3, "P1_0": 2.5,  "P2_0": 0.1},
    #{"nom": "Air1_2", "gaz": "Air",  "r0": 1.55e-3, "P1_0": 2.5,  "P2_0": 0.1},
    {"nom": "Air2",   "gaz": "Air",  "r0": 1.55e-3, "P1_0": 5.0,  "P2_0": 2.5},
    {"nom": "Air3",   "gaz": "Air",  "r0": 1.55e-3, "P1_0": 7.5,  "P2_0": 5.0},
    {"nom": "Air4",   "gaz": "Air",  "r0": 1.55e-3, "P1_0": 12.5, "P2_0": 10.0},
    {"nom": "Air5",   "gaz": "Air",  "r0": 2.00e-3, "P1_0": 5.0,  "P2_0": 2.5},
    {"nom": "Air6",   "gaz": "Air",  "r0": 0.81e-3, "P1_0": 5.0,  "P2_0": 2.5},
    {"nom": "Argon7", "gaz": "Ar",   "r0": 1.55e-3, "P1_0": 5.0,  "P2_0": 2.5},
    {"nom": "CO2_8",  "gaz": "CO2",  "r0": 1.55e-3, "P1_0": 5.0,  "P2_0": 2.5},
    {"nom": "AirF1",  "gaz": "Air",  "r0": None,    "filtre": True, "P1_0": 2.5,  "P2_0": 0.1},
    {"nom": "AirF2",  "gaz": "Air",  "r0": None,    "filtre": True, "P1_0": 5.0,  "P2_0": 2.5},
    {"nom": "AirF3",  "gaz": "Air",  "r0": None,    "filtre": True, "P1_0": 7.5,  "P2_0": 5.0},
    {"nom": "AirF4",  "gaz": "Air",  "r0": None,    "filtre": True, "P1_0": 12.5, "P2_0": 10.0},
]

# Expériences : liste de dictionnaires
experiences1 = [
    {"nom": "Air1",   "gaz": "Air",  "r0": 1.55e-3, "P1_0": 2.5,  "P2_0": 0.1},
   
    {"nom": "Air2",   "gaz": "Air",  "r0": 1.55e-3, "P1_0": 5.0,  "P2_0": 2.5},
    {"nom": "Air3",   "gaz": "Air",  "r0": 1.55e-3, "P1_0": 7.5,  "P2_0": 5.0},
    {"nom": "Air4",   "gaz": "Air",  "r0": 1.55e-3, "P1_0": 12.5, "P2_0": 10.0}
]

experiences2 = [
    {"nom": "Air1_2", "gaz": "Air",  "r0": 1.55e-3, "P1_0": 2.5,  "P2_0": 0.1},
    {"nom": "Air5",   "gaz": "Air",  "r0": 2.00e-3, "P1_0": 5.0,  "P2_0": 2.5},
    {"nom": "Air6",   "gaz": "Air",  "r0": 0.81e-3, "P1_0": 5.0,  "P2_0": 2.5}
]
experiences3 = [
    
    {"nom": "Argon7", "gaz": "Ar",   "r0": 1.55e-3, "P1_0": 5.0,  "P2_0": 2.5},
    {"nom": "CO2_8",  "gaz": "CO2",  "r0": 1.55e-3, "P1_0": 5.0,  "P2_0": 2.5},
]

experiences4 = [
    {"nom": "AirF1",  "gaz": "Air",  "r0": None,    "filtre": True, "P1_0": 2.5,  "P2_0": 0.1},
    {"nom": "AirF2",  "gaz": "Air",  "r0": None,    "filtre": True, "P1_0": 5.0,  "P2_0": 2.5},
    {"nom": "AirF3",  "gaz": "Air",  "r0": None,    "filtre": True, "P1_0": 7.5,  "P2_0": 5.0},
    {"nom": "AirF4",  "gaz": "Air",  "r0": None,    "filtre": True, "P1_0": 12.5, "P2_0": 10.0},
]




# =============================================================================
# Fonctions utilitaires
# =============================================================================
def calculer_lambda(P_kPa, gaz):
    """
    Calcule le libre parcours moyen lambda (m).
    P_kPa : pression en kPa
    gaz   : type de gaz (Air, Ar, CO2)
    """
    d = d_mol.get(gaz, 3.7e-10)  # Diamètre moléculaire, par défaut celui de l'air
    # Conversion de kPa -> Pa
    P_pa = P_kPa * 1e3
    return k_B * T / (np.sqrt(2) * np.pi * d**2 * P_pa)

def lire_fichier_donnees(nom):
    """
    Lit un fichier .txt depuis ./ecoulement/nom.txt
    Doit contenir 3 colonnes : Temps, P1, P2 (sans en-tête).
    Retourne un DataFrame avec colonnes : "Temps", "P1", "P2".
    """
    fichier = os.path.join("ecoulement", nom + ".txt")
    if not os.path.exists(fichier):
        raise FileNotFoundError(f"Introuvable: {fichier}")
    # On suppose que chaque fichier a 3 colonnes sans header : Temps (s), P1 (kPa), P2 (kPa)
    return pd.read_csv(fichier, delim_whitespace=True, header=None, names=["Temps", "P1", "P2"])

def calculer_constante_temps(df, P1_0, P2_0, epsilon=0.0):
    """
    Calcule la pente et l'intercept (et incertitudes) à partir du graphe
    semi-log de (P1(t) - P2_0)/(P1_0 - P2_0).

    df       : DataFrame (Temps, P1, P2)
    P1_0     : pression initiale (kPa)
    P2_0     : pression finale (kPa)
    epsilon  : seuil en dessous duquel on exclut les données (en fraction normalisée)

    Retourne (slope, slope_err, intercept, r_value).
    """
    denom = P1_0 - P2_0
    if denom == 0:
        return np.nan, np.nan, np.nan, np.nan

    P1_norm = (df["P1"] - P2_0) / denom
    valid_mask = (P1_norm > 0) & (P1_norm > epsilon)

    # Si on n'a pas assez de points pour un ajustement fiable, on retourne NaN
    if valid_mask.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan

    slope, intercept, r_value, p_value, std_err = linregress(
        df["Temps"][valid_mask],
        np.log(P1_norm[valid_mask])
    )

    return slope, std_err, intercept, r_value

# =============================================================================
# Programme principal
# =============================================================================
if __name__ == "__main__":
    # Liste où l'on enregistre tous les résultats
    tableau_resultats = []

    plt.figure(figsize=(8, 6))

    for exp in experiences:
        nom_exp = exp["nom"]
        gaz = exp["gaz"]
        r0_m = exp.get("r0", None)     # rayon en m
        P1_0 = exp["P1_0"]            # kPa
        P2_0 = exp["P2_0"]            # kPa
        filtre = exp.get("filtre", False)

        # Calcul de la pression moyenne
        Pmoy = (P1_0 + P2_0) / 2.0

        # Calcul du libre parcours moyen en mm
        lambda_moy = calculer_lambda(Pmoy, gaz)  # en m
        lambda_mm = lambda_moy * 1e3            # conversion en mm

        # Lecture des données
        try:
            df_data = lire_fichier_donnees(nom_exp)
            print(f"=== Données lues pour {nom_exp} ===")
            print(df_data.head())
            print()
        except FileNotFoundError as fe:
            print(fe)
            continue

        # Calcul de la pente (slope), erreur (slope_err), intercept et r_value
        slope, slope_err, intercept, r_value = calculer_constante_temps(
            df_data, P1_0, P2_0,
            epsilon=0.02  # exclure les points < 2% du signal normalisé
        )

        # Régime visqueux ou moléculaire ?
        if r0_m is not None:
            regime = "Visqueux" if lambda_moy < r0_m else "Moléculaire"
        else:
            regime = "Inconnu/Filtre"

        # Stockage des résultats dans un dictionnaire
        tableau_resultats.append({
            "Exp": nom_exp,
            "Gaz": gaz,
            "r0 (mm)": f"{r0_m*1e3:.3f}" if r0_m else ("Filtre" if filtre else "Inconnu"),
            "P1_0 (kPa)": P1_0,
            "P2_0 (kPa)": P2_0,
            "Pmoy (kPa)": Pmoy,
            "lambda (mm)": f"{lambda_mm:.3e}",
            "Regime": regime,
            "Slope (1/s)": f"{slope:.5f}" if not np.isnan(slope) else "NaN",
            "Slope_err (1/s)": f"{slope_err:.5f}" if not np.isnan(slope_err) else "NaN",
            "Intercept": f"{intercept:.5f}" if not np.isnan(intercept) else "NaN",
            "r_value": f"{r_value:.5f}" if not np.isnan(r_value) else "NaN"
        })

        # Tracé des données sur un graphe semi-log
        denom = (P1_0 - P2_0)
        if denom != 0:
            P1_norm = (df_data["P1"] - P2_0) / denom
            mask_plot = (P1_norm > 0)  # on exclut les valeurs <= 0
            plt.semilogy(df_data["Temps"][mask_plot],
                        P1_norm[mask_plot],
                        'o',   # marqueurs pour les données brutes
                        markersize=1,
                        label=f"{nom_exp} data")

            # Tracé de la droite de régression si la pente n'est pas NaN
            #if not np.isnan(slope):
            #    t_lin = np.linspace(df_data["Temps"].min(), df_data["Temps"].max(), 100)
            #    y_fit = np.exp(intercept + slope * t_lin)
            #    plt.semilogy(t_lin, y_fit, '-',  # ligne continue pour le fit
            #                label=f"{nom_exp} fit")

    # Création d'un DataFrame Pandas pour rassembler tous les résultats
    df_res = pd.DataFrame(tableau_resultats)

    # Affichage complet des lignes/colonnes
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("=== Résultats synthétiques ===")
    print(df_res)

    # Tableau extrait pour répondre au cahier des charges : gaz, Pmoy, rayon, lambda, régime
    df_sub = df_res[["Gaz", "Pmoy (kPa)", "r0 (mm)", "lambda (mm)", "Regime"]]

    print("\n=== Libre parcours moyen, type de gaz, pression moyenne, rayon du tube ===")
    print(df_sub.to_string(index=False))

    # Mise en forme du graphe final
    plt.xlabel("Temps (s)")
    plt.ylabel(r"$(P_1(t) - P_{2,0}) / (P_{1,0} - P_{2,0})$")
    plt.title("Pression normalisée en fonction du temps (échelle semi-log)")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()

