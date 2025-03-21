import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# =============================================================================
# Paramètres globaux et données
# =============================================================================
L = 0.75  # Longueur du tube en mètres
T = 299.15   # Température en K (approx)
k_B = 1.38e-23  # Constante de Boltzmann en J/K
mu_air = 1.81e-5  # Viscosité dynamique de l'air (Pa.s)

# Diamètres moléculaires (m)
d_mol = {
    "Air": 3.7e-10,        
    "Ar": 3.4e-10,
    "CO2": 4.5e-10
}

# =============================================================================
# Définition des groupes d'expériences
# =============================================================================
# Groupe 1 : Air sans filtre, r0 = 1.55e-3 m
experiences1 = [
    {"nom": "Air1",   "gaz": "Air",  "r0": 1.55e-3, "P1_0": 2.5,  "P2_0": 0.1},
    {"nom": "Air2",   "gaz": "Air",  "r0": 1.55e-3, "P1_0": 5.0,  "P2_0": 2.5},
    {"nom": "Air3",   "gaz": "Air",  "r0": 1.55e-3, "P1_0": 7.5,  "P2_0": 5.0},
    {"nom": "Air4",   "gaz": "Air",  "r0": 1.55e-3, "P1_0": 12.5, "P2_0": 10.0}
]

# Groupe 2 : Air sans filtre, r0 différent de 1.55e-3
experiences2 = [
    {"nom": "Air1_2", "gaz": "Air",  "r0": 1.55e-3, "P1_0": 2.5,  "P2_0": 0.1},
    {"nom": "Air5",   "gaz": "Air",  "r0": 2.00e-3, "P1_0": 5.0,  "P2_0": 2.5},
    {"nom": "Air6",   "gaz": "Air",  "r0": 0.81e-3, "P1_0": 5.0,  "P2_0": 2.5}
]

# Groupe 3 : Gaz différents de Air sans filtre
experiences3 = [
    {"nom": "Argon7", "gaz": "Ar",   "r0": 1.55e-3, "P1_0": 5.0,  "P2_0": 2.5},
    {"nom": "CO2_8",  "gaz": "CO2",  "r0": 1.55e-3, "P1_0": 5.0,  "P2_0": 2.5},
]

# Groupe 4 : Expériences avec filtre
experiences4 = [
    {"nom": "AirF1",  "gaz": "Air",  "r0": None,    "filtre": True, "P1_0": 2.5,  "P2_0": 0.1},
    {"nom": "AirF2",  "gaz": "Air",  "r0": None,    "filtre": True, "P1_0": 5.0,  "P2_0": 2.5},
    {"nom": "AirF3",  "gaz": "Air",  "r0": None,    "filtre": True, "P1_0": 7.5,  "P2_0": 5.0},
    {"nom": "AirF4",  "gaz": "Air",  "r0": None,    "filtre": True, "P1_0": 12.5, "P2_0": 10.0},
]

# Regroupement des groupes avec leurs noms
groupes = [
    ("Groupe 1", experiences1),
    ("Groupe 2", experiences2),
    ("Groupe 3", experiences3),
    ("Groupe 4", experiences4)
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
    d = d_mol.get(gaz, 3.7e-10)
    P_pa = P_kPa * 1e3  # Conversion de kPa en Pa
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
    return pd.read_csv(fichier, delim_whitespace=True, header=None, names=["Temps", "P1", "P2"])

def calculer_constante_temps(df, P1_0, P2_0, epsilon=0.0):
    """
    Calcule la pente et l'intercept (avec incertitudes) à partir du graphe semi-log de (P1(t) - P2_0)/(P1_0 - P2_0).
    Retourne (slope, slope_err, intercept, r_value).
    """
    denom = P1_0 - P2_0
    if denom == 0:
        return np.nan, np.nan, np.nan, np.nan

    P1_norm = (df["P1"] - P2_0) / denom
    valid_mask = (P1_norm > 0) & (P1_norm > epsilon)
    if valid_mask.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan

    slope, intercept, r_value, p_value, std_err = linregress(
        df["Temps"][valid_mask],
        np.log(P1_norm[valid_mask])
    )
    return slope, std_err, intercept, r_value

# =============================================================================
# Analyse principale & tracés pour chaque groupe
# =============================================================================
if __name__ == "__main__":
    tableau_resultats = []

    # On itère sur chaque groupe
    for (nom_groupe, groupe) in groupes:
        plt.figure(figsize=(8, 6))  # Crée une nouvelle figure pour chaque groupe
        for exp in groupe:
            nom_exp = exp["nom"]
            gaz = exp["gaz"]
            r0_m = exp.get("r0", None)
            P1_0 = exp["P1_0"]
            P2_0 = exp["P2_0"]
            filtre = exp.get("filtre", False)

            Pmoy = (P1_0 + P2_0) / 2.0
            lambda_moy = calculer_lambda(Pmoy, gaz)  # en m
            lambda_mm = lambda_moy * 1e3            # conversion en mm

            try:
                df_data = lire_fichier_donnees(nom_exp)
                print(f"=== Données lues pour {nom_exp} ===")
                print(df_data.head())
                print()
            except FileNotFoundError as fe:
                print(fe)
                continue

            slope, slope_err, intercept, r_value = calculer_constante_temps(
                df_data, P1_0, P2_0, epsilon=0.02
            )

            if r0_m is not None:
                regime = "Visqueux" if lambda_moy < r0_m else "Moléculaire"
            else:
                regime = "Inconnu/Filtre"

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

            # Tracé des points en semi-logarithmique (axe Y logarithmique)
            denom = P1_0 - P2_0
            if denom != 0:
                P1_norm = (df_data["P1"] - P2_0) / denom
                mask_plot = (P1_norm > 0)
                plt.semilogy(df_data["Temps"][mask_plot],
                             P1_norm[mask_plot],
                             'o',
                             markersize=3,
                             label=f"{nom_exp}")
        plt.xlabel("Temps (s)")
        plt.ylabel(r"$(P_1(t) - P_{2,0}) / (P_{1,0} - P_{2,0})$")
        plt.title(f"Pression normalisée en fonction du temps - {nom_groupe}")
        plt.grid(True, which="both")
        plt.legend()
        plt.show()

    # Affichage global des résultats synthétiques
    df_res = pd.DataFrame(tableau_resultats)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("=== Résultats synthétiques ===")
    print(df_res)

    df_sub = df_res[["Gaz", "Pmoy (kPa)", "r0 (mm)", "lambda (mm)", "Regime"]]
    print("\n=== Libre parcours moyen, type de gaz, pression moyenne, rayon du tube ===")
    print(df_sub.to_string(index=False))
