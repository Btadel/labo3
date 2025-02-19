import pandas as pd
import matplotlib.pyplot as plt
import os

import numpy as np
from scipy.optimize import curve_fit

# =========================================================================
# Constantes physiques (à adapter selon vos valeurs / hypothèses)
# =========================================================================

k_B = 1.380649e-23   # Constante de Boltzmann (J/K)
T   = 293.0          # Température en K (ex. ~20°C)
# Exemple de diamètre moléculaire effectif pour l'air/azote (~3,7 Å) :
d_air  = 3.7e-10     # m (ordre de grandeur)
d_argon= 3.4e-10     # m
d_co2  = 4.0e-10     # m

# Remarque : le diamètre précis dépendra du gaz et de la littérature.
# On peut aussi trouver la "section efficace" pi*d^2 à partir de ces diamètres.

# Longueur du tube (à adapter) :
L = 0.75   # 75 cm => 0.75 m

# =========================================================================
# Paramètres expérimentaux : liste des expériences
# =========================================================================

# Exemple de structure pour regrouper paramètres divers (à adapter à vos besoins)
# Pour chaque bloc, on pourra préciser : gaz, pression(s), rayon, etc.
experiences = [
    # ---------------------------
    # (a) r0 = 1.55 mm, gaz = Air
    # ---------------------------
    {
        "nom": "A1",
        "gaz": "Air",
        "r0": 1.55e-3,        # 1.55 mm en mètres
        "P1_0": 2.5,          # Pression haute (ex. en Torr/mbar, à convertir)
        "P2_0": 0.1           # Pression basse
    },
    {
        "nom": "A2",
        "gaz": "Air",
        "r0": 1.55e-3,
        "P1_0": 5.0,
        "P2_0": 2.5
    },
    {
        "nom": "A3",
        "gaz": "Air",
        "r0": 1.55e-3,
        "P1_0": 7.5,
        "P2_0": 5.0
    },
    {
        "nom": "A4",
        "gaz": "Air",
        "r0": 1.55e-3,
        "P1_0": 12.5,
        "P2_0": 10.0
    },

    # ------------------------------------
    # (b) Gaz = Air, mais r0 varie (3 rayons)
    # ------------------------------------
    {
        "nom": "B1",
        "gaz": "Air",
        "r0": 2.00e-3,   # 2.0 mm
        "P1_0": 5.0,
        "P2_0": 2.5
    },
    {
        "nom": "B3",
        "gaz": "Air",
        "r0": 0.81e-3,   # 0.81 mm
        "P1_0": 5.0,
        "P2_0": 2.5
    },

    # ---------------------------------
    # (c) r0 = 1.55 mm, Gaz varie (Ar, Air, CO2)
    # ---------------------------------
    {
        "nom": "C1",
        "gaz": "Ar",
        "r0": 1.55e-3,
        "P1_0": 5.0,
        "P2_0": 2.5
    },
    {
        "nom": "C2",     # Encore Air, même que (a2) ou (b2)
        "gaz": "Air",
        "r0": 1.55e-3,
        "P1_0": 5.0,
        "P2_0": 2.5
    },
    {
        "nom": "C3",
        "gaz": "CO2",
        "r0": 1.55e-3,
        "P1_0": 5.0,
        "P2_0": 2.5
    },

    # ------------------------------------------------------
    # (d) Gaz = Air + filtre (r0 inconnu ou modifié par le filtre)
    # ------------------------------------------------------
    {
        "nom": "D1",
        "gaz": "Air",
        "r0": None,         # Incertitude (tube + filtre ?)
        "filtre": True,     # Indicateur de la présence du filtre
        "P1_0": 2.5,
        "P2_0": 0.1
    },
    {
        "nom": "D2",
        "gaz": "Air",
        "r0": None,
        "filtre": True,
        "P1_0": 5.0,
        "P2_0": 2.5
    },
    {
        "nom": "D3",
        "gaz": "Air",
        "r0": None,
        "filtre": True,
        "P1_0": 7.5,
        "P2_0": 5.0
    },
    {
        "nom": "D4",
        "gaz": "Air",
        "r0": None,
        "filtre": True,
        "P1_0": 12.5,
        "P2_0": 10.0
    }
]


def lire_fichier_txt(chemin_fichier: str) -> pd.DataFrame:
    """
    Lit un fichier texte (Temps, P1, P2) séparé par des espaces ou des tabulations.
    Renvoie un DataFrame pandas avec les colonnes : "Temps", "P1", "P2".
    """
    if not os.path.exists(chemin_fichier):
        raise FileNotFoundError(f"Fichier introuvable: {chemin_fichier}")
    
    # - delim_whitespace=True indique que le séparateur est un ou plusieurs espaces
    # - names=["Temps", "P1", "P2"] assigne un nom à chaque colonne
    # - header=None indique qu'il n'y a pas de ligne d'en-tête dans le fichier
    df = pd.read_csv(
        chemin_fichier, 
        delim_whitespace=True,  # ou sep='\t' si c’est tabulé
        header=None, 
        names=["Temps", "P1", "P2"],
        skipinitialspace=True
    )
    
    return df

def pression_en_Pa(p, unite="torr"):
    """
    Convertit une pression p (valeur) depuis 'torr', 'mbar' ou autre vers Pa.
    Adaptez la formule selon votre unité de mesure expérimentale.
    """
    if unite.lower() == "torr":
        # 1 torr ~ 133.322 Pa
        return p * 133.322
    elif unite.lower() == "mbar":
        # 1 mbar = 100 Pa
        return p * 100
    elif unite.lower() == "pa":
        return p
    else:
        raise ValueError(f"Unité inconnue: {unite}")

def calculer_libre_parcours_moyen(p_moy_Pa, gaz="Air"):
    """
    Calcule le libre parcours moyen (en m), à partir d'une pression en Pa
    et du type de gaz (pour choisir le diamètre moléculaire).
    """
    if gaz.lower() == "air":
        d = d_air
    elif gaz.lower() == "argon" or gaz.lower() == "ar":
        d = d_argon
    elif gaz.lower() == "co2":
        d = d_co2
    else:
        # Si gaz inconnu, mettez une valeur par défaut ou levez une exception
        d = d_air
    
    lambda_ = k_B * T / (np.sqrt(2)*np.pi * d**2 * p_moy_Pa)
    return lambda_

def fit_exponentiel(df, Pmoy):
    """
    df contient au moins les colonnes 'Temps' et 'P1'.
    On ajuste ln[(P1 - Pmoy)/(P1(0) - Pmoy)] = A + B*t.
    Retourne B, l'incertitude sur B, et la partie 'clean' du DataFrame.
    """
    # On prépare d'abord le vecteur x = t et y = ln[...].
    P1_0 = df["P1"].iloc[0]
    numerateur   = df["P1"] - Pmoy
    denominateur = (P1_0 - Pmoy)
    
    # Eviter les valeurs négatives ou nulles
    masque_valide = (numerateur > 0.0)
    df_valid = df[masque_valide].copy()
    
    # Calcul du log
    df_valid["Y"] = np.log(df_valid["P1"] - Pmoy) - np.log(denominateur)
    df_valid["X"] = df_valid["Temps"]
    
    # Ajustement linéaire : Y = A + B*X
    # On peut utiliser np.polyfit (sans pondération) ou curve_fit, etc.
    # np.polyfit(X, Y, 1) renvoie [B, A] => Y = B*X + A
    B, A = np.polyfit(df_valid["X"], df_valid["Y"], 1)
    
    # Pour estimer l'incertitude sur B et A, on peut utiliser la matrice de covariance
    # via np.polyfit(..., cov=True) dans certaines versions de numpy (>=1.21).
    # Sinon, on peut utiliser scipy.stats.linregress ou scipy.optimize.curve_fit.
    # Ici, un exemple avec curve_fit :

    def modele_lin(x, a, b):
        return a + b*x

    popt, pcov = curve_fit(modele_lin, df_valid["X"], df_valid["Y"])
    A_fit, B_fit = popt
    A_err, B_err = np.sqrt(np.diag(pcov))  # incertitudes
    
    return A_fit, A_err, B_fit, B_err, df_valid

def analyser_experiences(dossier_data, experiences, unite_p="torr"):
    """
    - Lit les données de chaque expérience
    - Calcule la pression moyenne Pmoy
    - Calcule le libre parcours moyen
    - Décide si on est en régime visqueux ou moléculaire
    - Fait l'ajustement linéaire en échelle semi-log
    - Construit des graphiques et un tableau de résultats
    """
    resultats = []

    for exp in experiences:
        chemin_fichier = os.path.join(dossier_data, exp["fichier"])
        df = lire_fichier_txt(chemin_fichier)

        # Calcul de la pression moyenne (P1 et P2 ?)
        # Par exemple, si vous définissez "Pmoy" = (P1(t)+P2(t))/2 ou c'est la moyenne des P2(t) ...
        # Ici, on suppose qu'on utilise P2(t) moyenne ou un Pmoy constant
        P2_moy   = df["P2"].mean()
        P2_moy_Pa = pression_en_Pa(P2_moy, unite=unite_p)

        # Stocker Pmoy pour l'offset
        Pmoy = P2_moy

        # Rayon
        r0_m  = exp["r0"]  # déjà en m
        # Pression moyenne en Pascal :
        Pmoy_Pa = P2_moy_Pa

        # Calcul du libre parcours moyen
        lambda_ = calculer_libre_parcours_moyen(Pmoy_Pa, gaz=exp["gaz"])

        # Décision sur le régime : comparer r0 avec ~ λ
        # si r0 >> λ => régime visqueux, si r0 << λ => régime moléculaire 
        # "nettement" en visqueux => λ << r0
        # "nettement" en moléculaire => λ >> r0
        if lambda_ << r0_m:
            regime = "Visqueux"
        elif lambda_ >> r0_m:
            regime = "Moleculaire"
        else:
            regime = "Transition"

        # Ajustement linéaire semi-log
        A_fit, A_err, B_fit, B_err, df_valid = fit_exponentiel(df, Pmoy)

        # Tau = -1/B
        tau = -1.0 / B_fit
        # incertitude sur tau, propagation d'erreur
        tau_err = tau**2 * B_err

        # Sauvegarde des résultats
        resultats.append({
            "Nom"      : exp["nom"],
            "Gaz"      : exp["gaz"],
            "r0 (m)"   : r0_m,
            "P_moy (unite)" : Pmoy,
            "P_moy (Pa)"    : Pmoy_Pa,
            "lambda (m)"    : lambda_,
            "Regime"   : regime,
            "Slope (B)" : B_fit,
            "Slope_err" : B_err,
            "Tau (s)"   : tau,
            "Tau_err (s)" : tau_err
        })

        # --- Graphique semi-log pour l'expérience courante ---
        plt.figure(figsize=(6,4))
        plt.plot(df_valid["X"], df_valid["Y"], 'o', label="Points validés")
        # Courbe ajustée
        x_fit = np.linspace(min(df_valid["X"]), max(df_valid["X"]), 100)
        y_fit = A_fit + B_fit * x_fit
        plt.plot(x_fit, y_fit, '-', label=f"Ajustement (B={B_fit:.4g} s^-1)")
        plt.xlabel("t (s)")
        plt.ylabel("ln[(P1 - Pmoy)/(P1(0) - Pmoy)]")
        plt.title(f"Semi-log : {exp['nom']}, gaz={exp['gaz']}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Conversion en DataFrame final
    df_res = pd.DataFrame(resultats)
    return df_res
