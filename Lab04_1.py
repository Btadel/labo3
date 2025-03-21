import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Paramètres globaux
L = 0.75  # Longueur du tube en mètres

# Viscosités des gaz en Pa.s (valeurs approximatives à 300K)
viscosites = {
    "Air": 1.8e-5,
    "Ar": 2.1e-5,
    "CO2": 1.4e-5
}

# Liste des expériences
experiences = [
    {"nom": "Air1", "gaz": "Air", "r0": 1.55e-3, "P1_0": 2.5, "P2_0": 0.1},
    {"nom": "Air2", "gaz": "Air", "r0": 1.55e-3, "P1_0": 5.0, "P2_0": 2.5},
    {"nom": "Air3", "gaz": "Air", "r0": 1.55e-3, "P1_0": 7.5, "P2_0": 5.0},
    {"nom": "Air4", "gaz": "Air", "r0": 1.55e-3, "P1_0": 12.5, "P2_0": 10.0},
    {"nom": "Air5", "gaz": "Air", "r0": 2.00e-3, "P1_0": 5.0, "P2_0": 2.5},
    {"nom": "Air6", "gaz": "Air", "r0": 0.81e-3, "P1_0": 5.0, "P2_0": 2.5},
    {"nom": "Argon7", "gaz": "Ar", "r0": 1.55e-3, "P1_0": 5.0, "P2_0": 2.5},
    {"nom": "CO2_8", "gaz": "CO2", "r0": 1.55e-3, "P1_0": 5.0, "P2_0": 2.5},
    {"nom": "AirF1", "gaz": "Air", "r0": None, "filtre": True, "P1_0": 2.5, "P2_0": 0.1},
    {"nom": "AirF2", "gaz": "Air", "r0": None, "filtre": True, "P1_0": 5.0, "P2_0": 2.5},
    {"nom": "AirF3", "gaz": "Air", "r0": None, "filtre": True, "P1_0": 7.5, "P2_0": 5.0},
    {"nom": "AirF4", "gaz": "Air", "r0": None, "filtre": True, "P1_0": 12.5, "P2_0": 10.0}
]

def lire_fichier(chemin):
    """Lit un fichier de données et retourne un DataFrame."""
    if not os.path.exists(chemin):
        raise FileNotFoundError(f"Fichier introuvable: {chemin}")
    return pd.read_csv(chemin, sep='\s+', header=None, names=["Temps", "P1", "P2"])

def ajustement_exponentiel(temps, P1, Pmoy):
    """Effectue un ajustement exponentiel pour déterminer la constante de temps tau."""
    y = (P1 - Pmoy) / (P1[0] - Pmoy)
    def modele(t, tau):
        return np.exp(-t / tau)
    params, covariance = curve_fit(modele, temps, y)
    return params[0], np.sqrt(np.diag(covariance))[0]

# Traitement des données et analyse
donnees_experiences = {}
for exp in experiences:
    fichier = f"./ecoulement/{exp['nom']}.txt"
    try:
        df = lire_fichier(fichier)
        Pmoy = (exp['P1_0'] + exp['P2_0']) / 2
        tau, incertitude = ajustement_exponentiel(df['Temps'], df['P1'], Pmoy)
        eta = viscosites.get(exp["gaz"], None)
        donnees_experiences[exp['nom']] = {"tau": tau, "incertitude": incertitude, "Pmoy": Pmoy, "r0": exp["r0"], "eta": eta}
    except FileNotFoundError as e:
        print(e)

# Séparation des expériences
sans_filtre = [exp for exp in experiences if not exp.get("filtre", False)]
avec_filtre = [exp for exp in experiences if exp.get("filtre", False)]

# Création des graphiques
plt.figure()
plt.scatter(1 / np.array([donnees_experiences[exp["nom"]]["Pmoy"] for exp in sans_filtre]), 
            [donnees_experiences[exp["nom"]]["tau"] for exp in sans_filtre], 
            label="Sans filtre")
if avec_filtre:
    plt.scatter(1 / np.array([donnees_experiences[exp["nom"]]["Pmoy"] for exp in avec_filtre]), 
                [donnees_experiences[exp["nom"]]["tau"] for exp in avec_filtre], 
                label="Avec filtre")
plt.xlabel("1 / Pmoy")
plt.ylabel("Tau")
plt.title("Variation de Tau en fonction de 1/Pmoy")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.scatter(1 / np.array([donnees_experiences[exp["nom"]]["r0"] for exp in sans_filtre])**4, 
            [donnees_experiences[exp["nom"]]["tau"] for exp in sans_filtre], 
            label="Données")
plt.xlabel("1 / r0^4")
plt.ylabel("Tau")
plt.title("Variation de Tau en fonction de 1/r0^4")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.scatter([donnees_experiences[exp["nom"]]["eta"] for exp in sans_filtre], 
            [donnees_experiences[exp["nom"]]["tau"] for exp in sans_filtre], 
            label="Données")
plt.xlabel("Viscosité η (Pa.s)")
plt.ylabel("Tau")
plt.title("Variation de Tau en fonction de η")
plt.legend()
plt.grid()
plt.show()
