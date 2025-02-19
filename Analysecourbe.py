import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

###############################################################################
# Paramètres génériques (communs)
###############################################################################
RESOLUTION = 5.2e-6   # [m/pixel] Taille d'un pixel sur le détecteur
LAMBDA     = 632.8e-9 # [m] Longueur d'onde du laser (ex. He-Ne)
F          = 0.10     # [m] Distance focale ou distance écran

###############################################################################
# Fonctions de calcul
###############################################################################

def normalisation_intensite(intensites):
    """
    Normalise un tableau d'intensités par son maximum.
    """
    return intensites / np.max(intensites)

def modele_interference_multiple(theta, b, d, N, lambd):
    """
    Modèle I/I0 pour N fentes de largeur b, espacées de d.

    Formule : (sin(alpha)/alpha)^2 * (sin(N*beta)/(N*sin(beta)))^2
      avec alpha = pi * b * sin(theta) / lambda
           beta  = pi * d * sin(theta) / lambda.
    """
    alpha = np.pi * b * np.sin(theta) / lambd
    beta  = np.pi * d * np.sin(theta) / lambd
    
    # Pour éviter la division par zéro
    alpha = np.where(alpha == 0, 1e-12, alpha)
    beta  = np.where(beta  == 0, 1e-12, beta)
    
    # Terme de diffraction (fente simple)
    diffraction = (np.sin(alpha) / alpha)**2
    
    # Terme d'interférence (N fentes)
    if N > 1:
        interference = (np.sin(N * beta) / (N * np.sin(beta)))**2
        return diffraction * interference
    else:
        return diffraction

def modele_complet(x_m, x0_m, b, I0, offset, N, d, f, lambd):
    """
    Modèle reliant x_m (en mètres) à l'intensité (adimensionnelle).
    
    1) On calcule l'angle theta = arctan((x_m - x0_m)/f).
    2) On applique le modèle fente(s) multiple(s).
    3) On multiplie par I0, on ajoute l'offset (fond).
    
    Paramètres:
      - x_m : positions en mètres
      - x0_m : centre du maximum en mètres
      - b : largeur de fente (m)
      - I0 : amplitude d'intensité
      - offset : fond
      - N : nombre de fentes
      - d : distance entre fentes (m)
      - f : distance focale / écran (m)
      - lambd : longueur d'onde (m)
    """
    theta = np.arctan((x_m - x0_m) / f)
    I_rel = modele_interference_multiple(theta, b, d, N, lambd)
    return I0 * I_rel + offset

###############################################################################
# LISTE DES FICHIERS ET DES PARAMÈTRES (à adapter à vos données réelles)
###############################################################################

data_files = [
    {
        "filename": "test1.txt",
        "N": 1,
        "d": 0.0,            # pas pertinent si N=1, on peut mettre n'importe quoi
        "b_init": 40e-6,     # estimation initiale
        "offset_init": 0.0,
        "comment": "Fente simple ~40 µm"
    },
    {
        "filename": "test2.txt",
        "N": 2,
        "d": 500e-6,        # distance entre fentes
        "b_init": 40e-6,
        "offset_init": 0.0,
        "comment": "Fentes doubles, b~40 µm, d=500 µm"
    },
    {
        "filename": "test3.txt",
        "N": 2,
        "d": 250e-6,
        "b_init": 40e-6,
        "offset_init": 0.0,
        "comment": "Fentes triples, b~40 µm, d=250 µm"
    },
    # etc. Ajoutez autant de dictionnaires que nécessaire
]

###############################################################################
# LISTES POUR RÉCOLTER LES RÉSULTATS
###############################################################################
all_b_fit      = []
all_x0m_fit    = []
all_I0_fit     = []
all_offset_fit = []
all_N          = []
all_d          = []
all_file_names = []

###############################################################################
# BOUCLE PRINCIPALE SUR LES FICHIERS
###############################################################################
plt.figure(figsize=(10, 6))  # figure où on superpose tout, par exemple

for i, data in enumerate(data_files):
    filename   = data["filename"]
    N_data     = data["N"]
    d_data     = data["d"]
    b_guess    = data["b_init"]
    off_guess  = data["offset_init"]
    comment    = data["comment"]

    # (1) Lecture du fichier
    y_raw = np.loadtxt(filename)
    
    # Indice des pixels : 0,1,2,...,len(y_raw)-1
    # Conversion en mètres : x_m = index * RESOLUTION
    nb_points = len(y_raw)
    x_m = np.arange(nb_points) * RESOLUTION
    
    # (2) Normalisation
    y_norm = normalisation_intensite(y_raw)

    # (3) Estimation de x0 (en mètres) comme la position du maximum
    idx_max  = np.argmax(y_raw)         # indice (en pixels)
    x0_guess = x_m[idx_max]            # conversion en mètres
    I0_guess = np.max(y_norm)          # l'intensité max (~1, après normalisation)

    # (4) Définition de la fonction pour curve_fit
    def fit_func(x, x0, b, I0, offset):
        return modele_complet(
            x, x0, b, I0, offset,
            N=N_data, d=d_data,
            f=F, lambd=LAMBDA
        )

    # (5) Ajustement
    pinit = [x0_guess, b_guess, I0_guess, off_guess]
    popt, pcov = curve_fit(fit_func, x_m, y_norm, p0=pinit, maxfev=10000)

    # Récupération des paramètres
    x0m_fit, b_fit, I0_fit, offset_fit = popt

    # Incertitudes (1 sigma)
    perr = np.sqrt(np.diag(pcov))
    err_x0m, err_b, err_I0, err_off = perr

    # Stockage pour analyse ultérieure
    all_b_fit.append(b_fit)
    all_x0m_fit.append(x0m_fit)
    all_I0_fit.append(I0_fit)
    all_offset_fit.append(offset_fit)
    all_N.append(N_data)
    all_d.append(d_data)
    all_file_names.append(filename)

    # (6) Calcul de la courbe ajustée pour le tracé
    x_fit = np.linspace(x_m.min(), x_m.max(), 500)  # un maillage plus fin en mètres
    y_fit = fit_func(x_fit, *popt)

    # (7) Trace sur la figure commune
    plt.plot(x_m, y_norm, '.', label=f"{comment} (exp)")
    plt.plot(x_fit, y_fit, '-', label=f"{comment} (fit)")

    # -- Affichage console des résultats --
    print("-----------------------------------------------------")
    print(f"Fichier: {filename} ({comment})")
    print(f"N = {N_data}, d = {d_data:.2e} m (si pertinent)")
    print("Paramètres ajustés :")
    print(f"  x0    = {x0m_fit:.3e} ± {err_x0m:.3e} m  (centre)")
    print(f"  b     = {b_fit:.3e} ± {err_b:.3e} m      (largeur)")
    print(f"  I0    = {I0_fit:.3f} ± {err_I0:.3f}      (amplitude)")
    print(f"  offset= {offset_fit:.3f} ± {err_off:.3f} ")
    
    # Calcul et affichage de la position du premier minimum pour les fentes simples
<<<<<<< HEAD
    if N_data == 1:  # Seulement pour les fentes simples
        x_min = (LAMBDA * F) / b_fit  # Calcul de la position du premier minimum
        print(f"  Position du premier minimum x_min = {x_min:.3e} m")
=======
    if N_data == 1: 
        x_min = (LAMBDA * F) / b_fit  
        err_x_min = (LAMBDA * F) / (b_fit**2) * err_b  
        print(f"  Position du premier minimum x_min = {x_min:.3e} ± {err_x_min:.3e} m")
>>>>>>> 5b75743bcc82ec1c4e42b645934a0db956f63295
    print("-----------------------------------------------------")

# Fin de la boucle

# Mise en forme de la figure commune
plt.xlabel("Position x (m)")
plt.ylabel("Intensité normalisée")
plt.title("Superposition de patrons pour différents fichiers (N, b, d)")
plt.legend(loc='best', fontsize='small')
plt.grid(True)
plt.show()


###############################################################################
# EXEMPLES D'ANALYSES SUPPLÉMENTAIRES
###############################################################################
# Vous disposez maintenant de :
#   all_b_fit, all_x0m_fit, all_I0_fit, all_offset_fit, all_N, all_d, all_file_names
#
# Vous pouvez par exemple :
#  - tracer la position du premier minimum en fonction de 1/b
#  - comparer la largeur ajustée b_fit avec une valeur attendue
#  - superposer l'enveloppe (fente simple) et le patron d'interférence (N>1)
#  - etc.