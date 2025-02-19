import numpy as np
import matplotlib.pyplot as plt

def theoretical_response_rlc(f, R, L, C):
    """
    Calcule (V/E, phi) théoriques pour un circuit RLC série, selon la fréquence f.
    f : fréquences en Hz (array)
    R : ohms
    L : henrys
    C : farads
    
    Retourne : VE_th, phi_th
    """
    omega = 2 * np.pi * f
    X_L = omega * L       # Réactance inductive
    X_C = 1.0 / (omega * C)  # Réactance capacitive
    
    # Magnitude de l'impédance
    Z = np.sqrt(R**2 + (X_L - X_C)**2)
    # Rapport V/E théorique
    VE_th = R / Z
    # Déphasage théorique
    phi_th = np.arctan((X_L - X_C) / R)
    
    return VE_th, phi_th

def plot_rlc_data(f, E_exp, V_exp, deltaT_exp, R, L, C,
                  V_err=0.005, E_err=0.005, t_err=0.0005,
                  title_suffix=""):
    """
    Compare (V/E, phi) expérimentaux et théoriques pour un circuit RLC.
    
    Paramètres
    ----------
    f : array-like (Hz)
    E_exp : array-like (V) - tension source mesurée
    V_exp : array-like (V) - tension R mesurée
    deltaT_exp : array-like (s) - décalage temporel entre e(t) et v(t)
    R : ohms
    L : henrys
    C : farads
    V_err, E_err, t_err : incertitudes absolues (optionnelles)
    title_suffix : str - texte supplémentaire pour le titre
    """
    # --- THÉORIE ---
    VE_th, phi_th = theoretical_response_rlc(f, R, L, C)

    # --- EXPÉRIMENTAL ---
    VE_exp = V_exp / E_exp
    # Déphasage expérimental (convention : - 2π * f * deltaT)
    phi_exp = -2.0 * np.pi * (deltaT_exp * f)
    
    # --- INCERTITUDES ---
    # (modèle simple de propagation)
    VE_err = VE_exp * np.sqrt((V_err / V_exp)**2 + (E_err / E_exp)**2)
    # d(phi)/d(deltaT) = -2π f  => incert. ~ 2π f * t_err
    phi_err = 2.0 * np.pi * f * t_err

    # --- TRACÉ : Rapport V/E ---
    plt.figure(figsize=(8, 6))
    plt.errorbar(f, VE_exp, yerr=VE_err, fmt='o', label='Expérimental', capsize=4)
    plt.plot(f, VE_th, label='Théorique', color='red')
    plt.xscale('log')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('V/E')
    plt.title(f'Circuit RLC : Comparaison V/E {title_suffix}')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.show()

    # --- TRACÉ : Déphasage ---
    plt.figure(figsize=(8, 6))
    plt.errorbar(f, phi_exp, yerr=phi_err, fmt='o', label='Expérimental', capsize=4)
    plt.plot(f, phi_th, label='Théorique', color='red')
    plt.xscale('log')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Phase (rad)')
    plt.title(f'Circuit RLC : Comparaison du déphasage {title_suffix}')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.show()


# ====================================================
#   2) UTILISATION POUR VOS DEUX JEUX DE MESURES RLC
# ====================================================

# -----------------------------------------
# Mesures 1 : R = 1000 Ω, L = 500 mH, C = 4.0 nF
# -----------------------------------------
mesures1 = np.array([
    [  100,   3.5264,  0.09203,  2.4   ],
    [  300,   3.5264,  0.27323,  0.836 ],
    [ 2000,   3.5273,  0.26774,  0.116 ],
    [ 2500,   3.5253,  0.4599,   0.090 ],
    [ 3000,   3.5120,  1.0062,   0.100 ],
    [ 3500,   3.3948,  2.7565,   0.012 ],
    [ 3750,   3.4842,  1.5998,   0.090 ],
    [ 4000,   3.3703,  1.01446,  0.04760],
    [ 5000,   3.5233,  0.41102,  0.04600],
    [10000,   3.5223,  0.11480,  0.02560]
    
    
    
])

# On sépare les colonnes
f1       = mesures1[:,0]   # Hz
E_exp_1  = mesures1[:,1]   # V
V_exp_1  = mesures1[:,2]   # V
dt_1_ms  = mesures1[:,3]   # (ms)

# Conversion en secondes si nécessaire :
dt_1 = dt_1_ms * 1e-3   # si vos dt sont effectivement en ms

plot_rlc_data(f1, E_exp_1, V_exp_1, dt_1,
              R=1000, L=0.5, C=4e-9,
              title_suffix="(R=1000Ω, L=500mH, C=4nF)")

# -----------------------------------------
# Mesures 2 : R = 500 Ω, L = 200 mH, C = 20.0 nF
# -----------------------------------------
mesures2 = np.array([
    [  100, 3.5261, 0.22156, 2.480 ],
    [  500, 3.5260, 0.11501, 0.512 ],
    [ 2000, 3.4839, 1.16435, 0.153 ],
    [ 2100, 3.4594, 1.4433,  0.153 ],
    [ 2200, 3.4161, 1.8284,  0.159 ],
    [ 2450, 3.2431, 2.8530,  0.202 ],
    [ 2300, 3.3439, 2.3320,  0.262 ],
    [ 2400, 3.2617, 2.7652,  0.226 ],
    [ 2500, 3.2507, 2.8145,  0.212 ],
    [ 2600, 3.3151, 2.4773,  0.203 ]    
])

f2       = mesures2[:,0]
E_exp_2  = mesures2[:,1]
V_exp_2  = mesures2[:,2]
dt_2_ms  = mesures2[:,3]   # idem, probablement en ms

# Conversion en secondes
dt_2 = dt_2_ms * 1e-3

plot_rlc_data(f2, E_exp_2, V_exp_2, dt_2,
              R=500, L=0.2, C=20e-9,
              title_suffix="(R=500Ω, L=200mH, C=20nF)")
