import numpy as np
import matplotlib.pyplot as plt

def reponse_theorique_rlc(frequence, R, L, C):
    """
    Calcule (V/E, phi) théoriques pour un circuit RLC série, selon la fréquence donnée.
    
    Paramètres :
    - frequence : fréquences en Hz (array)
    - R : résistance en ohms
    - L : inductance en henrys
    - C : capacité en farads
    
    Retourne :
    - VE_theorique : rapport tension mesurée sur tension source
    - phi_theorique : déphasage en radians
    """
    pulsation = 2 * np.pi * frequence
    reactance_L = pulsation * L       #  inductive
    reactance_C = 1.0 / (pulsation * C)  # capacitive
    
    # Magnitude de l'impédance
    impedance = np.sqrt(R**2 + (reactance_L - reactance_C)**2)
    # Rapport V/E théorique
    VE_theorique = R / impedance
    # Déphasage théorique
    phi_theorique = np.arctan((reactance_L - reactance_C) / R)
    
    return VE_theorique, phi_theorique

def tracer_rlc(frequence, tension_source_exp, tension_resistance_exp, decalage_temps_exp,
               R, L, C, incertitude_V=0.005, incertitude_E=0.005, incertitude_t=0.0005,
               titre_supplementaire=""):
    """
    Compare (V/E, phi) expérimentaux et théoriques pour un circuit RLC.
    
    Paramètres :
    - frequence : array-like (Hz)
    - tension_source_exp : array-like (V) - tension source mesurée
    - tension_resistance_exp : array-like (V) - tension mesurée sur la résistance
    - decalage_temps_exp : array-like (s) - décalage temporel entre e(t) et v(t)
    - R : résistance en ohms
    - L : inductance en henrys
    - C : capacité en farads
    - incertitude_V, incertitude_E, incertitude_t : incertitudes absolues (optionnelles)
    - titre_supplementaire : texte supplémentaire pour le titre du graphique
    """
    # --- THÉORIE ---
    VE_theorique, phi_theorique = reponse_theorique_rlc(frequence, R, L, C)

    # --- EXPÉRIMENTAL ---
    VE_experimental = tension_resistance_exp / tension_source_exp
    # Déphasage expérimental (convention : - 2π * f * deltaT)
    phi_experimental = -2.0 * np.pi * (decalage_temps_exp * frequence)
    
    # --- INCERTITUDES ---
    VE_incertitude = VE_experimental * np.sqrt((incertitude_V / tension_resistance_exp)**2 + (incertitude_E / tension_source_exp)**2)
    phi_incertitude = 2.0 * np.pi * frequence * incertitude_t

    # --- TRACÉ : Rapport V/E ---
    plt.figure(figsize=(8, 6))
    plt.errorbar(frequence, VE_experimental, yerr=VE_incertitude, fmt='o', label='Expérimental', capsize=4)
    plt.plot(frequence, VE_theorique, label='Théorique', color='red')
    plt.xscale('log')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('V/E')
    plt.title(f'Circuit RLC : Comparaison V/E {titre_supplementaire}')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.show()

    # --- TRACÉ : Déphasage ---
    plt.figure(figsize=(8, 6))
    plt.errorbar(frequence, phi_experimental, yerr=phi_incertitude, fmt='o', label='Expérimental', capsize=4)
    plt.plot(frequence, phi_theorique, label='Théorique', color='red')
    plt.xscale('log')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Phase (rad)')
    plt.title(f'Circuit RLC : Comparaison du déphasage {titre_supplementaire}')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.show()



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

frequence1 = mesures1[:,0]  
tension_source_1 = mesures1[:,1]  
tension_resistance_1 = mesures1[:,2]  
decalage_temps_1 = mesures1[:,3] * 1e-3  

tracer_rlc(frequence1, tension_source_1, tension_resistance_1, decalage_temps_1,
           R=1000, L=0.5, C=4e-9,
           titre_supplementaire="(R=1000Ω, L=500mH, C=4nF)")

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

frequence2 = mesures2[:,0]
tension_source_2 = mesures2[:,1]
tension_resistance_2 = mesures2[:,2]
decalage_temps_2 = mesures2[:,3] * 1e-3  

tracer_rlc(frequence2, tension_source_2, tension_resistance_2, decalage_temps_2,
           R=500, L=0.2, C=20e-9,
           titre_supplementaire="(R=500Ω, L=200mH, C=20nF)")