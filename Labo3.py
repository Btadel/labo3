import numpy as np
import matplotlib as plt 
import matplotlib.pyplot as plt
import pandas as pd


taille_ptube,incertitude_ptube=0.298,0.005
taille_gtube,incertitude_gtube=0.498,0.005
incertitude_frequence=2.5

Temperature= 26.5+273.2
incertitude_temperature=0.5

R= 8.314
masses_molaires = {
    "Air": 28.97e-3,     
    "Argon": 39.948e-3,  
    "CO2": 44.01e-3,     
    "SF6": 146.06e-3     
}


# Données
tube_petit = {
    "Air": [[596.875, 1184.375, 1784.375, 2378.125, 2950], [596.875, 1184.375, 1784.375, 2378.875, 2950], [596.875, 1184.375, 1784.375, 2378.875, 2950]],
    "Argon": [[546.875, 1103.125, 1653.125, 2196.875, 2778.125], [550, 1106.25, 1656.25, 2203.125, 2787.5], [556.25, 1093.75, 1643.75, 2209.37, 2778.12]],
    "CO2": [[462.5, 925, 1384.375, 1871.875, 2331.25], [465.625, 918.75, 1378.125, 1837.5, 2325], [459.375, 918.75, 1384.375, 1843.75, 2325]],
    "SF6": [[231.25, 465.625, 696.875, 931.25, 1105.625], [231.25, 462.5, 693.75, 931.25, 1105.625], [231.25, 465.625, 696.875, 931.25, 1105.625]],
}

tube_grand = {
    "Air": [[250, 490.625, 731.25, 978.125, 1218.75], [250, 490.625, 737.5, 978.125, 1225], [250, 490.625, 737.5, 918.125, 1225]],
    "Argon": [[331.25, 656.25, 978.125, 1315.62, 1637.5], [328.125, 659.375, 978.125, 1318.75, 1631.25], [328.125, 653.125, 978.125, 1303.12, 1637.50]],
    "CO2": [[271.875, 543.75, 818.75, 1090.62, 1365.62], [271.875, 543.75, 818.75, 1090.62, 1365.62], [271.875, 543.75, 818.75, 1090.62, 1365.62]],
    "SF6": [[134.375, 275, 415.625, 550, 684.375], [134.375, 281.25, 412.5, 550, 687.5], [140.625, 278.125, 412.5, 546.875, 693.75]],
}



#calcule de la vitesse du son (tube avec ectrémitéer fermer )
def v_son (frequence,tailletube,incertitude_frequence,incertitude_tube):
    indices = np.arange(1, len(frequence) + 1)  # [1, 2, 3, ...]
    vitesse=2.0*frequence*tailletube
    propagation_inc=((2.0*frequence*incertitude_tube)**2+(2.0*tailletube*incertitude_frequence)**2)**0.5
    vitesse_corrigee = vitesse / indices
    propagation_corrigee = propagation_inc / indices

    return vitesse_corrigee,propagation_corrigee

def graph(donnees, tube):
 
    plt.figure(figsize=(10, 6))
    
    for gas, mesures in donnees.items():
        
        mesure = np.array(mesures)
        print(mesure)
        
        # Moyenne et l'incertitude (écart-type)
        moyenne = np.mean(mesure, axis=0)
        incertitude = np.std(mesure, axis=0)
        
        pics = np.arange(1, len(moyenne)+1)
        
        # Courbe
        plt.errorbar(pics, moyenne, yerr=incertitude, elinewidth=10,label=gas, capsize=30, marker='o')

    

    plt.rcParams.update({'font.size': 12})
    plt.title(f"Fréquences mesurées dans le {tube}")
    plt.xlabel("Pic")
    plt.ylabel("Fréquence (Hz)")
    plt.legend()
    plt.grid(True)
    plt.show()

graph(tube_grand,'caca')

def gamma_gaz(vitesse,Temperature,Masse_molaire,incertitude_vitesse,incertitude_temperature,R):
    gamma=((Masse_molaire*vitesse**2)/(R*Temperature))
    gammav_inc= (vitesse/Temperature)*(Masse_molaire/R)*2.0
    gammat_inc= -(Masse_molaire/R)*(vitesse**2/Temperature**2)
    gamma_inc=np.sqrt((gammav_inc*incertitude_vitesse)**2+(gammat_inc*incertitude_temperature)**2)
    
    return gamma, gamma_inc
    



results = []

for gaz, frequence_lists in tube_petit.items():
    #print(gaz)
    #print(frequence_lists)
    vitesses = []
    incertitudes_vitesses = []
    Masse_molaire = masses_molaires[gaz]
    gammas = []
    incertitudes_gammas = []

    for frequence in frequence_lists:
        
        frequence = np.array(frequence)
        #print(frequence)
        
        vitesse, incert_vitesse = v_son(frequence, taille_ptube, incertitude_frequence, incertitude_ptube)
        
        gamma, incert_gamma = gamma_gaz(vitesse, Temperature, Masse_molaire, incert_vitesse, incertitude_temperature, R)
        
        # Stocker les résultats
        vitesses.append(np.mean(vitesse))
        incertitudes_vitesses.append(np.mean(incert_vitesse))
        gammas.append(np.mean(gamma))
        incertitudes_gammas.append(np.mean(incert_gamma))
    
   
    mean_vitesse = np.mean(vitesses)
    mean_incert_vitesse = np.sqrt(np.sum(np.array(incertitudes_vitesses)**2)) / len(incertitudes_vitesses)

    mean_gamma = np.mean(gammas)
    mean_incert_gamma = np.sqrt(np.sum(np.array(incertitudes_gammas)**2)) / len(incertitudes_gammas)
    
   
    results.append({
        "Gaz": gaz,
        "Vitesse du son (m/s)": f"{mean_vitesse:.2f} ± {mean_incert_vitesse:.2f}",
        "Gamma (γ)": f"{mean_gamma:.2f} ± {mean_incert_gamma:.2f}"
    })


#results_df = pd.DataFrame(results)

#results_df.to_csv("resultats_cp_cv.csv", index=False)
#print(results_df)

