import numpy as np
import matplotlib.pyplot as plt

def graph(donnees, tube):
 
    plt.figure(figsize=(10, 6))
    
    for gas, mesures in donnees.items():
        
        mesure = np.array(mesures)
        
        # Moyenne et l'incertitude (écart-type)
        moyenne = np.mean(mesure, axis=0)
        incertitude = np.std(mesure, axis=0)
        
        pics = np.arange(1, len(moyenne)+1)
        
        # Courbe
        plt.errorbar(pics, moyenne, yerr=incertitude, label=gas, capsize=10,elinewidth=2, marker='o',markersize=6)
    
    plt.title(f"Fréquences mesurées dans le {tube}")
    plt.xlabel("Pic")
    plt.ylabel("Fréquence (Hz)")
    plt.legend()
    plt.grid(True)
    plt.show()

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

graph(tube_petit, "petit tube")
graph(tube_grand, "grand tube")