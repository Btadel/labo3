import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Constantes
lambda_nm = 632.8  # longueur d'onde en nm
lambda_m = lambda_nm * 1e-9  # conversion en m
a = 0.04e-3  # largeur de la fente en m
d = 0.25e-3  # distance entre les fentes en m

# Fonction pour calculer I/I0 pour la diffraction (fente unique)
def diffraction_intensity(theta, a, lambda_m):
    beta = (np.pi * a * np.sin(theta)) / lambda_m
    I_I0 = (np.sin(beta) / beta)**2
    I_I0[beta == 0] = 1  # éviter les NaN lorsque beta est nul
    return I_I0

# Fonction pour calculer I/I0 pour l'interférence et diffraction (deux fentes)
def interference_diffraction_intensity(theta, a, d, lambda_m):
    beta = (np.pi * a * np.sin(theta)) / lambda_m
    delta = (np.pi * d * np.sin(theta)) / lambda_m
    I_I0 = ((np.sin(beta) / beta)**2) * (np.cos(delta)**2)
    I_I0[beta == 0] = 1  # éviter les NaN lorsque beta est nul
    return I_I0

# Génération des données
theta = np.linspace(-np.pi / 16, np.pi / 16, 1000)  # θ en radians, zoom sur le graphique

# Calcul des intensités
I_diffraction = diffraction_intensity(theta, a, lambda_m)
I_interference_diffraction = interference_diffraction_intensity(theta, a, d, lambda_m)

# Création du graphique
plt.figure(figsize=(10, 6))
plt.plot(theta, I_diffraction, label="Diffraction (une fente)", linewidth=1)
plt.plot(theta, I_interference_diffraction, label="Interférence|diffraction (deux fentes)", linewidth=1)
plt.title("Interférence et Diffraction")
plt.xlabel("Angle θ (radians)")
plt.ylabel("Intensité relative (I/I0)")
plt.legend()
plt.grid(True)
plt.show()

# Sauvegarde en PDF
#pdf_filename = "interference_diffraction.pdf"
#with PdfPages(pdf_filename) as pdf:
#    pdf.savefig()  # Sauvegarde la figure actuelle dans le fichier PDF
#    plt.close()

#print(f"Graphique sauvegardé dans le fichier {pdf_filename}")
