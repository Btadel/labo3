import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

filename = 'test2.txt'
y = np.loadtxt(filename)  # intensités
x = np.arange(len(y))     # pixels

RESOLUTION = 5.2e-6  #taille des pixels en metre
LAMBDA = 632.8e-9    #longueur d'onde du laser metre
D = 500e-6           #distance entre les fentes metre
N = 1                #nombre de fentes
F = 0.1              #nistance focale metre

#transformation pixel à distance
def transformation_pixels_en_distance(pixels, resolution):
    return pixels * resolution

# Normalisation 
def normalisation_intensite(intensites):
    return intensites / np.max(intensites)


def full(x, x_0, f, b, N, d, lambd, resolution):
    theta = np.arctan((x - x_0) * resolution / f)
    alpha = np.pi * b * np.sin(theta) / lambd
    beta = np.pi * d * np.sin(theta) / lambd
    alpha = np.where(alpha == 0, 1e-12, alpha)
    beta = np.where(beta == 0, 1e-12, beta) # Éviter division par 0
    if N == 1:
        theo = (np.sin(alpha) / alpha) ** 2
    else:
        theo = (np.sin(alpha) / alpha) ** 2 * (np.sin(N * beta) / (N * np.sin(beta))) ** 2
    return theo

# Fonction théorique simplifiée 
def theob(x, x_0, b, I_0, off):
    resolution = RESOLUTION
    lambd = LAMBDA
    f = F
    d = D
    N = 2
    y = I_0 * full(x, x_0, f, b, N, d, lambd, resolution) + off
    return y



# Détermination automatique de x_0 (centre)
x_0_auto = np.argmax(y)  # Position du maximum d'intensité
print(f"x_0 déterminé automatiquement : {x_0_auto} (pixel)")

# Transformation et normalisation
y_normalisees = normalisation_intensite(y)

# Ajustement manuel
plt.plot(x, y_normalisees, 'k.', label='Données expérimentales')
b = 40e-6
ytheo = theob(x, x_0_auto, b, np.max(y), 0)
plt.plot(x, ytheo / np.max(ytheo), 'r-', label='Ajustement manuel (x0 auto)')
plt.legend()
plt.xlabel('Position (pixels)')
plt.ylabel('Intensité normalisée')
plt.title('Ajustement manuel avec x0 automatique')
plt.grid()
plt.show()

# Ajustement avec curve_fit
pinit = [x_0_auto, 40e-6, np.max(y), 0]
popt, pcov = curve_fit(theob, x, y_normalisees, p0=pinit, maxfev=10000)

x0_fit, b_fit, I0_fit, offset_fit = popt
err_b = np.sqrt(np.diag(pcov)[1])

print(f"x0 (centre) = {x0_fit:.2f} pixels")
print(f"b (largeur fente) = {b_fit:.2e} ± {err_b:.2e} m")
print(f"I0 (intensité max) = {I0_fit:.2f}")
print(f"Offset (décalage) = {offset_fit:.2f}")

# Courbe ajustée
y_fit = theob(x, *popt)

plt.plot(x, y_normalisees, 'k.', label='Données expérimentales')
plt.plot(x, y_fit, 'r-', label='Ajustement curve_fit')
plt.legend()
plt.xlabel('Position (pixels)')
plt.ylabel('Intensité normalisée')
plt.title('Ajustement avec curve_fit')
plt.grid()
plt.show()
