import numpy as np
import matplotlib.pyplot as plt

# Indices
n1 = 1.0      # air
n2 = 1.51     # BK7

# Génère un tableau d'angles d'incidence (en degrés)
theta1_deg = np.arange(20, 81, 1)  # de 20° à 80° inclus
theta1_rad = np.radians(theta1_deg)

# Calcul de theta2 via la loi de Snell
# n1 sin(theta1) = n2 sin(theta2)  =>  theta2 = arcsin((n1/n2)*sin(theta1))
theta2_rad = np.arcsin((n1 / n2) * np.sin(theta1_rad))

# Calcul de la réflectance Rs et Rp
# R_s = ((n1 cos theta1 - n2 cos theta2)/(n1 cos theta1 + n2 cos theta2))^2
# R_p = ((n2 cos theta1 - n1 cos theta2)/(n2 cos theta1 + n1 cos theta2))^2

cos_theta1 = np.cos(theta1_rad)
cos_theta2 = np.cos(theta2_rad)

Rs = ((n1 * cos_theta1 - n2 * cos_theta2) / (n1 * cos_theta1 + n2 * cos_theta2))**2
Rp = ((n2 * cos_theta1 - n1 * cos_theta2) / (n2 * cos_theta1 + n1 * cos_theta2))**2

# Tracé des courbes
plt.figure()
plt.plot(theta1_deg, Rs, label='R_s (polarisation perpendiculaire)')
plt.plot(theta1_deg, Rp, label='R_p (polarisation parallèle)')

plt.xlabel('Angle d\'incidence θ1 (degrés)')
plt.ylabel('Réflectance R1')
plt.title('Réflectance Fresnel pour un dioptre air-BK7')
plt.legend()
plt.grid(True)
plt.show()
