#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 11:07:04 2025

@author: rosalie
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

#DONNEES
#pas du reseau du manufacturier
d_manu=1/15000
#longueurs d'onde doublet jaune
lam_jaune1=576.96e-9 
lam_jaune2=579.07e-9 

#tableau couleurs des raies
raies=['mauve','bleue','verte','lime','jaune','rouge']

#incertitude angle (deg)
inc_ang=39/60

# %%
#MESURES
#partie 2
#angles de deviation a droite (deg)
phi_droite=np.array([64+9/60, 63, 60+22/60, 58+55/60, 57+55/60, 55+20/60])

#angles de deviation a gauche (deg)
phi_gauche=np.array([92+18/60, 93+30/60, 96+4/60, 97+29/60, 98+32/60, 101])

#partie 3
alpha_droite=18+14/60
alpha_gauche=138

#partie 4
#position angulaire extreme (deg)
ang_ext=np.array([74+30/60, 79+46/60, 77+45/60, 78+6/60, 79, 78+39/60])

#ligne de visee
ang_vis=np.array([148+34/60, 146+51/60, 146+16/60, 145+31/60, 145+58/60, 144+26/60])

# %%
#CALCULS
#angles de diffraction
ang_diff=np.abs(phi_droite-phi_gauche)/2

#d experimental
d_exp=((lam_jaune1+lam_jaune2)/2)/np.sin(ang_diff[4]*(np.pi/180))

#lambda nominal et experimental
lam_nom=d_manu*np.sin(ang_diff*(np.pi/180))
lam_exp=d_exp*np.sin(ang_diff*(np.pi/180))

#minimum de deviation
delm=(ang_vis-ang_ext)*(np.pi/180)

#angle alpha
alpha=((alpha_gauche-alpha_droite)/2)*(np.pi/180)

#indice de diffraction
n=(np.sin((delm+alpha)/2))/(np.sin(alpha/2))

#graphique
x_exp=1/lam_exp**2
x_nom=1/lam_nom**2

# %%
#Graphique experimental
color=['purple', 'blue', 'green', 'limegreen', 'yellow', 'red']
for i in range(0,6):
    x=x_exp[i]
    y=n[i]
    
    plt.plot(x,y, marker='*', color=color[i], label=raies[i])

plt.title('Indice de refraction en fonction de l\'inverse \n du carre de la longueur d\'onde (avec d experimental)')
plt.xlabel('1/lam**2')
plt.ylabel('n')
plt.legend()
plt.grid()
plt.show()

#%%
#Graphique nominal
for i in range(0,6):
    x2=x_nom[i]
    y=n[i]
    
    plt.plot(x2,y, marker='*', color=color[i], label=raies[i])

plt.title('Indice de refraction en fonction de l\'inverse \n du carre de la longueur d\'onde(avec d nominal)')
plt.xlabel('1/lam**2')
plt.ylabel('n')
plt.legend()
plt.grid()
plt.show()
#%%
#Prendre la partie lineaire pour les coefficients de Cauchy (pente=B, ordonnee a l'origine=A)

x_reg1=[x_exp[3],x_exp[2],x_exp[1],x_exp[0]]
y_reg1=[n[3],n[2],n[1],n[0]]

res1 = linregress(x_reg1,y_reg1)
B1=res1[0] #car element 0 est la pente
A1=res1[1] #car element 1 est l'ordonnee a l'origine

x_reg2=[x_nom[3],x_nom[2],x_nom[1],x_nom[0]]
y_reg2=[n[3],n[2],n[1],n[0]]

res2 = linregress(x_reg2,y_reg2)
B2=res2[0] #car element 0 est la pente
A2=res2[1] #car element 1 est l'ordonnee a l'origine

n_Cauchy_exp=A1+B1*x_exp
n_Cauchy_nom=A2+B2*x_nom

    
