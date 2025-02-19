 # This Python file uses the following encoding: utf-8
import os, sys
from matplotlib import pylab
from pylab import *
 
# Exemple de régression linéaire avec les formules dans Bevington

# Lire les donneés
# Présumez un fichier avec trois colonnes, x, y, et l'erreur dans y
# Voir le fichier pente.py pour une version sans incertitude en y
x,y,dy=loadtxt('lindatxydy.dat',unpack=True)

# Régression linéaire avec les formules 6.11, 6.21, et 6.22 du livre de Bevington
# a,b = les paramètres du fit y=a+b*x; da, db, les incertitudes dans a et b
delta=sum((1/dy)**2)*sum(x**2/dy**2)-(sum(x/dy**2))**2
a=(sum(x**2/dy**2)*sum(y/dy**2)-sum(x/dy**2)*sum(x*y/dy**2))/delta
b=(sum(1/dy**2)*sum(x*y/dy**2)-sum(x/dy**2)*sum(y/dy**2))/delta
da=sqrt(sum(x**2/dy**2)/delta)
db=sqrt(sum(1/dy**2)/delta)
 
yf=a+b*x

print(u'Régression linéaire par formule de Bevington avec incertitude variable')
print(u'Paramètres   : décalage=%.2f  pente=%.2f \nIncertitudes : sigma(décal.)=%.2f sigma(pente)=%.2f' % (a,b,da,db))

# Graphique
title(u'Exemple de Régression Linéaire')
plot(x,y,'go')
plot(x,yf,'r.-')
errorbar(x,y,yerr=dy,fmt='.')
xlabel('x'); ylabel('y')
legend(['données','régression'])

show()
