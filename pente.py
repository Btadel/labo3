 # This Python file uses the following encoding: utf-8
import os, sys
from pylab import *
 
# Exemple de régression linéaire avec les formules dans Bevington
# Voir la fonction pente-err pour la version avec incertitudes variables dans y

# Lire les donneés
# Présumez un fichier avec deux colonnes, x, y, et l'erreur dans y
x,y=loadtxt('lindatxy.dat',unpack=True)
N=x.size

# Régression linéaire avec les formules 6.11,  6.23 du livre de Bevington
# a,b = les paramètres du fit y=a+b*x; da, db, les incertitudes dans a et b
delta=N*sum(x**2)-(sum(x))**2
a=(sum(x**2)*sum(y)-sum(x)*sum(x*y))/delta
b=(N*sum(x*y)-sum(x)*sum(y))/delta
s2=sum((y-a-b*x)**2)/(N-2)
da=sqrt(s2/delta*sum(x**2))
db=sqrt(N*s2/delta)
err=sqrt(s2)
 
yf=a+b*x

print(u'Régression linéaire par formule de Bevington (incertitudes uniformes)')
print(u'Paramètres   : décalage=%.2f pente=%.2f \nIncertitudes : sigma(décal.)=%.2f sigma(pente)=%.2f sigma=%.3f' % (a,b,da,db,err))

# Graphique
title(u'Exemple de Régression Linéaire')
plot(x,y,'go'); plot(x,yf,'r.-')
xlabel('x'); ylabel('y')
legend(['données','régression'])

show()
