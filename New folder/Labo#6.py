import numpy as np
import matplotlib.pyplot as plt


theta=np.arange(0,360,1,dtype=float)
lombda=6,328e-7 #nm
N=1
b=1
d=1
alpha=(np.pi * b * np.sin(theta))/lombda

beta=(np.pi * d * np.sin(theta))/lombda
IsurI=((np.sin(alpha)/alpha)**2)*((np.sin(N*beta)/N*np.sin(beta))**2)

def formule(lombda,b,d,theta):
    graph=[]
    theta2=[]
    for angle in theta():
        theta2.append(angle)
        N=1
        alpha=(np.pi * b * np.sin(theta))/lombda

        beta=(np.pi * d * np.sin(theta))/lombda
        IsurI=graph.append(((np.sin(alpha)/alpha)**2)*((np.sin(N*beta)/N*np.sin(beta))**2))
    plt.plot(theta2,graph)
    
