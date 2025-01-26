import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np

# Code (lab 6, interférence et diffraction) pour lire un image (et le bruit de fond) et extraire une courbe d'intensité en fonction de position sur le caméra.
# La position sur le caméra est en termes de pixel, il faut multiplier avec la taille de pixel pour la distance en m.
# Vous devez éditer ce fichier pour identifier les noms des images à évaluer et le nom du fichier à sauvegarder


if __name__ == '__main__':
        # Définitions
        imagename   ='test7.tif'		# image à évaluer (inclure "path" si ailleurs)
        bruitdefond ='bruit7.tif'	        # image du bruit de fond associé (inclure "path" si ailleurs)
        filename    ='test7.txt'           # nom de ficher pour écrire la courbe extrait des images
        # Lire les images
        image1      = mpimg.imread(imagename) 	#lire l'image imagename
        image1      = image1[:,:,0]
        image1      = np.float64(image1) 	# la convertir en format double
        bkgnd       = mpimg.imread(bruitdefond) # lire l'image de bruit de fond
        bkgnd       = bkgnd[:,:,0]
        bkgnd       = np.float64(bkgnd) 	# la convertir en format double

        # Vérifie qu'il n'y a pas de données saturées dans l'image
        # Le numériseur donne des intensités qui varient entre 0 (noir) à 255 (blanc) et il est important qu'aucun des pics ne soit saturé.
        if (np.max(image1) == 255):
                print("\n ATTENTION: votre image comprend des données saturées.")
                print("Il serait préférable de reprendre de nouvelles données.\n")

        cimage1     = image1-bkgnd 		# corriger l'image
        plt.imshow(cimage1) 			# montrer l'image

        # Choisir la région d’interêt pour sommer le signal du patron selon les colonnes
        # print("Cliquez une premiere fois n'importe où sur la fenêtre pour l'activer")
        print("Cliquez ensuite deux fois (haut et bas) pour sélectionner la région de l'image à intégrer")

        pts = plt.ginput(2)			# extrait deux points sur l'image
        pts_y = [x[1] for x in pts]		# extrait les valeurs y de l'image
        y1= int(round(float(min(pts_y))))	# la plus petite 
        y2= int(round(float(max(pts_y))))	# la plus grande 

        # NB, si vous avez plusieurs images à évaluer, toujours à la même hauteur sur la caméra (disons entre 300 et 400), 
        # vous pouvez remplacer les lignes précédentes par une simple déclaration :
        #y1 = 400
        #y2 = 700

        # sommation sur la dimension 0 (lignes)
        y= np.sum(cimage1[y1:y2,:],axis=0)

        plt.close()				# fermer la fenêtre précédente
        ncol= y.shape[0]			# déterminer le nombre de colonnes dans l'image
        x= np.arange(ncol)			# créer un vecteur de valeurs comprises entre 0 et ncol -1 
        fig= plt.figure()
        plt.plot(x,y)				# affiche le patron
        plt.ylabel('Instensité (comptes)')
        plt.xlabel('Position sur le détecteur (pixel)')
        plt.show()
        np.savetxt(filename,y)


