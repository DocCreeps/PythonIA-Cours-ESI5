# -*- coding: utf-8 -*-
"""
Created on Wed May  4 09:44:11 2022
Reconnaissance chiffre
@author: BEBO
"""

#!/usr/bin/env python
# coding: utf-8

# # Reconnaissance de chiffres manuscrits avec scikit-learn
# @ Colin Bernet IP2I Lyon modifie B.Bonche

# ## Échantillon de données de chiffres manuscrits
# 
# Une version basse résolution de cet échantillon est fourni avec scikit-learn.
# 
# On commence par charger l'échantillon : 
    
    
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

digits = datasets.load_digits()
print("dataset\n",digits)
print("\n")

# Nb Images = 50

for i in range (50):
  print("\n")
  print(digits.images[i])
  plt.imshow(digits.images[i],cmap='binary')
  plt.title(digits.target[i])
  plt.axis('off')
  plt.show()
  print("\n")


# Puis on imprime la première image : 

print("Premières images")
print("\n")
print(digits.images[0])
print("\n")


# Comme toutes les images de l'échantillon, celle-ci est une image de 8x8 pixels, noir et blanc 
#(un seul niveau de couleur par pixel). On peut l'afficher de la manière suivante,
# en indiquant également l'étiquette correspondante (le chiffre auquel correspond l'image) : 


plt.imshow(digits.images[0],cmap='binary')
plt.title(digits.target[0])
plt.axis('off')
plt.show()

input() # pour avour un arret dans le programme

# Nous allons entraîner un réseau de neurones simple à reconnaître les chiffres dans ces images.
# Ce réseau va prendre en entrée des tableaux 1D de 8x8=64 valeurs. Nous devons donc convertir nos images 2D en tableaux 1D :



x = digits.images.reshape((len(digits.images), -1))
# x contient toutes les images en version 1D.
# nous affichons ici la première, que nous avons déjà vue : 
    
print(x[0])
print("\n")



# Le réseau va agir comme une fonction permettant de passer d'un tableau de 64 valeurs en entrée à une valeur en sortie,
#son estimation du chiffre. Voici les valeurs de sortie : 


y = digits.target


print("\nNb images dans l'echantillon ",len(digits.images)) # Nombre d'images dans le set d'échantillons


# ## Définition et entraînement du réseau

# Nous décidons de créer un réseau de neurones relativement simple, avec une seule couche de 15 neurones : 



from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(15,))


# Nous allons entraîner ce réseau sur les 1000 premières images de notre échantillon, 
# et réserver les images suivantes pour tester les performances du réseau : 



x_train = x[:1000]
y_train = y[:1000]

x_test = x[1000:]
y_test = y[1000:]



mlp.fit(x_train, y_train) #  entrainement


# Et voilà ! nous pouvons maintenant regarder ce que donne le réseau pour les images suivantes, 
#qui n'ont pas été vues par le réseau lors de l'entraînement : 


mlp.predict(x_test[:10])


y_test[:10]


# Pour les 10 premières images de test, les estimations sont excellentes ! 

# ## Performances

# Nous pouvons maintenant évaluer le réseau pour toutes les images de test



y_pred = mlp.predict(x_test)


# Puis rechercher les images pour lesquelles le réseau s'est trompé : 



error = (y_pred != y_test)


# Voici le calcul du taux d'erreur :


np.sum(error) / len(y_test)

print("\nNb Erreurs", np.sum(error/len(y_test)))


# environ 11%, ce qui veut dire que 89% des prédictions sont correctes: 

# Nous pouvons enfin sélectionner les mauvaises prédictions pour les afficher :


x_error = x_test[error].reshape((-1, 8,8))
y_error = y_test[error]
y_pred_error = y_pred[error]
i = 1
plt.imshow(x_error[i],cmap='binary')
plt.title(f'cible: {y_error[i]}, prediction: {y_pred_error[i]}')
plt.axis('off')
plt.show()


# Comme on peut le voir, il est difficile de classifier ces images, même pour un humain.
# 
# Pour de meilleures performances, il faudrait utiliser des images de plus haute résolution 
#et un réseau de neurones plus complexe, comme un réseau convolutionnel.
# 