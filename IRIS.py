import numpy as np
import pandas

# Initialisation des donnnées
data = pandas.read_csv("iris.csv", header=0, delimiter=";").to_numpy()
x_enter = data[:, [0, 1, 2, 3]]
y = data[:, [4]]

# Changement de l'échelle de nos valeurs pour être entre 0 et 1
x_enter = x_enter/np.amax(x_enter, axis=0)
X = np.split(x_enter, [149])[0] # Données sur lesquelles on va s'entrainer, les 150 premières de notre matrice
xPrediction = np.split(x_enter, [149])[1] # Donnée sur laquel on veut trouver la classe

class Neural_Network(object):
    def __init__(self):
        
        #Nos paramètres
        self.inputSize = 4 # Nombre de neurones d'entrer
        self.outputSize = 1 # Nombre de neurones de sortie
        self.hiddenSize = 4 # Nombre de neurones cachés

        #Nos poids
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (2x3) Matrice de poids entre les neurones d'entrer et cachés
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) Matrice de poids entre les neurones cachés et sortie


    #Fonction de propagation avant
    def forward(self, X):
        self.z = np.dot(X, self.W1) # Multiplication matricielle entre les valeurs d'entrer et les poids W1
        self.z2 = self.sigmoid(self.z) # Application de la fonction d'activation (Sigmoid)
        self.z3 = np.dot(self.z2, self.W2) # Multiplication matricielle entre les valeurs cachés et les poids W2
        o = self.sigmoid(self.z3) # Application de la fonction d'activation, et obtention de notre valeur de sortie final
        return o

    # Fonction d'activation
    def sigmoid(self, s):
        return 1/(1 + np.exp(-s))
    
    # Dérivée de la fonction d'activation
    def sigmoidPrime(self, s):
        return s * (1 - s)
      
    #Fonction de rétropropagation
    def backward(self, X, y, o):
    
        self.o_error = y - o # Calcul de l'erreur
        self.o_delta = self.o_error*self.sigmoidPrime(o) # Application de la dérivée de la sigmoid à cette erreur
        
        self.z2_error = self.o_delta.dot(self.W2.T) # Calcul de l'erreur de nos neurones cachés 
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # Application de la dérivée de la sigmoid à cette erreur
        
        self.W1 += X.T.dot(self.z2_delta) # On ajuste nos poids W1
        self.W2 += self.z2.T.dot(self.o_delta) # On ajuste nos poids W2
      
    #Fonction d'entrainement 
    def train(self, X, y):
  
        o = self.forward(X)
        self.backward(X, y, o)
    
    #Fonction de prédiction
    def predict(self):
  
        print("Donnée prédite apres entrainement: ")
        print("Entrée : \n" + str(xPrediction))
        print("Sortie : \n" + str(self.forward(xPrediction)))
        
        if(self.forward(xPrediction) < 1/3):
            print("La fleur est de classe Iris-setosa ! \n")
        elif(self.forward(xPrediction) < 2/3):
            print("La fleur est de classe Iris-versicolor ! \n")
        else:
            print("La fleur est de classe Iris-virginica ! \n")

NN = Neural_Network()

for i in range(1000): #Choisissez un nombre d'itération, attention un trop grand nombre peut créer un overfitting !
    print("# " + str(i) + "\n")
    print("Valeurs d'entrées: \n" + str(X))
    print("Sortie actuelle: \n" + str(y))
    print("Sortie prédite: \n" + str(np.matrix.round(NN.forward(X),2)))
    print("\n")
    NN.train(X,y)

NN.predict()


