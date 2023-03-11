

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 11:15:29 2022

@author: Doria
"""

import numpy as np
import pandas as pd

data_bears = pd.read_csv("dataset Bears.csv", usecols=['AGE', 'MONTH', 'SEX', 'HEADLEN', 'HEADWTH', 'NECK', 'LENGTH', 'CHEST'], sep=';')
data_bearsWeight = pd.read_csv("dataset Bears.csv", usecols = ['WEIGHT2'], sep=';')
data_bears_actualWeight = pd.read_csv("dataset Bears.csv", usecols = ['WEIGHT'], sep=';')

randage = np.random.randint(data_bears['AGE'].min(), data_bears['AGE'].max() + 1,1)
randMonth = np.random.randint(data_bears['MONTH'].min(), data_bears['MONTH'].max()+ 1,1)
randSex = np.random.randint(data_bears['SEX'].min(), data_bears['SEX'].max()+ 1,1)
randHeadlen = np.random.randint(data_bears['HEADLEN'].min(), data_bears['HEADLEN'].max()+ 1,1)
randHeadwth = np.random.randint(data_bears['HEADWTH'].min(), data_bears['HEADWTH'].max()+ 1,1)
randNeck = np.random.randint(data_bears['NECK'].min(), data_bears['NECK'].max()+ 1,1)
randLength = np.random.randint(data_bears['LENGTH'].min(), data_bears['LENGTH'].max()+ 1,1)
randChest = np.random.randint(data_bears['CHEST'].min(), data_bears['CHEST'].max()+ 1,1)


data_bears.loc[len(data_bears.index)] = [randage,randMonth,randSex,randHeadlen,randHeadwth,randNeck,randLength,randChest] 



# Traitement des ours
x_enter = np.array(data_bears, dtype = float) 
y = np.array(data_bearsWeight, dtype= float) 
x_enter = x_enter/np.amax(x_enter,axis = 0)

X = np.split(x_enter,[len(data_bearsWeight)])[0] 
xPrediction = np.split(x_enter,[len(data_bearsWeight)])[1]

class Neural_Network(object):
    
    def __init__(self):
        self.inputSize = 8 # nombre de couches du R.N.
        self.outputSize = 1
        self.hiddenSize = 9
        
        self.w1 = np.random.randn(self.inputSize,self.hiddenSize) # poids premiere matrice 2x3
        self.w2 = np.random.randn(self.hiddenSize,self.outputSize) # poids deuxième matrice 3x1
        
    def sigmoid(self,s):   # choix de la sigmoid aussi tgh
            return 1/(1 + np.exp(-s))
        
    def sigmoidPrime(self,s): #dérivée de la fonction sigmoid
            return s * (1-s)
        
    def forward(self,X):
        self.z = np.dot(X,self.w1)  # z valeur d'entrée
        self.z2 = self.sigmoid(self.z)  # Z2 valeur cachée
        self.z3 = np.dot(self.z2,self.w2) # z3 valeur de sortie
        O = self.sigmoid(self.z3)
        return O
 
   
    
    def backward(self,X,y,O):
        self.o_error = y - O
        self.o_delta = self.o_error * self.sigmoidPrime(O)
        
        self.z2_error = self.o_delta.dot(self.w2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        
        self.w1 += X.T.dot(self.z2_delta)
        self.w2 += self.z2.T.dot(self.o_delta)
        
    def train(self,X,y):
        O = self.forward(X)
        self.backward(X,y,O)
    
    def predict(self):
      

        
        print("Age:" + str(randage) + " mois")
        if (randMonth == 1):
            month = "Janvier"
        elif(randMonth == 2):
            month = "Février"
        elif(randMonth == 3):
            month = "Mars"
        elif(randMonth == 4):
            month = "Avril"
        elif(randMonth == 5):
            month = "Mai"
        elif(randMonth == 6):
            month = "Juin"
        elif(randMonth == 7):
            month = "Juillet"
        elif(randMonth == 8):
            month = "Août"
        elif(randMonth == 9):
            month = "Septembre"
        elif(randMonth == 10):
            month = "Octobre"
        elif(randMonth == 11):
            month = "Novembre"
        elif(randMonth == 12):
            month = "Décembre" 
        
        print("Mois de naissance: " + month)
        if(randSex == 1): 
            sex = "Masculin"
        else:
            sex = "Féminin"
            
        print("Sexe: " + sex)
        print("Longueur tête: " + str(randHeadlen) + " inches ")
        print("Largeur tête: " + str(randHeadwth) + " inches ")
        print("Cou: " + str(randNeck) + " inches ")
        print("Longueur: " + str(randLength)+ " inches ")
        print("Torse: " + str(randChest)+ " inches ")
        weight = self.forward(xPrediction)
        print("Poids prédit:" + str(weight * data_bears_actualWeight['WEIGHT'].max()) + " pounds")

        
######################################################

NN = Neural_Network()

for i in range(25000):

    NN.forward(X)

    NN.train(X,y)

NN.predict()