# -*- coding: utf-8 -*-
import math
import numpy as np

class Layer:
        
    def __init__(self, nbInput, nbNeurones, nbEpoch, learningRate, biais=1):
        self.nbInput = nbInput
        self.nbEpoch = nbEpoch
        self.nbNeurones = nbNeurones
        self.learningRate = learningRate
        self.biais = biais*np.ones([nbNeurones, 1])
        self.poid = np.zeros([nbNeurones, nbInput])
        
        
    def predict(self, entree):
        resultat = self.poid.dot(np.array(entree).reshape(len(entree),1)) + self.biais
        for i in range(len(resultat)):
            resultat[i] = self.sigmoide(resultat[i])
            
        return resultat
    
    
    def train(self, entree, sortie):
        erreur_poid = np.zeros([self.nbNeurones,len(entree[0])])
        erreur_biais = np.zeros([self.nbNeurones,1])
        
        for k in range(self.nbEpoch):
            for i in range(len(entree[0])):
                pred = self.predict(entree[i])
                
                for j in range(self.nbNeurones):
                    err_neurone_poid = np.array(2*(pred[j] - sortie[i][j])*self.sigmo_prime(np.array(self.poid[j]).dot(np.array(entree[i]).reshape(len(entree[i]),1) + self.biais[j]))*entree[i])
                    err_neurone_poid = err_neurone_poid.reshape(len(entree[i]),1)
                    err_neurone_biais = 2*(pred[j] - sortie[i][j])*self.sigmo_prime(np.array(self.poid[j]).dot(np.array(entree[i]).reshape(len(entree[i]),1) + self.biais[j]))
                    erreur_biais[j,0] = err_neurone_biais
                    for l in range(self.nbInput):
                        erreur_poid[j,l] = err_neurone_poid[l]
                    
        self.poid -= self.learningRate*erreur_poid
        self.biais -= self.learningRate*erreur_biais
                
            
    
    def backprop(self, poid, biais):
        self.poid = poid
        self.biais = biais
        
    def sigmoide(self, x):
        return 1/(1+math.exp(-x))

    def sigmo_prime(self, x):
        return (math.exp(-x))/(1+math.exp(-x))**2
    
    def __str__(self):
        chaine = "Layer a " + str(self.nbInput) + " inputs, " + str(self.nbNeurones) + " neurones. Poids : " + str(self.poid.shape) + ", biais : " + str(self.biais.shape) + "\n\nPoids\n"
        for element in self.poid:
            chaine += str(element) + "\n"
        chaine += "\nBiais :\n"
        
        for element in self.biais:
            chaine += str(element) + "\n"
        return chaine
    
    