# -*- coding: utf-8 -*-
import math

class Neurone:
        
    def __init__(self, nbInput, nbEpochs, learningRate, biais=1):
        self.nbInput = nbInput
        self.nbEpochs = nbEpochs
        self.learningRate = learningRate
        self.poids = [0 for i in range(nbInput + 1)]
        self.biais = biais
    
    def predict(self, entree):
        resultat = self.biais*self.poids[-1]
        for i in range(len(entree)):
            resultat += self.poids[i] * entree[i]
        
        return self.sigmoide(resultat)
		
    
    def train(self, entree, sortie):
        for i in range(self.nbEpochs):
            for j in range(len(entree)):
                prediction = self.predict(entree[j])
                
                if(prediction != sortie[j]):
                    for k in range(len(entree[j])):
                        self.poids[k] += self.learningRate*(sortie[j] - prediction)*entree[j][k]
                        self.poids[-1] += self.learningRate*(sortie[j] - prediction)*self.biais		
		
    def __str__(self):
        chaine = ""
        for i in range(len(self.poids)):
            chaine += "Poids " + str(i) + " = " + str(self.poids[i]) + "\n"
        return chaine
    
    def sigmoide(x):
        return 1/(1+math.exp(-x))
