# -*- coding: utf-8 -*-
from Layer import *

class Reseau:
        
    def __init__(self, nbInput, nbSortie, nbNeurones, nbHiddenLayers=2):
        self.nbInput = nbInput
        self.nbSortie = nbSortie
        self.nbHiddenLayers = nbHiddenLayers
        self.nbNeurones = nbNeurones
        self.network = [Layer(self.nbInput, self.nbNeurones)]
        for i in range(1, nbHiddenLayers):
            self.network.append(Layer(self.network[i - 1].nbNeurones, self.nbNeurones))
    
    
    def predict(self, entree):
        new_entry = entree
        for layer in self.network:
            
            new_entry = layer.predict(new_entry)
            
        return entree
		
    
    def train(self, entree, sortie):
        
        for i in range(len(entree)):
            prediction = self.predict(entree[i])
            
            if(prediction != sortie[i]):
                for k in range(len(entree[i])):
                    self.poids[k] += self.learningRate*(sortie[i] - prediction)*entree[i][k]
                    self.poids[-1] += self.learningRate*(sortie[i] - prediction)*self.biais		
		
    def __str__(self):
        chaine = "Reseau a " + str(self.nbInput) + " inputs, " + str(self.nbHiddenLayers) + " hidden layers et " + str(self.nbSortie) + " sorties\n\n"
        i = 0
        for couche in self.network:
            #chaine += "Couche " + str(i) + " : " + str(couche.nbNeurones) + " neurones\n"
            chaine += "\t" + str(couche)
            i += 1
        return chaine
    
net = Reseau(3, 4, 12, 6)
print(net)
print(net.predict(np.array([3,4,5])))