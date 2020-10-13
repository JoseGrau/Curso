# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 08:13:57 2020

@author: ps10803
"""

from preprocesing_ecommerce_data import get_data

from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

#Cogemos los datos de prueba de ecommerce
X, Y = get_data()

#Separamos los datos en el conjunto de entrenamiento y el de prueba
X, Y = shuffle(X, Y)

Ntrain = int(0.7*len(X))
Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

#Creación de la red neuronal
model = MLPClassifier(hidden_layer_sizes = (20,20), max_iter = 2000)

#Entrenar el modelo
model.fit(Xtrain, Ytrain)

#Comprobarla precisión
train_accuracy = model.score(Xtrain, Ytrain)
test_accuracy = model.score(Xtest, Ytest)
print("Train accuracy = ", train_accuracy, "\nTest accuracy = ", test_accuracy)