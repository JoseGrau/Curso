#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:42:05 2020

@author: josegrau
"""

#E-commerce course project for logistic regression

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from preprocesing_ecommerce_data import get_data

#función para obtener la matriz de índices

def y2indicator(y, K):
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

#Coger, mezclar y separar los datos

X, Y = get_data()
X, Y = shuffle(X, Y)
Y = Y.astype(np.int32)

D = X.shape[1]
K = len(set(Y))

Xtrain = X[:-100]
Ytrain = Y[:-100]
Ytrain_ind = y2indicator(Ytrain, K)

Xtest = X[-100:]
Ytest = Y[-100:]
Ytest_ind = y2indicator(Ytest, K)

W = np.random.randn(D, K)
b = np.zeros(K)

#Función softmax

def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis = 1, keepdims = True)

#Función para obtener el output con probabilidades

def forward (X, W, b):
    return softmax(X.dot(W) + b)

#Función para transformar probabilidades en predicciones

def predict (P_Y_given_X):
    return np.argmax(P_Y_given_X, axis=1)

#Función aciertos/fallos

def classification_rate(Y, P):
    return np.mean(Y == P)

#Cross entropy

def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY))

#Guardamos los valores de la función coste y establecemos learning rate
train_costs = []
test_costs = []
learning_rate = 0.001

#Aquí el bucle para entrenar

for i in range(10000):
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)
    
    ctrain = cross_entropy(Ytrain_ind, pYtrain)
    ctest = cross_entropy(Ytest_ind, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)
    
    W -= learning_rate*(Xtrain.T).dot(pYtrain - Ytrain_ind)
    b -= learning_rate*(pYtrain - Ytrain_ind).sum(axis = 0)
    
    if i % 1000 == 0:
        print(i, ctrain, ctest)

print("Final train classification rate: ", classification_rate(Ytrain, predict(pYtrain)))
print("Final test classification rate: ", classification_rate(Ytest, predict(pYtest)))

legend1, = plt.plot(train_costs, label="Train Cost")
legend2, = plt.plot(test_costs, label ="Test Cost")
plt.legend()