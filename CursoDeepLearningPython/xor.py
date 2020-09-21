#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 19:46:04 2020

@author: josegrau
"""

#xor classification problem with neural network

import numpy as np
import matplotlib.pyplot as plt

#no usamos softmax al ser clasificación binaria

#usamos en ambas capas la función sigmoide como activación

def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-(X.dot(W1) + b1)))
    a = Z.dot(W2) + b2
    Y = 1 / (1 + np.exp(-a))
    return Y, Z

#Para predecir, en vez del máximo, basta redondear la función forward

def predict(X, W1, b1, W2, b2):
    Y, _ = forward(X, W1, b1, W2, b2)
    return np.round(Y)

def derivative_w2(Z, T, Y):
    return (T - Y).dot(Z)

def derivative_b2(T, Y):
    return (T - Y).sum()

def derivative_w1(X, Z, T, Y, W2):
    dZ = np.outer(T - Y, W2) * (1 - Z * Z)
    return X.T.dot(dZ)

def derivative_b1(Z, T, Y, W2):
    dZ = np.outer(T - Y, W2) * (1 - Z * Z)
    return dZ.sum(axis=0)

#cross-entropy

def cost(T, Y):
    tot = 0
    for n in range(len(T)):
        if T[n] == 1:
            tot += np.log(Y[n])
        else:
            tot += np.log(1 - Y[n])
    return tot

#Y aquí el main


X = np.array([[0,0],[1,0],[0,1],[1,1]])
Y = np.array([0,1,1,0])
W1 = np.random.randn(2,4)
b1 = np.random.randn(4)
W2 = np.random.randn(4)
b2 = np.random.randn(1)
costs = []
    
learning_rate = 0.0005
regularization = 0.
last_error_rate = None
    
for i in range(100000):
    pY, Z = forward(X, W1, b1, W2, b2)
    c = cost(Y, pY)
    prediction = predict(X, W1, b1, W2, b2)
    er = (prediction - Y).mean()
    if er != last_error_rate:
        last_error_rate = er
        print("Error rate = ", er)
        print("Valor real: ", Y)
        print("Predicción: ", prediction)
        
    if costs and c < costs[-1]:
        print("Early Exit")
        break
    
    costs.append(c)
    W2 += learning_rate*(derivative_w2(Z, Y, pY)-regularization*W2)
    b2 += learning_rate*(derivative_b2(Y, pY)-regularization*b2)
    W1 += learning_rate*(derivative_w1(X,Z,Y,pY,W2)-regularization*W1)
    b1 += learning_rate*(derivative_b1(Z,Y,pY,W2)-regularization*b1)
    if i %10000 == 0:
            print(c)
    
print("Final classification rate: ", 1-np.abs(prediction-Y).mean())
plt.plot(costs)
    
