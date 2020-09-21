#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 15:56:16 2020

@author: josegrau
"""

#Dounut problem classification problem with neural network

import numpy as np
import matplotlib.pyplot as plt

#todas las funciones son iguales que en el problema xor

#no usamos softmax al ser clasificación binaria

#usamos en ambas capas la función sigmoide como activación

def forward(X, W1, b1, W2, b2):
    # tanh
    # Z = np.tanh(X.dot(W1) + b1)
    #sigmoid
    Z = 1 / (1 + np.exp( -(X.dot(W1) + b1) ))
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
    dZ = np.outer(T - Y, W2) * (1 - Z) * Z #sigmoid
    return X.T.dot(dZ)

def derivative_b1(Z, T, Y, W2):
    dZ = np.outer(T - Y, W2) * (1 - Z) * Z #sigmoid
    return dZ.sum(axis=0)

#cross-entropy binaria mejorada

def cost(T, Y):
    return np.sum(T*np.log(Y) + (1-T)*np.log(1-Y))

#Y aquí el main

#Definimos el donut
N = 10000
R_inner = 5
R_outer = 10

R1 = np.random.randn(N//2) + R_inner
theta = 2*np.pi*np.random.randn(N//2)
X_inner = np.concatenate([[R1*np.cos(theta)],[R1*np.sin(theta)]]).T

R2 = np.random.randn(N//2) + R_outer
theta = 2*np.pi*np.random.randn(N//2)
X_outer = np.concatenate([[R2*np.cos(theta)],[R2*np.sin(theta)]]).T

X = np.concatenate([X_inner, X_outer])
Y = np.array([0]*(N//2) + [1]*(N//2))

#Definimos los parametros de la red neuronal
n_hidden = 8
W1 = np.random.randn(2,n_hidden)
b1 = np.random.randn(n_hidden)
W2 = np.random.randn(n_hidden)
b2 = np.random.randn(1)

#El resto funciona igual al caso xor, ajustando los parámetros y cambiando alguna cosa

costs = []
    
learning_rate = 0.00005
regularization = 0.2
last_error_rate = None
    
for i in range(6000):
    pY, Z = forward(X, W1, b1, W2, b2)
    c = cost(Y, pY)
    prediction = predict(X, W1, b1, W2, b2)
    er = np.abs(prediction - Y).mean()
    
    costs.append(c)
    W2 += learning_rate*(derivative_w2(Z, Y, pY)-regularization*W2)
    b2 += learning_rate*(derivative_b2(Y, pY)-regularization*b2)
    W1 += learning_rate*(derivative_w1(X,Z,Y,pY,W2)-regularization*W1)
    b1 += learning_rate*(derivative_b1(Z,Y,pY,W2)-regularization*b1)
    if i %100 == 0:
            print("i =", i, "cost =", c, "classification rate =", 1-er)
    

plt.plot(costs)
