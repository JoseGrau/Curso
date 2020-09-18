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
    Z = 1 / 1 + np.exp(-(X.dot(W1) + b1))
    Y = 1 / 1 + np.exp(-(Z.dot(W2) + b2))
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

