#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:10:15 2020

@author: josegrau
"""

#SoftMax

import numpy as np

#One sample
a = np.random.randn(5)
expa = np.exp(a)
answer = expa/np.sum(expa)

#100 samples
A = np.random.randn(100,5)
expA = np.exp(A)
answer1 = expA/np.sum(expA, axis = 1, keepdims = True)