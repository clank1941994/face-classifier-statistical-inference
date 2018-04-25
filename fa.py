#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:02:26 2018

@author: siddharth
"""

import glob
import cv2
import numpy as np 
from numpy import array
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from scipy.special import digamma
from scipy.special import gamma
from math import exp, expm1
from scipy.optimize import fmin
from scipy.special import gammaln
from scipy.optimize import fminbound
from scipy.special import loggamma
from init_data import get_data


[X_face_train,X_not_train,X_face_test,X_not_Test,X_MOG] = get_data()

def multi_prob(X,Mean,Covariance):
    
    l = len(Mean)
    A = np.subtract(X,Mean).T
    p = np.matmul(A,np.linalg.inv(Covariance))
    p = np.matmul(p,A)
    p = np.exp(-0.5*p)
    #p = p/np.sqrt(np.linalg.det(Covariance))
    return p        

#%%
def Expectation(X,K):
    I = len(X)
    D = len(X[0])
    
    M = np.sum(X,axis = 0)/I
    
    P = np.random.rand(D,K)
    
    X_minus_M = X - M
    
    C = np.sum(np.square(X_minus_M),axis = 0)/I
    
   
    inv_sig = np.diag(1/C)
    phi_transpose_times_sig_inv = np.matmul(np.transpose(P),inv_sig)
    temp = np.linalg.inv(np.matmul(phi_transpose_times_sig_inv,P) + np.eye(K))
    E_hi = np.matmul(np.matmul(temp,phi_transpose_times_sig_inv),np.reshape(X_minus_M,[D,I]))
    E_hi_hitr = np.zeros((K,K,I))
    
    for i in range(0,I):
        e = E_hi[:,i]
        E_hi_hitr[:,:,0] = temp + np.matmul(np.reshape(e,[K,1]),np.reshape(e,[1,K]))
    
    return [E_hi, E_hi_hitr]

def fit_fa(X,K,iterations):
    I = len(X)
    D = len(X[0])
    
    M = np.sum(X,axis = 0)/I
    
    #P = np.random.rand(D,K)
    
    X_minus_M = X - M
    
    C = np.sum(np.square(X_minus_M),axis = 0)/I
    
    iterations_count = 0
    
    while True:
        print("Iteration No:",iterations_count)
        [E_hi, E_hi_hitr] = Expectation(X,K)
        print("Expectation Updated")    
        ##Maximization step
        phi_1 = np.zeros((D,K))
        for i in range(0,I):
            phi_1 += np.matmul(np.reshape(X_minus_M[i,:],[D,1]),np.reshape(E_hi[:,i],[1,K]))
            
        phi_2 = np.zeros((K,K))
        for i in range(0,I):
            phi_2 += E_hi_hitr[:,:,i]
        
        phi_2 = np.linalg.inv(phi_2)
        P = np.matmul(phi_1,phi_2)
        
        sig_diag = np.zeros(D)
        for i in range(0,I):
            xm = np.transpose(X_minus_M[i,:])
            sig_1 = np.square(xm)
            sig_2 = np.multiply(np.matmul(P,E_hi[:,i]),xm)
            sig_diag = sig_diag + sig_1 - sig_2
            
        C = sig_diag/I
        
        print("Maximization Updated")
        iterations_count = iterations_count + 1
        
        if iterations_count == iterations:
            break
        
    return [M,P,C]


[M_face,P_face,C_face] = fit_fa(X_face_train,80,100)
[M_not,P_not,C_not] = fit_fa(X_not_train,80,100)


#%%
count = 0
p1face = np.zeros(351)
p2face = np.zeros(351)
p1not = np.zeros(351)
p2not = np.zeros(351)
for i in range(0,351):
    p1face[i] = multi_prob(X_face_test[i,:],M_face,(np.matmul(P_face,np.transpose(P_face))+np.diag(C_face)))
    
    p2face[i] = multi_prob(X_face_test[i,:],M_not,(np.matmul(P_not,np.transpose(P_not))+np.diag(C_not)))
    if p1face[i]>p2face[i]:
        count += 1

print(count)

#%%
count = 0
for i in range(0,351):
    p1not[i] = multi_prob(X_not_Test[i,:],M_face,(np.matmul(P_face,np.transpose(P_face))+np.diag(C_face)))
    
    p2not[i] = multi_prob(X_not_Test[i,:],M_not,(np.matmul(P_not,np.transpose(P_not))+np.diag(C_not)))
    if p2not[i]>p1not[i]:
        count += 1

print(count)            
            
#%%
meanImage = np.array(M_face,dtype = "uint8").reshape(10,10)
cv2.imwrite("Mean_fa_dist.png",cv2.resize(meanImage,None,fx = 10,fy = 10))

CovImage = np.sqrt(C_face)
CovImage = np.array(CovImage,dtype = "uint8").reshape(10,10)
cv2.imwrite("Cov_fa_dist.png",cv2.resize(CovImage,None,fx = 50,fy = 50))


meannot = np.array(M_not,dtype = "uint8").reshape(10,10)
cv2.imwrite("Mean_fa_dist_not.png",cv2.resize(meannot,None,fx = 10,fy = 10))

CovImagenot = np.sqrt(C_not)
CovImagenot = np.array(CovImagenot,dtype = "uint8").reshape(10,10)
cv2.imwrite("Cov_fa_dist_not.png",cv2.resize(CovImagenot,None,fx = 50,fy = 50))

#%%
prob_face  = p1face/(p1face+p2face)
prob_not  = p2not/(p1not+p2not)
labels = np.append(np.ones(351),np.zeros(351))
Posterior = np.append(prob_face,prob_not)

from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(labels,Posterior,pos_label=0)
plt.plot(fpr,tpr,color = 'blue')
plt.title("REceiver Operating Characteristic")

plt.savefig("fa_mixture.png")          
