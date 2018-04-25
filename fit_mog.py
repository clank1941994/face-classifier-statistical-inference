#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 15:41:16 2018

@author: siddharth
"""
import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np 
from numpy import array
import matplotlib.pyplot as plt
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


def multi_prob_sing(X,Mean,Covariance):
    Covariance = Covariance *np.eye(100)
    l = len(Mean)
    A = np.subtract(X,Mean).T
    print(np.shape(A))
    p = np.matmul(A,np.linalg.inv(Covariance))
    p = np.matmul(p,A)
    p = np.exp(-0.5*p)
    #p = p/np.sqrt(np.linalg.det(Covariance))
    return p        

#%%
#################################################
def multi_prob(X,Mean,Covariance):
    #Covariance = Covariance*np.eye(100)
    l = len(Mean)
    p = np.zeros(len(X))
    for i in range(0,len(X)):
        A = np.subtract(X[i,:],Mean)
        
        B = np.matmul(A,np.linalg.lstsq(Covariance,np.eye(100), rcond = None)[0])
 #       B = np.matmul(A,np.linalg.lstsq(Covariance,np.eye(100))[0]) 
        p[i] = np.matmul(B,np.transpose(A))
        p[i] = np.exp(-0.5*p[i])
       # p[i] = p[i]/np.sqrt((np.linalg.det(Covariance)))
        
    return p        
#################################################




def fit_mog(X,K, precision):
    lam = np.matlib.repmat(1/K, K, 1)
    D = len(X[0])
    I = len(X)
    K_random_unique_integers = np.random.permutation(I)
    K_random_unique_integers = K_random_unique_integers[0:K]
    mu = X[K_random_unique_integers,:]
    
    # Initialize the variances in sig to the variance of the dataset.
    sig = np.zeros((D,D,K))
    dataset_mean = np.sum(X,axis = 0)/I
    dataset_variance = np.zeros((D,D))
    for i in range(0,I):
        mat = X[i,:] - dataset_mean
        mat  = np.matmul(np.reshape(mat,[D,1]),np.reshape(mat,[1,D]))
        dataset_variance = dataset_variance + mat
    
    dataset_variance = dataset_variance/I
    for i in range(0,K):
        sig[:,:,i] = dataset_variance
    
    iterations = 0
    previous_L = 100000
    
    while True:
        l = np.zeros((I,K))
        r = np.zeros((I,K))
        for k in range(0,K):
            l[:,k] = lam[k] * multi_prob(X,mu[k,:],sig[:,:,k])        
    
        s = np.sum(l,axis = 1)
        for i in range(0,I):
            r[i,:] = l[i,:]/s[i]
        
        ##Maximization step
        r_summed_rows = np.sum(r,axis = 0)
        r_summed_all = np.sum(np.sum(r,axis = 0),axis = 0)
        for k in range(0,K):
            lam[k] = r_summed_rows[k]/ r_summed_all
            
            new_mu = np.zeros(D)
            for i in range(0,I):
                new_mu = new_mu + r[i,k] * X[i,:]
            
            mu[k,:] = new_mu/r_summed_rows[k]
            
            
            new_sigma = np.zeros((D,D))
            for i in range(0,I):
                mat = X[i,:] - mu[k,:]
                mat  = np.matmul(np.reshape(mat,[D,1]),np.reshape(mat,[1,D])) *r[i,k]
                new_sigma = new_sigma + mat
            
            sig[:,:,k] = new_sigma/r_summed_rows[k]
            
        
        temp = np.zeros((I,K));
        for k in range(0,K):
            temp[:,k] = lam[k] * multi_prob(X, mu[k,:], sig[:,:,k])
        
        temp = np.sum(temp,axis= 1)
        temp = np.log(temp);        
        L = np.sum(temp);  
        iterations = iterations + 1
        print("Iteration :",iterations)
        print(np.absolute(L- previous_L))
        if(np.absolute(L - previous_L)<precision):
            break
        
        previous_L = L

        
        
            
    return [lam,mu,sig]

[X_face_train,X_not_train,X_face_test,X_not_Test,X_MOG] = get_data()

[L,M,C] = fit_mog(X_face_train,5,300)
#%%
[L2,M2,C2] = fit_mog(X_not_train,5,200)

#%%
false_positives = np.zeros(1000)
false_negatives = np.zeros(1000)
thresh = 0.1

prob_face = np.zeros((351,5));
prob_not = np.zeros((351,5));
for k in range(0,5):
    prob_face[:,k] = L[k] * multi_prob(X_face_test, M[k,:], C[:,:,k]*np.eye(100))
    prob_not[:,k] = L2[k] * multi_prob(X_face_test, M2[k,:], C2[:,:,k]*np.eye(100))
    
    
prob_face = np.sum(prob_face,axis= 1)  
prob_not = np.sum(prob_not,axis= 1)   

p_face = prob_face/(prob_face+prob_not)


for i in range(0,1000):
    false_negatives[i] = (p_face<1 - (i+1)/1000).sum()


prob_face = np.zeros((351,5));
prob_not = np.zeros((351,5));
for k in range(0,5):
    prob_face[:,k] = L[k] * multi_prob(X_not_Test, M[k,:], C[:,:,k]*np.eye(100))
    prob_not[:,k] = L2[k] * multi_prob(X_not_Test, M2[k,:], C2[:,:,k]*np.eye(100))
    
    
prob_face = np.sum(prob_face,axis= 1)  
prob_not = np.sum(prob_not,axis= 1)   

p_not = prob_not/(prob_face+prob_not)


for i in range(0,1000):
    false_positives[i] = (p_not<1 - (i+1)/1000).sum()



#%%
meanImage = np.array(L[1]*M[1,:]+L[3]*M[3,:],dtype = "uint8").reshape(10,10)
cv2.imwrite("Mean_MOG.png",cv2.resize(meanImage,None,fx = 20,fy = 20))
CovImage = L[1]*C[:,:,1]+L[3]*C[:,:,3]
CovImage = np.sqrt(CovImage.diagonal())
CovImage = np.array(CovImage,dtype = "uint8").reshape(10,10)
cv2.imwrite("Cov_MOG.png",cv2.resize(CovImage,None,fx = 10,fy = 10))






meanImagenot = np.array(L2[0]*M2[0,:]+L2[3]*M2[3,:],dtype = "uint8").reshape(10,10)
cv2.imwrite("Mean_MOGnot.png",cv2.resize(meanImagenot,None,fx = 10,fy = 10))
CovImagenot = L2[0]*C2[:,:,0]+L2[3]*C2[:,:,3]
CovImagenot = np.sqrt(CovImagenot.diagonal())
CovImagenot = np.array(CovImagenot,dtype = "uint8").reshape(10,10)
cv2.imwrite("Cov_MOGnot.png",cv2.resize(CovImagenot,None,fx = 10,fy = 10))



#%%
labels = np.append(np.ones(351),np.zeros(351))
Posterior = np.append(p_face,p_not)

from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(labels,Posterior,pos_label=0)
plt.plot(fpr,tpr,color = 'blue')
plt.title("REceiver Operating Characteristic")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("MOG.png")