import numpy as np
from scipy.special import gamma
from scipy.special import gammaln
from numpy.linalg import inv
from numpy import log
from scipy.special import digamma
from numpy.linalg import det
from scipy.optimize import fminbound
from init_data import get_data
from scipy.special import loggamma

def cost(V,z,u,D):
    
    I = len(z)
    t1 = log(V/2)
    t2 = digamma(V/2)
    t3 = log((V+D)/2)
    t5 = digamma((V+D)/2)
    t4  = 0
    for j in range(0,I):

        t4 +=z[j]*(log(u[j]) - u[j])/np.sum(z) 

    c = t1 - t2 + 1 -t3 + t4 + t5
    return c

def t_cost_calculation(V, z, u):
    # e_h = hidden_variable_estimates[0]
    # e_logh = hidden_variable_estimates[1]
    half_dof = V / 2
    I = len(z)
    t1 = half_dof * np.log(half_dof)
    t2 = gammaln(half_dof)
    finalCost = 0

    for i in range(I):
        t3 = (half_dof - 1) * z[i]
        t4 = half_dof * u[i]
        finalCost = finalCost + t1 - t2 + t3 - t4

    finalCost = -finalCost

    return finalCost


def prob(X,M,C,V):
    
    D = len(M)
#    delta = np.matmul(np.reshape(X-M,[1,D]),inv(C))
#    delta = np.matmul(delta,np.reshape(X-M,[D,1]))
#    t1 = gamma((V+D)/2)
#    t2 = 1/(np.sqrt(det(C)))
#    t3 = (3.14*V)**(D/2)
#    t3 = t3*gamma(V/2)
#    t4 = (1 + delta/V)**((V+D)/2)
#    p = t1*t2/(t3*t4)
    inv_std_face = np.linalg.inv(C)
    

    t1 = gammaln((V + D) / 2)
    t2 = D*np.log(V*np.pi)/2
    t3 = np.linalg.slogdet(C)[1]/2
    t4 = loggamma(V/2)

    fixed_pre_term = (t1 - t2 - t3 - t4)

    diff = X - M
    diff = np.array(diff)[np.newaxis]
    e1 = np.matmul(diff, inv_std_face)
    e2 = np.matmul(e1, diff.T)
    e3 = np.log(1+e2/V)
    e4 = fixed_pre_term - ((V + D)/2 * e3)
    

    return e4[0,0]
    
    


def Expectation(X,M,C,V,K):
    lam = np.ones(K)/K
    I = len(X)
    D = len(X[0])
    z = np.zeros((K,I))
    u = np.zeros((K,I))
    
    for j in range(0,I):
        for i in range(0,K):
            z[i,j] = lam[i]*prob(X[j,:],M[i,:],C[:,:,i],V[i])
        
    z = z/np.sum(z,axis = 0)
    for j in range(0,I):
        for i in range(0,K):
            delta = np.matmul(np.reshape(X[j,:]-M[i,:],[1,D]),inv(C[:,:,i]))
            delta = np.matmul(delta,np.reshape(X[j,:]-M[i,:],[D,1]))
    
            u[i,j] = (V[i] + D)/(V[i] + delta)
    
    return [z,u]

def Maximization(z,u,X,K):
    lam = np.zeros(K)
    I = len(X)
    D = len(X[0])    
    mu = np.zeros((K,D))
    sig = np.zeros((D,D,K))
    lam = np.sum(z,axis = 1)/I
    V = np.zeros(K)
    zu = np.multiply(z,u)
    for j in range(0,I):
        for i in range(0,K):
            mu[i,:] += z[i,j]*u[i,j]*X[j,:]/np.sum(zu[i,:])
    
    for j in range(0,I):
        for i in range(0,K):
            sig[:,:,i] += z[i,j]*u[i,j]*np.matmul(np.reshape(X[j,:] - mu[i,:],[D,1]),np.reshape(X[j,:] - mu[i,:],[1,D]))/np.sum(z[i,:])
        
    
    for i in range(0,K):
        V[i] = fminbound(t_cost_calculation,0,999,args=(z[i,:],u[i,:]))
        
        
        
    return [mu,sig,lam,V]



## Initialize variables
           
[X_face_train,X_not_train,X_face_test,X_not_Test,X_MOG] = get_data()

I = len(X_face_train)
D = len(X_face_train[0])
K = 5 

M_face = np.random.rand(K,D)*255
C = np.eye(D)*200
C_face = np.zeros((D,D,K))
for i in range(0,K):
    C_face[:,:,i] = C
v_face = np.ones(K) * 5
lam_face = np.ones(K)/K
for i in range(0,10):
    [z,u] = Expectation(X_face_train,M_face,C_face,v_face,K)
    print("Expectation completed")
    [M_face,C_face,lam_face,v_face] = Maximization(z,u,X_face_train,K)
    print("Maximization Completed")
    print("Iteration for EM:",i+1)
    
  



 