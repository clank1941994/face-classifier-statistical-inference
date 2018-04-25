
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

def stud(X,M,C,v):
    A = np.matmul(np.reshape(X-M,[1,100]),np.linalg.inv(C))
    B = np.matmul(A,np.reshape(X-M,[100,1]))
    p = B[0,0]
    p = (1+p/v)**(-(v+100)/2)
    p = p*gamma((v+100)/2)
    p = p/np.sqrt(np.linalg.det(C))
    p = p/gamma(v/2)
    return p


def t_cost(deg_f,E,E_log):
    cost = 0
    for i in range(0,len(E)):
        cost += deg_f/2 * np.log(deg_f/2) - gammaln(deg_f/2) - (deg_f/2 - 1)*E_log[i] + deg_f/2 *E[i]
    
    cost = -cost
    return cost    

def t_cost_calculation(deg_f, E, E_log):
    # e_h = hidden_variable_estimates[0]
    # e_logh = hidden_variable_estimates[1]
    half_dof = deg_f / 2
    I = len(E)
    t1 = half_dof * np.log(half_dof)
    t2 = gammaln(half_dof)
    finalCost = 0

    for i in range(I):
        t3 = (half_dof - 1) * E_log[i]
        t4 = half_dof * E[i]
        finalCost = finalCost + t1 - t2 + t3 - t4

    finalCost = -finalCost

    return finalCost




    
def Expectation(train_img,M,C,deg_f,dim):
    delta = np.zeros(len(train_img))
    E = np.zeros(len(train_img))
    E_log = np.zeros(len(train_img))
    for i in range(0,len(train_img)):
        A = np.matmul(np.reshape(train_img[i,:]-M,[1,100]),np.linalg.inv(C))
        B = np.matmul(A,np.reshape(train_img[i,:]-M,[100,1]))
        delta[i] =B[0,0]
        E[i] = (deg_f+dim)/(deg_f+delta[i])
        E_log[i] = digamma((deg_f+dim)/2) - np.log((deg_f+delta[i])/2)
        
    return [E, E_log]
                
def Maximization(train_img,M,C,E,E_log,deg_f):
    for i in range(0,len(train_img)):
        M +=E[i]*train_img[i,:]
    M = M/np.sum(E)

    for i in range(0,len(train_img)):
        C += np.matmul(np.reshape(train_img[i,:]-M,[100,1]),np.reshape(train_img[i,:]-M,[1,100]))
    C = C/np.sum(E)

#    deg_f = fminbound(t_cost,0,999,args = (E,E_log))
    deg_f = fminbound(t_cost_calculation, 0, 999, args=(E, E_log))
    
    return [M,C,deg_f]

def log_likelihood(train_img,M,C,deg_f,dim):
    delta = np.zeros(len(train_img))
    suml = 0
    for i in range(0,len(train_img)):
        A = np.matmul(np.reshape(train_img[i,:]-M,[1,100]),np.linalg.inv(C))
        B = np.matmul(A,np.reshape(train_img[i,:]-M,[100,1]))
        delta[i] =B[0,0]
        
        suml += np.log(1+delta[i]/deg_f)/2
        
        
    L = gammaln((deg_f+dim)/2) - dim*np.log(deg_f*3.14)/2 - np.linalg.slogdet(C) - gammaln(deg_f/2)
    L = L*len(train_img)
    
    L = L - (deg_f+dim)*suml
    
    
def log_likelihood2(M,C,deg_f,dim,img):

    inv_std_face = np.linalg.inv(C)
    

    t1 = gammaln((deg_f + dim) / 2)
    t2 = dim*np.log(deg_f*np.pi)/2
    t3 = np.linalg.slogdet(C)[1]/2
    t4 = loggamma(deg_f/2)

    fixed_pre_term = (t1 - t2 - t3 - t4)

    diff = img - M
    diff = np.array(diff)[np.newaxis]
    e1 = np.matmul(diff, inv_std_face)
    e2 = np.matmul(e1, diff.T)
    e3 = np.log(1+e2/deg_f)
    e4 = fixed_pre_term - ((deg_f + dim)/2 * e3)
    

    return e4
    




#%%
    
[X_face_train,X_not_train,X_face_test,X_not_Test,X_MOG] = get_data()
M_face = np.random.rand(len(X_face_train[0]))*255
c = np.ones(len(X_face_train[0]))*2000
C_face = np.diag(c)
v_face = 50000
dim  = 100

count = 0
while(count<10):
    
    [E,E_log] = Expectation(X_face_train,M_face,C_face,v_face,dim)       
        
    [M_face,C_face,v_face] = Maximization(X_face_train,M_face,C_face,E,E_log,v_face)
    
    print("Degrees of freedom:",v_face)
    
    count = count + 1
#%%
M_not = np.random.rand(len(X_face_train[0]))*255
c = np.ones(len(X_face_train[0]))*2000
C_not = np.diag(c)
v_not = 1000
dim  = 100

count = 0
while(count<10):
    
    [E,E_log] = Expectation(X_not_train,M_not,C_not,v_not,dim)       
        
    [M_not,C_not,v_not] = Maximization(X_not_train,M_not,C_not,E,E_log,v_not)
    
    print("Degrees of freedom:",v_not)
    
    count = count + 1

#%%


count = 0
p1 = np.zeros(351)
p2 = np.zeros(351)
p1not = np.zeros(351)
p2not = np.zeros(351)
for i in range(0,351):
#    p1 = stud(X_face_test[i,:],M_face,C_face,v_face)
#    p2 = stud(X_face_test[i,:],M_not,C_not,v_not)
#   
    p1[i] = log_likelihood2(M_face,C_face,v_face,100,X_face_test[i,:])[0,0]
    p2[i] = log_likelihood2(M_not,C_not,v_not,100,X_face_test[i,:])[0,0]
    if p1[i]>p2[i]:
        count+=1
        
    
print(count)    
#%%
count = 0
for i in range(0,351):
#    p1 = stud(X_face_test[i,:],M_face,C_face,v_face)
#    p2 = stud(X_face_test[i,:],M_not,C_not,v_not)
#   
    p1not[i] = log_likelihood2(M_face,C_face,v_face,100,X_not_Test[i,:])[0,0]
    p2not[i] = log_likelihood2(M_not,C_not,v_not,100,X_not_Test[i,:])[0,0]
    if p1not[i]<p2not[i]:
        count+=1
        
    
print(count)    

#%%
meanImage = np.array(M_face,dtype = "uint8").reshape(10,10)
cv2.imwrite("Mean_T_dist.png",cv2.resize(meanImage,None,fx = 10,fy = 10))
c = np.diag(C_face)
CovImage = np.sqrt(c)
CovImage = np.array(CovImage,dtype = "uint8").reshape(10,10)
cv2.imwrite("Cov_T_dist.png",cv2.resize(CovImage,None,fx = 50,fy = 50))


meannot = np.array(M_not,dtype = "uint8").reshape(10,10)
cv2.imwrite("Mean_T_dist_not.png",cv2.resize(meannot,None,fx = 10,fy = 10))
c = np.diag(C_not)
CovImagenot = np.sqrt(c)
CovImagenot = np.array(CovImagenot,dtype = "uint8").reshape(10,10)
cv2.imwrite("Cov_T_dist_not.png",cv2.resize(CovImagenot,None,fx = 50,fy = 50))


#%%
probface = p1/(p1+p2)
probnot = p2not/(p1not + p2not)
labels = np.append(np.ones(351),np.zeros(351))
Posterior = np.append(probface*1.3,probnot*2.5)

from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(labels,Posterior,pos_label=0)
plt.plot(fpr,tpr,color = 'blue')
plt.title("REceiver Operating Characteristic")

plt.savefig("T_dist_ROC_mixture.png")          






        