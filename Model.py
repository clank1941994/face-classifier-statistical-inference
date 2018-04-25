import glob
import cv2
import numpy as np 
from numpy import array
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from math import exp, expm1
scaler = StandardScaler()
scaler2 = StandardScaler()
import matplotlib.pyplot as plt

#################################################
def multi_prob(X,Mean,Covariance):
    
    l = len(Mean)
    A = np.subtract(X,Mean).T
    print(np.shape(A))
    p = np.matmul(A,np.linalg.inv(Covariance))
    p = np.matmul(p,A)
    p = np.exp(-0.5*p)
    p = p/np.sqrt(np.linalg.det(Covariance))
    return p        
#################################################

i = 0

Cov = np.eye(25)
Cov2 = np.eye(25)
X_face_train = np.zeros([1011,4096])
X_face_test = np.zeros([351,4096])

X_not_train = np.zeros([1011,4096])
X_not_test = np.zeros([351,4096])
for img in glob.glob("/home/siddharth/Computer Vision/Data/Train_Positive/*.ppm"):
    I = cv2.imread(img)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    #I = cv2.resize(I,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    print("Iteration ",i)
    
    X_face_train[i,:] = np.ravel(I)
    
    i = i + 1

i = 0
for img in glob.glob("/home/siddharth/Computer Vision/Data/Test_Positive/*.ppm"):
    I = cv2.imread(img)
    print("Iteration %d",i)
    
    I = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
    #I = cv2.resize(I,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    X_face_test[i,:] = np.ravel(I)
    
    i = i + 1
i = 0
for img in glob.glob("/home/siddharth/Computer Vision/Data/Train_Negative/*.ppm"):
    I = cv2.imread(img)
    print("Iteration %d",i)
    
    I = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
    #I = cv2.resize(I,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    X_not_train[i,:] = np.ravel(I)
    
    i = i + 1
i = 0
for img in glob.glob("/home/siddharth/Computer Vision/Data/Test_Negative/*.ppm"):
    I = cv2.imread(img)
    print("Iteration %d",i)
    
    I = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
    #I = cv2.resize(I,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    X_not_test[i,:] = np.ravel(I)
    
    i = i + 1

    
scaler.fit(X_face_train)
train_face_img = scaler.transform(X_face_train)

scaler2.fit(X_not_train)
train_not_img = scaler2.transform(X_not_train)

#pca = PCA(.95)
pca = PCA(n_components=25)
pca2 = PCA(n_components=25)

#pca = PCA(.95)
#pca2 = PCA(.95)


pca.fit(train_face_img)
train_face_img = pca.transform(train_face_img)

pca2.fit(train_not_img)
train_not_img = pca2.transform(train_not_img)

M = np.sum(train_face_img,axis = 0)
M = M/1000

C = np.cov(np.transpose(train_face_img))

M2 = np.sum(train_not_img,axis = 0)
M2 = M2/1000

C2 = np.cov(np.transpose(train_not_img))


Cov = Cov*C
Cov2 = Cov2*C2
test_face_img = scaler.transform(X_face_test)
test_face_img2 = scaler2.transform(X_face_test)
test_face_img = pca.transform(test_face_img)
test_face_img2 = pca2.transform(test_face_img2)

test_not_img = scaler.transform(X_not_test)
test_not_img2 = scaler2.transform(X_not_test)
test_not_img = pca.transform(test_not_img)
test_not_img2 = pca2.transform(test_not_img2)


p = np.zeros(351)
pn = np.zeros(351)
prob = np.zeros(351)
prob_not = np.zeros(351)

#%%

false_negatives = np.zeros(1000)

c = 0
for i in range(0,351):
    p[i] = multi_prob(test_face_img[i,:],M,Cov)
    pn[i] = multi_prob(test_face_img2[i,:],M2,Cov2)
    prob[i] = np.divide(p[i],np.add(p[i],pn[i]))
    if prob[i]<0.5:
        c =c+1
#%%
for i in range(0,1000):
    false_negatives[i] = (prob<((i+1)/1000)).sum()        
#%%
false_positives = np.zeros(1000)
c = 0
for i in range(0,351):
    p[i] = multi_prob(test_not_img[i,:],M,Cov)
    pn[i] = multi_prob(test_not_img2[i,:],M2,Cov2)
    prob_not[i] = np.divide(pn[i],np.add(p[i],pn[i]))
    if prob[i]>0.5:
        c =c+1
    
for i in range(0,1000):
    false_positives[i] = (prob_not<((i+1)/1000)).sum()        


#%%
plt.plot(false_positives,false_negatives)
plt.xlabel("False Positives")
plt.ylabel("False Negatives")
plt.title("Receiver Operating Characteristic")
plt.savefig("Multivariate_Normal.png")





#false_positives = 0
#false_negatives = np.zeros(1000)
#thresh = 0.1
#
#prob_face = np.zeros(351);
#prob_not = np.zeros(351);
#for k in range(0,351):
#    prob_face[k] =   multi_prob(test_face_img[k,:], M, Cov)
#    prob_not[k] =  multi_prob(test_face_img2[k,:], M2, C2)
#    
#    
#
#p_face = prob_face/(prob_face+prob_not)
#
#
#for i in range(0,1000):
#    if(p_face<(i+1)/1000):

#%%
meanImage = np.array(M,dtype = "uint8").reshape(5,5)
cv2.imwrite("Mean_single_multivariate.png",cv2.resize(meanImage,None,fx = 10,fy = 10))
c = np.diag(C)
CovImage = np.sqrt(c)
CovImage = np.array(CovImage,dtype = "uint8").reshape(5,5)
cv2.imwrite("Cov_single_multivariate.png",cv2.resize(CovImage,None,fx = 50,fy = 50))


meannot = np.array(M2*10e16,dtype = "uint8").reshape(5,5)
cv2.imwrite("Mean_single_multivariate_nonface.png",cv2.resize(meannot,None,fx = 10,fy = 10))
c = np.diag(C2)
CovImage = np.sqrt(c)
CovImage = np.array(CovImage,dtype = "uint8").reshape(5,5)
cv2.imwrite("Cov_single_multivariate_nonface.png",cv2.resize(CovImage,None,fx = 50,fy = 50))



#%%


      #%%
labels = np.append(np.ones(351),np.zeros(351))
Posterior = np.append(prob,prob_not)

from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(labels,Posterior,pos_label=0)
plt.plot(fpr,tpr,color = 'blue')
plt.title("REceiver Operating Characteristic")

plt.savefig("Multivariate_Normal.png")          








