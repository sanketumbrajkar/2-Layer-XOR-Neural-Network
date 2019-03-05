#---LOGIC FOR XOR GATE---#
#2 layer neural network
#~AUTHOR-SANKET UMBRAJKAR

import numpy as np
import time 

#variables
n_hidden=10 #number of hidden neurons
n_input=10 #number of inputs
n_output=10 #number of output

n_sample=300 #sample data

#hyperparameters
learning_rate=0.01
momentum=0.9


#np.random.seed generates the same random
#values in all the iterations of code
#NON DETERMINISTIC SEEDING
np.random.seed(50) 

#ACTIVATION FUCNTIONS
#Sigmoid function turns numbers into probabilities
def sigmoid(x):
    return 1.0/(1.0 +np.exp(-x))

#tanh is hyperbolic tangent
def tanh_prime(x):
    return 1-np.tanh(x)**2


#TRAINING FUNCTION
#x is input data
#t is Transpose to multiply matrices
#V,W are layers to our network
#bv and bw are biasaes for each of the layer    
def train(x,t,V,W,bv,bw):    
    #forward propogation
    #matrix multiply + biases
    A=np.dot(x,V)+bv
    Z=np.tanh(A)
    
    B=np.dot(Z,W)+bw
    Y=sigmoid(B)
    
    #backward propogation
    Ew=Y-t
    Ev=tanh_prime(A) * np.dot(W,Ew)
    
    #predict loss
    dW=np.outer(Z,Ew)
    dV=np.outer(x,Ev)
    #what does the outer funciton do?
    
    #cross-entropy(for classification)
    loss= -np.mean(t*np.log(Y)+(1-t)*np.log(1-Y))
    
    return loss,(dV,dW,Ev,Ew)

#PREDICTION FUNCTION
def predict(x,V,W,bv,bw):
    A=np.dot(x,V)+bv
    B=np.dot(np.tanh(A),W)+bw
    return (sigmoid(B)>0.5).astype(int)



    
#CREATE LAYERS
V=np.random.normal(scale=0.1,size=(n_input,n_hidden))
W=np.random.normal(scale=0.1,size=(n_hidden,n_output))

#initialising biases
bv=np.zeros(n_hidden)
bw=np.zeros(n_output)

params= [V,W,bv,bw]

#generate our data
X=np.random.binomial(1,0.5,(n_sample,n_input))
T=X^1

#TRAINING TIME

for epoch in range(100):
    err=[]
    update=[0]*len(params)
    
    t0=time.clock()
    #for each data point, update our weights
    
    for i in range(X.shape[0]):
        loss,grad=train(X[i],T[i],*params)
        #update loss
        for j in range(len(params)):
            params[j]=update[j]
            
        for j in range(len(params)):
            update[j]=learning_rate*grad[j]+momentum*update[j]
        
        err.append(loss)


    print('Epoch: %d,Loss %.8f,Time:%fs' %(epoch,np.mean(err),time.clock()-t0))

#try to predict something
X=np.random.binomial(1,0.5,n_input)
print('XOR prediction')  
print(X)
print(predict(X,*params))      