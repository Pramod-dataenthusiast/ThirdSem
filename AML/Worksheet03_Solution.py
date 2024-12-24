#!/usr/bin/python

import numpy as np

from sklearn import datasets
iris = datasets.load_iris() # use the Iris data.

X = iris.data
y = iris.target

# N is the sample size
# Di is the number of input features
# Do is the number of output features
# H1 is the number of units in the hidden layer
[N, Di] = X.shape # N objects, Di input dimension
Do = 3 # three output dimensions (one for each category)
H1 = 10 # dimension of hidden layer

alpha = 0.00001 # set the learning rate for gradient descent

y = np.reshape(y, (N,1)) # change y type to matrix


## convert the set of categories in y to a set of one hot variables
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
y = onehot_encoder.fit_transform(y)

np.random.seed(1)

# Randomly initialize weights
w2 = np.random.randn(Di, H1)
w1 = np.random.randn(H1, Do)

#-----------------------------------------------------------------
# Activation functions and gradients
def relu(x):
	return(np.maximum(x, 0))

def reluGrad(x):
    if (x < 0):
        return(0)
    return(1)
    

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoidGrad(x):
	return(sigmoid(x)*(1 - sigmoid(x)))

#-----------------------------------------------------------------
# Loss functions and gradients

def linearLoss(y, yhat):
    return(np.square(yhat - y).sum())

def linearLossGrad(y, yhat):
    return(2*(yhat - y))


def softmax(x):
    # subtract maximum to ensure no large values
    # does not effect result
    m = np.max(x) 
    z = [np.exp(a - m) for a in x]
    return(z/np.sum(z))

def softmax2(x):
    # normal softmax (without max subtraction)
    m = np.max(x)
    z = [np.exp(a) for a in x]
    return(z/np.sum(z))

def softmaxLoss(y, yhat):
    # compute soft probabilities for each row
    z = np.apply_along_axis(softmax, 1, yhat)
    # -sum(count*log(prob)) for likelihood
    return(-(np.log(z)*y).sum())

def softmaxLossGrad(y, yhat):
    # compute dl/dyhat_i
    z = np.apply_along_axis(softmax, 1, yhat)
    # -sum(count*log(prob)) for likelihood
    return(z - y)

#-----------------------------------------------------------------
# function for computing accuracy of predictions.
def accuracy(y, yhat):
    zhat = np.apply_along_axis(np.argmax, 1, yhat)
    z = np.apply_along_axis(np.argmax, 1, y)
    return(np.mean(zhat == z))

#-----------------------------------------------------------------
# Assign functions for use in the network

# set neuron activation function
# ReLU
activation = np.vectorize(relu)
activationGrad = np.vectorize(reluGrad)
# Sigmoid
#activation = np.vectorize(sigmoid)
#activationGrad = np.vectorize(sigmoidGrad)

# set loss function
# Linear
#lossFunction = linearLoss
#lossGrad = linearLossGrad
# Softmax
lossFunction = softmaxLoss
lossGrad = softmaxLossGrad

#-----------------------------------------------------------------
# training the network

for iteration in range(10000):
    # Forward propagation to compute the estiamte of y
    h1 = X.dot(w2) # multiply input by weights and sum
    fh1 = activation(h1) # make sure activation function is chosen
    yhat = fh1.dot(w1) # multiply hidden layer by weights and sum
    
    # Compute and print loss
    loss = lossFunction(y, yhat)

    print('[ iter:', iteration, '] loss:', loss, end='\r', flush=True)

    ## backpropagate the error compute the gradient with respect to
    ## the first set of weights
    dJdy = lossGrad(y, yhat) # loss gradient
    dydw1 = fh1
    dJdw1 = dydw1.T.dot(dJdy)

    ## backpropagate the error compute the gradient with respect to
    ## the second set of weights
    dydh = w1

    # create the matrix of 0 and 1 for the relu gradient
    dhdg = activationGrad(h1)
    dgdw2 = X
    dJdh = dJdy.dot(dydh.T)
    dJdg = dJdh*dhdg
    dJdw2 = dgdw2.T.dot(dJdg)
    #print(dJdw2)
    
    # update the weights using gradient descent
    w1 = w1 - alpha*dJdw1
    w2 = w2 - alpha*dJdw2


print('\n')

print("Training accuracy:", accuracy(y, yhat))
