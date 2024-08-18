import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W

def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    # 1/1+e^(-z) - sigmoid function
    return 1.0/(1.0+np.exp(-1*z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""
    global noOfUniqueFeatures
    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.

    def extractDataFromMnist(mat):
      train = []
      test = []
      train_count = 0
      test_count = 0
      for key in mat:
        data = mat[key]
        if 'train' in key:
          label = np.full((data.shape[0],1),train_count)
          train_count += 1
          train.append(np.hstack((data,label)))
        if 'test' in key:
          label = np.full((data.shape[0],1),test_count)
          test_count += 1
          test.append(np.hstack((data,label)))
      train = np.vstack(tuple(train))
      test = np.vstack(tuple(test))
      train.astype('float64')
      test.astype('float64')
      np.random.shuffle(train)
      np.random.shuffle(test)
      return train,test

    def splitDataAndLabels(data):
      return (data[:,0:-1],data[:,-1])
    
    # def featureSelection(boolVector):
      
    (train,test) = extractDataFromMnist(mat)

    NO_OF_TRAINING_DATA = 50000

    dataForTraining = train[0:NO_OF_TRAINING_DATA]
    dataForValidation = train[NO_OF_TRAINING_DATA:]

    (train_data,train_label) = splitDataAndLabels(dataForTraining)
    (validation_data,validation_label) = splitDataAndLabels(dataForValidation)
    (test_data,test_label) = splitDataAndLabels(test)

    # Normalizing Image values from 0 to 1
    train_data = train_data / 255.0
    validation_data = validation_data / 255.0
    test_data = test_data / 255.0
    
    # Feature selection
    # Here first column is taken as the reference for the features
    allFeatures = np.concatenate((train_data,validation_data),axis=0)
    redundantFeatures = np.all(allFeatures == allFeatures[0],axis=0)
    noOfUniqueFeatures = np.count_nonzero(np.invert(redundantFeatures))

    print("No of unique features : ", noOfUniqueFeatures)

    allFeatures = allFeatures[:,~redundantFeatures]
    test_data = test_data[:,~redundantFeatures]
    train_data = allFeatures[0:NO_OF_TRAINING_DATA]
    validation_data = allFeatures[NO_OF_TRAINING_DATA:]

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, the training data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    def feedForward(input,weight,biasVector):
      dataWithBias =  np.concatenate((biasVector,input),axis=1)
      valueOfNodes = np.dot(dataWithBias,weight.T)
      outputOfNodes = sigmoid(valueOfNodes)
      return (outputOfNodes,dataWithBias)

    biasValue = 1
    noOfInputs = training_data.shape[0]
    # Layer 1
    biasVector_1 = np.full((noOfInputs,1),biasValue)
    (outputOfLayer1Nodes,dataWithBias_1) = feedForward(training_data,w1,biasVector_1)
    biasVector_2 = np.full((outputOfLayer1Nodes.shape[0],1),biasValue)
    (outputOfLayer2Nodes,dataWithBias_2) = feedForward(outputOfLayer1Nodes,w2,biasVector_2)
    groundTruth = np.zeros((noOfInputs,n_class))

    for i in range(noOfInputs):
      groundTruth[i][training_label[i]] = 1
    
    # Error function -> -1/n*( Summation of( (y*log(y`)) + (1-y)*log(1-y`) ) )

    error = (-1/noOfInputs) * np.sum(
      np.multiply(groundTruth, np.log(outputOfLayer2Nodes)) + 
      np.multiply(1.0 - groundTruth, np.log(1.0 - outputOfLayer2Nodes)) 
    )

    diff = outputOfLayer2Nodes - groundTruth
    gradientOfL2 = np.dot(diff.T,dataWithBias_2)
    l2Input = np.dot(diff,w2) * (dataWithBias_2 * ( 1.0 - dataWithBias_2))
    gradientOfL1 = np.dot(np.transpose(l2Input),dataWithBias_1)
    gradientOfL1 = gradientOfL1[1:,:]

    regularizationVal =  (lambdaval * (np.sum(w1**2) + np.sum(w2**2))) / (2*noOfInputs)
    obj_val = error + regularizationVal
    
    grad_w1 = (gradientOfL1 + lambdaval * w1)/noOfInputs
    grad_w2 = (gradientOfL2 + lambdaval * w2)/noOfInputs

    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    return (obj_val, obj_grad)

def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""
    biasValue = 1
    def predictor(input,weight,bias):
      biasVector = np.full((input.shape[0],1),bias)
      dataWithBias =  np.concatenate((biasVector,input),axis=1)
      valueOfNodes = np.dot(dataWithBias,weight.T)
      outputOfNodes = sigmoid(valueOfNodes)
      return outputOfNodes
    
    outputOfL1Nodes = predictor(data,w1,biasValue)
    outputOfL2Nodes = predictor(outputOfL1Nodes,w2,biasValue)

    labels = np.argmax(outputOfL2Nodes, axis=1)

    return labels

"""**************Neural Network Script Starts here********************************"""

featureIndices = []

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
# set the regularization hyper-parameter
lambdaval = 0

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
