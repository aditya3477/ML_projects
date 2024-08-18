import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    # Reference : https://people.revoledu.com/kardi/tutorial/LDA/Numerical%20Example.html
    # Step 1 : Find unique classes in y Column Vector
    uniqueY = np.unique(y)
    # k -> length of unique values in y
    # d -> no of Columns in X
    # Step 2 : Find mean for each unique class in Y
    def calculateMean(yValue):
        # Step 2.1 : Get the indices of current Y value in Y Column Vector
        indicesOfCurrY = np.where(y == yValue)[0] # N x 1 becomes k x 1
        # Step 2.2 : Map the rows of X corresponding to the index
        rowsInX = X[indicesOfCurrY] # N x d becomes k x d
        # Step 2.3 : Calculate mean of each column
        return rowsInX.mean(0) # list of d values
    
    means = np.array([ list(calculateMean(i)) for i in uniqueY ]) # Mean matrix
    
    # Step 3 : Find the covariance of X transpose => d x N : N x d => d x d
    covmat = np.cov(X.T)
    
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    # Step 1 : Find unique classes in y Column Vector
    uniqueY = np.unique(y)
    # k -> length of unique values in y
    # d -> no of Columns in X
    means = np.empty([uniqueY.shape[0],X.shape[1]])
    covmats = []
    for i in range(len(uniqueY)):
        # Step 2.1 : Get the indices of current Y value in Y Column Vector
        indicesOfCurrY = np.where(y == uniqueY[i])[0] # N x 1 becomes k x 1
        # Step 2.2 : Map the rows of X corresponding to the index
        rowsInX = X[indicesOfCurrY] # N x d becomes k x d
        # Step 2.3 : Calculate mean of each column
        means[i] = rowsInX.mean(0)
        # Step 2.4 : Calculate covariance of each column
        covmats.append(np.cov(rowsInX.T))
    
    return means,covmats


def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    
    # IMPLEMENT THIS METHOD
    
    # Given : You can assume the priors of labels follows a uniform distribution
    # Step 1: Prediction of y values - Multivariate Gaussian distribution
    # Formula - argmax( { (1/Sqrt(((2*pi)^k)*Covariance))*exp^(-((x-mu)*Cov^-1*(x-mu)^T)/2) })
    
    # Step 1.1 : (1/Sqrt(((2*pi)^k)*determinant of Covariance))

    kValue = means.shape[1]
    
    constant = 1 / np.sqrt((2*np.pi**kValue)*det(covmat))
    
    # Step 1.2 : Calculate the Power of Exponent
    
    def calculatePdfForRow(i,j):
        xMinusMean = Xtest[i] - means[j] # [X - mu]
        powerOfExp = -1*(np.dot(np.dot(xMinusMean, inv(covmat)), xMinusMean.T))/2 
        return constant * np.exp(powerOfExp) 
    
    def calculateAllPdfs(i):
        return [ calculatePdfForRow(i,j) for j in range(means.shape[0]) ]
    
    allPdf = [ calculateAllPdfs(i) for i in range(Xtest.shape[0]) ]

    # Step 1.3 : Find argmax of all the values
    
    def findMaxValueInRow(row):
        return np.argmax(row)+1

    ypred = np.array([ findMaxValueInRow(row) for row in allPdf ])
    
    # Step 2 : Find the accuracy in predictions
    
    acc = np.mean(ypred == ytest.flatten()) # Comparing two 1D array side by side and averaging it
                  
    return acc, np.array(ypred)

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    
    rows = Xtest.shape[0]
    cols = means.shape[0]
    
    kValue = means.shape[1]
    
    pdf = np.empty([rows,cols])
    
    for i in range(rows):
        for j in range(cols):
            xMinusMean = Xtest[i] - means[j]
            powerOfExp = -1*(np.dot(np.dot(xMinusMean, inv(covmats[j])), xMinusMean.T))/2
            constant = 1 / np.sqrt((2*np.pi**kValue)*det(covmats[j]))
            pdf[i,j] = constant * np.exp(powerOfExp)
            
    def findMaxValueInRow(row):
        return np.argmax(row)+1

    ypred = np.array([ findMaxValueInRow(row) for row in pdf ])
    
    acc = np.mean(ypred == ytest.flatten()) # Comparing two 1D array side by side and averaging it
                  
    return acc, np.array(ypred)

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD
    # Reference : https://web.stanford.edu/~mrosenfe/soc_meth_proj3/matrix_OLS_NYU_notes.pdf
    # Formula for ordinary least squares estimator = ( X^T . X )^-1 . X^T . y
    Xtranspose = X.T # d x N matrix
    X_Xtranspose = np.dot(Xtranspose,X) # d x N . N x d matrix => d x d matrix
    invX_Xtranspose = np.linalg.inv(X_Xtranspose) # Inversing a Matrix d x d matrix
    
    # Weight vector Calculation
    # invX_Xtranspose . Xtranspose = d x d . d x N -> output d x N
    # output . y = d x N . N x 1 => d x 1 -> weight vector
    
    leastSquaresEstimator = np.dot(np.dot(invX_Xtranspose,Xtranspose),y) 
    
    return leastSquaresEstimator

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD                                                   
    return w

def testOLERegression(w,Xtest,yTest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    
    yPredictions = np.dot(Xtest,w)
    
    diff = (yTest - yPredictions)    
    N = 1 /len(Xtest)
    
    mse = np.dot( np.dot(N,diff.T), diff )
    
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD                                             
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD
    return Xp

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
