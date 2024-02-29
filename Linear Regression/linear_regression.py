"""
    this class  is a general base class which implements 
    linear regressor class for n dimension data

"""
import numpy as np
import os
import sys
import json
sys.path.insert(0,'/home/endless/Documents/Algorithm')
from error_handling import ErrorHandler


class LinearRegression(ErrorHandler):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.intercept = 0.0
        self.coeff = 0.0
        print('----------------------------')
        print('Linear Regression Classifier')
        print('----------------------------')

    """

        this method implements linear regression for n dimension data.
        Loss function(E) : y(T)*y - 2*y(T)*x*beta + beta(T)*x(T)*x*beta
        when we solve
                 dE/d(beta) = 0 than we get
                    dE/d(beta)  = (x(T)*x)inverse * (x(T)*y)
                    this is also a closed form solution

        this is also known as batch gradient descent and is not 
        always used because if data size is very large than it is not possible to
        upload all data at one time.

    """

    def fit(self,x_trian,y_train,verbose=True):
        x_trian = np.insert(x_trian,0,1,axis=1)
        beta = np.linalg.inv(np.dot(x_trian.T,x_trian)).dot(x_trian.T).dot(y_train)
        self.intercept = beta[0]
        self.coeff = beta[1:]

        if verbose:
            print("intercept : ",self.intercept)
            print("coefficient : ",self.coeff )

    def fit_regularization(self,x_train,y_train,verbose=True,lamda = 0.3):
        self.lamda = lamda
        x_trian = np.insert(x_trian,0,1,axis=1)
        ident = np.identity(x_train.shape[0])
        beta = np.linalg.inv(np.dot(x_trian.T,x_trian)+ ident*self.lamda ).dot(x_trian.T).dot(y_train)
        self.intercept = beta[0]
        self.coeff = beta[1:]

        if verbose:
            print("intercept : ",self.intercept)
            print("coefficient : ",self.coeff )
            print("Regularisation parameter constant : ",self.lamda)
        

    
    """
        this method takes x_test (numpy array) as an argument and 
        predict the output 
    """
    def predict(self,x_test):
        return np.dot(x_test,self.coeff) + self.intercept


if __name__ == '__main__':

    lr = LinearRegression()
    arr = np.random.rand(100,3)
    y = np.random.rand(50,3)
    x_train = arr[:,0:3]
    y_train = arr[:,-1]
    lr.fit(x_trian=x_train,y_train=y_train)
    print(lr.predict(y))

