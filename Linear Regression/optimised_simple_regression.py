"""
    this class implements the general linear regression classifier using stochastic gradient algorithm.
    this works for any n dimensional dataset.

"""


import numpy as np
import os
import sys
import json
sys.path.insert(0,'/home/endless/Documents/Algorithm')
from error_handling import ErrorHandler

class LinearRegressorClassifier(ErrorHandler):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        print('---------------------------')
        print('Linear Regressor Classifier')
        print('---------------------------')

    def fit(self,x_train,y_train,epochs=10,lr= 0.01):
        self.lr = lr
        self.epochs = epochs
        self.intercept = 0.0
        self.coeff = np.random.rand(x_train.shape[1])

        for i in range(self.epochs):
            for cnt in range(x_train.shape[0]):
                idx = np.random.randint(0,x_train.shape[0])
                y_hat = np.dot(x_train[idx],self.coeff) + self.intercept 
                beta_not = -2 * ( y_train[idx] - y_hat ) 
                self.intercept = self.intercept - self.lr*beta_not

                beta = -2 *np.dot((y_train[idx]-y_hat),x_train[idx])
                self.coeff = self.coeff - self.lr*beta

        print('self.coeff is ',self.coeff)
        print('self.intercept is ',self.intercept)

    """
        this method implements prediction method for predicting value
        for a specific given data point.
    """
    def predict(self,x_test):
        return np.dot(x_test,self.coeff) + self.intercept

if __name__ == '__main__':

    lr = LinearRegressorClassifier()
    arr = np.random.rand(100,3)
    y = np.random.rand(50,3)
    x_train = arr[:,0:3]
    y_train = arr[:,-1]
    lr.fit(x_train=x_train,y_train=y_train)
    print(lr.predict(y))