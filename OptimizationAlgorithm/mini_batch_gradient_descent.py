"""
    this class implements gradient descent optimisation algorithm for 
    a specific size of batch of data.
    this is a first order derivative optimisation algorithm which is used 
    to find out the minima of the loss function given (Default : Root Square Mean Square) // same as gradient_descent

"""
import matplotlib.pyplot as plt
import os
import sys
import json
import numpy as np
sys.path.insert(0, '/home/endless/Documents/Algorithm')
from error_handling import ErrorHandler

class MiniBatchGradientDescent(ErrorHandler):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        print('---------------------------------')
        print('Mini Batch Gradient Descent Class')
        print('---------------------------------')

    """
        this is similar to stochastic gradient descent algorithm difference is just in batch size.
        if batch_size = 1 than it is stochastic gradient descent otherwise it is 
        mini batch_gradient descent.
    """
    def call(self,x_train,y_train,epochs=10,lr=0.001,batch_size=25):

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.intercept = 0.0
        self.coeff = np.random.rand(x_train.shape[1])

        for i in range(self.epochs):

            for btc in range(x_train.shape[0]/self.batch_size):
                idx = np.random.randint(0,x_train.shape[0])
                y_hat = np.dot(x_train[idx],self.coeff) + self.intercept 
                print(y_hat)
                beta_not = -2 * ( y_train[idx] - y_hat ) 
                self.intercept = self.intercept - self.lr*beta_not

                beta = -2 *np.dot(((y_train[idx]-y_hat),x_train[idx])) 
                self.coeff = self.coeff - self.lr*beta

        print('self.coeff is ',self.coeff)
        print('self.intercept is ',self.intercept)

        return [self.intercept,self.coeff]


if __name__ == '__main__':
    sc = MiniBatchGradientDescent()
    data = np.random.randint(low=1,high=100,size=(50,2))
    x_train = data[:,0]
    y_train = data[:,-1]
    param = sc.call(x_train=x_train,y_train=y_train,learning_rate=0.001,epochs=100)
    print(param)
    param1 = sc.call(x_train=x_train,y_train=y_train,learning_rate=0.001,epochs=95)
    print(param1)
    plt.plot(x_train,y_train)
    plt.show()