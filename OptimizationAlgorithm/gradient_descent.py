"""
this class implements the gradient descent optimisation algorithm.
this is a first order derivative optimisation algorithm which is used 
to find out the minima of the loss function given (Default : Root Square Mean Square)
"""
import matplotlib.pyplot as plt
import os
import sys
import json
import numpy as np
sys.path.insert(0, '/home/endless/Documents/Algorithm')
from error_handling import ErrorHandler

class GradientDescent(ErrorHandler):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        print("----------------------")
        print("Gradient Descent Class")
        print("----------------------")

    """
        this method is valid for only 2 columns dataset or 2d dataset.
        Formula is :
            θ = θ - α * ∇J(θ)
                where α is learning rate
                and ∇J(θ) is derivative with respect to θ.

        this is an iterative approach it runs untill
        the slope and intercept converges.
        Sometimes due to very high learning and very low rate also it never converge.
        Learning rate acts as hyper parameters.
    """
    def call(self,x_train,y_train,learning_rate = 0.001,epochs=10):

        self.epochs = epochs
        self.lr = learning_rate
        # prev_slope = 0.0
        # prev_intercept = 0.0
        self.intercept = 0.0
        self.slope = 0.0
        for i in range(self.epochs):
            intercept_derivative = -2 * np.sum(y_train - self.slope*x_train-self.intercept)
            slope_derivative = -2* np.sum(np.dot((y_train-self.slope*x_train-self.intercept),x_train))
            self.intercept = self.intercept - self.lr*intercept_derivative
            self.slope = self.slope - self.lr*slope_derivative

        return [self.slope,self.intercept]
    



if __name__ == '__main__':
    sc = GradientDescent()
    data = np.random.randint(low=1,high=100,size=(50,2))
    x_train = data[:,0]
    y_train = data[:,-1]
    param = sc.call(x_train=x_train,y_train=y_train,learning_rate=0.001,epochs=100)
    print(param)
    param1 = sc.call(x_train=x_train,y_train=y_train,learning_rate=0.001,epochs=95)
    print(param1)
    plt.plot(x_train,y_train)
    plt.show()
