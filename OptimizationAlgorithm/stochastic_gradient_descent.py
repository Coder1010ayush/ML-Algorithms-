"""
    this class implements the stochastic gradient descent optimisation algorithm.
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


class Stochastic(ErrorHandler):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        print("---------------------------------")
        print("Stochastic Gradient Descent Class")
        print("---------------------------------")

    """
    this optimisation algorithm is a best way to approximate the minima of a 
    convex function.In this algorithm we change the weight and bias on the basis of a random
    choosen row.
    Mathematical formula remains same as batch gradient descent.
    Formula is :
            θ = θ - α * ∇J(θ)
                where α is learning rate
                and ∇J(θ) is derivative with respect to θ.
    """
    def call(self,x_train,y_train,epochs=50,lr=0.001):

        self.lr = lr
        self.epochs = epochs
        self.intercept = 0.0
        self.coeff = np.random.rand(x_train.shape[1])


        for i in range(self.epochs):
            for cnt in range(x_train.shape[0]):
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
    sc = Stochastic()
    data = np.random.randint(low=1,high=100,size=(150,4))
    x_train = data[:,:3]
    y_train = data[:,-1]
    param = sc.call(x_train=x_train,y_train=y_train,lr=0.001,epochs=2)
    print(param)
    param1 = sc.call(x_train=x_train,y_train=y_train,learning_rate=0.001,epochs=95)
    print(param1)
    plt.plot(x_train,y_train)
    plt.show()