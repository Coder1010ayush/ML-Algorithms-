"""
    this class is a general implementation of batch gradient descent for 
    n dimensional data points,essentially it is not preferable way because in 
    this method we update the coefficient after going through all the data points in one go 
    which may cause hardware limitation issue or very time taking process for long dataset.
    so enchanced way is using stochastic gradient descent


"""
import matplotlib.pyplot as plt
import sys
import os
import json
import numpy as np
sys.path.insert(0, '/home/endless/Documents/Algorithm')
from error_handling import ErrorHandler
class BatchGreadientDescent(ErrorHandler):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        print('------------------------------')
        print('Batch Greadient Descent Class')
        print('------------------------------')


    """
        Mathematical Intuition : 
            
        this method is valid for any n dimension dataset.
        Formula is :
            θ = θ - α * ∇J(θ)
                where α is learning rate
                and ∇J(θ) is derivative with respect to θ.
        beta(not) = -2/n * ( sigme(yi - yi( hat) ) ) -- intercept
        beta(m) = -2/n * ( sigme( yi - yi( hat) )* xi(m)  )  0<= i <=n and 1 <= m <= number_of_col 

    """
    def call(self,x_train,y_train,epochs=50,lr= 0.01):
        self.lr = lr
        self.epochs = epochs
        self.intercept = 0.0
        self.coeff = np.random.rand(x_train.shape[1])


        for i in range(self.epochs):
            y_hat = np.dot(x_train,self.coeff) + self.intercept 
            beta_not = -2 * np.mean( y_train - y_hat ) 
            self.intercept = self.intercept - self.lr*beta_not

            beta = -2 *( np.dot((y_train-y_hat),x_train) ) / x_train.shape[0]
            self.coeff = self.coeff - self.lr*beta

        print('self.coeff is ',self.coeff)
        print('self.intercept is ',self.intercept)

        return [self.intercept,self.coeff]


        


if __name__ == '__main__':
    sc = BatchGreadientDescent()
    data = np.random.randint(low=1,high=100,size=(150,100))
    x_train = data[:,:99]
    y_train = data[:,-1]
    param = sc.call(x_train=x_train,y_train=y_train,lr=0.001,epochs=100)
    print(param)
    param1 = sc.call(x_train=x_train,y_train=y_train,lr=0.001,epochs=95)
    print(param1)
    plt.plot(x_train,y_train)
    plt.show()
