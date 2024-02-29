"""
    this class implements linear regression from skretch 
    there is no optimization algorithm used just using linear 
    algebra (inverse and multiplication ).

"""
import numpy as np
import os
import yaml
import sys
sys.path.insert(0, '/home/endless/Documents/Algorithm')
from error_handling import ErrorHandler
from OptimizationAlgorithm.gradient_descent import GradientDescent
class SimpleLinearRegression(ErrorHandler):

    def __init__(self) -> None:
        self.gd = GradientDescent()
        self.intercept = 0.0
        self.slope = 0.0
        print("----------------------------")
        print("Linear Regression Classifier")
        print("----------------------------")

    
    """
        this method will take input columns and trains the model on that data points.
        here closed form solution (estimation is used.) and also this is just valid for 2 column dataset.
        input format : 
            x_train,y_train
        Loss function : 
            f(x) = sigma(yi - m*xi - b)  where 0<i<=n 
            after taking partial derivation w.r.t b and dividing by n (no. of rows in data)
            we get , 
                b = mean(y) - m* mean(x)

            after taking partial derivation w.r.t m and deviding by n(no.of rows in data)
            we get,
                m = sigma((mean(y)-yi) * (mean(x)- xi)) / sigma((x-xi)**2)
    """

    def fit(self,x_train,y_train,verbose = True):
        if self.shapeError(x_train=x_train,y_train=y_train):
            numerator = 0.0
            denominator = 0.0
            for cnt in range(0,x_train.shape[0]):
                numerator += (y_train[cnt]-np.mean(y_train))*(x_train[cnt]-np.mean(x_train))
                denominator += (x_train[cnt]-np.mean(x_train))*(x_train[cnt]-np.mean(x_train))
            
            self.slope = numerator/denominator
            self.intercept = np.mean(y_train) - self.slope*np.mean(x_train)
            if verbose == True:

                print("Slope : ",self.slope)
                print("Intercept : ",self.intercept)

    def fit_regularization(self,x_train,y_train,verbose = True,lamda = 0.3):
        self.lamda = lamda
        if self.shapeError(x_train=x_train,y_train=y_train):
            numerator = 0.0
            denominator = 0.0
            for cnt in range(0,x_train.shape[0]):
                numerator += (y_train[cnt]-np.mean(y_train))*(x_train[cnt]-np.mean(x_train))
                denominator += (x_train[cnt]-np.mean(x_train))*(x_train[cnt]-np.mean(x_train))
            
            self.slope = numerator/(denominator+self.lamda)
            self.intercept = np.mean(y_train) - self.slope*np.mean(x_train)
            if verbose == True:

                print("Slope : ",self.slope)
                print("Intercept : ",self.intercept)
                print("Regularization constant : ",self.lamda)

    def fit_gradient(self,x_train,y_train,epochs,learning_rate):
        ls = self.gd.call(x_train=x_train,y_train=y_train,learning_rate=0.001,epochs=50)
        self.intercept = ls[1]
        self.slope = ls[0]
        print('Using Gradient Descent ',ls)

    """
        this method will take a single data point and predict accordingly with minimum loss.
        this method takes input as np array of one dimension.
    """
    def predict(self,x_test):
        ls = []
        for cnt in range(0,x_test.shape[0]):
            outcome = self.slope*x_test[cnt] + self.intercept
            ls.append(outcome)
        return np.array(ls)



if __name__ == '__main__':
    sc = SimpleLinearRegression()
    data = np.random.randint(low=1,high=100,size=(50,2))
    x_train = data[:,0]
    y_train = data[:,-1]
    sc.fit(x_train=x_train,y_train=y_train)
    print(sc.predict(np.array([10,30,12])))


