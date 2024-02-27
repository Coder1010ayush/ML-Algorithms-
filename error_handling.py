"""
    this class is a base class which implements all kinds of error that will 
    produces during training and evaluation as well as during type checking.
    this class supports numpy array.

"""
class ShapeError(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)



class ErrorHandler(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)

    def shapeError(self,x_train,y_train):
        if x_train.shape[0] != y_train.shape[0]:

            raise ShapeError("shape of matrix is not consistant.")
        return True
    
    