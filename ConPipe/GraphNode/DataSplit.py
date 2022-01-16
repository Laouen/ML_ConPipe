
import inspect

from ConPipe.ModuleLoader import get_function


'''
TODO: make a split class wrapper to allow using all the sklearn split classes  
if inspect.isclass(self.data_split_function):
            train_idx, test_idx = next(self.data_split_function(
                **self.parameters
            ).split(X,y,group))

        else:
'''

class DataSplit():

    def __init__(self, function, parameters):
        self.parameters = parameters
        self.data_split_function = get_function(function)

    def run(self, *args, **kwargs):

        X_train, X_test, y_train, y_test = self.data_split_function(
            *args,
            **kwargs,
            **self.parameters
        )

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
