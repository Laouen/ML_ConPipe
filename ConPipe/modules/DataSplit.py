
import inspect

from ConPipe.ModuleLoader import ModuleLoader

class DataSplit():

    def __init__(self, function, parameters):
        self.loader = ModuleLoader()
        self.parameters = parameters
        self.data_split_function = self.loader.get_function(**function)        

    def run(self, X, y, group=None):
        
        if inspect.isclass(self.data_split_function):
            train_idx, test_idx = self.data_split_function(
                **self.parameters
            ).split(X,y,group)

        else:
            train_idx, test_idx = self.data_split_function(
                X, y, group,
                **self.parameters
            )

        return {
            'X_train': X[train_idx],
            'y_train': y[train_idx],
            'X_test': X[test_idx],
            'y_test': y[test_idx]
        }