from sklearn import model_selection
import inspect

from ConPipe.utils import find_function_from_modules

data_split_modules = [model_selection]

def get_data_split_func(funct_name):
    return find_function_from_modules(
        data_split_modules,
        funct_name
    )

class DataSplit():

    def __init__(self, config, verbose):
        self.config = config
        self.verbose = verbose

        self.data_split_func = get_data_split_func(
            self.config['train_test_split']
        )

    def fit(self, X, y):
        
        if inspect.isclass(self.data_split_func):
            train_idx, test_idx = self.data_split_func(
                **self.config['parameters']
            ).split(X,y)

        else:
            train_idx, test_idx = self.data_split_func(
                X, y,
                **self.config['parameters']
            )

        return {
            'X_train': X[train_idx],
            'y_train': y[train_idx],
            'X_test': X[test_idx],
            'y_test': y[test_idx]
        }

