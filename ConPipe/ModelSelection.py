from sklearn import ensemble, linear_model, svm
from sklearn import model_selection # import GridSearchCV, StratifiedKFold, GroupKFold
import pandas as pd

from ConPipe.exceptions import NotExistentMethodError
from ConPipe.utils import find_function_from_modules

# TODO: agregar los modulos que pase el usuario
model_modules = [ensemble, linear_model, svm]
cv_modules = [model_selection]
parameter_optimizer_modules = [model_selection]

def get_model(model_name):
    return find_function_from_modules(
        model_modules,
        model_name
    )

def get_cv(cv_name):
    return find_function_from_modules(
        cv_modules,
        cv_name
    )

def get_parameter_optimizer(param_opt_name):
    return find_function_from_modules(
        parameter_optimizer_modules,
        param_opt_name
    )

class ModelSelection():

    def __init__(self, config, verbose=1d):

        self.verbose = verbose
        self.config = config
        
        # TODO: Implementar modelos encadenados con sklearn Pipe de forma de permitir cosas como hacer
        # feature selecton dentro del hiper parameter optimization schema
        self.models = {
            model_name: get_model(model_name)(**self.config['param_grid'][model_name])
            for model_name in self.config['models']
        }
        
        cv = get_cv(self.config['cv'])
        search_module = get_parameter_optimizer(self.config['parameter_optimizer'])
        self.parameter_optimizers_ = {
            model_name: search_module(
                estimator=self.models[model_name],
                param_grid=self.config['param_grid'][model_name],
                scoring=self.config['scoring'],
                cv=cv(**self.config['cv_parameters']),
                refit=self.config['refit'],
                verbose=self.verbose,
                **self.config['parameter_optimizer_params']
            ) for model_name in self.config['models']
        }

        # Set after fit
        self.cv_results_ = None
        self.best_index_ = None
        self.best_estimator_ = None
        self.best_score_ = None
        self.best_params_ = None
    
    def fit(self, X, y=None, groups=None):
        if self.verbose > 1:
            print(f'Select best model and parameters')

        for optimizer in self.parameter_optimizers_: 
            optimizer.fit(
                X, y=y, groups=groups,
                **self.config['fit_params'][model_name]
            )
        ]

        if hasattr(self.parameter_optimizers[0], 'cv_results_'):
            self.cv_results_ = pd.concat([
                pd.DataFrame(optimizer.cv_results_).assign(model_name=model_name)
                for model_name,optimizer in self.parameter_optimizers_.items()
            ])

            self.cv_results_.sort_values(
                by='rank_test_score',
                axis=0,
                inplace=True,
                ignore_index=True
            )

            self.cv_results_.loc[:, 'rank_test_score'] = self.cv_results_.index + 1


        self.best_index_ = 0 
        best_model = self.cv_results_.loc[self.best_index_,'model_name']
        self.best_optimizer_ = self.parameter_optimizers_[best_model]
        self.best_estimator_ = self.best_optimizer_.best_estimator_
        self.best_score_ = self.best_optimizer_.best_score_
        self.best_params_ = self.best_optimizer_.best_params_

    def __check_if_has_method(self, method_name):
        if not hasattr(self.best_optimizer_, method_name):
            NotExistentMethodError(f'The passed optimizer has no method {method_name}')
    
    def decision_function(self, X):
        self._check_if_has_method('decision_function')
        return self.best_optimizer_.decision_function(X)

    def inverse_transform(self, Xt):
        self._check_if_has_method("inverse_transform")
        return self.best_optimizer_.inverse_transform(Xt)

    def predict(X):
        self._check_if_has_method("predict")
        return self.best_optimizer_.predict(Xt)

    def predict_log_proba(X):
        self._check_if_has_method("predict_log_proba")
        return self.best_optimizer_.predict_log_proba(X)

    def predict_proba(X):
        self._check_if_has_method("predict_proba(")
        return self.best_optimizer_.predict_proba(X)

    def score(X, y=None):
        self._check_if_has_method("score")
        return self.best_optimizer_.score(X, y=y)

    def score_samples(X):
        self._check_if_has_method("score_samples")
        return self.best_optimizer_.score_samples(X)

    def transform(X):
        self._check_if_has_method("transform")
        return self.best_optimizer_.transform(X)

    def set_params(**params):
        self._check_if_has_method("set_params")
        self.best_optimizer_.set_params(**params)
        return self
