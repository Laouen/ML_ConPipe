from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, GroupKFold
import pandas as pd

# TODO: Agregar más models
MODELS = {
    'LogisticRegression': LogisticRegression,
    'RandomForestClassifier': RandomForestClassifier
}

# TODO: Agregar mas optimizers
PARAMETER_OPTIMIZERS = {
    'GridSearchCV': GridSearchCV
}

# TODO: Agregar más cross validations
CROSS_VALIDATIONS = {
    'StratifiedKFold': StratifiedKFold,
    'GroupKFold': GroupKFold
}

class MLModelSelection():

    def __init__(self, config, verbose):

        self.verbose = verbose
        self.config = config
        
        # TODO: Implementar modelos encadenados con sklearn Pipe de forma de permitir cosas como hacer
        # feature selecton dentro del hiper parameter optimization schema
        self.models = {
            model_name: MODELS[model_name](**self.config['param_grid'][model_name])
            for model_name in self.config['models']
        }
        
        cv = CROSS_VALIDATIONS[self.config['cv']]
        search_module = PARAMETER_OPTIMIZERS[self.config['parameter_optimizer']]
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

        self.cv_results_ = pd.concat([
            pd.DataFrame(optimizer.cv_results_).assign(model_name=model_name)
            for model_name,optimizer in self.parameter_optimizers_.items()
        ])

        self.best_index_ = self.cv_results_[['mean_test_score']].idxmax(axis=0, skipna=True) 
        best_model = self.cv_results_.loc[self.best_index_]['model_name']
        self.best_optimizer_ = self.parameter_optimizers_[best_model]
        self.best_estimator_ = self.best_optimizer_.best_estimator_
        self.best_score_ = self.best_optimizer_.best_score_
        self.best_params_ = self.best_optimizer_.best_params_

    def decision_function(self, X):
        return self.best_optimizer_.decision_function(X)

    def inverse_transform(self, Xt):
        return self.best_optimizer_.inverse_transform(Xt)

    def predict(X):
        return self.best_optimizer_.predict(Xt)

    def predict_log_proba(X):
        return self.best_optimizer_.predict_log_proba(X)

    def predict_proba(X):
        return self.best_optimizer_.predict_proba(X)

    def score(X, y=None):
         return self.best_optimizer_.score(X, y=y)

    def score_samples(X)
        return self.best_optimizer_.score_samples(X)

    def transform(X):
        return self.best_optimizer_.transform(X)

    def set_params(**params):
        self.best_optimizer_.set_params(**params)
        return self