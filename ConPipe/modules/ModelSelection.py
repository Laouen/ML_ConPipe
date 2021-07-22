import pandas as pd

from ConPipe.exceptions import NotExistentMethodError
from ConPipe.Logger import Logger
from ConPipe.ModuleLoader import ModuleLoader

class ModelSelection():

    def __init__(self, parameter_optimizer, parameter_optimizer_params, scoring, cv, cv_parameters, models):

        self.logger = Logger()
        self.loader = ModuleLoader()

        self.fit_params = {
            model_name: model_params['fit_params']
            for model_name, model_params in models.items()
        }

        self.param_grids = {
            model_name: model_params['param_grid']
            for model_name, model_params in models.items()
        }
        
        # TODO: Implementar modelos encadenados con sklearn Pipe de forma de permitir cosas como hacer
        # feature selecton dentro del hiper parameter optimization schema
        self.models = {
            model_name: self.loader.get_class(
                module=model_params['module'],
                class_name=model_params['class_name']
            )(**model_params['param_grid'])
            for model_name, model_params in models.items()
        }
        
        cv = self.loader.get_class(**cv)
        search_module = self.loader.get_class(**parameter_optimizer)
        self.parameter_optimizers_ = {
            model_name: search_module(
                estimator=model,
                param_grid=self.param_grids[model_name],
                scoring=scoring,
                cv=cv(**cv_parameters),
                verbose=self.logger.verbose,
                **parameter_optimizer_params
            ) for model_name, model in self.models.items()
        }

        # Set after fit
        self.cv_results_ = None
        self.best_index_ = None
        self.best_estimator_ = None
        self.best_score_ = None
        self.best_params_ = None
    
    def run(self, X, y=None, groups=None):
        self.logger(1, f'Select best model and parameters')

        for model_name, optimizer in self.parameter_optimizers_.items(): 
            optimizer.fit(
                X, y=y, groups=groups,
                **self.fit_params[model_name]
            )

        if hasattr(self.parameter_optimizers_.values()[0], 'cv_results_'):
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

        return {
            'estimator': self.best_estimator_,
            'cv_results_': self.cv_results_
        }

    def _check_if_has_method(self, method_name):
        if not hasattr(self.best_optimizer_, method_name):
            NotExistentMethodError(f'The passed optimizer has no method {method_name}')
    
    def decision_function(self, X):
        self._check_if_has_method('decision_function')
        return self.best_optimizer_.decision_function(X)

    def inverse_transform(self, Xt):
        self._check_if_has_method("inverse_transform")
        return self.best_optimizer_.inverse_transform(Xt)

    def predict(self, Xt):
        self._check_if_has_method("predict")
        return self.best_optimizer_.predict(Xt)

    def predict_log_proba(self, X):
        self._check_if_has_method("predict_log_proba")
        return self.best_optimizer_.predict_log_proba(X)

    def predict_proba(self, X):
        self._check_if_has_method("predict_proba(")
        return self.best_optimizer_.predict_proba(X)

    def score(self, X, y=None):
        self._check_if_has_method("score")
        return self.best_optimizer_.score(X, y=y)

    def score_samples(self, X):
        self._check_if_has_method("score_samples")
        return self.best_optimizer_.score_samples(X)

    def transform(self, X):
        self._check_if_has_method("transform")
        return self.best_optimizer_.transform(X)

    def set_params(self, **params):
        self._check_if_has_method("set_params")
        self.best_optimizer_.set_params(**params)
        return self
