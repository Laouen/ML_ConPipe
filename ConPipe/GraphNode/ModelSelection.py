import pandas as pd

from ConPipe.exceptions import NotExistentMethodError
from ConPipe.Logger import Logger
from ConPipe.ModuleLoader import get_class, get_function
from sklearn.metrics import make_scorer

class ModelSelection():

    def __init__(self, parameter_optimizer, scoring, cv, models):

        self.logger = Logger()

        models = {
            model_name: model_params
            for model_name, model_params in models.items()
            if model_params != False
        }

        self.fit_params = {
            model_name: model_params.get('fit_params', {})
            for model_name, model_params in models.items()
        }

        # TODO: check that all models have a param_grid or raise value error

        self.param_grids = {
            model_name: model_params['param_grid']
            for model_name, model_params in models.items()
        }
        
        # Note: Agregarlo como feature futura en GitHub
        # TODO: Implementar modelos encadenados con sklearn Pipe de forma de permitir cosas como hacer
        # feature selecton dentro del hiper parameter optimization schema
        
        self.models = {
            model_name: get_class(model_params['class'])(
                **model_params.get('constructor_params', {})
            ) for model_name, model_params in models.items()
        }
        
        scoring_function = get_function(scoring['function'])
        if 'parameters' in scoring:
            scoring_function = make_scorer(
                scoring_function,
                **scoring.get('parameters', {})
            )

        cv_class = get_class(cv['class'])

        search_module = get_class(parameter_optimizer['class'])
        self.parameter_optimizers_ = {
            model_name: search_module(
                estimator=model,
                param_grid=self.param_grids[model_name],
                scoring=scoring_function,
                cv=cv_class(**cv.get('parameters', {})),
                verbose=self.logger.verbose,
                **parameter_optimizer.get('parameters', {})
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

        self.cv_results_ = []
        for model_name, optimizer in self.parameter_optimizers_.items(): 
            optimizer.fit(
                X, y=y, groups=groups,
                **self.fit_params[model_name]
            )
            
            self.cv_results_.append(pd.DataFrame(optimizer.cv_results_))
            self.cv_results_[-1]['model_name'] = model_name

        self.cv_results_ = pd.concat(self.cv_results_)

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

        self.logger(1, f'Best model: {best_model}')
        self.logger(1, f'Best parameters: {self.best_params_}')

        return {
            'estimator': self.best_estimator_,
            'cv_results_': self.cv_results_
        }
