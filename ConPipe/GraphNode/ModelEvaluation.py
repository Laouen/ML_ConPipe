import pandas as pd
import os

from ConPipe.Logger import Logger
from ConPipe.module_loaders import get_function

class ModelEvaluation():

    def __init__(self, scores, charts, output_path, tag, class_labels, fit_model):

        self.charts = charts
        self.output_path = output_path
        self.tag = tag
        self.class_labels = class_labels
        self.fit_model = fit_model
        self.logger = Logger()

        # Get all al score functions ready to run
        self.score_pred_functions = {}
        self.score_proba_functions = {}
        self.score_parameters = {}
        for score_name, score_module in scores.items():
            
            func = get_function(score_module['function']) 
            
            if score_module['score_type'] == 'pred':
                self.score_pred_functions[score_name] = func
            elif score_module['score_type'] == 'proba':
                self.score_proba_functions = func

            parameters = {} if 'parameters' not in score_module else score_module['parameters']
            self.score_parameters[score_name] = parameters

        # Get all chart functions ready to run
        self.chart_functions = {}
        self.chart_parameters = {}
        for chart_name, chart_module in self.charts.items():
            self.chart_functions[chart_name] = get_function(chart_module['function'])
            parameters = {} if 'parameters' not in chart_module else chart_module['parameters']
            self.chart_parameters[chart_name] = parameters


    # Ver si le agregamos que evalue X_train, y_train con CV o algo
    def run(self, estimator, X_test, y_test, X_train=None, y_train=None):
        
        if self.fit_model:
            if X_train is None or y_train is None:
                raise ValueError("X_train and y_train can't be None if fit_model is true")
            estimator.fit(X_train, y_train)

        classes = estimator.classes_
        class_labels = [
            self.class_labels[c]
            for c in classes
        ]

        y_pred = estimator.predict(X_test)
        y_probas = estimator.predict_proba(X_test)

        self.make_charts(y_test, y_pred, y_probas, classes, class_labels)
        self.calculate_scores(y_test, y_pred, y_probas)

    
    def make_charts(self,y_test, y_pred, y_probas, classes, class_labels):
        for chart_name, chart_function in self.chart_functions.items():
            self.logger(2, f'making chart {chart_name}')
            chart_function(
                y_true=y_test,
                y_pred=y_pred,
                y_probas=y_probas,
                classes=classes,
                class_labels=class_labels,
                output_path=self.output_path,
                **self.chart_parameters[chart_name]
            )

    def calculate_scores(self, y_test, y_pred, y_probas):
        # Calculate scores with y_pred
        scores = [
            {
                'score_name': score_name,
                'score_val': score_function(
                    y_test,
                    y_pred,
                    **self.score_parameters[score_name]
                )
            } for score_name, score_function in self.score_pred_functions.items()
        ]

        # Calculate scores with y_probas[:,1]
        scores += [
            {
                'score_name': score_name,
                'score_val': score_function(
                    y_test,
                    y_probas[:,1],
                    **self.score_parameters[score_name]
                )
            } for score_name, score_function in self.score_proba_functions.items()
        ]

        # TODO: calulate scores for multiclass and multioutput
        # TODO: implement one vs rest for multiclass, currently only works for positive class

        pd.DataFrame(scores).to_csv(
            os.path.join(self.output_path, f'{self.tag}_scores.csv'),
            sep=';',
            index=False
        )
        
