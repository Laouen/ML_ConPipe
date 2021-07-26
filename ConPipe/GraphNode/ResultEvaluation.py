import pandas as pd
import matplotlib.pyplot as plt
import os

from ConPipe.Logger import Logger
from ConPipe.module_loaders import get_function


class ResultEvaluation():

    def __init__(self, scores, charts, output_path, tag, classes, class_labels=None):

        self.output_path = output_path
        self.tag = tag
        self.classes = classes
        self.class_labels = class_labels
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
                self.score_proba_functions[score_name] = func

            parameters = {} if 'parameters' not in score_module else score_module['parameters']
            self.score_parameters[score_name] = parameters

        # Get all chart functions ready to run
        self.chart_functions = {}
        self.chart_parameters = {}
        for chart_name, chart_module in charts.items():
            self.chart_functions[chart_name] = get_function(chart_module['function'])
            parameters = {} if 'parameters' not in chart_module else chart_module['parameters']
            self.chart_parameters[chart_name] = parameters

    def run(self, y_true, y_pred, y_probas, classes):

        class_labels = [
            self.class_labels[c]
            for c in classes
        ]

        self._make_charts(y_true, y_pred, y_probas, classes, class_labels)
        self._calculate_scores(y_true, y_pred, y_probas)

    def _make_charts(self, y_true, y_pred, y_probas, classes, class_labels):
        for chart_name, chart_function in self.chart_functions.items():
            self.logger(2, f'making chart {chart_name}')
            plt.clf()
            chart_function(
                y_true=y_true.copy(),
                y_pred=y_pred.copy(),
                y_probas=y_probas.copy(),
                classes=classes,
                class_labels=class_labels,
                **self.chart_parameters[chart_name]
            )

            plt.savefig(
                os.path.join(
                    self.output_path,
                    f'{self.tag}_{chart_name}.png'
                )
            )

    def _calculate_scores(self, y_true, y_pred, y_probas):
        
        scores = []
        # Calculate scores with y_pred
        for score_name, score_function in self.score_pred_functions.items():
            self.logger(2, f'Calculating score {score_name}')
            scores.append({
                'score_name': score_name,
                'score_val': score_function(
                    y_true,
                    y_pred,
                    **self.score_parameters[score_name]
                )
            })
        
        # Calculate scores with y_probas
        for score_name, score_function in self.score_proba_functions.items():
            self.logger(2, f'Calculating score {score_name}')
            scores.append({
                'score_name': score_name,
                'score_val': score_function(
                    y_true,
                    y_probas,
                    **self.score_parameters[score_name]
                )
            })

        self.logger(1, 'Obtained scores:')
        self.logger(1, scores)
        
        pd.DataFrame(scores).to_csv(
            os.path.join(self.output_path, f'{self.tag}_scores.csv'),
            sep=';',
            index=False
        )
        
