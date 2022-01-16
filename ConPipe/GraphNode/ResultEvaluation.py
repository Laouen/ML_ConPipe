import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

from ConPipe.Logger import Logger
from ConPipe.ModuleLoader import get_function


class ResultEvaluation():

    def __init__(self, scores, charts, output_path, tag, classes=None, class_labels=None):

        # TODO: Check if all the mandatory parameters are set and correct or raise Value error

        self.output_path = output_path
        self.tag = tag
        self.classes = classes
        self.class_labels = class_labels
        self.logger = Logger()

        Path(self.output_path).mkdir(
            parents=True,
            exist_ok=True
        )

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

            parameters = score_module.get('parameters', {})
            self.score_parameters[score_name] = parameters

        # Get all chart functions ready to run
        self.chart_functions = {}
        self.chart_parameters = {}
        for chart_name, chart_module in charts.items():
            self.chart_functions[chart_name] = get_function(chart_module['function'])
            parameters = {} if 'parameters' not in chart_module else chart_module['parameters']
            self.chart_parameters[chart_name] = parameters

    def run(self, y_true, y_pred, y_probas, classes=None, class_labels=None):
        # Note: Classes is asumed to come in same order returned by the estimator when predic_probas(). 
        # Thus, y_probas[i] are the probabilities of classes[i]

        # Class_labels must be a dict where class_label[classes[i]] is the label of the class classes[i]

        # Use dynamic (passed as parameter from other node outputs) or static class and labels (defined in the node constructor parameters)
        if class_labels is None and self.class_labels is None:
            raise ValueError('class_label must be defined either in the node constructor of as the run parameter, but they are currently both None')

        if classes is None and self.classes is None:
            raise ValueError('classes must be defined either in the node constructor of as the run parameter, but they are currently both None')

        classes = classes if classes is not None else self.classes
        class_labels = class_labels if class_labels is not None else self.class_labels
        class_labels = [class_labels[c] for c in classes] # order class labels to match the order set in classes

        # Calculate the chart and scores
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
        
