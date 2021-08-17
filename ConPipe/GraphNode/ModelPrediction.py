import numpy as np

from ConPipe.Logger import Logger
from ConPipe.module_loaders import get_class

from sklearn.base import clone


class ModelPrediction():

    def __init__(self, fit_model=False, cv=None):

        self.fit_model = fit_model

        self.cv = None
        if cv is not None:
            self.cv = get_class(cv['class'])(**cv['parameters'])
        
        self.logger = Logger()
    
    def run(self, estimator, X, y, group_test=None, X_train=None, y_train=None):

        if self.fit_model and self.cv is not None:
            raise ValueError('fit_model cannot be True if cv is not None')

        if self.fit_model:
            if X_train is None or y_train is None:
                raise ValueError(
                    "X_train and y_train can't be None if fit_model is True")
            
            y_true, y_pred, y_probas, classes = self._run_re_fit(
                estimator,
                X, y,
                X_train,
                y_train
            )

        elif self.cv is not None:
            y_true, y_pred, y_probas, classes = self._run_cv(
                estimator,
                X, y,
                group_test
            )

        else:

            y_true, y_pred, y_probas, classes = self._run_simple(
                estimator,
                X, y
            )

        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_probas': y_probas,
            'classes': classes
        }

    def _run_re_fit(self, estimator, X, y, X_train, y_train):

        estimator = clone(estimator)

        estimator.fit(X_train, y_train)

        y_pred = estimator.predict(X)
        y_probas = estimator.predict_proba(X)

        return [
            y,
            y_pred,
            y_probas,
            estimator.classes_
        ]
    
    def _run_cv(self, estimator, X, y, group=None):

        estimator = clone(estimator)

        split_data = [X, y]
        if group is not None:
            split_data.append(group)

        y_true_all = []
        y_pred_all = []
        y_probas_all = []
        for train_idx, test_idx in self.cv.split(*split_data):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            y_probas = estimator.predict_proba(X_test)

            y_true_all.append(y_test)
            y_pred_all.append(y_pred)
            y_probas_all.append(y_probas)
        
        return [
            np.concatenate(y_true_all),
            np.concatenate(y_pred_all),
            np.concatenate(y_probas_all),
            estimator.classes_
        ]
    
    def _run_simple(self, estimator, X, y):

        y_pred = estimator.predict(X)
        y_probas = estimator.predict_proba(X)

        return [
            y,
            y_pred,
            y_probas,
            estimator.classes_
        ]
