from sklearn import metrics
import matplotlig.pyplot as plt
import seaborn as sns

from ConPipe.exceptions import NotExistentMethodError
from ConPipe.utils import find_function_from_modules

metric_modules = [metrics]

def find_score(metric_name):
    return find_function_from_modules(
        metric_modules,
        metric_name
    )

# TODO: include user modules
chart_modules = []

class ModelEvaluation():

    def __init__(self, config, verbose=1):

        self.config = config
        self.verbose = verbose
        
        self.score_functions = {
            score_name: find_score(score_name)
            for score_name in self.config['scores']
        }

    # Ver si le agregamos que evalue X_train, y_train con CV o algo
    def fit(self, estimator, X_test, y_test, refit=False, X_train=None, y_train=None):
        
        if refit:
            # TODO: raise exception if X or y is None
            estimator.fit(X_train, y_train)

        self.classes_ = estimator.classes_
        self.class_labels = [
            self.config['class_labels'][c]
            for c in self.classes_
        ]

        y_pred = estimator.predict(X_test)
        y_probas = estimator.predict_proba(X_test)

        for chart_name in self.config['charts']:
            if hasattr(self, chart_name):
                getattr(self, chart_name)(y_test, y_pred, y_probas)
            
            for chart_module in chart_modules:
                if hasattr(chart_module, chart_name):
                    # TODO: pass more parameters like the self.class_labels, self.classes_ etc.
                    getattr(chart_module, chart_name)(y_test, y_pred, y_probas)
        
        self.calculate_scores(y_test, y_pred, y_probas)

    
    def calculate_scores(self, y_test, y_pred, y_probas):
        # Calculate scores with y_pred
        scores = [
            {
                'score_name': score_name,
                'score_val': score_function(
                    y_test,
                    y_pred,
                    **self.config['parameters'][score_name]
                )
            } for score_name: score_function in self.score_functions.items()
            if self.config['prediction_type'][score_name] != 'proba'
        ]

        # calulate scores with y_proba
        # TODO: implement one vs rest for multiclass, currently only works for positive class
        # TODO: implement this better for binary, multiclass and multioutput
        poss_class_index = np.where(
            np.all(
                estimator.classes_ == self.config['poss_class'],
                axis=1
            )
        )[0][0]
        scores += [
            {
                'score_name': score_name, 
                'score_val': score_function(
                    y_test,
                    y_probas[:, poss_class_index],
                    **self.config['parameters'][score_name]
                )
            } for score_name: score_function in self.score_functions.items()
            if self.config['prediction_type'][score_name] == 'proba'
        ]

        return pd.DataFrame(scores)
        
    def roc_chart(self, y_test, y_pred, y_probas):
        n_classes = len(self.classes_)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve((y_true == i).astype(int), y_probas[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        # Plot of a ROC curve for a specific class
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        for i in range(n_classes):
            sns.lineplot(
                x=fpr[i],
                y=tpr[i],
                label=f'{self.class_labels[i]} AUC = {roc_auc[i]:.2f}'
            )
        
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.savefig(os.path.join(self.config['output_path'],f'test_roc.png'))

    def confusion_matrix_chart(self, y_test, y_pred, y_probas=None):

        cm = pd.DataFrame(
            metrics.confusion_matrix(y_test, y_pred),
            columns=self.class_labels,
            index=self.class_labels,
        )

        # TODO: send chart parameters to config
        sns.heatmap(cm, annot=True, cmap='flare', fmt='g')
        plt.title('Confusion matrix')
        plt.savefig(os.path.join(self.config['output_path'], f'test_confusion_matrix.png'))
