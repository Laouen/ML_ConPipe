import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np

def roc_chart_binary(y_true, y_pred, y_probas, classes, class_labels):

    if len(classes) != 2:
        raise ValueError('classes must be arrays with 2 elements')

    fpr, tpr, _ = roc_curve((y_true == classes[1]).astype(int), y_probas[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot of a ROC curve for a specific class
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(
        fpr,
        tpr,
        label=f'AUC = {str(roc_auc)[:4]}'
    )
    
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    

def roc_chart_multilabel(y_true, y_pred, y_probas, classes, class_labels):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i,c in enumerate(classes):
        fpr[i], tpr[i], _ = roc_curve((y_true == c).astype(int), y_probas[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    plt.plot([0, 1], [0, 1], 'k--')
    for i,c in enumerate(class_labels):
        plt.plot(
            fpr[i],
            tpr[i],
            label=f'{c} AUC = {str(roc_auc[i])[:4]}'
        )
    
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])


def roc_chart(y_true, y_pred, y_probas, classes, class_labels):
    if np.unique(y_true).shape[0] == 2:
        roc_chart_binary(y_true, y_pred, y_probas, classes, class_labels)
    else:
        roc_chart_multilabel(y_true, y_pred, y_probas, classes, class_labels)


def confusion_matrix_chart(y_true, y_pred, y_probas, classes, class_labels, annot=True, cmap='flare', fmt='g'):

    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        columns=class_labels,
        index=class_labels,
    )

    sns.heatmap(cm, annot=annot, cmap=cmap, fmt=fmt)
    plt.title('Confusion matrix')
