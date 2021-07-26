import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

def roc_chart(y_true, y_pred, y_probas, classes, class_labels):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i,c in enumerate(classes):
        fpr[i], tpr[i], _ = roc_curve((y_true == c).astype(int), y_probas[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    print(f'fpr:', fpr[0])
    print(f'tpr:', tpr[0])

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


def confusion_matrix_chart(y_true, y_pred, y_probas, classes, class_labels, annot=True, cmap='flare', fmt='g'):

    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        columns=class_labels,
        index=class_labels,
    )

    sns.heatmap(cm, annot=annot, cmap=cmap, fmt=fmt)
    plt.title('Confusion matrix')
