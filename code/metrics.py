import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve


def compute_mean_auroc(labels):
    """Computes mean AUROC over folds provided.

    Args:
        labels (df): columns include 'probas' (from model) and 'true'
            (ground truth). One row for each fold.

    Returns:
        mean_auroc (float): mean AUROC over folds

    """
    labels_true = labels['true']
    labels_probas = labels['probas']

    mean_fpr = np.linspace(0, 1, 101)
    tprs = []
    aucs = []

    for i in range(len(labels_true)):
        fold_ytrue = labels_true.iloc[i]
        fold_yprobas = labels_probas.iloc[i]

        fpr, tpr, _ = roc_curve(fold_ytrue, fold_yprobas)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tpr_interp[-1] = 1.0
        tprs.append(tpr_interp)

    mean_tpr = np.mean(tprs, axis=0)
    mean_auroc = auc(mean_fpr, mean_tpr)
    return mean_auroc


def compute_total_ap(labels):
    """Computes total average precision over folds provided.

    Args:
        labels (df): columns include 'probas' (from model) and 'true'
            (ground truth). One row for each fold.

    Returns:
        total_ppv (float): positive predictive value across all folds
        total_ap (float): average precision over all folds
    """
    ytrue = np.concatenate(labels['true'])
    yprobas = np.concatenate(labels['probas'])

    total_ppv = np.sum(ytrue)/len(ytrue)
    precision, recall, _ = precision_recall_curve(ytrue, yprobas)
    total_ap = auc(recall, precision)
    return total_ppv, total_ap


# TODO DELETE?
# def calculate_youdens_index(labels):
#     """Compute threshold defined by Youden's J statistic.
    
#     Args:
#         labels (df): columns include 'probas' (from model) and 'true'
#             (ground truth). One row for each fold.

#     Returns:
#         threshold (float): Youden's index threshold.
#     """
#     labels_true = labels['true']
#     labels_probas = labels['probas']
#     fpr, tpr, thresholds = roc_curve(labels_true, labels_probas)
#     threshold = thresholds[np.argmax(tpr - fpr)]
#     return threshold
