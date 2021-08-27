"""
Copyright (c) 2021, Stanford Neuromuscular Biomechanics Laboratory
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, 
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer 
in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived 
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT 
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

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
