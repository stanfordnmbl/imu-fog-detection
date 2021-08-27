# Copyright (c) 2021, Stanford Neuromuscular Biomechanics Laboratory
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, 
# this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer 
# in the documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived 
# from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT 
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import pandas as pd
import tensorflow as tf
from datapreprocessing import selectFeats
from model import run_model


def losocv_split(subjectIDs):
    """Create leave-one-subject-out cross-validation train/test splits.

    Args:
        subjectIDs (list): subjectID corresponding to each example.

    Returns:
        splits (list of lists): each fold's train and test indices.
        subjectIDset (list): unique IDs, in held-out-test-set order

    """

    subjectIDset = list(set(subjectIDs))
    splits = []
    for subject in subjectIDset:
        test_idx = [i for i in range(len(subjectIDs)) if subjectIDs[i]==subject]
        train_idx = [i for i in range(len(subjectIDs)) if subjectIDs[i]!=subject]
        
        splits.append([train_idx,test_idx])

    return splits, subjectIDset


def get_sample_weights(y, subjectIDs):
    """Computes sample weights such that each label for each subject
    has equal weight. Subject's examples have weights totalling 1, and
    each label within a subject totals to 0.5.

    Args:
        y (np array): label corresponding to each example.
        subjectIDs (list): subjectID corresponding to each example.

    Returns: sample_weights (np array): weight corresponding to each
        example.

    """
    subjects = list(set(subjectIDs))
    sample_weights = np.empty_like(subjectIDs, dtype=float)
    
    # loop through subjectIDs to count number of each label
    for subject in subjects:
        # get labels specific to subject
        subj_idx = [i for i in range(len(subjectIDs))
            if subjectIDs[i]==subject]
        ysubj = y[subj_idx]

        # compute weights for each subject (sum to 1)
        subj_y_counts = np.zeros((2,)) # 2 for 2 classes
        for i in range(2):
            subj_y_counts[i] = np.count_nonzero(ysubj==i) # count number of each label
            if subj_y_counts[i]==0:
                raise Exception('subject missing a class example.') # missing subject example for 1+ classes
        subj_weights = 1/(2*subj_y_counts)
        
        # populate full sample weights matrix
        subj_sample_weights = np.zeros_like(ysubj,dtype=float)
        for i in range(len(ysubj)):
            subj_sample_weights[i] = subj_weights[ysubj[i]] 
        sample_weights[subj_idx] = subj_sample_weights
        
    return sample_weights


def create_dsets(data, train_idx, test_idx, batch_size=512):
    """Create tf train and test datasets for a single fold.

    Args:
        data (dict): holds all data matrices/lists.
        train_idx (list): elements are bools. True for train data,
            False for test data.
        test_idx (list): elements are bools. True for test data,
            False for train data.
        batch_size (int): batch size for model training.

    Returns:
        train_dset (tf dataset): train_X, train_y, train_sample_weights
            as components.
        test_dset (tf dataset): test_X, test_y, test_sample_weights as
            components.

    """
    X = data['X']
    y = data['y']
    subjectIDs = np.array(data['subjectIDs'])
    augment_idx = data['augment_idx']

    train_X = X[train_idx]
    train_y = y[train_idx]
    train_subjectIDs = subjectIDs[train_idx]

    test_X = X[test_idx]
    test_y = y[test_idx]
    test_subjectIDs = subjectIDs[test_idx]
    test_augment_idx = augment_idx[test_idx]

    # remove augmented examples from test set
    non_augmented = (test_augment_idx==0)
    test_X = test_X[non_augmented]
    test_y = test_y[non_augmented]
    test_subjectIDs = test_subjectIDs[non_augmented]

    # get sample weights
    train_sample_weights = get_sample_weights(train_y, train_subjectIDs)
    test_sample_weights = get_sample_weights(test_y, test_subjectIDs)
    
    train_dset = tf.data.Dataset.from_tensor_slices(
        (train_X, train_y, train_sample_weights)).shuffle(
        buffer_size=len(train_X),seed=0).batch(batch_size)

    test_dset = tf.data.Dataset.from_tensor_slices(
        (test_X, test_y, test_sample_weights)).shuffle(
        buffer_size=len(test_X),seed=0).batch(batch_size)

    return train_dset, test_dset


def train_models_losocv(data, IMUs, result_dir, batch_size, n_epoch,
    verbose=False):
    """Perform leave-one-subject-out cross-validation for a sensor
    set specified by arg IMUs.

    Args:
        data (dict): holds all data matrices/lists.
        IMUs (list): individual sensors included in sensor set.
        result_dir (str): directory in which results will reside.
        batch_size (int): batch size for model training.
        n_epoch (int): max number of epochs for model training.
        verbose (bool): True to print test statements, False otherwise.

    Returns:
        labels (dict): columns include 'probas' (from model) and 'true'
            (ground truth). One row for each fold.

    """
    # select features of interest
    data = selectFeats(data, IMUs)

    subjectIDs = data['subjectIDs']
    split, test_order = losocv_split(subjectIDs)

    labels = pd.DataFrame()
    fold_num = 0
    for train_idx, test_idx in split:        
        if verbose:
            print('\nFold %i' %fold_num)

        subject = str(int(test_order[fold_num]))
        train_dset, test_dset = create_dsets(data, train_idx, test_idx,
            batch_size)
        fold_labels = run_model(train_dset, test_dset, subject,
            result_dir, batch_size, n_epoch, verbose=verbose)
        labels = labels.append(fold_labels, ignore_index=True)
        
        fold_num += 1

    return labels
