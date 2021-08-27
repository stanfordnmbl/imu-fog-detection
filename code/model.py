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
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv1D, Dropout, MaxPooling1D,
    Flatten, Dense)
from readwrite import make_dir


def init_model(n_timesteps, n_features):
    """Initialize model.

    Args:
        n_timesteps (int): number of timesteps per example.
        n_features (int): number of features per example.

    Returns:
        model (tf keras model): initialized model.

    """

    FILTERS = 16
    KERNEL_LEN = 17
    DENSE_NODES = 10

    model = Sequential()

    model.add(Conv1D(filters=FILTERS, kernel_size=KERNEL_LEN,
        activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=FILTERS, kernel_size=KERNEL_LEN,
        activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))

    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(DENSE_NODES, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def extract_dataset_components(dset):
    """Extract X and y components of dset. Note that components may be
    shuffled if they were shuffled in the dataset.

    Args:
        dset (tf dataset): X, y, as first two components.

    Returns:
        X (np array): first component of the dataset
        y (np array): second component of the dataset

    """
    iterator = list(dset.as_numpy_iterator())

    first_iteration = True
    for batch in iterator:
        if first_iteration==True:
            X = batch[0]
            y = batch[1]
            first_iteration = False
        else:
            X = np.concatenate((X, batch[0]))
            y = np.concatenate((y, batch[1]))

    return X, y


def save_model_and_weights(model, subject, result_dir, verbose=False):
    """Given a model with weights, save to model and weight files.

    Args:
        model (tf keras model): model with weights.
        subject (str): held-out subject identifier
        result_dir (str): directory in which results will reside.
        verbose (bool): True to print test statements, False otherwise.

    Returns:
        none.

    """
    make_dir(result_dir)

    model_filename = (result_dir + '/'
        + 'subject' + subject + 'model.json')
    model_json = model.to_json()
    with open(model_filename, "w") as json_file:
        json_file.write(model_json)

    weights_filename = (result_dir + '/'
        + 'subject' + subject + 'weights.h5')
    model.save_weights(weights_filename)

    if verbose:
        print('Saved subject %s model and weights to disk' %subject)

    return


def load_model_and_weights(subject, result_dir):
    """Load model and weights for subject fold.

    Args:
        subject (str): held-out subject identifier
        result_dir (str): directory in which results reside.
    Returns:
        model with weights.

    """
    # load model
    model_filename = (result_dir + '/' 
        + 'subject' + subject + 'model.json')
    json_file = open(model_filename, 'r')
    model = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(model)

    # load weights
    model_weights_filename = (result_dir + '/'
        + 'subject' + subject + 'weights.h5')
    model.load_weights(model_weights_filename)
    return model


def make_predictions(model, test_dset):
    """Make predictions on test dataset with model.

    Args:
        model (tf keras model): trained model.
        test_dset (tf dataset): dataset on which to make predictions.

    Return:
        labels (dict): keywords 'true' (true labels) and 'probas'
            (predictions from model).

    """
    if not isinstance(test_dset, tuple):
        test_X, test_y = extract_dataset_components(test_dset)
    else:
        test_X = test_dset[0]
        test_y = test_dset[1]

    yprobas = model(test_X, training=False)

    labels = {'true': test_y, 'probas': yprobas}
    return labels


def run_model(train_dset, test_dset, subject, result_dir, batch_size,
    n_epoch, verbose=False):
    """Fit model and make predictions on test dataset.

    Args:
        train_dset (tf dataset): X, y, weights as components.
        test_dset (tf dataset): X, y, weights as components.
        subject (str): held-out subject identifier
        result_dir (str): directory in which results will reside.
        batch_size (int): batch size for model training.
        n_epoch (int): max number of epochs for model training.
        verbose (bool): True to print test statements, False otherwise.

    """
    LR = 0.001
    ES_PATIENCE = 20

    ex_X, _, _ = next(iter(train_dset))
    n_timesteps = ex_X.shape[1]
    n_features = ex_X.shape[2]

    model = init_model(n_timesteps, n_features)
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    auroc = tf.keras.metrics.AUC(name='auc')
    ap = tf.keras.metrics.AUC(curve='PR', name='ap') # avg precision
    model.compile(optimizer=opt, loss='binary_crossentropy',
        metrics=[auroc, ap])

    # train with early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
        patience=ES_PATIENCE, restore_best_weights=True)
    if verbose:
        fit_verbose = 'auto'
    else:
        fit_verbose = 0
    model.fit(train_dset, epochs=n_epoch, validation_data=test_dset,
        callbacks=[es], verbose=fit_verbose)

    # save model and weights
    save_model_and_weights(model, subject, result_dir, verbose)

    labels = make_predictions(model, test_dset)
    return labels