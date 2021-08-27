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
from datapreprocessing import build_matrices
from readwrite import make_dir, read_IMU_configs, read_data
from crossval import train_models_losocv
from metrics import compute_mean_auroc, compute_total_ap


def run_sensor_set(data, iteration, IMUs, result_dir, batch_size,
    n_epoch, verbose=False):
    """Train model and save results.

    Args:
        data (dict): holds all data matrices/lists.
        iteration (int): iteration number
        IMUs (list): individual sensors included in sensor set
        result_dir (str): directory in which results will reside.
        batch_size (int): batch size for model training.
        n_epoch (int): max number of epochs for model training.
        verbose (bool): True to print test statements, False otherwise.

    Returns:
        mean_auroc (float): mean AUROC across all folds
        total_ppv (float): positive predictive value across all folds
        total_ap (float): average precision over all folds

    """
    if verbose:
        print('\nIteration %i' %iteration)

    save_dir = result_dir + 'iteration' + str(iteration) + '/'

    labels = train_models_losocv(data, IMUs, save_dir, batch_size,
        n_epoch, verbose=verbose)

    rocs = []
    prs = []

    loss_filename = save_dir + 'loss.png'
    roc_filename = save_dir + 'roc.png'
    pr_filename = save_dir + 'pr.png'

    mean_auroc = compute_mean_auroc(labels)
    total_ppv, total_ap = compute_total_ap(labels)
    return mean_auroc, total_ppv, total_ap

def run_sensor_set_many(data, IMU_set, IMUs, result_dir, n_iter,
    batch_size, n_epoch, verbose=False):
    """Run experiment for a single sensor set.

    Args:
        data (dict): holds all data matrices/lists.
        IMU_set (str): description of sensor set.
        IMUs (list): individual sensors included in sensor set.
        result_dir (str): directory in which results will reside.
        n_iter (int): number of times to run each model.
        batch_size (int): batch size for model training.
        n_epoch (int): max number of epochs for model training.
        verbose (bool): True to print test statements, False otherwise.

    Returns:
        none.

    """
    save_dir = result_dir + IMU_set + '/'
    make_dir(save_dir)

    aurocs = []
    aps = []
    for i in range(n_iter):
        mean_auroc, total_ppv, total_ap = run_sensor_set(data, i,
            IMUs, save_dir, batch_size, n_epoch, verbose=verbose)
        aurocs.append(mean_auroc)
        aps.append(total_ap)

    np.save(save_dir + 'aurocs.npy', np.array(aurocs))
    np.save(save_dir + 'aps.npy', np.array(aps))
    np.save(save_dir + 'ppv.npy', total_ppv)

    return


def main():
    """Run experiments and saves data for sensors included in
    specified data directory.

    Args:
        none, though constants should be modified:
            DATA_DIR (str): directory in which data resides.
                See ReadME.
            RESULT_DIR (str): directory in which results will reside.
            ITER (int): number of times to run each model, to estimate
                effects of stochastisity.
            BATCH_SIZE (int): batch size for model training.
            EPOCHS (int): max number of epochs for model training.
            IMU_CONFIG_FILENAME (str): Excel sheet containing IMU
                configuration definitions.
            VERBOSE (bool): True to print test statements, False
                otherwise.

    Returns:
        none.

    """
    DATASETS = ['imus6_subjects7', 'imus11_subjects4'] 
    WINDOW_OVERLAPS = [0.5, 0.85] # to generate approximately 10k examples
    DATA_DIR = '../data/'
    RESULT_DIR = '../results/'
    GENERATE_DATA = True
    ITER = 30
    BATCH_SIZE = 512
    EPOCHS = 1000
    VERBOSE = True

    raw_data_dir = DATA_DIR + 'raw/'
    preprocessed_data_dir = DATA_DIR + 'preprocessed/'

    make_dir(preprocessed_data_dir)
    make_dir(RESULT_DIR)

    for i in range(len(DATASETS)):
        dataset = DATASETS[i]
        imu_config_filename = DATA_DIR + dataset + '_configs.xlsx'
        dataset_raw_data_dir = raw_data_dir + dataset + '/'
        dataset_preprocessed_data_dir = (preprocessed_data_dir
            + dataset + '/')

        if GENERATE_DATA:
            window_overlap = WINDOW_OVERLAPS[i]
            build_matrices(dataset_raw_data_dir, 
                dataset_preprocessed_data_dir, window_overlap,
                verbose=VERBOSE)

        dataset_result_dir = RESULT_DIR + dataset + '/'
        make_dir(dataset_result_dir)
    
        IMU_set_dict = read_IMU_configs(imu_config_filename)

        for IMU_set in IMU_set_dict.keys():
            data = read_data(dataset_preprocessed_data_dir)
            if VERBOSE:
                print('\n\n' + IMU_set)
            run_sensor_set_many(data, IMU_set, IMU_set_dict[IMU_set], 
                dataset_result_dir, ITER, BATCH_SIZE, EPOCHS, VERBOSE)
    return


if __name__ == '__main__':
    main()
    