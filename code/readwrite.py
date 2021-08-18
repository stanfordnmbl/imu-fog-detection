import os
import zipfile
import pandas as pd
import numpy as np


def make_dir(dir):
    """Create directory if nonexistent.

    Args:
        dir (str): directory to create.

    Returns:
        none.

    """
    if not os.path.exists(dir):
        os.mkdir(dir)
    return


def read_IMU_configs(filename):
    """Creates dictionary of IMU configurations.

    Args:
        filename (str): Excel file with IMU configuration definitions.

    Returns:
        IMU_sets (dict): keywords contain IMU configuration names.
            Values contain sensors included in the configuration.

    """
    df = pd.read_excel(filename, engine='openpyxl', index_col=0)
    df = df.fillna('')
    IMU_set_names = df.index.to_list()
    allIMUs = np.array(list(df))

    IMU_sets = {}
    for IMU_set_name in IMU_set_names:
        a = (df.loc[IMU_set_name].to_numpy()!='')
        IMU_sets[IMU_set_name] = list(allIMUs[a])
    return IMU_sets


def read_data(data_dir):
    """Read data matrices and store.

    Args:
        data_dir (str): directory in which data resides. See ReadME.

    Returns:
        data (dict): holds all data matrices/lists.

    """
    X = np.load(data_dir + 'X.npy')
    y = np.load(data_dir + 'y.npy')
    headers = list(np.load(data_dir + 'headers.npy'))
    subjectIDs = np.load(data_dir + 'subjectIDs.npy')
    walkIDs = np.load(data_dir + 'walkIDs.npy')
    augment_idx = np.load(data_dir + 'augment_idx.npy')
    data = {'X': X, 'y': y, 'headers': headers, 'subjectIDs':
        subjectIDs, 'walkIDs': walkIDs, 'augment_idx': augment_idx}

    return data
