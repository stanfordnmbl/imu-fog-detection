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
