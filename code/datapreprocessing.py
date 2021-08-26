from glob import glob
import numpy as np
import pandas as pd
from scipy import stats
from transforms3d.axangles import axangle2mat
from readwrite import make_dir


def get_walkID(filename):
    """Create walkID from filename.

    Args:
        filename (str): contains 3 substrings
            'trial_[trial_num]'
            'walklr_[walk_side]'
            'visit_[visit_num']

    Returns:
        walkID (str): formatted as [trial_num]-[walk_side]-[visit_num]
    """
    trial = filename.split('trial_')[-1].split('.')[0]
    walk = filename.split('walklr_')[-1].split('_')[0]
    visit = filename.split('visit_')[-1].split('_')[0]
    walkID = trial + '-' + walk + '-' + visit
    return walkID


def get_start_indices(n_timesteps, window_len, window_overlap):
    """Obtain window start indices.

    Args:
        n_timesteps (int): number of total available timesteps.
        window_len (int): number of timesteps to include per window.
        window_overlap (float): 0 to 1, how much consecutive windows
            should overlap.

    Returns:
        start_indices (np array): indices at which windows will start.
    """
    n_timesteps_valid = n_timesteps - window_len + 1
    step_size = int(window_len * (1 - window_overlap))
    if step_size <= 0:
        step_size = 1

    start_indices = np.arange(0, n_timesteps_valid, step_size,
        dtype=int)
    return start_indices


def normalize(X):
    """Normalize the columns of X (zero mean, unit variance).

    Args:
        X (np array): data array.

    Returns:
        X_norm (np array): data array with normalized column values.
    
    """
    EPSILON = 1e-12  # to avoid divide by zero
    X = np.nan_to_num(X)
    X_norm = ((X - np.nanmean(X, axis=0))
             / (np.nanstd(X, axis=0) + EPSILON))
    return X_norm


def build_matrices(read_dir, write_dir, window_overlap, verbose=False):
    """Build windowed and labeled dataset.

    Args:
        read_dir (str): dir containing 'pt*.xlsx' files
        write_dir (str): dir in which to save generated data files
        window_overlap (float): fraction of window to overlap from one
            window to the next. 0 to 1.
        verbose (bool): True to print test statements, False otherwise.

    Returns:
        none.

    """
    FREQ_SAMPLED = 128 # Hz
    FREQ_DESIRED = 64 # Hz
    WINDOW_DUR = 2 # s
    N_AUGMENTATION = 1 # integer

    window_data = []
    window_labels = []
    subjectIDs = []
    walkIDs = []
    augment_idx = []

    files_to_read = glob(f'{read_dir}/pt*.xlsx')
    n_files = len(files_to_read)
    for i in range(n_files):
        f = files_to_read[i]
        if verbose:
            print(f'Reading file {i} of {n_files}: {f}')
        walk_df = pd.read_excel(f, engine='openpyxl')
        walkID = get_walkID(f)
        headers = list(walk_df)
        if verbose:
            print(f'Total time steps in file: {len(walk_df)}')

        # downsample to desired frequency
        nth_row = int(FREQ_SAMPLED/FREQ_DESIRED)
        walk_df = walk_df.iloc[::nth_row]

        walk_data = walk_df.to_numpy(dtype='float')

        window_len = WINDOW_DUR * FREQ_DESIRED
        start_indices = get_start_indices(len(walk_data), window_len,
            window_overlap)

        for k in start_indices:
            this_window_data = walk_data[k:k+window_len,:]

            # label window, remove irrelevant columns, normalize columns
            subjectID = this_window_data[0,0]
            this_window_data = this_window_data[:,2:] # remove subjectID, time
            window_label = int(stats.mode(this_window_data[:,-1])[0][0])
            this_window_data = this_window_data[:,:-1] # remove timestep labels
            this_window_data = normalize(this_window_data)

            # add real window and augmented window(s)
            for j in range(N_AUGMENTATION+1):
                if j == 0:
                    augment_flag = 0
                    window_data.append(this_window_data)
                else:
                    augment_flag = 1
                    this_augmentation = rotate(this_window_data)
                    window_data.append(this_augmentation)
                window_labels.append(window_label)
                subjectIDs.append(subjectID)
                walkIDs.append(walkID)
                augment_idx.append(augment_flag)

        # make headers match data
        headers.remove('subject_ID')
        headers.remove('time')
        headers.remove('freeze_label')

        if verbose:
            print(f'\t{len(window_labels)} total windows generated '
                'from files so far.')

    # save data
    make_dir(write_dir)
    np.save(write_dir + 'y.npy', np.array(window_labels))
    np.save(write_dir + 'X.npy', window_data)
    np.save(write_dir + 'subjectIDs.npy', np.array(subjectIDs))
    np.save(write_dir + 'walkIDs.npy', np.array(walkIDs))
    np.save(write_dir + 'augment_idx.npy', np.array(augment_idx))
    np.save(write_dir + 'headers.npy', np.array(headers))
    return


def rotate(X):
    """Rotates each IMU in X by a random rotation.

    Args:
        X (np array): every sequential set of 6 columns contains a
            single IMU's data, with xyz components of a single sensor
            (accelerometer or gyro) adjacent to one another.

    Returns:
        X (np array): rotated data.

    """
    N_AXES = 3 # x, y, z
    N_DATA_STREAMS = 6 # ax, ay, az, gx, gy, gz
    MAX_ROT = np.pi/12

    n_col = X.shape[1]
    IMU_start_indices = np.arange(0, n_col-(N_DATA_STREAMS-1),
        N_DATA_STREAMS)
    for col_idx in IMU_start_indices:
        axis = np.random.uniform(low=-1, high=1, size=N_AXES)
        angle = np.random.uniform(low=-MAX_ROT, high=MAX_ROT)
        rot_mat = axangle2mat(axis, angle)

        sensor_start_indices = [col_idx, col_idx+3] # ax and gx cols
        for idx in sensor_start_indices:
            X[:,idx:idx+N_AXES] = np.matmul(X[:,idx:idx+N_AXES],
                                            rot_mat)

    return X


def selectFeats(data, IMUs):
    """Return updated data dictionary containing only data from sensors
    of interest.

    Args:
        data (dict): contains X and headers as keys.
        IMUs (list): individual sensors included in sensor set.

    Returns:
        data (dict): updated with X and headers related to sensors of
            interest.

    """
    X = data['X']
    headers = np.array(data['headers'])

    keep_mask = [True if any(IMU in IMU_name for IMU in IMUs) else False
        for IMU_name in headers]

    data['X'] = X[:,:,keep_mask]
    data['headers'] = headers[keep_mask]
    return data
