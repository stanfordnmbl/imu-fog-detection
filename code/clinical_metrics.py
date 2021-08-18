import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg

from crossval import losocv_split, create_dsets
from datapreprocessing import selectFeats
from model import load_model_and_weights, make_predictions
from readwrite import read_data

def get_walk_data(walk, data):
    """Retrieve data specific to a given walk.

    Args:
        walk (str): walk identifier.
        data (dict): holds all data matrices/lists, including unique
            walk identifiers in 'subject_walk_IDs'.

    Returns:
        walk_data (dict): all data matrices/lists for a single subject.

    """
    X = data['X']
    y = data['y']
    subjectIDs = data['subjectIDs']
    subject_walk_IDs = data['subject_walk_IDs']
    augment_idx = data['augment_idx']

    # remove augmented examples from test set
    non_augmented = (augment_idx==0)
    X = X[non_augmented]
    y = y[non_augmented]
    subjectIDs = subjectIDs[non_augmented]
    subject_walk_IDs = subject_walk_IDs[non_augmented]

    # get walk-specific data
    walk_specific = (subject_walk_IDs==walk)
    walk_data = {}
    walk_data['X'] = X[walk_specific]
    walk_data['y'] = y[walk_specific]
    walk_data['subjectID'] = subjectIDs[0]
    walk_data['headers'] = data['headers']
    return walk_data


def predict_ankle_model(data):
    """Generate ankle model predictions for data.

    Args:
        data (dict): all data matrices/lists for a single subject.

    Returns:
        labels (dict): columns include 'probas' (from model) and 'true'
            (ground truth). One row for each fold.

    """
    RESULT_DIR = '../results/imus6_subjects7/sensors01_rankle/'\
                     'iteration0/'

    data = selectFeats(data, ['ankle_r'])
    test_dset = (data['X'], data['y'])

    subject = str(int(data['subjectID']))
    model = load_model_and_weights(subject, RESULT_DIR)

    labels = make_predictions(model, test_dset)
    return labels


def smooth_labels(labels_probas, n_samples):
    """Apply moving average filter.

    Args:
        labels_probas (np array): outputs from the model, 0 to 1.
        n_samples (int): number of samples over which to average.

    Returns:
        labels_smoothed (np array): smoothed labels_probas.
    """
    n = len(labels_probas)
    half_window = int(np.floor(n_samples/2))
    idx = half_window # start index
    labels_smoothed = np.empty_like(labels_probas)
    while idx <= n - half_window:
        labels_smoothed[idx] = np.mean(
            labels_probas[idx-half_window:idx+half_window])
        idx += 1

    # back-fill and forward-fill ends
    labels_smoothed[:half_window] = labels_smoothed[half_window]
    labels_smoothed[n-half_window:] = labels_smoothed[n-half_window]

    return labels_smoothed


def get_freeze_onsets(labels_pred):
    """Find indices of 1s following 0s.

    Args:
        labels_pred (ndarray): contains 0s and 1s.

    Returns:
        onsets (ndarray): indices of 1s following 0s.

    """
    if labels_pred[0] == 1: # check for initial state
        onsets = [0]
    else:
        onsets = []

    transitions = (labels_pred[:-1] < labels_pred[1:])
    onsets.extend(np.where(transitions==True)[0] + 1)
    return np.array(onsets)


def get_freeze_offsets(labels_pred):
    """Find indices of 0s following 1s.

    Args:
        labels_pred (ndarray): contains 0s and 1s.

    Returns:
        offsets (ndarray): indices of 0s following 1s.

    """
    transitions = (labels_pred[:-1] > labels_pred[1:])
    offsets = list(np.where(transitions==True)[0] + 1)

    if labels_pred[-1] == 1: # check for final state
        offsets.append(len(labels_pred))
    return np.array(offsets)


def remove_short_freezes(labels_pred, n_samples):
    """Remove freezes with duration less than n_samples.

    Args:
        labels_pred (ndarray): contains 0s and 1s.
        n_samples (int): number of samples over which to average.

    Returns:
        labels_pred (ndarray): input, with freezes less than n_samples
            removed.

    """
    freeze_onsets = get_freeze_onsets(labels_pred)
    freeze_offsets = get_freeze_offsets(labels_pred)

    for onset_idx in freeze_onsets:
        temp_idx = np.where(freeze_offsets>onset_idx)[0][0]
        offset_idx = freeze_offsets[temp_idx]

        if offset_idx-onset_idx < n_samples:
            labels_pred[onset_idx:offset_idx] = 0
    return labels_pred


def get_binary_predictions(labels, thresh):
    """Smooth probabilities, threshold, and remove short freezes.
    
    Args:
        labels (dict): columns include 'probas' (from model) and 'true'
            (ground truth). One row for each fold.
        thresh (float): threshold for nonFOG/FOG labels.

    Returns:
        labels (dict): input, with added column 'pred', containing
            binary labels.

    """
    labels_true = labels['true']
    labels_probas = labels['probas']

    labels_probas = smooth_labels(labels_probas, 3)
    labels_pred = (labels_probas >= thresh)
    labels_pred = remove_short_freezes(labels_pred, 1)
    labels['pred'] = labels_pred
    return labels


def compute_percent_FOG(freeze_array):
    """Compute percent FOG from array.

    Args:
        freeze_array (ndarray): model labels, 0s and 1s.

    Returns:
        perc_FOG (float): percent of array with label 1.

    """
    frac_FOG = np.nansum(freeze_array)/np.sum(~np.isnan(freeze_array))
    perc_FOG = frac_FOG * 100
    return perc_FOG


def compute_n_FOG(freeze_array):
    """compute number of FOG events from array.

    Args:
        freeze_array (ndarray): model labels, 0s and 1s.

    Returns:
        n_FOG (int): number of 0 to 1 transitions.

    """
    transitions = (freeze_array[:-1] > freeze_array[1:])
    n_FOG = np.sum(transitions)
    return n_FOG


def compute_clinical_metrics(labels):
    """Compute percent FOG and number of FOG events from model and
    human labels.

    Args:
        labels (ndarray): labels, 0s and 1s.

    Returns:
        metrics (pd DataFrame): contains model and human metrics.
    """
    labels_pred = labels['pred']
    labels_true = labels['true']
    metrics = pd.DataFrame()
    metrics['model_percent_FOG'] = [compute_percent_FOG(labels_pred)]
    metrics['model_n_FOG'] = [compute_n_FOG(labels_pred)]
    metrics['human_percent_FOG'] = [compute_percent_FOG(labels_true)]
    metrics['human_n_FOG'] = [compute_n_FOG(labels_true)]
    return metrics


def get_summary_metrics(data, thresholds, verbose):
    """Compute clinical metrics across thresholds for the dataset.

    Args:
        data (dict): holds all data matrices/lists.
        thresholds (list): thresholds over which to compute metrics.
        verbose (bool): True to print test statements, False otherwise.

    Returns:
        summary_metrics (pd DataFrame): consists of metrics for each
            walk for each subject across different thresholds.

    """
    # get unique identifiers for each walk
    data['subjectIDs'] = data['subjectIDs'].astype(int)
    subjectIDs = np.char.array(data['subjectIDs'].astype(str))
    walkIDs = np.char.array(data['walkIDs'])
    data['subject_walk_IDs'] = subjectIDs + '-' + walkIDs
    walks = set(data['subject_walk_IDs'])

    # compute metrics across thresholds
    summary_metrics = pd.DataFrame()
    for i in range(len(thresholds)):
        thresh = thresholds[i]
        if verbose:
            print(f'Threshold {i+1} of {len(thresholds)}')
        for walk in walks:
            walk_data = get_walk_data(walk, data)
            labels = predict_ankle_model(walk_data)
            labels = get_binary_predictions(labels, thresh)
            walk_metrics = compute_clinical_metrics(labels)
            walk_metrics['thresh'] = [thresh]
            walk_metrics['walk'] = [walk]
            walk_metrics['subject'] = [walk.split('-')[0]]
            summary_metrics = summary_metrics.append(walk_metrics,
                ignore_index=True)

    return summary_metrics


def compute_ICC(metrics_df, metric):
    """Compute intraclass correlation coefficient between model- and
    human-determined metrics.

    TODO subject or walk as target? If you revert to target, need to
    have target as argument (since you use walk as target for
    optimizing threshold)

    Args:
        metrics_df (pd DataFrame): consists of threshold (that
            optimizes metric) and corresponding model- and human-
            determined metrics.
        metric (str): name of metric.

    Returns:
        ICC (pd DataFrame): contains ICC values.

    """
    model_vals = metrics_df[['subject', 'walk', 'model_' + metric]].copy()
    model_vals = model_vals.rename(columns={'model_'+metric: metric})
    model_vals['true'] = 'model'

    true_vals = metrics_df[['subject', 'walk', 'human_' + metric]].copy()
    true_vals = true_vals.rename(columns={'human_'+metric: metric})
    true_vals['true'] = 'rater'

    df_ICC = model_vals.append(true_vals)

    ICC = pg.intraclass_corr(df_ICC, targets='walk', raters='true', ratings=metric)
    ICC.set_index('Type')

    return ICC


def get_optimal_thresh(subject, metrics_df):
    """Identify thresholds that optimize each the ICC of each clinical
    metric for the given subject.

    Args:
        subject (int): subject number of interest.
        metrics_df (pd DataFrame): summary metric dataframe consisting
            of metrics for each walk for each subject across different
            thresholds.

    Returns:
        optimal_thresh_p_FOG_metrics (pd DataFrame): threshold that
            optimizes percent time FOG, and corresponding metrics.
        optimal_thresh_n_FOG_metrics (pd DataFrame): threshold that
            optimizes number of FOG events, and corresponding metrics.

    """
    subject_rows = metrics_df['walk'].str.startswith(str(subject))
    subject_metrics = metrics_df[subject_rows].copy()
    subject_metrics['subject'] = subject

    thresholds = set(metrics_df['thresh'])
    max_ICC_p_FOG = -1
    max_ICC_n_FOG = -1
    for thresh in thresholds:
        print(thresh)
        thresh_rows = subject_metrics['thresh']==thresh
        thresh_metrics = subject_metrics[thresh_rows]

        ICC_p_FOG = compute_ICC(thresh_metrics, 'percent_FOG')['ICC'].iloc[0]
        ICC_n_FOG = compute_ICC(thresh_metrics, 'n_FOG')['ICC'].iloc[0]

        if ICC_p_FOG > max_ICC_p_FOG:
            max_ICC_p_FOG = ICC_p_FOG
            optimal_thresh_p_FOG_metrics = thresh_metrics

        if ICC_n_FOG > max_ICC_n_FOG:
            max_ICC_n_FOG = ICC_n_FOG
            optimal_thresh_n_FOG_metrics = thresh_metrics

    return optimal_thresh_p_FOG_metrics, optimal_thresh_n_FOG_metrics


def plot_clinical_metrics(summary_metrics, result_dir):
    """Generate paper figure.

    Args:
        summary_metrics (pd DataFrame): consists of metrics for each
            walk for each subject across different thresholds.
        result_dir (str): directory in which results will reside.

    Returns:
        None.

    """
    COLORS = ['#CC3311', '#EE7733', '#DDCC77', '#999933', '#117733',
        '#0077BB', '#882255']

    fig, ax = plt.subplots(1, 2, figsize=(8,4))# TODO, dpi=1000)
    for i in [0,1]:
        ax[i].set_xlabel('model')
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].plot([0, 100], [0, 100], '--', color='#CCCCCC', zorder=0)
    ax[0].set_ylabel('ground truth')

    # identify optimal threshold for each subject model
    optimal_thresholds_p_FOG = pd.DataFrame() # for percent FOG
    optimal_thresholds_n_FOG = pd.DataFrame() # for number of FOG
    subjects = list(set(summary_metrics['subject'].astype(int)))
    for i in range(len(subjects)):
        subject = subjects[i]
        color = COLORS[i]
        p_FOG_metrics, n_FOG_metrics = get_optimal_thresh(subject,
            summary_metrics)
        optimal_thresholds_p_FOG = optimal_thresholds_p_FOG.append(
            p_FOG_metrics, ignore_index=True)
        optimal_thresholds_n_FOG = optimal_thresholds_n_FOG.append(
            n_FOG_metrics, ignore_index=True)

        ax[0].scatter(p_FOG_metrics['model_percent_FOG'],
            p_FOG_metrics['human_percent_FOG'], marker='o',
            label=str(i+1), color=COLORS[i])
        ax[1].scatter(n_FOG_metrics['model_n_FOG'],
            n_FOG_metrics['human_n_FOG'], marker='o', label=str(i+1),
            color=COLORS[i])
    
    # compute ICC
    p_FOG_ICC = compute_ICC(optimal_thresholds_p_FOG, 'percent_FOG')
    p_FOG_ICC.to_csv(result_dir + 'p_FOG_optimized_metrics.csv',
        index=False)
    p_FOG_ICC = p_FOG_ICC['ICC'].iloc[0]
    n_FOG_ICC = compute_ICC(optimal_thresholds_n_FOG, 'n_FOG')
    n_FOG_ICC.to_csv(result_dir + 'n_FOG_optimized_metrics.csv',
        index=False)
    n_FOG_ICC = n_FOG_ICC['ICC'].iloc[0]

    # finish plot
    leg = plt.legend(bbox_to_anchor=(1.05, 0), loc='lower left',
        frameon=False)
    leg.set_title('participant')
    leg._legend_box.align = "left"
    ax[0].set_xlim([-5, 100])
    ax[0].set_ylim([-5, 100])
    ax[0].set_xticks([0, 20, 40, 60, 80, 100])
    ax[0].set_yticks([0, 20, 40, 60, 80, 100])

    ax[1].set_xlim([-1, 15])
    ax[1].set_ylim([-1, 15])
    ax[1].set_xticks([0, 5, 10, 15])
    ax[1].set_yticks([0, 5, 10, 15])

    ax[0].text(0, 0.95*100, 'ICC = %.2f' %p_FOG_ICC,
        horizontalalignment='left')
    ax[1].text(0, 0.95*15, 'ICC = %.2f' %n_FOG_ICC,
        horizontalalignment='left')

    ax[0].text(-17.5, 100*1.05, 'a) Percent Time FOG',
        horizontalalignment='left', size=15, fontweight='bold')
    ax[1].text(-2.5, 15*1.05, 'b) Number of FOG Events',
        horizontalalignment='left', size=15, fontweight='bold')
    plt.savefig(result_dir + 'clinical_metrics.png', bbox_inches='tight')
    return


def main():
    """Compute clinical metrics and identify optimal thresholds.
    Generate summary figure.
    """
    DATA_DIR = '../data/preprocessed/imus6_subjects7/'
    RESULT_DIR = '../results/imus6_subjects7/sensors01_rankle/'\
        'iteration0/'
    VERBOSE = True
    THRESHOLDS = np.linspace(0, 1, 101)

    data = read_data(DATA_DIR)
    # summary_metrics = get_summary_metrics(data, THRESHOLDS, VERBOSE) # TODO uncomment
    summary_metrics = pd.read_csv(RESULT_DIR + 'ankle_metrics_over_thresh.csv') # TODO remove and delete file
    plot_clinical_metrics(summary_metrics, RESULT_DIR)
    return


if __name__ == "__main__":
    main()