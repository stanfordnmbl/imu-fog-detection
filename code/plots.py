import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rc
import pandas as pd


rc('font',**{'family':'sans-serif','sans-serif':['Arial']})


METRICS = ['auroc','ap']
N_SUBJ = [7, 4]
RESULT_DIR = '../results/'
DATASET_RESULT_DIRS = [RESULT_DIR + 'imus6_subjects7/',
                       RESULT_DIR + 'imus11_subjects4/']

# set up key, value pairs for combining certain sensor sets into
# one result
LABELS_TO_COMBINE_IMUS6 = {
    '6_chest + lumbar + ankles + feet': 
        ['sensors06_chest_lumbar_ankles_feet'],
    '1_chest': ['sensors01_chest'], 
    '1_ankle': ['sensors01_lankle', 'sensors01_rankle'],
    '1_foot': ['sensors01_lfoot', 'sensors01_rfoot'],
    '1_lumbar': ['sensors01_lumbar'],
    '2_lumbar + ankle': ['sensors02_lumbar_lankle',
                         'sensors02_lumbar_rankle'],
    '2_ankles': ['sensors02_ankles'],
    '2_feet': ['sensors02_feet'],
    '3_chest + feet': ['sensors03_chest_feet'],
    '3_lumbar + ankles': ['sensors03_lumbar_ankles'],
    '3_lumbar + feet': ['sensors03_lumbar_feet']
    }
LABELS_TO_COMBINE_OTHER = {
    '11_all sensors': ['sensors11'],
    '1_head': ['sensors01_head'],
    '1_thigh': ['sensors01_lthigh', 'sensors01_rthigh'],
    '1_wrist': ['sensors01_lwrist', 'sensors01_rwrist'],
    '2_lumbar + wrist': ['sensors02_lumbar_lwrist',
                         'sensors02_lumbar_rwrist'],
    '2_wrist + ankle': ['sensors02_lwrist_lankle',
                        'sensors02_rwrist_rankle'],
    '2_chest + wrist': ['sensors02_chest_lwrist',
                        'sensors02_chest_rwrist'],
    '3_chest + wrist + foot': ['sensors03_chest_lwrist_lfoot',
                               'sensors03_chest_rwrist_rfoot'],
    '3_wrist + ankles': ['sensors03_lwrist_ankles',
                         'sensors03_rwrist_ankles'],
    '3_wrist + feet': ['sensors03_lwrist_feet','sensors03_rwrist_feet']
    }

# preferred sensors, from survey results
PREFERRED_SENSORS_IMUS6 = ['3_lumbar + ankles', '2_ankles', '1_lumbar']
PREFERRED_SENSORS_IMUS11 = ['3_wrist + ankles', '2_wrist + ankle', 
                            '2_ankles', '1_wrist']

# colors
GREEN = '#117733'
RED = '#CC3311'
BLUE = '#0077BB'
ORANGE = '#EE7733'
GREY = '#CCCCCC'


def get_preferred_indices(y_pos, labels, preferred_sensors):
    """Get indices of preferred sensor sets in plot.

    Args:
        y_pos (list): list of y positions used for sensors in plot.
        labels (list-like): sensor set labels associated with y_pos.
        preferred_sensors (list-like): preferred sensor sets.

    Returns:
        indices (list): y positions associated with preferred sets.

    """
    labels = list(labels)
    indices = []
    for sensor in preferred_sensors:
        label_idx = labels.index(sensor)
        indices.append(y_pos[label_idx])
    return indices


def get_best_technical_set(df):
    """Identify set with greatest AUROC in dataframe.

    Args:
        df (pd Dataframe): must contain 'auroc_mean' and 'full_label'.

    Returns:
        best_set (str): label of best technical set.

    """
    df = df.sort_values(by=['auroc_mean'], ascending=False,
                        ignore_index=True)
    best_set = df['full_label'].iloc[0]
    return best_set
    

def get_minimal_IMU_set(df):
    """Identify set with fewest sensors with AUROC within 5% that of
    the best technical set.

    Args:
        df (pd Dataframe): must contain 'n_sensor', 'auroc_mean', 
            and 'full_label'.

    Returns:
        minimal_set (str): label of minimal IMU set.

    """
    df = df.sort_values(by=['n_sensor', 'auroc_mean'],
                        ascending=[True,False], ignore_index=True)
    max_roc = df['auroc_mean'].max()
    thresh = max_roc - 0.05*max_roc
    df = df[df['auroc_mean']>=thresh]
    minimal_set = df['full_label'].iloc[0]
    return minimal_set


# create a figure for each outcome metric
for metric in METRICS:
    fig, ax = plt.subplots(2, 1, figsize=(10,12),
        gridspec_kw={'height_ratios': [4, 7]}, dpi=1000)

    # create a subplot for each dataset
    for i in range(len(N_SUBJ)):
        n = N_SUBJ[i]
        dataset_result_dir = DATASET_RESULT_DIRS[i]

        if n==4:
            labels_to_combine = {**LABELS_TO_COMBINE_IMUS6,
                                 **LABELS_TO_COMBINE_OTHER}
            preferred_sensors = PREFERRED_SENSORS_IMUS11
        elif n==7:
            labels_to_combine = LABELS_TO_COMBINE_IMUS6
            preferred_sensors = PREFERRED_SENSORS_IMUS6
        else:
            raise Exception('Expected 4 or 7 as element of nSubj. '\
                'Received:', n)

        # get average and SD of each sensor set's performance
        labels = []
        auroc_means = []
        auroc_sds = []
        ap_means = []
        ap_sds = []
        for key in labels_to_combine:
            mean_aurocs = []
            total_aps = []

            subdirs = labels_to_combine[key]
            for subdir in subdirs:
                path = dataset_result_dir + subdir + '/'

                aurocs = np.load(path + 'aurocs.npy')
                aps = np.load(path + 'aps.npy')
                ppv = np.load(path + 'ppv.npy')

                mean_aurocs.extend(aurocs)
                total_aps.extend(aps)

            labels.append(key)
            auroc_means.append(np.mean(mean_aurocs))
            auroc_sds.append(np.std(mean_aurocs))
            ap_means.append(np.mean(total_aps))
            ap_sds.append(np.std(total_aps))

        df = pd.DataFrame({'full_label':labels,
                           'auroc_mean': auroc_means,
                           'auroc_sd': auroc_sds,
                           'ap_mean': ap_means,
                           'ap_sd': ap_sds})
        df[['n_sensor','sensor_label']] = (df['full_label']
            .str.split('_', n=1, expand=True))
        df['n_sensor'] = pd.to_numeric(df['n_sensor'])
        
        # sort df and save to csv
        df = df.sort_values(by=['n_sensor', 'auroc_mean'],
            ascending=True, ignore_index=True)
        df.to_csv(dataset_result_dir + str(n) + 'subj_summary.csv')

        # set plot properties
        best_set = get_best_technical_set(df)
        minimal_set = get_minimal_IMU_set(df)
        df['color'] = [GREEN if df['full_label'][i]==best_set else GREY
                       for i in range(len(df))]
        df['color'][df.index[df['full_label']==minimal_set].tolist()] = RED

        # find ypos and nsensor midpoints
        n_sensor_counts = df['n_sensor'].value_counts().to_numpy()
        sensor_set_ypos = []
        midpts = []
        counter = 0
        for count in n_sensor_counts:
            sensor_set_ypos.extend(np.asarray(range(count)) + counter)

            old_counter = counter
            counter += count + 1
            midpts.append(np.mean([old_counter, counter-1]) - 0.5)

        # plot
        ax[i].barh(sensor_set_ypos, df[metric + '_mean'],
            xerr = df[metric + '_sd'], color='k')
        bars = ax[i].patches
        for bar, color in zip(bars, df['color']):
            bar.set_color(color)
        ylim = [-1.5, sensor_set_ypos[-1]+1.5]

        # set xlim and plot baseline (if applicable)
        if metric=='auroc':
            metric_label = 'area under the receiver operating '\
                            'characteristic'
            xmin = 0.5
            xmax = 0.85
            ax[i].set_xlim([xmin, xmax])
        elif metric=='ap':
            metric_label = 'average precision'
            ax[i].plot([ppv, ppv], ylim, color=BLUE, linestyle='--')
            ax[i].text(ppv, sensor_set_ypos[-1]+1.75, 'PPV = %.2f' %ppv,
                horizontalalignment='center')
            xmin = 0
            xmax = 0.7
            ax[i].set_xlim([xmin, xmax])
        else:
            raise Exception('Unknown metric (expected "roc" or '\
                '"pr"):', metric)
        ax[i].set_xlabel(metric_label)

        # set y labels
        ax[i].set_ylim(ylim)
        ax[i].set_yticks(sensor_set_ypos)
        ax[i].set_yticklabels(df['sensor_label'])
        sensor_num_labels = ['1 IMU','2 IMUs','3 IMUs']
        for j in range(len(sensor_num_labels)):
            if metric == 'auroc':
                xpos = 0.41
            else: # ap
                xpos = -0.18
            ax[i].text(xpos,midpts[j], sensor_num_labels[j],
                rotation='vertical', verticalalignment='center')

        # indicate patient preference
        ypos = get_preferred_indices(sensor_set_ypos, df['full_label'],
                                     preferred_sensors)
        for y in ypos:
                ax[i].scatter(xmin + 0.015*(xmax-xmin), y + 0.02,
                              color=ORANGE, s=100, zorder=100)
        if n==4:
            ax[i].text(xpos, len(labels_to_combine)+5,
                'b) 11 IMUs', horizontalalignment='left', size=15,
                fontweight='bold')            
        elif n==7:
            ax[i].text(xpos, len(labels_to_combine)+4,
                'a) 6 IMUs', horizontalalignment='left', size=15,
                fontweight='bold')

        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        
        green_patch = mpatches.Patch(color=GREEN,
            label='best technical set')
        red_patch= mpatches.Patch(color=RED,
            label='minimal sensor set')
        orange_dot = plt.Line2D(range(1), range(1), color='white',
            marker='o', markerfacecolor=ORANGE, markersize=12,
            label='top-ranked participant set')

    plt.legend(handles=[green_patch, red_patch, orange_dot],
        loc='lower right', frameon=False)
    plt.savefig(RESULT_DIR + metric + '_summary.png', bbox_inches='tight')
