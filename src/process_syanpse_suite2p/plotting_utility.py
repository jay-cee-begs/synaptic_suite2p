#I probably will rename these files so I can tell them apart when I have a lot of tabs open...

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
import pickle
import pynapple as nap

"""
below is an example structure for a dictionary for all the experiment files 
suite2one might implement this code as the following in a processing_pipeline:
    experiment_structure = {
    "control": {
        "replicate_1": [f"control_A_well_0{j}Dur180sInt100msBin500ms_filtered.pkl" for j in range(1, 5)],
        "replicate_2": [f"control_A_well_0{j}Dur180sInt100msBin500ms_filtered.pkl" for j in range(5, 9)],
    },
    "treatment": {
        "replicate_1": [f"treatment_B_well_0{j}Dur180sInt100msBin500ms_filtered.pkl" for j in range(1, 5)],
        "replicate_2": [f"treatment_B_well_0{j}Dur180sInt100msBin500ms_filtered.pkl" for j in range(5, 9)],
    }
}
experiment_structure = {


maybe consider a dictionary of dictionaries?? although I am not sure the benefit of this immediately
}
"""
_experiment_structure_example = {
    "control": {
        "dataset1": ["file1", "file2"]
    },
    "APV": {
        "dataset2": ["file3", "file4"]
    },
    "PDBu": {
        "dataset3": ["file1", "file2"]
    },
    "CNQX": {
        "dataset4": ["file3", "file4"]
    },
}


_available_tests = {
    "mann-whitney-u": stats.mannwhitneyu,
    "wilcoxon": stats.wilcoxon,
    "paired_t": stats.ttest_rel,
}
def get_significance_text(series1, series2, test="mann-whitney-u", bonferroni_correction=1, show_ns=False, 
                          cutoff_dict={"*":0.05, "**":0.01, "***":0.001, "****":0.00099}, return_string="{text}\n{pvalue:.4f}"):
    statistic, pvalue = _available_tests[test](series1, series2)
    levels, cutoffs = np.vstack(list(cutoff_dict.items())).T
    levels = np.insert(levels, 0, "n.s." if show_ns else "")
    text = levels[(pvalue < cutoffs.astype(float)).sum()]
    return return_string.format(pvalue=pvalue, text=text) #, text=text

def add_significance_bar_to_axis(ax, series1, series2, center_x, line_width):
    significance_text = get_significance_text(series1, series2, show_ns=True)
    
    original_limits = ax.get_ylim()
    
    ax.errorbar(center_x, original_limits[1], xerr=line_width/2, color="k", capsize=12)
    ax.text(center_x, original_limits[1], significance_text, ha="center", va="bottom", fontsize = 32)
    
    extended_limits = (original_limits[0], (original_limits[1] - original_limits[0]) * 1.2 + original_limits[0])
    ax.set_ylim(extended_limits)
    
    return ax

def aggregated_feature_plot(experiment_df, feature="SpikesFreq", agg_function="median", comparison_function="mean",
                            palette="Set3", significance_check=False, group_order=None, control_group = None, ylim = 0, y_label = "", x_label = ""):
    """
    Add a 'group_order' parameter that takes a list of groups in the desired order.
    """
    grouped_df = experiment_df.groupby(["group", "file_name"]).agg(agg_function).reset_index(drop=False) #"dataset",

    if control_group is not None:
        control_avg = grouped_df[grouped_df['group'] == control_group][feature].agg(comparison_function)

        grouped_df[feature] = grouped_df[feature].apply(lambda x: (x / control_avg) * 100)
    else:
        grouped_df = grouped_df

    fig = plt.figure(figsize=(48, 16))
    ax = fig.add_subplot()
    color_palette = sns.color_palette(palette)
    def get_kde(data):
        """
        Custom function to calculate the quartiles and add dashed lines to the violin plots.
        """
        q25, q50, q75 = np.percentile(data, [25, 50, 75])
        ax.axvline(q25, linestyle="--", color="black", alpha=1, ymin=0, ymax=1)
        ax.axvline(q50, linestyle="--", color="black", alpha=1, ymin=0, ymax=1)
        ax.axvline(q75, linestyle="--", color="black", alpha=1, ymin=0, ymax=1)
        return sns.kdeplot(data, color="black", ax=ax, bw_adjust=0.5, clip=(0, 1))

    # Use 'group_order' in the sns.violinplot call to control the order of groups on the x-axis.
    sns.violinplot(x="group", y=feature, data=grouped_df, ax=ax, palette=palette, order=group_order,
                    inner="quartile", width=0.5) #, scale = 'area' inner="quartile",   get_kde=get_kde, , fontsize=44
    # Use 'group_order' in the sns.violinplot call to control the order of groups on the x-axis.
    # sns.violinplot(x="group", y=feature, data=grouped_df, ax=ax, palette=palette, order=group_order) #hue="dataset"

    # marker_width = 0.5
    # Update tick_positions to match the provided 'group_order' if it's specified, ensuring the custom order is used.
    if group_order:
        tick_positions = {group: pos for pos, group in enumerate(group_order)}
    else:
        tick_positions = {ax.get_xticklabels()[index].get_text(): ax.get_xticks()[index] for index in
                          range(len(ax.get_xticklabels()))}

    # for group, group_df in grouped_df.groupby("group"):
    #     for dataset_index, (dataset, dataset_df) in enumerate(group_df.groupby("dataset")):
    #         feature_mean = dataset_df[feature].agg(comparison_function)
    #         ax.plot([tick_positions[group] - marker_width/2, tick_positions[group] + marker_width/2],
    #                 [feature_mean, feature_mean], "-", color=color_palette[dataset_index], lw=2)

    # Your significance check and plotting logic remains unchanged

    if significance_check:
        sub_checks = [significance_check] if not any(isinstance(element, list) for element in significance_check) else significance_check
        for sub_check in sub_checks:
            add_significance_bar_to_axis(ax, 
                                 grouped_df[grouped_df["group"] == sub_check[0]][feature], 
                                 grouped_df[grouped_df["group"] == sub_check[1]][feature],
                                (tick_positions[sub_check[0]] + tick_positions[sub_check[1]]) / 2,
                                abs(tick_positions[sub_check[0]] - tick_positions[sub_check[1]]))

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)

    ax.set_ylim([ylim, ax.get_ylim()[1]])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(y_label, fontsize = 64)
    ax.set_xlabel(x_label, fontsize = 64)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=44)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=44)

    return fig






def build_experiment_dfs(input_path, experiment_structure):
    experiment_cell_stats, experiment_binned_stats = pd.DataFrame(), pd.DataFrame()

    for group in experiment_structure.keys():
        for dataset in experiment_structure[group].keys():
            try:
                for file_name in experiment_structure[group][dataset]:
                    # try: TODO trouble shoot why file not found error pops up here
                        file_dict = pd.read_pickle(os.path.join(input_path, file_name))
                        cell_stats, binned_stats = file_dict["cell_stats"], file_dict["binned_stats"]
                        
                        for stats, experiment_stats in zip((cell_stats, binned_stats), 
                                                        (experiment_cell_stats, experiment_binned_stats)):
                            stats[["group", "dataset", "file_name"]] = group, dataset, file_name
                        experiment_cell_stats = pd.concat((experiment_cell_stats, cell_stats))
                        experiment_binned_stats = pd.concat((experiment_binned_stats, binned_stats))
            except Exception as e:
                        print(f"Error processing file {file_name}: {e}")
                
    return experiment_cell_stats, experiment_binned_stats

def get_all_pkl_outputs_in_path(path):
    processed_files = []
    file_names = []
    for current_path, directories, files in os.walk(path):
        for file_name in files:
            full_path = os.path.join(current_path, file_name)
            processed_files.append(full_path)
        file_names.append(files)
    return processed_files, file_names[0]


def pynapple_plots(file_path, output_directory, max_amplitude):#, video_label):
    import warnings
    warnings.filterwarnings('ignore')
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    df_cell_stats = data['cell_stats']
    
    
    my_tsd = {}
    for idx in df_cell_stats['SynapseID'][0:]:
        my_tsd[idx] = nap.Tsd(t=df_cell_stats[df_cell_stats['SynapseID']==idx]['PeakTimes'][idx],
                            d=df_cell_stats[df_cell_stats['SynapseID']==idx]['Amplitudes'][idx],time_units='s')
        
    Interval_1 = nap.IntervalSet(0,180)

    # Interval_2 = nap.IntervalSet(250,290)
    # Interval_3 = nap.IntervalSet(290,450)
    
    interval_set = [Interval_1]#,
                #Interval_2]
    
    #Make the figure
    plt.figure(figsize=(6,6))
    plt.subplot(2,1,1)
    for i, idx in enumerate(df_cell_stats['SynapseID']):
#     
        plt.eventplot(df_cell_stats[df_cell_stats['SynapseID']==idx]['PeakTimes'],lineoffsets=i,linelength=0.8)
#     
        plt.ylabel('SynapseID')
        plt.xlabel('Time (s)')
        plt.tight_layout()
    plt.subplot(2,1,2)
    for i in range(1): #change range for multiple intervals
        # plt.title(file_path)
        # plt.title(f'interval {i+1}')
        for idx in my_tsd.keys():
            plt.plot(my_tsd[idx].restrict(interval_set[i]).index,my_tsd[idx].restrict(interval_set[i]).values,color=f'C{idx}',marker='o',ls='',alpha=0.5)
        plt.ylabel('Amplitude')
        plt.ylim(0,max_amplitude)
        plt.xlabel('Spike time (s)')
        plt.tight_layout()

    base_file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    #Check if output 
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    figure_output_path = os.path.join(output_directory, f'{base_file_name}_figure.png')

    plt.savefig(figure_output_path)
    plt.show()

            ## You can then just group the amplitude as you want for later analysis

    transient_count = []
    for idx in my_tsd.keys():
        transient_count.append(my_tsd[idx].restrict(interval_set[0]).shape[0])
        
