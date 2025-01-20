import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm
from scipy.signal import find_peaks
from scipy.ndimage import binary_dilation, binary_fill_holes
from PIL import Image
from BaselineRemoval import BaselineRemoval
import pickle
import pynapple as nap

from gui_config import gui_configurations as configurations
from analyze_suite2p import suite2p_utility
from analyze_suite2p import detector_utility


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
        "replicate01": ["file1", "file2"]
    },
    "treatment1": {
        "replicate01": ["file3", "file4"]
    },
    "treatment2": {
        "replicate01": ["file1", "file2"]
    },
    "treatment3": {
        "replicate01": ["file3", "file4"]
    },
}


_available_tests = {
    "mann-whitney-u": stats.mannwhitneyu,
    "wilcoxon": stats.wilcoxon,
    "paired_t": stats.ttest_rel,
}
def get_significance_text(series1, series2, test="mann-whitney-u", bonferroni_correction=1, show_ns=False, 
                          cutoff_dict={"*":0.05, "**":0.01, "***":0.001, "****":0.00099}, return_string="{text}\n{pvalue:.4f}"):
    
    """Function to calculate significance for data to plot
    INPUTS:
    series 1: first sample
    series 2: second sample
    test: chosen from scipy.stats"""

    statistic, pvalue = _available_tests[test](series1, series2)
    levels, cutoffs = np.vstack(list(cutoff_dict.items())).T
    levels = np.insert(levels, 0, "n.s." if show_ns else "")
    text = levels[(pvalue < cutoffs.astype(float)).sum()]
    return return_string.format(pvalue=pvalue, text=text) #, text=text

def add_significance_bar_to_axis(ax, series1, series2, center_x, line_width):
    """Function to add significance stars to figures"""
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
    Plot summary statistics for aggregated calcium imaging dataframe
    INPUTS:
    experiment_df: parent dataframe
    feature: column to aggregate in df
    agg_function: mean or median (median less sensitive to outliers)
    comparison_function: comparing between groups
    signficance_check: list of lists of groups to compare
    group_order: order of experimental conditions
    control_group: for normalizing values to a control conditions
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
    """Function to build experiment dataframe from pickle files; currently performs for binned stats and cell stats; only need cell stats in future"""
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
    """raster plot function for all graphs; also can plot event amplitude"""
    
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
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    figure_output_path = os.path.join(output_directory, f'{base_file_name}_figure.png')

    plt.savefig(figure_output_path)
    plt.show()


    transient_count = []
    for idx in my_tsd.keys():
        transient_count.append(my_tsd[idx].restrict(interval_set[0]).shape[0])
        
def getImg(ops):
    """Accesses suite2p ops file (itemized) and pulls out a composite image to map ROIs onto"""
    Img = ops["max_proj"] # Also "max_proj", "meanImg", "meanImgE"
    mimg = Img # Use suite-2p source-code naming
    mimg1 = np.percentile(mimg,1)
    mimg99 = np.percentile(mimg,99)
    mimg = (mimg - mimg1) / (mimg99 - mimg1)
    mimg = np.maximum(0,np.minimum(1,mimg))
    mimg *= 255
    mimg = mimg.astype(np.uint8)
    return mimg

    #redefine locally suite2p.gui.utils import boundary
def boundary(ypix,xpix):
    """ returns pixels of mask that are on the exterior of the mask """
    ypix = np.expand_dims(ypix.flatten(),axis=1)
    xpix = np.expand_dims(xpix.flatten(),axis=1)
    npix = ypix.shape[0]
    if npix>0:
        msk = np.zeros((np.ptp(ypix)+6, np.ptp(xpix)+6), bool) 
        msk[ypix-ypix.min()+3, xpix-xpix.min()+3] = True
        msk = binary_dilation(msk)
        msk = binary_fill_holes(msk)
        k = np.ones((3,3),dtype=int) # for 4-connected
        k = np.zeros((3,3),dtype=int); k[1] = 1; k[:,1] = 1 # for 8-connected
        out = binary_dilation(msk==0, k) & msk

        yext, xext = np.nonzero(out)
        yext, xext = yext+ypix.min()-3, xext+xpix.min()-3
    else:
        yext = np.zeros((0,))
        xext = np.zeros((0,))
    return yext, xext

#gets neuronal indices

def getStats(suite2p_dict, frame_shape, output_df, use_iscell = False):
    stat = suite2p_dict['stat']
    iscell = suite2p_dict['iscell']
    MIN_COUNT = 2 # minimum number of detected spikes for ROI inclusion
    MIN_SKEW = 1.0
    # min_pixel = 25
    # min_footprint = 0
    pixel2neuron = np.full(frame_shape, fill_value=np.nan, dtype=float)
    scatters = dict(x=[], y=[], color=[], text=[])
    nid2idx = {}
    nid2idx_rejected = {}
    synapse_ID = []
    print(f"Number of detected ROIs: {stat.shape[0]}")
    
    if not use_iscell:

        for n in range(stat.shape[0]):
            peak_count = output_df.iloc[n]["PeakCount"]
            skew = stat.iloc[n]['skew']
            # footprint = stat.iloc[n]['footprint']
            # npix = stat.iloc[n]['npix']

            if peak_count >= MIN_COUNT and skew >=MIN_SKEW:
                synapse_ID.append(n)
                nid2idx[n] = len(scatters["x"]) # Assign new idx
            else:
                nid2idx_rejected[n] = len(scatters["x"])
            
            ypix = stat.iloc[n]['ypix'].flatten() - 1 #[~stat.iloc[n]['overlap']] - 1
            xpix = stat.iloc[n]['xpix'].flatten() - 1 #[~stat.iloc[n]['overlap']] - 1

            valid_idx = (xpix>=0) & (xpix < frame_shape[1]) & (ypix >=0) & (ypix < frame_shape[0])
            ypix = ypix[valid_idx]
            xpix = xpix[valid_idx]
            yext, xext = boundary(ypix, xpix)
            scatters['x'] += [xext]
            scatters['y'] += [yext]
            pixel2neuron[ypix, xpix] = n
    else:
        for n in range(stat.shape[0]):

            if iscell[n,0]:
                nid2idx[n] = len(scatters["x"]) # Assign new idx
            else:
                nid2idx_rejected[n] = len(scatters["x"])

            ypix = stat.iloc[n]['ypix'].flatten() - 1 #[~stat.iloc[n]['overlap']] - 1
            xpix = stat.iloc[n]['xpix'].flatten() - 1 #[~stat.iloc[n]['overlap']] - 1

            valid_idx = (xpix>=0) & (xpix < frame_shape[1]) & (ypix >=0) & (ypix < frame_shape[0])
            ypix = ypix[valid_idx]
            xpix = xpix[valid_idx]
            yext, xext = boundary(ypix, xpix)
            scatters['x'] += [xext]
            scatters['y'] += [yext]
            pixel2neuron[ypix, xpix] = n

    return scatters, nid2idx, nid2idx_rejected, pixel2neuron, synapse_ID

def dispPlot(MaxImg, scatters, nid2idx, nid2idx_rejected,
             pixel2neuron, F, Fneu, save_path, axs=None):
             if axs is None:
                fig = plt.figure(constrained_layout=True)
                NUM_GRIDS=12
                gs = fig.add_gridspec(NUM_GRIDS, 1)
                ax1 = fig.add_subplot(gs[:NUM_GRIDS-2])
                fig.set_size_inches(12,14)
             else:
                 ax1 = axs
                 ax1.set_xlim(0, MaxImg.shape[0])
                 ax1.set_ylim(MaxImg.shape[1], 0)
             ax1.imshow(MaxImg, cmap='gist_gray')
             ax1.tick_params(axis='both', which='both', bottom=False, top=False, 
                             labelbottom=False, left=False, right=False, labelleft=False)
             print("Synapse count:", len(nid2idx))
             norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True) 
             mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_rainbow) 

             def plotDict(n2d2idx_dict, override_color = None):
                 for neuron_id, idx in n2d2idx_dict.items():
                     color = override_color if override_color else mapper.to_rgba(scatters['color'][idx])
                            # print(f"{idx}: {scatters['x']} - {scatters['y'][idx]}")
                            
                     sc = ax1.scatter(scatters["x"][idx], scatters['y'][idx], color = color, 
                                      marker='.', s=1)
             plotDict(nid2idx, 'g')
            #  plotDict(nid2idx_rejected, 'm')
             ax1.set_title(f"{len(nid2idx)} Synapses used (green) out of {len(nid2idx)+len(nid2idx_rejected)} potential detected (magenta - rejected)") 

             plt.savefig(save_path)
             plt.close(fig)

def create_suite2p_ROI_masks(stat, frame_shape, nid2idx, output_path):
    """Function designed to do what was done above, except mask the ROIs for detection in other programs (e.g. FlouroSNNAP)"""
    #Make an empty array to contain the nid2idx masks
    roi_masks = np.zeros(frame_shape, dtype=int)

    #Iterate through the ROIs in nid2idx and fill in the masks
    for n in nid2idx.keys():
        ypix = stat.iloc[n]['ypix'].flatten() - 1
        xpix = stat.iloc[n]['xpix'].flatten() - 1

        #Ensure the indices are within the bounds of the frame_shape

        valid_idx = (xpix >= 0) & (xpix<frame_shape[1]) & (ypix >=0) & (ypix < frame_shape[0])
        ypix = ypix[valid_idx]
        xpix = xpix[valid_idx]

        #Set ROI pixels to mask

        roi_masks[ypix, xpix] = 255 # n + 1 helps to differentiate masks from background
    # plt.figure(figsize=(10, 10))
    # plt.imshow(roi_masks, cmap='gray', interpolation='none')
    # # plt.colorbar(label='ROI ID')
    # plt.title('ROI Mask')
    # plt.tight_layout()
    # plt.show()
    im = Image.fromarray(roi_masks)
    im.save(output_path)
    return im, roi_masks



def single_spine_peak_plotting(input_f,input_fneu):
    
    corrected_trace = input_f - (0.7*input_fneu) ## neuropil correction
    # corrected_trace = remove_bleaching(corrected)
    deltaF= []

    amount = int(0.125*len(corrected_trace))
    middle = 0.5*len(corrected_trace)
    F_sample = (np.concatenate((corrected_trace[0:amount], corrected_trace[int(middle-amount/2):int(middle+amount/2)], 
                corrected_trace[len(corrected_trace)-amount:len(corrected_trace)])))  #dynamically chooses beginning, middle, end 12.5%, changeable
    F_baseline = np.mean(F_sample)
    deltaF.append((corrected_trace-F_baseline)/F_baseline)

    iqr_noise = detector_utility.filter_outliers(deltaF) #iqr noise
    mu, SD = norm.fit(iqr_noise) #median and sd of noise of trace based on IQR
    threshold = mu + 3.5 * SD

    deltaF = np.array(deltaF)
    deltaF = np.squeeze(deltaF)
    # corrected_trace = deltaF
    # deltaF_corr = remove_bleaching(deltaF)
    deltaF_corr = BaselineRemoval(deltaF)
    deltaF_corr = deltaF_corr.ZhangFit()
    peaks, _ = find_peaks(deltaF, height = abs(2.5*(abs(np.median(deltaF)) + abs(deltaF.min()))), distance = 10) #frequency
    amplitudes = deltaF[peaks] - np.median(deltaF) #amplitude

    negative_points = np.where((deltaF < np.median(deltaF)))[0]
    # print(negative_points)
    decay_points = []
    decay_time = []
    for peak1, peak2 in zip(peaks[:-1], peaks[1:]):
        negative_between_peaks = negative_points[negative_points>peak1]
        negative_between_peaks = negative_between_peaks[negative_between_peaks<peak2]
        if len(negative_between_peaks)>0:
            decay_points.append(negative_between_peaks[0])
        if len(negative_between_peaks)==0:
            decay_points.append(np.nan)

    if len(peaks)>0:
        negative_after_last_peak = negative_points[negative_points>peaks[-1]]
        if len(negative_after_last_peak)>0:
            decay_points.append(negative_after_last_peak[0])
        else:
            decay_points.append(np.nan)

    else:

        decay_points = []

    for peak,decay in zip(peaks, decay_points):
        decay_time.append(np.abs(decay - peak)/20)

        # scipy find_peaks function
        #then plot the traces you generate
    plt.figure(figsize=[24,7])
    plt.plot(deltaF)
    plt.plot(peaks, deltaF[peaks], "x")
    # plt.plot(decay_points, corrected_trace[decay_points], "x")
    plt.plot(np.full_like(deltaF, 2.5*(abs(np.median(deltaF)) + abs(deltaF.min()))), "--",color = "grey")
    plt.plot(np.full_like(deltaF, 2*(abs(np.median(deltaF)) + abs(deltaF.min()))), "--",color = "grey")
    plt.plot(np.full_like(deltaF, 3*(abs(np.median(deltaF)) + abs(deltaF.min()))), "--",color = "grey")
    plt.plot(np.full_like(deltaF, np.median(deltaF)), "--", color = 'r')
    print('the peak time stamps are :{}' .format(peaks))
    print('the peak returns to baseline at frame: {}' .format(decay_points))
    print('the decay time in seconds for each peak is: {}' .format(decay_time))
    print('the amplitude of each peak is: {}' .format(amplitudes))
    plt.show()
    delta_peaks, _ = find_peaks(deltaF_corr, height = abs(2.5*(abs(np.median(deltaF_corr)) + abs(deltaF_corr.min()))), distance = 10) #frequency
    plt.figure(figsize=[24,7])

    plt.plot(deltaF_corr)
    plt.plot(delta_peaks, deltaF_corr[delta_peaks], "x")
    # plt.plot(decay_points, corrected_trace[decay_points], "x")
    plt.plot(np.full_like(deltaF_corr, 2.5*(abs(np.median(deltaF_corr)) + abs(deltaF_corr.min()))), "--",color = "grey")
    plt.plot(np.full_like(deltaF_corr, 2*(abs(np.median(deltaF_corr)) + abs(deltaF_corr.min()))), "--",color = "grey")
    plt.plot(np.full_like(deltaF_corr, 3*(abs(np.median(deltaF_corr)) + abs(deltaF_corr.min()))), "--",color = "grey")
    plt.plot(np.full_like(deltaF_corr, np.median(deltaF_corr)), "--", color = 'r')
    plt.show()

    plt.figure(figsize=[24,7])
    plt.hist(deltaF_corr, bins = 1000)
    plt.axvline(np.median(deltaF_corr), color = "red")
    plt.axvline(SD, linestyle = '--', color = "blue")
    plt.axvline(3*SD + np.median(deltaF_corr), color = 'green')
    plt.legend()

