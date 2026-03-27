import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as stats
from scipy.signal import find_peaks
from scipy.ndimage import binary_dilation, binary_fill_holes
from PIL import Image
from BaselineRemoval import BaselineRemoval
from pathlib import Path

from analyze_suite2p import detector_utility, config_loader

_DEFAULT_CONFIG = config_loader.load_json_config_file()
config = _DEFAULT_CONFIG

def single_spine_peak_plotting(deltaF, threshold_multiplier):
    """
    Function to plot single synaptic trace with threshold and example peak detection
    NOTE: Function can be run iteratively if run in a `for` loop

    Args:
    ----------
        deltaF: NumPy Array
            Normalized Fluroescence from deltaF.npy file or calculated otherwise
        threshold_multiplier: float
            Number of MAD-estimated standard deviations above baseline to set peak detection threshold
            
    Returns:
    ----------
        None  
    """
    sigma, deltaF_baseline = detector_utility.estimate_single_trace_baseline_noise_mad(deltaF, event_threshold=2)
    
    baseline_reference = np.median(deltaF_baseline)
    peak_detection_multiplier = threshold_multiplier
    threshold = np.median(deltaF_baseline) + (peak_detection_multiplier * sigma)

    peaks, _ = find_peaks(deltaF, height = threshold, distance = 5, prominence = baseline_reference + sigma, width = (2,None))
    amplitudes = deltaF[peaks] - baseline_reference #amplitude
    plt.figure(figsize=[10,10])
    plt.plot(deltaF)
    plt.plot(peaks, deltaF[peaks], "x")

    plt.plot(np.full_like(deltaF, threshold), "--",color = "red", label = 'Threshold')
    plt.plot(np.full_like(deltaF, np.median(deltaF_baseline)), "--", color = 'black')

    print('the peak time stamps are :{}' .format(peaks))
    print('the amplitude of each peak is: {}' .format(amplitudes))
    plt.legend()
    plt.show()


def plot_raw_deltaF_vs_airPLS_correction(deltaF, lambda_values = [100,1000], ylim = (-0.2,0.9)):
    """
    Function to illustrate difference between normalized deltaF with and without baseline correction via airPLS

    Args:
    ----------
        deltaF (np.ndarray):
            Normalized fluorescence trace (e.g., from deltaF.npy).
        lambda_values (int or list[int], optional):
            Smoothing parameters used for baseline correction.
        ylim (tuple[float, float], optional):
            Y-axis limits for the plot.
    
    Returns:
    ----------
        None
    """
    # Define lambda values to test
    lambda_values = [1000, 100]

    # Create a figure for visualization
    plt.figure(figsize=(12, 6))
    plt.plot(deltaF, label='Original dF/F Trace', color='black', linewidth=1, alpha = 0.3)

    # Apply baseline correction with different lambda values and plot
    for lam in lambda_values:
        corrected_signal = BaselineRemoval(deltaF)
        trace = corrected_signal.ZhangFit(lambda_=lam, repitition=100)
        plt.plot(trace, label=f'Corrected (lambda={lam})', alpha = 0.5)

    # Show plot
    plt.legend()
    plt.title('Effect of Lambda on Baseline Correction (ZhangFit)')
    plt.xlabel('Time')
    plt.ylabel('dF/F')
    plt.ylim(ylim)
    plt.grid()
    plt.show()

def get_all_pkl_outputs_in_path(path):
    """
    Function to automatically gather all pkl files into a single list
    
    Args:
    ----------
    path: str
        Path to experiment folder that contains contains pkl files (e.g., in `pkl_files` folder) 
        
    Returns:
    ----------
    processed_files: List
        List of all .pkl files including full file path for easy access
    file_names: List
        List of all .pkl files that were found
    """
    processed_files = []
    file_names = []
    for root, directories, files in os.walk(path):
        for file_name in files:
            if file_name.endswith('.pkl'):
                full_path = os.path.join(root, file_name)
                processed_files.append(Path(full_path))
                file_names.append(file_name)
    return processed_files, file_names


def pynapple_plots(file_path, output_directory, treatment_vid = False, treatment_no = 2, synapse_count = 1000, plot_amplitudes = False,
                   max_amplitude = 5.0, plot_shape = 'square', save_fig = False):#, max_amplitude = 4):#, video_label):
    """
    Plotting function for implementing Pynapple plots for rasterplots and amplitudes of individual events
    This function can be called in sequence using a 'for' loop.

    Args:
    ----------
    file_path: str
        Path to individual pkl file
    output_directory: str
        Path to where output rasterplots should be saved (if save_fig)
    treatment_vid: bool
        Option to account for concatenated video with treatment in middle of provided pkl file frames
    treatment_no: int
        Number of treatments found in a treatment_vid
        NOTE: current implementation assumes that if treatment is 2 (baseline vs. treatment) 
        the treatment begins at the halfway point in the trace
    synapse_count: int
        Maximum number of synapses for setting y-axis limits
    plot_amplitudes: bool
        Decide whether to plot only rasterplots or also amplitude maps showing amplitudes of individual transients
    max_amplitude: float
        Maximum amplitude for establishing ylim for amplitude plots
        NOTE: can be ignored if plot_amplitudes = False
    plot_shape: str
        String to decide what size plot should be included (current options: 'square', 'rectangle', and 'rectangle_skinny')
        NOTE: all other shapes will cause the function to fail unless they are defined within the function beforehand
    save_fig: bool
        Boolean whether to save figure (when True) or display figures (when False)
        
    Returns:
    ----------
    if save_fig is True: .png and .svg files
        Saves pynapple output plots within experiment directory in subfolder 'rasterplots'
    
    if save_fig is False: plt.show()
    """
    import os
    import warnings
    import pickle
    import pynapple as nap
    import matplotlib.pyplot as plt

    warnings.filterwarnings('ignore')
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    df_cell_stats = data['cell_stats']
    my_tsd = {}
    for idx in df_cell_stats['SynapseID'][0:]:
        my_tsd[idx] = nap.Tsd(t=df_cell_stats[df_cell_stats['SynapseID']==idx]['PeakTimes'][idx],
                            d=df_cell_stats[df_cell_stats['SynapseID']==idx]['Amplitudes'][idx],time_units='s')

    #Make the figure
    plot_shapes = ['square','rectangle', "rectangle_skinny"]
    if plot_shape not in plot_shapes:
        raise ValueError(f"Acceptable Plot shapes are {plot_shapes}")
        
    if plot_shape =='square':
        plt.figure(figsize = (6,6))
    if plot_shape == 'rectangle_skinny':
        plt.figure(figsize=(12,4))
    if plot_shape == 'rectangle':
        plt.figure(figsize=(12,6))
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ["Arial"]
    if plot_amplitudes:
        plt.subplot(2,1,1) #
    else:
        plt.subplot(1,1,1)
    for i, idx in enumerate(df_cell_stats['SynapseID']):
#     
        spike_times = df_cell_stats[df_cell_stats['SynapseID'] == idx]['PeakTimes'].values[0]
        plt.scatter(spike_times, [i]*len(spike_times), color = '#4e4d4d', linewidths=0.1, s=1, alpha=0.6) #"#4e4d4d" edgecolors='black'f'C{idx}'
        if treatment_vid:
            if treatment_no == 0:
                pass
            if treatment_no == 1:
                plt.axvline(180, linestyle = 'dashed', color = 'red', linewidth = 0.1)
            if treatment_no == 2:
                plt.axvline(180, linestyle = 'dashed', color = 'red', linewidth = 0.1)
                plt.axvline(360, linestyle = 'dashed', color = 'red', linewidth = 0.1)

        plt.ylabel('SynapseID')
        plt.xlabel('Time (s)')
        if synapse_count is not None:
            plt.ylim(0,synapse_count)
        else:
            plt.ylim(0,1000)
        plt.tight_layout()

        plt.gca().spines[['top', 'right']].set_visible(False)  # cleaner plot

    if plot_amplitudes:
        plt.subplot(2,1,2)
        for i in range(1): #change range for multiple intervals
            for idx in my_tsd.keys():
                tsd = my_tsd[idx]
                plt.plot(tsd.index, tsd.values, color=f'C{idx}',marker='o',ls='',alpha=0.5)
            plt.ylabel('Amplitude')
            plt.ylim(0, max_amplitude)
            plt.xlabel('Spike Time (s)')
            plt.tight_layout()

    base_file_name = os.path.splitext(os.path.basename(file_path))[0]
    base_file_name = base_file_name.split("Dur180s")[0]

    if save_fig:
            
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        figure_output_path = os.path.join(output_directory, f'gray_{base_file_name}_figure_{plot_shape}.png')
        figure_output_path2 = os.path.join(output_directory, f'gray_{base_file_name}_figure_{plot_shape}.svg')

        plt.savefig(figure_output_path, dpi = 300)
        plt.savefig(figure_output_path2)
    else:
        plt.show()


def plot_synapse_traces(suite2p_dict, frame_rate = 20, trace_offset = 5, list = None, treatment_vid = False, treatment_no = 1, show_peaks = False, save_fig = False):
    """
    Plot multiple synapse fluorescence traces with optional peak detection.

    Traces are vertically offset for visualization and optionally annotated
    with detected peaks and treatment time markers.

    Args:
    ----------
        suite2p_dict (dict):
            Dictionary containing suite2p outputs (e.g., deltaF, iscell).
        frame_rate (int, optional):
            Imaging frame rate in Hz.
        trace_offset (float, optional):
            Vertical offset between traces.
        list (list[int], optional):
            Indices of synapses to plot. If None, randomly selects 10.
        treatment_vid (bool, optional):
            Whether to display treatment time markers.
        treatment_no (int, optional):
            Number of treatment events (1 or 2).
        show_peaks (bool, optional):
            Whether to overlay detected peaks.
        save_fig (bool, optional):
            Whether to save the figure to disk.

    Returns:
    ----------
        None
    """
    iscell_mask = suite2p_dict['iscell'][:,0] == 1  

    masked_dF = suite2p_dict['deltaF'][iscell_mask]
    
    if treatment_vid:
         if treatment_no ==1:
              treatment1 = 180
         if treatment_no ==2:
              treatment1 = 180
              treatment2 = 360
    if list is None:
        lst = np.random.choice(masked_dF.shape[0], size=10, replace=False)
    else:
        lst = list
    
    print(lst)

    plt.figure(figsize = (10,7))
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ["Arial"]
    ax = plt.gca()
    # --- Colorblind-friendly palette ---
    colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', 
            '#0072B2', '#D55E00', '#CC79A7', '#999999', '#000000', '#999933']
    
    frame_rate = 20
    time = np.arange(masked_dF.shape[1]) / frame_rate
    print("time range (s):", time[0], "->", time[-1])
    plt_traces = masked_dF[lst]
    for i, trace in enumerate(plt_traces):
        offset_trace = trace + i * trace_offset
        ax.plot(time, offset_trace, color='black', alpha=0.6)
        if show_peaks:
            peak_list = detector_utility.single_synapse_peak_detection(trace, return_peaks = True)
            ax.plot(peak_list, trace[peak_list], 'o', color = 'red')
    if treatment_vid == True and treatment_no ==  1:
         ax.axvline(treatment1, color='red', linestyle='--')
    if treatment_vid == True and treatment_no == 2:
         ax.axvline(treatment1, color='red', linestyle='--')
         ax.axvline(treatment2, color='red', linestyle='--')

    scalebar_time = 10  # seconds
    scalebar_df = 2     # dF/F units
    x0 = time[-1] + 2
    y0 = -2

    # Horizontal and vertical bars
    ax.plot([x0, x0 + scalebar_time], [y0, y0], 'k', lw=2)
    ax.plot([x0 + scalebar_time, x0 + scalebar_time], [y0, y0 + scalebar_df], 'k', lw=2)

    # Scale bar labels
    ax.text(x0 + scalebar_time / 2, y0 - 0.3, fr"{scalebar_time}$\ s$", ha='center', va='top')
    
    ax.text(x0 + scalebar_time + 1, y0 + scalebar_df / 3,  fr"{scalebar_df} $\Delta F / F_0$", va='center', ha='left')
    
    # --- Minimalist figure: no axes ---
    ax.axis('off')
    # --- Optional: save to file ---
    if save_fig:
            plt.savefig(os.path.join(suite2p_dict['file_name'], 'synapse_trace_examples_BW.svg'))
            plt.savefig(os.path.join(suite2p_dict['file_name'], 'synapse_trace_examples_BW.png'))

    plt.tight_layout()
    plt.show()
        
def getImg(ops, config):
    """
    Generate a normalized image from suite2p ops for ROI visualization.

    Args:
    ----------
        ops (dict):
            Suite2p ops dictionary containing imaging outputs.
        config (object):
            Configuration object with analysis parameters.

    Returns:
    ----------
        np.ndarray:
            Normalized 8-bit image for visualization.
    """
    Img = ops[config.analysis_params.Img_Overlay] # Option of  "max_proj" or "meanImg"
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
    """
    Compute the boundary pixels of a given ROI mask.
    Function is taken directly from suite2p src code.

    Args:
    ----------
        ypix (np.ndarray):
            Y-coordinates of ROI pixels.
        xpix (np.ndarray):
            X-coordinates of ROI pixels.

    Returns:
    ----------
        tuple[np.ndarray, np.ndarray]:
            Arrays of y and x coordinates representing the boundary pixels.
    """
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

def getStats(suite2p_dict, frame_shape, output_df, config, use_iscell = False):
    """
    Classify ROIs and compute spatial/statistical properties.

    ROIs are categorized into synaptic, dendritic, or rejected based on
    thresholds for peak count, skewness, and compactness.

    Args:
    ----------
        suite2p_dict (dict):
            Dictionary containing suite2p outputs (stat, F, Fneu, iscell).
        frame_shape (tuple[int, int]):
            Shape of the imaging frame (height, width).
        output_df (pandas.DataFrame):
            DataFrame containing peak detection results.
        config (object):
            Configuration object with analysis thresholds.
        use_iscell (bool, optional):
            If True, classification is based only on iscell flag.

    Returns:
    ----------
        tuple:
            scatters (dict): ROI boundary coordinates.
            nid2idx (dict): Mapping of ROI IDs to indices.
            nid2idx_rejected (dict): Rejected ROI indices.
            pixel2neuron (np.ndarray): Pixel-to-ROI mapping.
            synapse_ID (list): List of accepted synapse IDs.
            nid2idx_dendrite (dict): Dendritic ROI indices.
            nid2idx_synapse (dict): Synaptic ROI indices.
    """
    stat = suite2p_dict['stat']
    iscell = suite2p_dict['iscell']
    F = suite2p_dict['F']
    Fneu = suite2p_dict['Fneu']
    MIN_COUNT = int(config.analysis_params.peak_count_threshold) # minimum number of detected spikes for ROI inclusion
    MIN_SKEW = float(config.analysis_params.skew_threshold)
    MIN_COMPACT = float(config.analysis_params.compactness_threshold)
    MIN_PIXEL = 25 #int(config.analysis_params.pixel_threshold)
    # MAX_PIXEL = 100
    # MIN_FOOTPRINT = 0
    pixel2neuron = np.full(frame_shape, fill_value=np.nan, dtype=float)
    scatters = dict(x=[], y=[], color=[], text=[])
    nid2idx = {}
    nid2idx_rejected = {}
    nid2idx_dendrite = {}
    nid2idx_synapse = {}
    synapse_ID = []
    print(f"Number of detected ROIs: {stat.shape[0]}")
    
    if not use_iscell:

        for n in range(stat.shape[0]):
            peak_count = output_df.iloc[n]["PeakCount"]
            skew = stat.iloc[n]['skew']
            footprint = stat.iloc[n]['footprint']
            compact = stat.iloc[n]['compact']
            npix = stat.iloc[n]['npix']
            f = F[n]
            fneu = Fneu[n]

            if peak_count >= MIN_COUNT and skew >=MIN_SKEW:
                synapse_ID.append(n)
                nid2idx[n] = len(scatters["x"]) # Assign new idx

                if compact <= MIN_COMPACT:
                    nid2idx_synapse[n] = len(scatters["x"])
                else:
                    # if npix > 50:
                        nid2idx_dendrite[n] = len(scatters["x"])
                    # else:
                    #     nid2idx_rejected[n] = len(scatters["x"])
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

    return scatters, nid2idx, nid2idx_rejected, pixel2neuron, synapse_ID, nid2idx_dendrite, nid2idx_synapse

def dispPlot(MaxImg, scatters, nid2idx, nid2idx_rejected,nid2idx_dendrite, nid2idx_synapse,
             pixel2neuron, F, Fneu, save_path, fill_ROIs=False, axs=None):
             """
                Display ROI overlays on a background image.

                ROIs are visualized with different colors depending on classification
                (synaptic vs dendritic).

                Args:
                ----------
                    MaxImg (np.ndarray):
                        Background image (e.g., max projection).
                    scatters (dict):
                        ROI boundary coordinates.
                    nid2idx (dict):
                        Mapping of ROI IDs to indices.
                    nid2idx_rejected (dict):
                        Rejected ROI indices.
                    nid2idx_dendrite (dict):
                        Dendritic ROI indices.
                    nid2idx_synapse (dict):
                        Synaptic ROI indices.
                    pixel2neuron (np.ndarray):
                        Pixel-to-ROI mapping array.
                    F (np.ndarray):
                        Fluorescence traces.
                    Fneu (np.ndarray):
                        Neuropil signals.
                    save_path (str):
                        File path to save the output image.
                    fill_ROIs (bool, optional):
                        Whether to fill ROI regions instead of outlining.
                    axs (matplotlib.axes.Axes, optional):
                        Existing axes to plot on.

                Returns:
                ----------
                    None
             """
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
             print("Total ROI count: ", len(nid2idx))
             print("Synaptic Puncta: ", len(nid2idx_synapse))
             print("Dendritic Events: ", len(nid2idx_dendrite))
             norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True) 
             mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_rainbow) 

             def plotDict(n2d2idx_dict, override_color = None):
                 for neuron_id, idx in n2d2idx_dict.items():
                     color = override_color if override_color else mapper.to_rgba(scatters['color'][idx])
                            # print(f"{idx}: {scatters['x']} - {scatters['y'][idx]}")
                     if fill_ROIs:
                        ax1.fill(scatters["x"][idx], scatters["y"][idx], color = color, alpha = 0.5)     
                     else:
                        sc = ax1.scatter(scatters["x"][idx], scatters['y'][idx], color = color, 
                                      marker='.', s=5)
             plotDict(nid2idx_synapse, 'teal')
             plotDict(nid2idx_dendrite, 'orange')
             ax1.set_title(f"{len(nid2idx_synapse)} Synaptic Puncta (red) and {len(nid2idx_dendrite)} Dendritic Events (gold); {len(nid2idx)} Total ROIs") 

             plt.savefig(save_path)
             plt.close(fig)

def create_suite2p_ROI_masks(stat, frame_shape, nid2idx, output_path):
    """
    Generate and save ROI masks for external analysis tools.

    Creates a binary mask image where ROI pixels are labeled and saves
    it as an image file.

    Args:
    ----------
        stat (pandas.DataFrame):
            Suite2p stat DataFrame containing ROI pixel coordinates.
        frame_shape (tuple[int, int]):
            Shape of the imaging frame (height, width).
        nid2idx (dict):
            Mapping of ROI IDs to indices.
        output_path (str):
            File path to save the ROI mask image.

    Returns:
    ----------
        tuple:
            PIL.Image.Image: Saved image object.
            np.ndarray: ROI mask array.
    """
    
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
    plt.figure(figsize=(10, 10))
    plt.imshow(roi_masks, cmap='gray', interpolation='none')
    plt.title('ROI Mask')
    plt.tight_layout()
    plt.show()
    im = Image.fromarray(roi_masks)
    im.save(output_path)
    return im, roi_masks