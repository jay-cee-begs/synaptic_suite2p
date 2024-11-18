import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from scipy.signal import find_peaks
import pandas as pd
from scipy.ndimage import binary_dilation, binary_fill_holes
import scipy.stats as stats
import pickle
from PIL import Image
import seaborn as sns #needed for aggregated feature plots
# import pynapple as nap #TODO if you need Pynapple plots, you cannot use alongside cascade as it will break the code
from configurations import *
import scipy.signal as signal
from scipy.optimize import curve_fit
from BaselineRemoval import BaselineRemoval

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

def getStats(stat, frame_shape, output_df):
    MIN_COUNT = 2 #TODO: define for iscell; also figure exclude noncell
    MIN_SKEW = 1.0
    min_pixel = 25
    max_compact = 1.4
    min_footprint = 0
    max_footprint = 3
    pixel2neuron = np.full(frame_shape, fill_value=np.nan, dtype=float)
    scatters = dict(x=[], y=[], color=[], text=[])
    nid2idx = {}
    nid2idx_rejected = {}
    print(f"Number of detected ROIs: {stat.shape[0]}")
    for n in range(stat.shape[0]): #TODO change back to be based on csv file (estimated_spike count)
        estimated_spikes = output_df.iloc[n]["PeakCount"]
        skew = stat.iloc[n]['skew']
        footprint = stat.iloc[n]['footprint']

        if estimated_spikes >= MIN_COUNT and skew >=MIN_SKEW and footprint > min_footprint:

        # # min_skew, max_skew = min(skew, min_skew), max(skew, max_skew)
        # npix = stat.iloc[n]['npix']
        # footprint = stat.iloc[n]['footprint']
        # compact = stat.iloc[n]['compact']
        # # min_skew, max_skew = min(skew, min_skew), max(skew, max_skew)
        # if skew >= MIN_SKEW and footprint > min_footprint and npix > min_pixel and compact < max_compact:

        
            nid2idx[n] = len(scatters["x"]) # Assign new idx
        else:
            nid2idx_rejected[n] = len(scatters["x"])
        
        
        
        #----------------------------------------------------------------------------------------------------------#
            


        ypix = stat.iloc[n]['ypix'].flatten() - 1 #[~stat[n]['overlap']] - 1
        xpix = stat.iloc[n]['xpix'].flatten() - 1 #[~stat[n]['overlap']] - 1
        # print(f"Before filtering - xpix max: {xpix.max()}, ypix max: {ypix.max()}")

        valid_idx = (xpix>=0) & (xpix < frame_shape[1]) & (ypix >=0) & (ypix < frame_shape[0])
        ypix = ypix[valid_idx]
        xpix = xpix[valid_idx]
        # print(f'After filtering - xpix max: {xpix.max()}, ypix max: {ypix.max()}')
        yext, xext = boundary(ypix, xpix)
        scatters['x'] += [xext]
        scatters['y'] += [yext]
        pixel2neuron[ypix, xpix] = n
        scatters["color"].append(skew)
        scatters["text"].append(f"Cell #{n} - Skew: {skew}")
    # print("Min/max skew: ", min_skew, max_skew)
    # Normalize colors between 0 and 1
    # color_raw = np.array(scatters["color"])
    # scatters["color"] = (color_raw - min_skew) / (max_skew - min_skew)
    return scatters, nid2idx, nid2idx_rejected, pixel2neuron

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
             print("Synaptic count:", len(nid2idx))
            #  norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True) 
            #  mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_rainbow) 

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

def create_suite2p_ROI_masks(stat, frame_shape, nid2idx):
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


#####-------------------------------------------------------------------------------------------------------------------------#####


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


    
""" https://suite2p.readthedocs.io/en/latest/outputs.html explains the code to load a video's F / Fneu files (and others)
    With the F / Fneu files; we can iterate this function to see the outputs using the following:
    for f, fneu in zip(F, Fneu):
        single_spine_peak_plotting(f, fneu)
        
            corrected_trace = input_f - (0.7*input_fneu)
    corrected_trace = BaselineRemoval(corrected_trace)
    corrected_trace = corrected_trace.ZhangFit(repitition = 100)"""
   
    
def single_synapse_baseline_correction_and_peak_return(input_f, input_fneu, return_peaks = False, return_decay_frames = False, 
                                                       return_amplitudes = False, return_decay_time = False, return_peak_count = False):
    
    corrected_trace = input_f - (0.7*input_fneu) ## neuropil correction
    # corrected_trace = remove_bleaching(corrected)
    deltaF= []

    amount = int(0.125*len(corrected_trace))
    middle = 0.5*len(corrected_trace)
    F_sample = (np.concatenate((corrected_trace[0:amount], corrected_trace[int(middle-amount/2):int(middle+amount/2)], 
                corrected_trace[len(corrected_trace)-amount:len(corrected_trace)])))  #dynamically chooses beginning, middle, end 12.5%, changeable
    #TODO Changed np.mean to np.median()
    F_baseline = np.median(F_sample)
    deltaF.append((corrected_trace-F_baseline)/F_baseline)

    deltaF = np.array(deltaF)
    deltaF = np.squeeze(deltaF)
    deltaF = BaselineRemoval(deltaF)
    deltaF = deltaF.ZhangFit()
    # peaks, _ = find_peaks(deltaF, height = 2.5*(abs(np.median(deltaF)) + abs(deltaF.min())), distance = 10) #frequency
    peaks, _ = find_peaks(deltaF, height = 2*(abs(np.std(deltaF))+ abs(np.median(deltaF))), distance = 10)
    amplitudes = deltaF[peaks] - np.median(deltaF) #amplitude
    peak_count = len(peaks)

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
            decay_time.append(np.abs(decay - peak)/frame_rate) #import framerate

        decay_points = np.array(decay_points) #decay frames crossing baseline
        decay_time = np.array(decay_time) #seconds after a calcium peak to return to baseline

    if return_peaks == True:
        return peaks
    if return_decay_frames == True:
        return decay_points
    if return_amplitudes == True:
        return amplitudes
    if return_decay_time == True:
        return decay_time
    if return_peak_count ==True:
        return peak_count
    # if calculate_tau == True:
    #     return decay_time
 #The following are  lines of code written by Marti, who I generally trust to write cohesive and concise code
# for coding; If you find my functions above to be inadequate you may impliment these instead
    
    

def detect_spikes_by_mod_z(input_trace, **signal_kwargs):
    median = np.median(input_trace)
    deviation_from_med = np.array(input_trace) - median
    mad = np.median(np.abs(deviation_from_med))
    mod_zscore = deviation_from_med/(1.4826*mad)
    return signal.find_peaks(mod_zscore, **signal_kwargs)[0]


def plot_spikes(raw_trace, detector_func, detector_trace=None, **detector_kwargs):
    if detector_trace is None:
        detector_input_trace = raw_trace.copy()
    else:
        detector_input_trace = detector_trace.copy()
    spikes = detector_func(detector_input_trace, **detector_kwargs)
    plt.plot(range(len(raw_trace)), raw_trace, color="blue")
    for spk in spikes:
        plt.axvline(spk, color="red")
    plt.show()
    
    
    
    
# rolling_min / remove_bleaching are basic polynomial-based baseline corrections; this will not remove noise either
# these can be compared to the more complicated ZhangFit iteratively-weighted approach (airPLS method)

def rolling_min(input_series, window_size):
    r = input_series.rolling(window_size, min_periods=1)
    m = r.min()
    return m


def remove_bleaching(input_trace):
    min_trace = rolling_min(pd.Series(input_trace), window_size=int(len(input_trace)/10))
    fit_coefficients = np.polyfit(range(len(min_trace)), min_trace, 2)
    fit = np.poly1d(fit_coefficients)
    return input_trace - fit(range(len(input_trace)))


import pandas as pd
import numpy as np
import os

    #this is where all the detector functions will be used; at least initially


"""
SUITE2P_STRUCTURE describes the sequence of directories to traverse to arrive at the files named in the key.
For example: "F": ["suite2p", "plane0", "F.npy"] --> File describing F is found at ./suite2p/plane0/F.npy.
All locations are relative to the suite2p output suite2p_output, which is output into the suite2p_output / file location where the original 
.tiff image for analysis was located. suite2p output can also be identified by checking if it follows this structure.
Both spks and iscell are not used because: spks are not calculated; no need for deconvolution
iscell is not used because all ROIs are considered and filtered later based on individual stats
"""
SUITE2P_STRUCTURE = {
    "F": ["suite2p", "plane0", "F.npy"],
    "Fneu": ["suite2p", "plane0", "Fneu.npy"],
    #'spks': ["suite2p", "plane0", "spks.npy"],
    "stat": ["suite2p", "plane0", "stat.npy"],
    "iscell": ["suite2p", "plane0", "iscell.npy"],
    "ops": ["suite2p", "plane0", "ops.npy"]#,
    # "deltaF": ["suite2p","plane0","deltaF.npy"]
}
""" spks is not really necessary with our current set up since the spont. events are all pretty uniform, and are below 
    AP threshold (and therefore will not need to be deconvolved into action potentials themselves)"""

def load_npy_array(npy_path):
    """Function to load an np array from .npy file (e.g. F.npy / Fneu.npy)"""
    return np.load(npy_path, allow_pickle=True) #functionally equivalent to np.load(npy_array) but iterable; w/ Pickle


def load_npy_df(npy_path):
    """Function to load pd.DataFrame from npy file"""
    return pd.DataFrame(np.load(npy_path, allow_pickle=True)) #load suite2p outputs as pandas dataframe


def load_npy_dict(npy_path):
    """function to load dictionary from .npy file (e.g. stat.npy)"""
    return np.load(npy_path, allow_pickle=True)[()] #load .npy as dictionary

    

"""
The following 3 func. are used to translate_suite2p_outputs_to_csv;
check_for_suite2p_output is defined below: primarily for if iscell is not included (it always is)

Then, we append the suite2p_output location of suite2p outputs into the current path (found_output_paths = files in os.walk(path))
found_output_paths.append(current_path)
"""

def check_for_suite2p_output(folder_name_list):
    for folder in folder_name_list:
        location = os.path.join(folder, *SUITE2P_STRUCTURE["F"])
        if os.path.exists(location):
            continue
        if not os.path.isfile(os.path.join(folder, location)):
            return False
    return True


def get_all_suite2p_outputs_in_path(folder_path): ## accounts for possible errors if deltaF files have been created before
    file_names = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith("F.npy" ):
                    file_names.append(os.path.join(root, file)[:-21])

    if len(file_names)> 0:
        return file_names
    
    else:
        print("Suite2p files for this dataset do not exist yet")




def load_suite2p_output(path, use_iscell=False):
    """here we define our suite2p dictionary from the SUITE2P_STRUCTURE...see above"""
    suite2p_dict = {
        "F": load_npy_array(os.path.join(path, *SUITE2P_STRUCTURE["F"])),
        "Fneu": load_npy_array(os.path.join(path, *SUITE2P_STRUCTURE["Fneu"])),
        "stat": load_npy_df(os.path.join(path, *SUITE2P_STRUCTURE["stat"]))[0].apply(pd.Series),
        "ops": load_npy_array(os.path.join(path, *SUITE2P_STRUCTURE["ops"])).item()#,
        # "deltaF": load_npy_array(os.path.join(path, *SUITE2P_STRUCTURE["deltaF"]))
    }

    if use_iscell == False:
        suite2p_dict["IsUsed"] = [(suite2p_dict["stat"]["skew"] >= 1)] 

        suite2p_dict["IsUsed"] = pd.DataFrame(suite2p_dict["IsUsed"]).iloc[:,0:].values.T
        suite2p_dict["IsUsed"] = np.squeeze(suite2p_dict["IsUsed"])
    else:
        suite2p_dict["IsUsed"] = load_npy_df(os.path.join(path, *SUITE2P_STRUCTURE["iscell"]))[0].astype(bool)

    return suite2p_dict
"""
Possible to append this function further for synapse exclusion
 for example, append the document based on 
suite2p_dict["stat"] using values for ["skew"]/["npix"]/["compactness"]
"""


def translate_suite2p_dict_to_df(suite2p_dict):
    """this is the principle function in which we will create our .csv file structure; and where we will actually use
        our detector functions for spike detection and amplitude extraction"""
        
    spikes_per_neuron = [single_synapse_baseline_correction_and_peak_return(f_trace, fneu_trace, return_peaks = True) 
                             for (f_trace, fneu_trace) in zip(suite2p_dict["F"], suite2p_dict["Fneu"])]
    decay_points_after_peaks = [single_synapse_baseline_correction_and_peak_return(f_trace, fneu_trace, return_decay_frames = True)
                             for (f_trace, fneu_trace) in zip(suite2p_dict["F"], suite2p_dict["Fneu"])]
    spike_amplitudes = [single_synapse_baseline_correction_and_peak_return(f_trace, fneu_trace, return_amplitudes = True) 
                             for (f_trace, fneu_trace) in zip(suite2p_dict["F"], suite2p_dict["Fneu"])]
    decay_times = [single_synapse_baseline_correction_and_peak_return(f_trace, fneu_trace, return_decay_time = True)
                             for (f_trace, fneu_trace) in zip(suite2p_dict["F"], suite2p_dict["Fneu"])]
    peak_count = [single_synapse_baseline_correction_and_peak_return(f_trace, fneu_trace, return_peak_count= True)
                             for (f_trace, fneu_trace) in zip(suite2p_dict["F"], suite2p_dict["Fneu"])]
#spikes_per_neuron from single_cell_peak_return OUTPUT = list of np.arrays        
    df = pd.DataFrame({"IsUsed": suite2p_dict["IsUsed"],
                       "Skew": suite2p_dict["stat"]["skew"],
                       "PeakTimes": spikes_per_neuron,
                       "PeakCount": peak_count,
                    #    "PeakFreq": spikes_per_neuron TODO figure out if we can calculate all the coversions here before the pkl file
                       "Amplitudes": spike_amplitudes,
                        "DecayTimes": decay_times,
                       "DecayFrames": decay_points_after_peaks,
                       "Total Frames": len(suite2p_dict["F"].T)})
                       
    df.index.set_names("SynapseID", inplace=True)
    df["IsUsed"] = df["PeakCount"] >= 2
    df["Active_Synapses"] = df["IsUsed"].sum()

    # df.fillna(0, inplace = True) potentially for decay time calculations
    return df

def translate_suite2p_outputs_to_csv(input_path, overwrite=False, check_for_iscell=False, update_iscell = True):
    """This will create .csv files for each video loaded from out data fram function below.
        The structure will consist of columns that list: "Amplitudes": spike_amplitudes})
        
        col1: ROI #, col2: IsUsed (from iscell.npy); boolean, col3: Skew (from stats.npy); could be replaced with any 
        stat >> compactness, col3: spike frames (relative to input frames), col4: amplitude of each spike detected measured 
        from the baseline (the median of each trace)"""
    
    suite2p_outputs = get_all_suite2p_outputs_in_path(input_path)

    output_path = input_path+r"\csv_files"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    for suite2p_output in suite2p_outputs:
        output_directory = os.path.basename(suite2p_output)
        translated_path = os.path.join(output_path, f"{output_directory}.csv")
        if os.path.exists(translated_path) and not overwrite:
            print(f"CSV file {translated_path} already exists!")
            continue
                    #CHANGE POTENTIALLY
        suite2p_dict = load_suite2p_output(suite2p_output)
        suite2p_df = translate_suite2p_dict_to_df(suite2p_dict)

        suite2p_df.to_csv(translated_path)
        ###TODO CHANGE ASAP
        print(f"csv created for {suite2p_output}")

        # suite2p_dict = load_suite2p_output(suite2p_output, groups, input_path, use_iscell=check_for_iscell)
        # suite2p_dict = load_suite2p_output(suite2p_output, use_iscell=False)
        ops = suite2p_dict["ops"]
        Img = getImg(ops)
        scatters, nid2idx, nid2idx_rejected, pixel2neuron = getStats(suite2p_dict["stat"], Img.shape, suite2p_df)
        iscell_path = os.path.join(folder, *SUITE2P_STRUCTURE['iscell'])
        if update_iscell == True:
            updated_iscell = suite2p_dict['iscell']
            for idx in nid2idx:
                updated_iscell[idx,0] = 1.0
            for idxr in nid2idx_rejected:
                updated_iscell[idxr,0] = 0.0
            np.save(iscell_path, updated_iscell)
            print(f"Updated iscell.npy saved for {folder}")
        else:
            continue

        image_save_path = os.path.join(input_path, f"{suite2p_output}_plot.png") #TODO explore changing "input path" to "suite2p_output" to save the processing in the same 
        dispPlot(Img, scatters, nid2idx, nid2idx_rejected, pixel2neuron, suite2p_dict["F"], suite2p_dict["Fneu"], image_save_path)

    print(f"{len(suite2p_outputs)} .csv files were saved under {configurations.main_folder+r'/csv_files'}")
