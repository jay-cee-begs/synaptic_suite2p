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
import scipy.signal as signal
from scipy.optimize import curve_fit
from gui_config import gui_configurations as configurations
from BaselineRemoval import BaselineRemoval
from scipy.stats import norm
from analyze_suite2p import suite2p_utility


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
    synapseID = []
    print(f"Number of detected ROIs: {stat.shape[0]}")
    for n in range(stat.shape[0]): #TODO change back to be based on csv file (estimated_spike count)
        estimated_spikes = output_df.iloc[n]["PeakCount"]
        skew = stat.iloc[n]['skew']
        footprint = stat.iloc[n]['footprint']

        if estimated_spikes >= MIN_COUNT and skew >=MIN_SKEW and footprint > min_footprint:
            synapseID.append(n)
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
    return scatters, nid2idx, nid2idx_rejected, pixel2neuron, synapseID

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

    iqr_noise = filter_outliers(deltaF) #iqr noise
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


    
""" https://suite2p.readthedocs.io/en/latest/outputs.html explains the code to load a video's F / Fneu files (and others)
    With the F / Fneu files; we can iterate this function to see the outputs using the following:
    for f, fneu in zip(F, Fneu):
        single_spine_peak_plotting(f, fneu)
        
            corrected_trace = input_f - (0.7*input_fneu)
    corrected_trace = BaselineRemoval(corrected_trace)
    corrected_trace = corrected_trace.ZhangFit(repitition = 100)"""
def filter_outliers(trace):
    q1,q3 = np.percentile(trace, [25,75])
    iqr = q3-q1
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr
    filtered_values = trace[(trace >= lower_bound) & (trace <= upper_bound)]
    return filtered_values   
    
def single_synapse_baseline_correction_and_peak_return(input_f, input_fneu, return_peaks = False, 
                                                       return_decay_frames = False, 
                                                       return_amplitudes = False, 
                                                       return_decay_time = False,
                                                       return_peak_count = False):
    """this function takes a single time series data series and converts into deltaF / F; it then will return, frames where peaks occurred, amplitudes of peaks
    the number of peaks detected, the decay frames (TBD) and the decay time converted into sections"""
    deltaF = suite2p_utility.calculate_deltaF(input_f)
    

    negative_points = np.where((deltaF < np.median(deltaF)))[0]
    iqr_noise = filter_outliers(deltaF) #iqr noise
    mu, std = norm.fit(iqr_noise) #median and sd of noise of trace based on IQR
    threshold = mu + 3.5 * std
    # peaks, _ = find_peaks(deltaF, height = 2.5*(abs(np.median(deltaF)) + abs(deltaF.min())), distance = 10) #frequency
    peaks, _ = find_peaks(deltaF, height = threshold, distance = 10)
    amplitudes = deltaF[peaks] - np.median(deltaF) #amplitude
    peak_count = len(peaks)
    # print(negative_points)
    decay_points = []
    decay_time = []
    
    if return_decay_time:
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
            decay_time.append(np.abs(decay - peak)/configurations.frame_rate) #import framerate

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
    if return_peak_count == True:
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


