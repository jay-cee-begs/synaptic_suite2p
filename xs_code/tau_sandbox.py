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
# from configurations import *
import scipy.signal as signal
from scipy.optimize import curve_fit
# from configurations import *
from BaselineRemoval import BaselineRemoval
from sklearn.metrics import r2_score
from scipy.stats import norm


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

        if estimated_spikes >= MIN_COUNT and skew >=MIN_SKEW:

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
def filter_outliers(trace):
    q1,q3 = np.percentile(trace, [25,75])
    iqr = q3-q1
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr
    filtered_values = trace[(trace >= lower_bound) & (trace <= upper_bound)]
    return filtered_values

def rolling_min_sd(input_series, window_size):
    input_pandas = pd.Series(input_series)
    rolling_sd = input_pandas.rolling(window_size, min_periods=25).std()
    min_sd_idx = rolling_sd.idxmin()
    min_sd = rolling_sd.min()
    # m = r.min()
    return min_sd

def single_spine_peak_plotting(input_f,input_fneu):
    """Code for visualizing the thresholding and peak detection for dF/F baseline corrected traces; can be modified to be corrected by simple polynomial as well (see rolling_min, remove_bleaching functions)"""
    #---create deltaF array---#

    corrected_trace = input_f - (0.7*input_fneu) ## neuropil correction
    plt.plot(corrected_trace)
    plt.title("Neuropil Corrected Trace")
    plt.show()

    deltaF= []
    amount = int(0.125*len(corrected_trace))
    middle = 0.5*len(corrected_trace)
    F_sample = (np.concatenate((corrected_trace[0:amount], corrected_trace[int(middle-amount/2):int(middle+amount/2)], 
                corrected_trace[len(corrected_trace)-amount:len(corrected_trace)])))  #dynamically chooses beginning, middle, end 12.5%, changeable
    F_baseline = np.mean(F_sample)
    deltaF.append((corrected_trace-F_baseline)/F_baseline)
    deltaF = np.array(deltaF)
    deltaF = np.squeeze(deltaF)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(20, 20))
    ax1.plot(deltaF, color = 'blue', label = 'Normalized Fluorescence (DeltaF)')
    ax1.plot(np.full_like(deltaF, np.median(deltaF)), '--', color = 'red')
    # ax1.plot(np.full_like(3*rolling_min_sd(deltaF, window_size=int(len(deltaF)/50))))
    ax1.legend()

    #---Baseline_Correction---#
    bleached_deltaF = remove_bleaching(deltaF)
    ax2.plot(bleached_deltaF, color = 'orange',label=f'Poly_corrected_DeltaF')
    ax2.plot(np.full_like(bleached_deltaF, np.median(bleached_deltaF)), '--', color ='red')
    ax2.legend()

    deltaF_corr = BaselineRemoval(deltaF)
    deltaF_corr = deltaF_corr.ZhangFit()
    ax3.plot(deltaF_corr, color = 'green', label=f'ZhangFit_DeltaF')
    ax3.plot(np.full_like(deltaF_corr, np.median(deltaF_corr)), '--', color = 'red')
    ax3.legend()
    plt.show()

    #---calculate noise standard deviations--_#
    rolling_min_deltaF_sd = 3*rolling_min_sd(deltaF_corr, window_size=int(len(deltaF_corr)/50)) #smallest SD throughout trace
    iqr_noise = filter_outliers(bleached_deltaF) #iqr noise
    negative_points = np.where((deltaF_corr < np.median(deltaF_corr)))[0] #points less than median; flipped to make positive
    SD = 3*abs(np.std((bleached_deltaF[negative_points]))) + np.median(bleached_deltaF)
    #---histogram of all frames with iqr noise; gaussian curve and threshold over
    plt.figure(figsize=[15,5])
    mu, std = norm.fit(iqr_noise) #median and sd of noise of trace based on IQR
    plt.hist(bleached_deltaF, bins = 1000)
    threshold = np.median(bleached_deltaF) + 4 * std
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std) #gaussian fit over histograpm
    plt.plot(x, p, 'k', linewidth=2, label='Gaussian fit') #plt gaussian curve
    plt.axvline(x=np.median(bleached_deltaF), color='red', linewidth = 2, label = f'Median: {np.median(bleached_deltaF):.2f}')
    plt.axvline(x=threshold, color='r', linestyle='--', linewidth=2, label=f'IQR_Threshold: {threshold:.2f}') #threshold for peak detection 3 SD away
    plt.axvline(x=rolling_min_deltaF_sd + np.median(bleached_deltaF), color = 'green',linestyle='--', linewidth=2, label=f'Rolling_min_SD: {rolling_min_deltaF_sd:.2f}')
    plt.axvline(x=SD, color = 'blue',linestyle='--', linewidth=2, label=f'negative_points_SD: {SD:.2f}')
    plt.legend()
    plt.show()
    
    

    peaks, _ = find_peaks(bleached_deltaF, height = threshold, distance = 10) #frequency; 3*SD + abs(np.median(deltaF_corr))
    amplitudes = deltaF_corr[peaks] - np.median(bleached_deltaF) #amplitude
    #TODO need to figure out how to calculate the decay time efficiently; is it decay to noise; halfwidth AUC?
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
        

    print('the peak time stamps are :{}' .format(peaks))
    print('the peak returns to baseline at frame: {}' .format(decay_points))
    print('the decay time in seconds for each peak is: {}' .format(decay_time))
    print('the amplitude of each peak is: {}' .format(amplitudes))
    # decay_points = decay_points[~np.isnan(decay_points)]
    # plt.figure(figsize=[20,20])
    # plt.plot(bleached_deltaF)
    # plt.plot(peaks, bleached_deltaF[peaks], "x")
    # # plt.plot(decay_points, corrected_trace[decay_points], "x")
    # # plt.plot(np.full_like(bleached_deltaF, SD), "--",color = "black")
    # # plt.plot(np.full_like(bleached_deltaF, rolling_min_deltaF_sd), "--",color = "magenta")
    # plt.plot(np.full_like(bleached_deltaF, np.median(bleached_deltaF)), "--", color = 'r')
    # plt.plot(np.full_like(bleached_deltaF, threshold), "--", color = 'red')
    # plt.ylabel("dF/F")
    # plt.xlabel("Frame Number (20 frames = 1 sec)")
    # plt.title("1-D Polynomial Corrected DeltaF/F")

    # plt.show()
    d_mu, d_std = norm.fit(filter_outliers(deltaF_corr)) #median and sd of noise of trace based on IQR

    plt.figure(figsize=[20,20])
    # mu, std = norm.fit(filter_outliers(deltaF_corr)) #median and sd of noise of trace based on IQR

    d_peaks, _ = find_peaks(deltaF_corr, height = d_mu + 3.5*d_std, distance = 10) #frequency; 3*SD + abs(np.median(deltaF_corr))

    plt.plot(deltaF_corr)
    # plt.hist(deltaF_corr, bins = 1000)
    plt.plot(d_peaks, deltaF_corr[d_peaks], "x")
    # plt.plot(decay_points, corrected_trace[decay_points], "x")
    plt.plot(np.full_like(deltaF_corr, np.median(deltaF_corr)), "--", color = 'b')
    plt.plot(np.full_like(deltaF_corr, np.median(deltaF_corr) + 3.5*d_std), "--", color = 'red')
    plt.title("ZhangFit DeltaF/F")
    plt.ylabel("dF/F")
    # plt.ylim(0,1.5)
    plt.xlabel("Frame Number (20 frames = 1 sec)")
    plt.show()

    negative_points = np.where((bleached_deltaF < np.median(deltaF_corr)))[0]
    SD = np.std(abs(deltaF_corr[negative_points])) + np.median(deltaF_corr)
    peaks, _ = find_peaks(bleached_deltaF, height = SD*3, distance = 10)

    # plt.hist(deltaF_corr, bins = 1000)
    # plt.axvline(np.median(deltaF_corr), color = "red")
    # plt.axvline(SD + np.median(deltaF_corr), linestyle = '--', color = "blue")
    # plt.axvline(SD, linestyle = '--', color = "blue")
    # plt.ylabel("Number of Frames\nper bin (n=1000)")
    # plt.xlabel("dF/F")
    # plt.legend()

    # plt.axvline(3*SD + np.median(deltaF_corr), color = 'green')
    # plt.show()
        # scipy find_peaks function
        #then plot the traces you generate
    # plt.figure(figsize=[24,7])
    # plt.plot(corrected_trace)
    # corr_peaks, _ = find_peaks(corrected_trace, 2*(abs(np.std(corrected_trace)) + np.median(corrected_trace)),distance = 10)
    # plt.plot(corr_peaks, corrected_trace[corr_peaks], "x")
    # # plt.plot(np.full_like(corrected_trace, 2.5*(abs(np.median(corrected_trace)) + abs(corrected_trace.min()))), "--",color = "green")
    # # plt.plot(np.full_like(corrected_trace, 2.5*(np.std(corrected_trace))), '--', color = "blue")
    # # plt.plot(np.full_like(corrected_trace, 2*(np.median(corrected_trace))), "--",color = "green")
    # # plt.plot(np.full_like(corrected_trace, 3*(np.median(corrected_trace))), "--",color = "green")

    # plt.plot(np.full_like(corrected_trace, np.median(corrected_trace)), "--", color = 'r')

    # plt.show()
    #deltaF trace peaks
    # plt.figure(figsize=[24,7])
    # plt.plot(deltaF)
    # plt.plot(peaks, deltaF[peaks], "x")
    # # plt.plot(decay_points, corrected_trace[decay_points], "x")
    # plt.plot(np.full_like(deltaF, 2.5*(np.median(deltaF))+np.median(deltaF)), "--",color = "green")
    # # plt.plot(np.full_like(deltaF, 2*(np.median(deltaF))+np.median(deltaF)), "--",color = "green")
    # # plt.plot(np.full_like(deltaF, 3*(np.std(deltaF))+np.median(deltaF)), "--",color = "green")
    # plt.plot(np.full_like(deltaF, np.median(deltaF)), "--", color = 'r')
    # print('the peak time stamps are :{}' .format(peaks))
    # print('the peak returns to baseline at frame: {}' .format(decay_points))
    # print('the decay time in seconds for each peak is: {}' .format(decay_time))
    # print('the amplitude of each peak is: {}' .format(amplitudes))
    # plt.show()

    # #deltaF with baseline correction
    # delta_peaks, _ = find_peaks(deltaF_corr, height = 2.5*(np.median(deltaF_corr))+np.median(deltaF_corr), distance = 10) #frequency
    # plt.figure(figsize=[24,7])

    # plt.plot(deltaF_corr)
    # plt.plot(delta_peaks, deltaF_corr[delta_peaks], "x")
    # # plt.plot(decay_points, corrected_trace[decay_points], "x")
    # plt.plot(np.full_like(deltaF_corr, 2.5*(np.median(deltaF_corr))+np.median(deltaF_corr)), "--",color = "green")
    # # plt.plot(np.full_like(deltaF_corr, 2*(np.median(deltaF_corr))+np.median(deltaF_corr)), "--",color = "green")
    # # plt.plot(np.full_like(deltaF_corr, 3*(np.median(deltaF_corr))+np.median(deltaF_corr)), "--",color = "green")
    # plt.plot(np.full_like(deltaF_corr, np.median(deltaF_corr)), "--", color = 'r')
    # plt.show()


    
""" https://suite2p.readthedocs.io/en/latest/outputs.html explains the code to load a video's F / Fneu files (and others)
    With the F / Fneu files; we can iterate this function to see the outputs using the following:
    for f, fneu in zip(F, Fneu):
        single_spine_peak_plotting(f, fneu)
        
            corrected_trace = input_f - (0.7*input_fneu)
    corrected_trace = BaselineRemoval(corrected_trace)
    corrected_trace = corrected_trace.ZhangFit(repitition = 100)"""


# Define exponential functions
def ExpPlB(x, m, t, b):
    return m * np.exp(-t * x) + b

def ExpZero(x, m, t):
    return m * np.exp(-t * x)   # since b = 0

def doubleExpFit(x, a, b, c, d):
    return a * np.exp(-b * x) + c * np.exp(-d * x)


frame_rate = 20

def single_synapse_baseline_correction_and_peak_return(input_f, input_fneu, return_peaks = False, return_decay_frames = False, 
                                                       return_amplitudes = False, return_decay_time = False, return_peak_count = False,
                                                       calculate_tau = False):
    
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
    iqr_noise = filter_outliers(deltaF) #iqr noise
    mu, std = norm.fit(iqr_noise) #median and sd of noise of trace based on IQR
    threshold = np.median(deltaF) + 4 * std
    peaks, _ = find_peaks(deltaF, height = abs(np.median(deltaF) + 2.5*(abs(np.median(deltaF)))), distance = 10) #frequency
    amplitudes = deltaF[peaks] - np.median(deltaF) #amplitude
    filtered_peaks =[]
    filtered_amplitudes =  []
    for peak, amplitude, in zip(peaks,amplitudes):
        if amplitude < 0.1:
            filtered_peaks.append(peak)
            filtered_amplitudes.append(amplitude)
    peak_count = len(filtered_peaks)
    if return_peaks == True:
        return filtered_peaks
    if return_amplitudes == True:
        return filtered_amplitudes


# """
    threshold_points = np.where((deltaF < threshold))[0]
#     # print(negative_points)
    decay_points = []
    decay_time = []
#     tau_values_expPlB = []
#     tau_values_expZero = []
#     tau_values_doubleExp = []
#     r2_values_expPlB = []
#     r2_values_expZero = []
#     r2_values_doubleExp = []
    for peak1, peak2 in zip(peaks[:-1], peaks[1:]):
        negative_between_peaks = threshold_points[threshold_points>peak1]
        negative_between_peaks = negative_between_peaks[negative_between_peaks<peak2]
        if len(negative_between_peaks)>0:
            decay_points.append(negative_between_peaks[0])
        if len(negative_between_peaks)==0:
            decay_points.append(np.nan)
#     if calculate_tau == False:
            
#         if len(peaks)>0:
#             negative_after_last_peak = negative_points[negative_points>peaks[-1]]
#             if len(negative_after_last_peak)>0:
#                 decay_points.append(negative_after_last_peak[0])
#             else:
#                 decay_points.append(np.nan)
#         else:
#             decay_points = []

#         for peak,decay in zip(peaks, decay_points):
#             decay_time.append(np.abs(decay - peak)/frame_rate) #import framerate

#         decay_points = np.array(decay_points) #decay frames crossing baseline
#         decay_time = np.array(decay_time) #seconds after a calcium peak to return to baseline
#     if calculate_tau:
#         for peak, decay in zip(peaks, decay_points):
#             if not np.isnan(decay):
#                 x_data = np.arange(peak, decay) / frame_rate
#                 y_data = deltaF[peak:decay]

#                 # Fit ExpPlB
#                 try:
#                     popt, _ = curve_fit(ExpPlB, x_data, y_data, p0=(1, 1, 0))
#                     y_fit = ExpPlB(x_data, *popt)
#                     r2_values_expPlB.append(r2_score(y_data, y_fit))
#                     tau_values_expPlB.append(1 / popt[1])  # tau = 1 / t
#                 except:
#                     r2_values_expPlB.append(np.nan)
#                     tau_values_expPlB.append(np.nan)

#                 # Fit ExpZero
#                 try:
#                     popt, _ = curve_fit(ExpZero, x_data, y_data, p0=(1, 1))
#                     y_fit = ExpZero(x_data, *popt)
#                     r2_values_expZero.append(r2_score(y_data, y_fit))
#                     tau_values_expZero.append(1 / popt[1])  # tau = 1 / t
#                 except:
#                     r2_values_expZero.append(np.nan)
#                     tau_values_expZero.append(np.nan)

#                 # Fit doubleExpFit
#                 try:
#                     popt, _ = curve_fit(doubleExpFit, x_data, y_data, p0=(1, 1, 1, 1))
#                     y_fit = doubleExpFit(x_data, *popt)
#                     r2_values_doubleExp.append(r2_score(y_data, y_fit))
#                     tau_values_doubleExp.append((1 / popt[1], 1 / popt[3]))  # tau1 = 1 / b, tau2 = 1 / d
#                 except:
#                     r2_values_doubleExp.append(np.nan)
#                     tau_values_doubleExp.append((np.nan, np.nan))

#         avg_tau_expPlB = np.nanmean(tau_values_expPlB)
#         avg_tau_expZero = np.nanmean(tau_values_expZero)
#         avg_tau_doubleExp = np.nanmean([tau[0] for tau in tau_values_doubleExp if not np.isnan(tau[0])]), np.nanmean([tau[1] for tau in tau_values_doubleExp if not np.isnan(tau[1])])
#         avg_r2_expPlB = np.nanmean(r2_values_expPlB)
#         avg_r2_expZero = np.nanmean(r2_values_expZero)
#         avg_r2_doubleExp = np.nanmean(r2_values_doubleExp)
        
#         return peaks, amplitudes
#     # {
#     #         "avg_tau_expPlB": avg_tau_expPlB,
#     #         "avg_tau_expZero": avg_tau_expZero,
#     #         "avg_tau_doubleExp": avg_tau_doubleExp,
#     #         "avg_r2_expPlB": avg_r2_expPlB,
#     #         "avg_r2_expZero": avg_r2_expZero,
#     #         "avg_r2_doubleExp": avg_r2_doubleExp,
#     #         "peaks": peaks if return_peaks else None,
#     #         "decay_points": decay_points if return_decay_frames else None,
#     #         "amplitudes": amplitudes if return_amplitudes else None,
#     #         "decay_time": decay_time if return_decay_time else None,
#     #         "peak_count": peak_count if return_peak_count else None
#     #     }
#     # # if return_peaks == True:
#     #     return peaks
#     # if return_decay_frames == True:
#     #     return decay_points
#     # if return_amplitudes == True:
#     #     return amplitudes
#     # if return_decay_time == True:
#     #     return decay_time
#     # if return_peak_count ==True:
#     #     return peak_count
#     # if calculate_tau == True:
#     #     return decay_time
#  #The following are  lines of code written by Marti, who I generally trust to write cohesive and concise code
# # for coding; If you find my functions above to be inadequate you may impliment these instead
    
#    """ 

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


