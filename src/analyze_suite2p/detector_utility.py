import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
from PIL import Image
import scipy.signal as signal
from gui_config import gui_configurations as configurations
from scipy.stats import norm
from analyze_suite2p import suite2p_utility



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
    
#TODO make sure that deltaF conversion here actually works
def single_synapse_baseline_correction_and_peak_return(deltaF, return_peaks = False, 
                                                       return_decay_frames = False, 
                                                       return_amplitudes = False, 
                                                       return_decay_time = False,
                                                       return_peak_count = False):
    """this function takes a single time series data series and converts into deltaF / F; it then will return, frames where peaks occurred, amplitudes of peaks
    the number of peaks detected, the decay frames (TBD) and the decay time converted into sections"""
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


