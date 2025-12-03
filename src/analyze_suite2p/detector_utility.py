import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import scipy.signal as signal
from scipy.stats import norm
from analyze_suite2p import config_loader
from BaselineRemoval import BaselineRemoval

config = config_loader.load_json_config_file()

def calculate_deltaF(F_file):
    """Function to calculated dF from F and Fneu of suite2p based on Sun & Sudhof, 2019 dF/F calculations
    inputs: 
    F_file: F.npy file that serves as a template for understanding the fluorescence of individual ROIs"""

    savepath = rf"{F_file}".replace("\\F.npy","") ## make savepath original folder, indicates where deltaF.npy is saved
    F = np.load(rf"{F_file}", allow_pickle=True)
    Fneu = np.load(rf"{F_file[:-4]}"+"neu.npy", allow_pickle=True)
    deltaF= []
    for f, fneu in zip(F, Fneu):
        corrected_trace = f - (0.7*fneu) ## neuropil correction
        trace_median = np.median(corrected_trace)
        trace_mad = np.median(np.abs(corrected_trace - trace_median))
        norm_sigma = 1.4826*trace_mad
        baseline_mask = np.abs(corrected_trace - trace_median) < event_threshold * norm_sigma

        F0 = np.median(normalized_F[baseline_mask])
        # amount = int(0.125*len(corrected_trace))
        # middle = 0.5*len(corrected_trace)
        # F_sample = (np.concatenate((corrected_trace[0:amount], corrected_trace[int(middle-amount/2):int(middle+amount/2)], 
        #             corrected_trace[len(corrected_trace)-amount:len(corrected_trace)])))  #dynamically chooses beginning, middle, end 12.5%, changeable
        # #TODO decide if mean, median or mode is best for deltaF calculations
        # F_baseline = np.percentile(F_sample, 10)
        normalized_F = (corrected_trace-F0)/F0
        baseline_correction = BaselineRemoval(normalized_F)
        ZhangFit_normalized = baseline_correction.ZhangFit(lambda_= 10)
        deltaF.append(ZhangFit_normalized)
        
    deltaF = np.array(deltaF)
    deltaF = np.squeeze(deltaF)
    np.save(f"{savepath}/deltaF.npy", deltaF, allow_pickle=True)
    print(f"delta F calculated for {F_file[len(config.general_settings.main_folder)+1:-21]}")
    print(f"delta F traces saved as deltaF.npy under {savepath}\n")
    return deltaF


def estimate_baseline_noise_mad(F_trace, frame_rate, event_threshold = 3, min_baseline_sec = 10):
    """
    Estimate noise sigma from baseline-only windows using MAD.
    
    Parameters
    ----------
    F : 1D numpy array
        Baseline-corrected ΔF/F trace.
    frame_rate : float
        Sampling rate (Hz).
    event_thresh : float
        Threshold (in MAD units) to detect obvious events. Default is 3; 
        less produces too few baseline points
    min_baseline_sec : float
        Minimum duration (seconds) of a baseline window.
        Default: 10 s

    Returns
    -------
    sigma : float
        Estimated noise standard deviation.
    baseline_mask : boolean array
        Mask of samples classified as baseline.
    """

    trace_median = np.median(F_trace)
    mad = np.median(np.abs(F_trace - trace_median))
    sigma = 1.4826 * mad
    event_mask = np.abs(F_trace - trace_median) > event_threshold * sigma()

    trace_baseline = ~event_mask
    min_baseline_length = int(np.ceil(min_baseline_sec * frame_rate))
    baseline_mask = np.zeros_like(trace_baseline, dtype=bool)

    start = None
    for i, val in enumerate(trace_baseline):
        if val and start is None:
            start = i
        elif not val and start is not None:
            end = i
            if (end - start) >= min_baseline_length:
                baseline_mask[start:end] = True
            start = None
    
    if start is not None:
        end = len(trace_baseline)
        if (end - start) >= min_baseline_length:
            baseline_mask[start:end] = True

    baseline_samples = F_trace[baseline_mask]

    if len(baseline_samples) < 10:
        print("Not enough points to produce reliable baseline for synapse trace...please check manually")
        
        baseline_samples = F_trace
    
    baseline_median = np.median(baseline_samples)
    baseline_mad = np.median(np.abs(baseline_samples - baseline_median))
    sigma = 1.4826 * baseline_mad
    
    return sigma, baseline_mask

def filter_outliers(trace):
    q1,q3 = np.percentile(trace, [25,75])
    iqr = q3-q1
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr
    filtered_values = trace[(trace >= lower_bound) & (trace <= upper_bound)]
    return filtered_values   


def single_synapse_baseline_correction_and_peak_return(deltaF, return_peaks = False, 
                                                       return_decay_frames = False, 
                                                       return_amplitudes = False, 
                                                       return_decay_time = False,
                                                       return_peak_count = False):
    """this function takes a single time series data series and converts into deltaF / F; it then will return, frames where peaks occurred, amplitudes of peaks
    the number of peaks detected, the decay frames (TBD) and the decay time converted into sections"""
    
    sigma, deltaF_baseline = estimate_baseline_noise_mad(deltaF, frame_rate = 20, event_threshold=3, min_baseline_sec=10)
    
    baseline_reference = np.median(deltaF_baseline)
    peak_detection_multiplier = float(config.analysis_params.peak_detection_threshold)
    threshold = np.median(deltaF_baseline) + (peak_detection_multiplier * sigma)

    peaks, _ = find_peaks(deltaF, height = threshold, distance = 5, prominence = baseline_reference + sigma, width = (2,None))
    amplitudes = deltaF[peaks] - baseline_reference #amplitude
    peak_count = len(peaks)
    negative_points = np.where((deltaF < threshold))[0]

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
            decay_time.append(np.abs(decay - peak)/config.general_settings.frame_rate) #import framerate

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


def rolling_min(input_series, window_size):
    r = input_series.rolling(window_size, min_periods=1)
    m = r.min()
    return m


def remove_bleaching(input_trace):
    min_trace = rolling_min(pd.Series(input_trace), window_size=int(len(input_trace)/10))
    fit_coefficients = np.polyfit(range(len(min_trace)), min_trace, 2)
    fit = np.poly1d(fit_coefficients)
    return input_trace - fit(range(len(input_trace)))


