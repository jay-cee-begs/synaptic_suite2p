import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import scipy.signal as signal
from scipy.stats import norm
from analyze_suite2p import config_loader
from BaselineRemoval import BaselineRemoval
import os

_DEFAULT_CONFIG = config_loader.load_json_config_file()
config = _DEFAULT_CONFIG

def calculate_deltaF(F_file, event_threshold = 2):
    """
    Convert raw fluorescence (F.npy) into change in fluorescence compared to baseline (dF / F0).

    Args:
    -----------
        F_file : str
            Path to NumPy array containing raw flourescence (F.npy) trace from suite2p.
        
        event_threshold: float
            Threshold (in MAD units) to mask obvious events by multiplying threshold by standard deviation. 
            The Default value is 3; smaller values will limit the number of baseline points used for correction.

    Returns:
    --------
        deltaF : 1D numpy array
            dF/F0 normalized fluorescence
            MAD baseline estimated
            ZhangFit / airPLS automated baseline correction
            deltaF is saved into the suite2p output folder generated from suite2p ROI detection.
    """

    savepath = rf"{F_file}".replace("\\F.npy","") ## make savepath original folder, indicates where deltaF.npy is saved
    F = np.load(rf"{F_file}", allow_pickle=True)
    Fneu = np.load(rf"{F_file[:-4]}"+"neu.npy", allow_pickle=True)
    deltaF= []
    for f, fneu in zip(F, Fneu):
        corrected_trace = f - (0.7*fneu) ## neuropil correction

        #Remove bleaching to generate change in Fluorescence
        baseline_corrected = BaselineRemoval(corrected_trace)
        airPLS_corrected = baseline_corrected.ZhangFit(lambda_= 10)

        #Determine baseline F0 value
        trace_median = np.median(corrected_trace)
        trace_mad = np.median(np.abs(corrected_trace - trace_median))
        norm_sigma = 1.4826*trace_mad
        baseline_mask = np.abs(corrected_trace - trace_median) < event_threshold * norm_sigma
        F0 = np.median(corrected_trace[baseline_mask])

        #calculate dF / F0
        normalized_F = (airPLS_corrected)/F0
        
        deltaF.append(normalized_F)
        
    deltaF = np.array(deltaF)
    deltaF = np.squeeze(deltaF)
    if not os.path.exists(f"{savepath}/deltaF.npy"):
        np.save(f"{savepath}/deltaF.npy", deltaF, allow_pickle=True)
        print(f"delta F traces saved as deltaF.npy under {savepath}\n")
    else:
        print(f"deltaF files already exist for {F_file[len(config.general_settings.main_folder)+1:-21]}")

    return deltaF


def estimate_single_trace_baseline_noise_mad(F_trace, event_threshold = 2):
    """
    Estimate noise sigma from baseline-only windows using MAD.
    
    Args:
    -----------
        F : 1D numpy array
            Baseline-corrected ΔF/F trace.
        frame_rate : float
            Sampling rate (Hz).
        event_threshold: float
            Preserved from calculate_deltaF function above.
            Threshold (in MAD units) to mask obvious events by multiplying by estimated noise standard deviation. 
            Default: 2 (SD above median)
            Smaller values will limit the number of baseline points used for correction.
        min_baseline_sec : float
            Minimum duration (seconds) of a baseline window.
            Default: 10 s

    Returns:
    --------
        sigma : float
            Estimated noise standard deviation.
        baseline_mask : boolean array
            Mask of samples classified as baseline.
    """

    trace_median = np.median(F_trace)
    mad = np.median(np.abs(F_trace - trace_median))
    sigma = 1.4826 * mad
    event_mask = np.abs(F_trace - trace_median) > event_threshold * sigma

    trace_baseline = ~event_mask

    baseline_samples = F_trace[trace_baseline]
    
    baseline_median = np.median(baseline_samples)
    baseline_mad = np.median(np.abs(baseline_samples - baseline_median))
    sigma = 1.4826 * baseline_mad
    
    return sigma, baseline_samples

def filter_outliers(trace):
    """
    Filter outliers (peaks) from calcium trace using the trace IQR

    Args:
    -----------
        trace : 1D numpy array
            Fluorescence trace (e.g., F.npy) from suite2p output.
        
    Returns:
    --------
        filtered_values : 1D array
            values from 1D array that fall within the original trace IQR.
    """
    q1,q3 = np.percentile(trace, [25,75])
    iqr = q3-q1
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr
    filtered_values = trace[(trace >= lower_bound) & (trace <= upper_bound)]
    return filtered_values   


def single_synapse_peak_detection(deltaF, return_peaks = False, 
                                                       return_decay_frames = False, 
                                                       return_amplitudes = False, 
                                                       return_decay_time = False,
                                                       return_peak_count = False, 
                                                       extract_peaks = False):
    """
    Identify time stamps and metrics of individual calcium spikes for a single ROI.
    
    Args:
    -----------
        deltaF: 1D numpy array
            Normalized fluroescence trace 
        return_peaks: bool, optional
            Returns time stamps (frame) for each peak
        return_decay_frames: bool, optional
            Returns time stamp (frame) for when each peak returns to threshold
            if no return to threshold --> returns NaN
        return_amplitudes: bool, optional
            Returns normalized amplitude for each detected peak
        return_decay_times: bool, optional
            Returns decay time from peak frame to crossing threshold (in seconds)
        return_peak_count: bool, optional
            Returns len(peaks)
        extract_peaks: bool, optional
            Returns peaks + 30 frames for peak library --> to be used for Tau calculations

    Returns:
    --------
        IF any==True:
            return_peaks: returns calcium spike time_stamps
            return_decay_frames: returns number of frames for calcium spike to decay to threshold
            return_amplitudes: returns amplitude of calcium spike in relation to baseline fluorescence (F0)
            return_decay_time: returns decay time in seconds (converts number of frames into seconds)
            return_peak_count: returns the total number of calcium spikes for the ROI fluorescence trace
            extract_peaks: returns deltaF window around calcium spike for calcium spike library 
    """
    
    sigma, deltaF_baseline = estimate_single_trace_baseline_noise_mad(deltaF, event_threshold=2)
    
    baseline_reference = np.median(deltaF_baseline)
    peak_detection_multiplier = 4.5# float(config.analysis_params.peak_detection_threshold)
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
    if extract_peaks:
        peak_dict = {}
        #TODO fix up these points for peak library
        # for peak in peaks:
        #     peak_dict.update(f'peak_{peak}': 'deltaF[peak:peak+30]')
        
        return peak_dict


def detect_spikes_by_mod_z(input_trace, **signal_kwargs):
    """
    Detect spikes by median absolute difference (MAD) of each frame from median of the trace.

    Args:
    -----------
        input_trace: 1D NumPy array
        **signal_kwargs: assorted see signal.find_peaks()
            Ex. width = (min,max), peak_prominence = type(float), height = type(float), threshold = type(float), distance = int/float
    Returns:
    --------
        peak time frames using signal.find_peaks() function

    """
    median = np.median(input_trace)
    deviation_from_med = np.array(input_trace) - median
    mad = np.median(np.abs(deviation_from_med))
    mod_zscore = deviation_from_med/(1.4826*mad)
    return signal.find_peaks(mod_zscore, **signal_kwargs)[0]


def plot_spikes(raw_trace, detector_func, detector_trace=None, **detector_kwargs):
    """
    Plot ROI calcium trace with overlayed detected spikes as red vertical lines.

    Args:
    -----------
        raw_trace: 1D NumPy array
        detector_func: Function
            Ex. scipy.signal.find_peaks() / detect_spikes_by_mod_z()
        detector_trace: bool, optional
        **detector_kwargs: assorted, optional
            Ex. scipy.signal.find_peaks(x, height = , threshold = , peak_prominence = , width = , distance = )

    Returns:
    -------- 
        matplotlib.pyplot.plot line graph
            blue: detector_trace (if true) or raw trace
            red: detected spikes
    """
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
    """
    Calculate rolling minimum value (input_series.rolling()) over different windows of the input trace.

    Args:
    -----------
        input_series: 1D NumPy array
            raw_trace / F.npy / deltaF.npy
        window_size: int
            Size of window to measure with each iteration

    Returns:
    -------- 
        m: int / float
            Smallest local minimum across all windows
    """
    r = input_series.rolling(window_size, min_periods=1)
    m = r.min()
    return m

def rolling_med(input_series, window_size):
    """
    Calculate rolling minimum value (input_series.rolling()) over different windows of the input trace.

    Args:
    -----------
        input_series: 1D NumPy array
            raw_trace / F.npy / deltaF.npy
        window_size: int
            Size of window to measure with each iteration

    Returns:
    -------- 
        m: int / float
            Smallest local minimum across all windows
    """
    r = input_series.rolling(window_size, min_periods=1)
    m = r.median()
    return m

def remove_bleaching(input_trace, baseline_correction, window = None):
    """
    Basic first-order polynomial function to remove bleaching from single ROI calcium imaging trace

    Args:
    -----------
        input_trace: 1D array
            raw fluorescence trace (F.npy or corrected: F.npy - 0.7*Fneu.npy)
            functions by processing one ROI at a time
        baseline_correction: str
            String name of function to call for removing bleaching from fluorescence trace
            Accepts 'rolling_min' or 'rolling_med' as possible values; all other values will break the function

    Returns:
    --------
        input_trace - fit(range(len(input_trace)))
            input trace adjusted by rolling minimum
            polynomial fit built on length of trace, rolling min values, and order of polynomial (e.g. 2nd)
            poly1d fits a 1 dimensional polynomial to the adjusted trace which is subtraced from the raw trace (input_Trace)

    """
    possible_corrections = ['rolling_min', 'rolling_med']
    if baseline_correction not in possible_corrections:
        print(f"Please enter a valid correction method: {possible_corrections}")
        return
    
    if baseline_correction == "rolling_min":
        if window is not None:
            corr_trace = rolling_min(pd.Series(input_trace), window_size = int(window))
        else:
            corr_trace = rolling_min(pd.Series(input_trace), window_size=int(len(input_trace)/10))
    if baseline_correction == "rolling_med":
        if window is not None:
            corr_trace = rolling_med(pd.Series(input_trace), window_size = int(window))
        else:
            corr_trace = rolling_med(pd.Series(input_trace), window_size = int(len(input_trace)/10))
    fit_coefficients = np.polyfit(range(len(corr_trace)), corr_trace, 2)
    fit = np.poly1d(fit_coefficients)
    return input_trace - fit(range(len(input_trace)))

