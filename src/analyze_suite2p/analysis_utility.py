
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from analyze_suite2p import suite2p_utility, detector_utility, plotting_utility, config_loader

_DEFAULT_CONFIG = config_loader.load_json_config_file()
config = _DEFAULT_CONFIG

def calculate_synapse_frequency(input_df):
    """
    Calculate spike frequency metrics for each ROI.

    Args:
    -----
        input_df : pandas.DataFrame
            DataFrame containing at least the 'PeakTimes' and 'Total_Frames'
            columns.

    Returns:
    --------
        pandas.DataFrame
            A copy of the input DataFrame with added columns:
            - SpikesCount
            - SpikesFreq
            - SpikesCV

    """
    output_df = input_df.copy()
    frame_rate = config.general_settings.frame_rate
    output_df["SpikesCount"] = output_df["PeakTimes"].str.len()
    output_df["SpikesFreq"] = output_df["SpikesCount"] / ((output_df["Total_Frames"] / frame_rate)) #divide by total # of frames NOT framerate
    output_df['SpikesCV'] = output_df['PeakTimes'].apply(lambda x: pd.Series(x).std()) / output_df['SpikesFreq'] * 100
    output_df['BaseSpikesCV'] = output_df['BasePeakTimes'].apply(lambda x: pd.Series(x).std()) / output_df['BaseSpikesFreq'] * 100
    output_df['PDBuSpikesCV'] = output_df['PDBuPeakTimes'].apply(lambda x: pd.Series(x).std()) / output_df['PDBuSpikesFreq'] * 100
    output_df['APVSpikesCV'] = output_df['APVPeakTimes'].apply(lambda x: pd.Series(x).std()) / output_df['APVSpikesFreq'] * 100

    return output_df

def calculate_synapse_isi(input_df): #isi == interspike interval
    """
    Calculate inter-spike interval (ISI) statistics for each ROI.

    Args:
    -----
        input_df : pandas.DataFrame
            DataFrame containing a 'PeakTimes' column of lists of spike frames.

    Returns:
    --------
        pandas.DataFrame
            A DataFrame with added ISI-related columns:
            - SpikesDiff
            - DiffAvg
            - DiffMedian
            - DiffCV

    """
    output_df = input_df.copy()
    output_df["SpikesDiff"] = output_df["PeakTimes"].apply(lambda x: list(pd.Series(x).diff().dropna()))
    output_df["DiffAvg"] = output_df["SpikesDiff"].apply(lambda x: pd.Series(x).mean())
    output_df["DiffMedian"] = output_df["SpikesDiff"].apply(lambda x: pd.Series(x).median())
    output_df["DiffCV"] = output_df["SpikesDiff"].apply(lambda x: pd.Series(x).std()) / output_df["DiffAvg"] * 100

    return output_df

def calculate_spike_amplitudes(input_df):
    """
    Calculate amplitude metrics for calcium spikes per ROI.

    Args:
    -----
        input_df : pandas.DataFrame
            DataFrame containing an 'Amplitudes' column with lists of peak
            amplitudes.

    Returns:
    --------
        pandas.DataFrame
            DataFrame with added amplitude metrics:
            - AvgAmplitude
            - SpkAmpMedian
            - SpkAmpCV

    """
    output_df = input_df.copy()
    output_df["AvgAmplitude"] = output_df["Amplitudes"].apply(lambda x: pd.Series(x).mean())
    output_df["SpkAmpMedian"] = output_df["Amplitudes"].apply(lambda x: pd.Series(x).median())
    output_df["SpkAmpCV"] = output_df["Amplitudes"].apply(lambda x: pd.Series(x).std()) / output_df["AvgAmplitude"] * 100

    return output_df

def calculate_decay_fraction(row):
    """
    Calculate the fraction of calcium events that decay back to baseline.

    Args:
    -----
        row : pandas.Series
            Single row containing 'DecayCount' and 'SpikesCount'.

    Returns:
    --------
        float
            Fraction of spikes with valid decay (or empty list if no spikes).

    """

    if row["SpikesCount"] != 0:
        return row['DecayCount'] / row["SpikesCount"]
    else:
        return []
        

def calculate_decay_values(input_df):
    """
    Calculate decay-related metrics for each ROI.

    Args:
    -----
        input_df : pandas.DataFrame
            DataFrame containing a 'DecayTimes' column, where each entry is a list
            of decay durations.

    Returns:
    --------
        pandas.DataFrame
            DataFrame with added decay metrics:
            - DecayCount
            - DecayedFraction
            - AvgDecayTime
            - AvgDecayCV

    """

    output_df = input_df.copy()
    output_df["DecayCount"] = output_df["DecayTimes"].apply(lambda arr: (len([x for x in arr if not pd.isna(x)])))
    output_df["DecayedFraction"] = output_df.apply(calculate_decay_fraction, axis =1)
    output_df["AvgDecayTime"] = output_df["DecayTimes"].apply(lambda x: pd.Series(x).dropna().mean())
    output_df["AvgDecayCV"] = output_df["DecayTimes"].apply(lambda x: pd.Series(x).dropna().std()) / output_df["AvgDecayTime"] * 100
    return output_df


def calculate_cell_stats(input_df, calculate_freq=True, calculate_isi=True, calculate_amplitudes=True, calculate_decays = True):
    """
    Compute spike frequency, ISI, amplitude, and decay metrics for each ROI.

    Args:
    -----
        input_df : pandas.DataFrame
            Input DataFrame of ROI-level Suite2p metrics.
        calculate_freq : bool, optional
            Whether to compute frequency metrics.
        calculate_isi : bool, optional
            Whether to compute ISI metrics.
        calculate_amplitudes : bool, optional
            Whether to compute amplitude metrics.
        calculate_decays : bool, optional
            Whether to compute decay metrics.

    Returns:
    --------
        pandas.DataFrame
            DataFrame with the selected statistical metrics added.

    """

    output_df = input_df.copy()
    if calculate_freq:
        output_df = calculate_synapse_frequency(output_df)
    if calculate_isi:
        output_df = calculate_synapse_isi(output_df)
    if calculate_amplitudes:
        output_df = calculate_spike_amplitudes(output_df)
    if calculate_decays:
        output_df = calculate_decay_values(output_df)
    return output_df

def translate_suite2p_dict_to_df(suite2p_dict, config):
    """
    Translate Suite2p output dictionaries into raw and processed DataFrames.

    Event detection, amplitude extraction, decay extraction, and ROI
    classification are performed using detector and plotting utilities.

    Args:
    -----
        suite2p_dict : dict
            Dictionary produced by suite2p_utility.load_suite2p_output().

    Returns:
    --------
        tuple of pandas.DataFrame
            (raw_df, processed_df)
            raw_df : unfiltered ROI data
            processed_df : ROIs with full computed metrics and filtering applied

    """

    def process_individual_synapse(deltaF):
        """
        Analyzes deltaF fluorescence to detect peaks
        
        This function is used to find peaks, the peak times,
        decay times, and normalized amplitudes

        Args:
        -----
            deltaF: 1D array

        Returns:
        --------
            peaks: 1D array
                Frame index for where peaks were detected
            amplitudes: 1D array
                List of amplitudes (peak - np.median(trace))
            decay_times: 1D array
                List of decay times in seconds for peak to decay to threshold, or NaN
            peak_count: int
                len(peaks) --> count of total peaks for a synapse
            decay_frames: 1D array
                number of frames for peak to decay back to threshold 
        """
        
        peaks = detector_utility.single_synapse_peak_detection(deltaF, return_peaks = True)
        amplitudes = detector_utility.single_synapse_peak_detection(deltaF, return_amplitudes=True)
        decay_times = detector_utility.single_synapse_peak_detection(deltaF, return_decay_time = True)
        peak_count = detector_utility.single_synapse_peak_detection(deltaF, return_peak_count=True)
        decay_frames = detector_utility.single_synapse_peak_detection(deltaF, return_decay_frames=True)
        return peaks, amplitudes, peak_count, decay_times, decay_frames

    results = []

    for idx, deltaF in enumerate(suite2p_dict["deltaF"]):
        result = process_individual_synapse(deltaF)
        
        results.append(result)
    spikes_per_neuron, spike_amplitudes, peak_count, decay_times, decay_frames = zip(*results)
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     results = list(executor.map(lambda args: process_individual_synapse(*args), zip(suite2p_dict["F"], suite2p_dict["Fneu"])))
    # spikes_per_neuron, decay_points_after_peaks, spike_amplitudes, decay_times, peak_count = zip(*results)
#spikes_per_neuron from single_cell_peak_return OUTPUT = list of np.arrays        
    total_frames = len(suite2p_dict['deltaF'].T)
    n_vids = 3
    vid_len = int(total_frames / 3)
    base_count = []
    baseline_peaks = []
    base_amp = []
    PDBu_count = []
    PDBu_peaks = []
    PDBu_amp = []
    APV_count = []
    APV_peaks = []
    APV_amp = []

    for roi_spikes, roi_amplitudes in zip(spikes_per_neuron, spike_amplitudes):
        roi_base_peaks = []
        roi_pdbu_peaks = []
        roi_apv_peaks = []

        roi_base_amp = []
        roi_pdbu_amp = []
        roi_apv_amp = []
        for peak, amplitude in zip(roi_spikes, roi_amplitudes):
    
            if peak <= vid_len:
                roi_base_peaks.append(peak)
                roi_base_amp.append(amplitude)
            
            elif vid_len < peak <= 2*vid_len:
                roi_pdbu_peaks.append(peak)
                roi_pdbu_amp.append(amplitude)
            else:
                roi_apv_peaks.append(peak)
                roi_apv_amp.append(amplitude)
        
        baseline_peaks.append(roi_base_peaks)
        base_amp.append(roi_base_amp)
        PDBu_peaks.append(roi_pdbu_peaks)
        PDBu_amp.append(roi_pdbu_amp)
        APV_peaks.append(roi_apv_peaks)
        APV_amp.append(roi_apv_amp)
        
        base_count.append(len(roi_base_peaks))
        PDBu_count.append(len(roi_pdbu_peaks))
        APV_count.append(len(roi_apv_peaks))
        
#############################################
####TODO Concatenated traces are currently hardcoded; this would need to be fixed in the future
    print(len(suite2p_dict["IsUsed"]))
    print(len(suite2p_dict["stat"]["skew"]))
    print(len(spikes_per_neuron))
    print(len(baseline_peaks))
    print(len(PDBu_peaks))
    print(len(APV_peaks))
    print(len(spike_amplitudes))
    print(len(base_amp))
    print(len(PDBu_amp))
    print(len(APV_amp))
    print(len(decay_times))
    print(len(decay_frames))
    print(len(base_count))
    
    df = pd.DataFrame({"IsUsed": suite2p_dict["IsUsed"],
                       "Skew": suite2p_dict["stat"]["skew"],
                       "PeakTimes": spikes_per_neuron,
                       "BasePeakTimes": baseline_peaks,
                       "PDBuPeakTimes": PDBu_peaks,
                       "APVPeakTimes": APV_peaks,
                       "PeakCount": peak_count, #TODO figure out if we can calculate all the coversions here before the pkl file
                       "BaseCount": base_count,
                       "PDBuCount": PDBu_count,
                       "APVCount": APV_count,
                       "Amplitudes": spike_amplitudes,
                       "BaseAmplitudes": base_amp,
                       "PDBuAmplitudes": PDBu_amp,
                       "APVAmplitudes": APV_amp,
                       "DecayTimes": decay_times,
                       "DecayFrames": decay_frames,
                       "Total_Frames": len(suite2p_dict["F"].T),
                       "Experimental_Group": suite2p_dict['Group'],
                       "Replicate_No.": suite2p_dict['sample'],
                       "File_Name": suite2p_dict['file_name']
                       })
                       
    df.index.set_names("SynapseID", inplace=True)
    Img = plotting_utility.getImg(suite2p_dict["ops"], config)
    scatters, nid2idx, nid2idx_rejected, pixel2neuron, synapse_ID, nid2dx_dendrite, nid2idx_synapse = plotting_utility.getStats(suite2p_dict, Img.shape, df, config, use_iscell = config.analysis_params.use_suite2p_ROI_classifier)
    
    df['classification'] = 'none'
    for n in range(len(df)):
        if n in nid2dx_dendrite:
            df.at[n,"classification"] = 'dendritic_event'
        elif n in nid2idx_synapse:
            df.at[n,"classification"] = 'synaptic_event'
    
    filtered_df = df[df['IsUsed']==True]

    processed_df = calculate_cell_stats(filtered_df)
    filtered_columns = processed_df.columns[0:7]
    processed_df = processed_df.drop(filtered_columns, axis = 1)
    processed_df = processed_df[processed_df['SpikesCount']>=1]

    return df, processed_df#, aggregate_stats

def translate_suite2p_outputs_to_csv(main_folder, config, check_for_iscell=False, update_iscell = True):
    """
    Convert Suite2p output folders into raw and processed CSV files.

    Args:
    -----
        main_folder : str
            Path containing Suite2p output folders.
        check_for_iscell : bool, optional
            Whether to classify ROIs using Suite2p's iscell.npy.
        update_iscell : bool, optional
            Whether to overwrite the iscell.npy file based on reclassification.

    Returns:
    --------
        None

    """
    suite2p_outputs = suite2p_utility.get_all_suite2p_outputs_in_path(main_folder, "samples", supress_printing=True)

    output_path = os.path.join(main_folder,"csv_files")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for suite2p_output in suite2p_outputs:
        output_directory = os.path.basename(suite2p_output)
        translated_path = os.path.join(output_path, f"{output_directory}.csv")
        processed_path = os.path.join(output_path, f"processed_{output_directory}.csv")
        
        suite2p_dict = suite2p_utility.load_suite2p_output(suite2p_output, config, use_iscell=check_for_iscell)
        
        raw_data, processed_data = translate_suite2p_dict_to_df(suite2p_dict, config)

        ops = suite2p_dict["ops"]
        Img = plotting_utility.getImg(ops, config)
        scatters, nid2idx, nid2idx_rejected, pixel2neuron, synapseID, nid2idx_dendrite, nid2idx_synapse = plotting_utility.getStats(suite2p_dict, Img.shape, raw_data, config, use_iscell=check_for_iscell)
        iscell_path = os.path.join(suite2p_output, *suite2p_utility.SUITE2P_STRUCTURE['iscell'])
        parent_iscell = suite2p_utility.load_npy_array(iscell_path)
        updated_iscell = parent_iscell.copy()
        if update_iscell:
            for idx in nid2idx:
                updated_iscell[idx,0] = 1.0
            for idxr in nid2idx_rejected:
                updated_iscell[idxr,0] = 0.0
            np.save(iscell_path, updated_iscell)
            print(f"Updated iscell.npy saved for {suite2p_output}")
        else:
            print("Using iscell from suite2p to classify ROIs")

        synapse_key = list(synapseID)
        raw_data['IsUsed'] = raw_data.index.isin(synapse_key)# .loc[synapse_key, 'IsUsed'] = True

        processed_data['synapse_ROI'] = len(nid2idx_synapse)
        processed_data['dendrite_ROI'] = len(nid2idx_dendrite)
        processed_data['total_ROIs'] = len(nid2idx)

        raw_data.to_csv(translated_path)
        processed_data.to_csv(processed_path)
        print(f"csvs created for {suite2p_output}")

        image_save_path = os.path.join(main_folder, f"{suite2p_output}_plot.png") #TODO add GUI config for choosing image type to save default should be .png and fix svg output so text is scaled correctly
        plotting_utility.dispPlot(Img, scatters, nid2idx, nid2idx_rejected, nid2idx_dendrite, nid2idx_synapse,
                                   pixel2neuron, suite2p_dict["F"], suite2p_dict["Fneu"], image_save_path, fill_ROIs=True)

    print(f"{len(suite2p_outputs)} .csv files were saved under {Path(config.general_settings.main_folder) / 'csv_files'}")

def create_experiment_summary(main_folder):
    """
    Create merged experiment-level summary CSV files from processed ROI data.

    Args:
    -----
        main_folder : str
            Path containing the 'csv_files' directory.

    Returns:
    --------
        tuple
            (aggregate_stats, merged_df)
            aggregate_stats : grouped summary metrics
            merged_df : concatenated per-recording data
    """
    csv_file_path = os.path.join(main_folder, 'csv_files')
    csv_files = list_all_files_of_type(csv_file_path, '.csv')
    processed_csvs = [file for file in csv_files if file.startswith('processed')]
    df_list = [pd.read_csv(os.path.join(csv_file_path, csv)) for csv in processed_csvs]
    merged_df = pd.concat(df_list, ignore_index=True)
    agg_columns = merged_df.select_dtypes(['float64', 'int'])
    aggregate_stats = agg_columns.groupby([merged_df['File_Name'], merged_df['classification']]).agg(['mean', 'std', 'median']).reset_index()

# Include non-numeric columns in the final aggregated dataframe
    aggregate_stats['Experimental_Group'] = merged_df.groupby(['File_Name', 'classification'])['Experimental_Group'].first().values
    aggregate_stats['Replicate_No.'] = merged_df.groupby(['File_Name', 'classification'])['Replicate_No.'].first().values
    main_group = main_folder.split('\\')[-1]
    merged_df.to_csv(os.path.join(main_folder, f'{main_group}_experiment_summary.csv'))
    aggregate_stats.to_csv(os.path.join(main_folder, f'{main_group}_aggregate_summary.csv'))
    
    return aggregate_stats, merged_df

def list_all_files_of_type(input_path, filetype):
    """
    List all files in a directory with a specific extension.

    Args:
    -----
        input_path : str
            Directory to search.
        filetype : str
            File extension filter.

    Returns:
    --------
        list of str
            Filenames matching the requested extension.

    """

    return [file for file in os.listdir(input_path) if file.endswith(filetype)]

def string_to_list_translator(input_string, strip_before_split="[ ]", split_on=" "):
    """
    Convert a delimited string into a cleaned list of tokens.

    Args:
    -----
        input_string : str
            String containing delimited values.
        strip_before_split : str, optional
            Characters to strip from both ends.
        split_on : str, optional
            String delimiter for splitting.

    Returns:
    --------
        list
            Cleaned list of non-empty string tokens.

    """

    split_string = input_string.strip(strip_before_split).split(split_on)
    return list(filter(None, split_string))

def spike_list_translator(input_string):
    """
    Convert a spike frame index string into a time-scaled NumPy array.

    Args:
    -----
        input_string : str
            Raw string containing integer frame indices.

    Returns:
    --------
        numpy.ndarray
            Array of spike times in seconds.

    """
    string_list = string_to_list_translator(input_string)
    return np.array(string_list).astype(int) * (1 / float(config.general_settings.frame_rate))

def amplitude_list_translator(input_string):
    """
    Convert an amplitude string into a NumPy float array.

    Args:
    -----
        input_string : str

    Returns:
    --------
        numpy.ndarray
            Array of spike amplitudes.

    """

    amp_string_list = string_to_list_translator(input_string)
    amp_string_list = np.array(amp_string_list).astype(float)
    return (amp_string_list)


def decay_frame_list_translator(input_string):
    """
    Convert a decay-frame string into a time-scaled NumPy array.

    Args:
    -----
        input_string : str

    Returns:
    --------
        numpy.ndarray
            Array of decay times in seconds.

    """

    decay_frame_list = string_to_list_translator(input_string)
    return np.array(decay_frame_list).astype(float)*(1/config.general_settings.frame_rate)

def decay_time_list_translator(input_string):
    """
    Convert a decay-time string into a NumPy array of floats.

    Args:
    -----
        input_string : str

    Returns:
    --------
        numpy.ndarray
            Array of decay times.

    """

    decay_time_list = string_to_list_translator(input_string) 
    return np.array(decay_time_list).astype(float)

def spike_df_iterator(input_path, return_name=True):
    """
    Iterate through CSV spike files and yield parsed DataFrames.

    Args:
    -----
        input_path : str
            Path to a folder containing CSV files.
        return_name : bool, optional
            Whether to include the filename in the output.

    Returns:
    --------
        tuple or pandas.DataFrame
            (DataFrame, filename) if return_name=True,
            otherwise DataFrame only.

    """

    for csv_file in list_all_files_of_type(input_path, "csv"):
        csv_path = os.path.join(input_path, csv_file)
        csv_df = pd.read_csv(csv_path, converters={
            "PeakTimes":spike_list_translator , 
            "BasePeakTimes":spike_list_translator , 
            "PDBuPeakTimes":spike_list_translator , 
            "APVPeakTimes":spike_list_translator , 
            "Amplitudes":amplitude_list_translator,
            "BaseAmplitudes":amplitude_list_translator,
            "PDBuAmplitudes":amplitude_list_translator,
            "APVAmplitudes":amplitude_list_translator,
            "DecayTimes": decay_time_list_translator, 
            "DecayFrames": decay_frame_list_translator}, na_filter =False) #Remember to change 'Decaytimes; in the csv to DecayTimes
        yield csv_df, csv_file if return_name else csv_df

        
   #Again, binned stats are not necessary for synapses, but regardless, we can leave this here for now
def calculate_binned_stats(input_df):
    """
    Compute population-level spike counts and instantaneous frequency in
    fixed-width time bins.

    Args:
    -----
        input_df : pandas.DataFrame
            DataFrame containing a 'PeakTimes' column.

    Returns:
    --------
        pandas.DataFrame
            Table of (bin range, spike count, frequency).

    """

    local_df = input_df.copy()

    bins = np.arange(0, config.general_settings.EXPERIMENT_DURATION + config.general_settings.BIN_WIDTH, config.general_settings.BIN_WIDTH) 
    population_spikes, _ = np.histogram(np.hstack(local_df["PeakTimes"].values), bins=bins)
    population_frequency = population_spikes / config.general_settings.BIN_WIDTH

    bin_stats = pd.DataFrame.from_dict({
        "Bin_Limits": [(bins[bin_index], bins[bin_index + 1]) for bin_index in range(len(bins) - 1)],
        "Spikes": population_spikes,
        "Frequency": population_frequency})
        
    return bin_stats



def process_spike_csvs_to_pkl(input_path):
    """
    Convert spike CSV files into pickled analysis dictionaries.

    Args:
    -----
        input_path : str
            Path containing a 'csv_files' directory.

    Returns:
    --------
        None

    """
    csv_path = os.path.join(input_path, 'csv_files')
    output_path = os.path.join(input_path, 'pkl_files')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    csv_file_list = []
    for root, dirs, files in os.walk(os.path.join(config.general_settings.main_folder, 'csv_files')):
        for file in files:
            if file.endswith('.csv') and "processed_" not in file:
                csv_file_list.append(file)
    
    for spike_df, file_name in spike_df_iterator(csv_path):
            if file_name in csv_file_list: 
                processed_path = os.path.join(output_path, 
                                        f"{os.path.splitext(file_name)[0]}"
                                        f"Dur{int(config.general_settings.EXPERIMENT_DURATION)}s"
                                        f"Int{int((1/config.general_settings.frame_rate)*1000)}ms"
                                        + ".pkl")
                
                
                processed_dict = {
                "cell_stats": calculate_cell_stats(spike_df)}#,#, #consider removing binned_stats for now unless there becomes a need for synapse synchronization / congruence
                # "binned_stats": calculate_binned_stats(spike_df)}
                processed_dict["cell_stats"] = processed_dict["cell_stats"][processed_dict['cell_stats']['IsUsed'] == True]

                pd.to_pickle(processed_dict, processed_path)


def generate_synapse_counts_and_summary_stats(experiment_folder):
    """
    Generate per-file summary statistics and synapse/dendrite counts.

    Args:
    -----
        experiment_folder : str
            Path to the experiment directory.

    Returns:
    --------
        tuple
            (groups, metrics, summary)
            groups : list of experimental groups
            metrics : list of metric names
            summary : grouped aggregate statistics

    """
    import os
    from analyze_suite2p import config_loader
    config = config_loader.load_json_config_file(os.path.join(experiment_folder, 'analysis_config.json'))
    experiment_name = str(config.general_settings.main_folder.split('\\')[-1])
    file_path = os.path.join(config.general_settings.main_folder,f"{experiment_name}_experiment_summary.csv")
    data = pd.read_csv(file_path)
    # synapses = data[["Experimental_Group", 
    #                  "File_Name", 
    #                  'synapse_ROI', 
    #                  "dendrite_ROI",
    #                  "total_ROIs",
    #                  "SpikesFreq",
    #                  "AvgAmplitude",
    #                  "AvgDecayTime" ]].drop_duplicates()
    # syanpses = synapses.groupby("File_Name").agg({"Experimental_Group": "first",
    #                                           "synapse_ROI":["mean"],
    #                                           "dendrite_ROI": ["mean"],
    #                                           "total_ROIs": ["mean"],
    #                                           "SpikesFreq": ["mean"],
    #                                           "AvgAmplitude": ["mean"],
    #                                           "AvgDecayTime": ["mean"]})
    synapses = data[["Experimental_Group", 
                     "Replicate_No.",
                     "File_Name",
                     "SpikesFreq", 
                     'synapse_ROI', 
                     "dendrite_ROI",
                     "total_ROIs"]].dropna()
    syanpses = synapses.groupby("File_Name").agg({"Experimental_Group": "first",
                                                  "Replicate_No.": "first",
                                                  "synapse_ROI":["mean"],
                                                  "dendrite_ROI": ["mean"],
                                                  "total_ROIs": ["mean"],
                                                  "SpikesFreq": ["mean"]})
    groups = synapses["Experimental_Group"].unique().tolist()
    mapped_groups = synapses["Experimental_Group"]
    # Metrics and grouping
    metrics = [
        "synapse_ROI",
        "dendrite_ROI",
        "total_ROIs",
        "SpikesFreq",
    #     "AvgAmplitude",
    #     "AvgDecayTime"
    ]
    save_path = file_path.split("\\")[0:-1]
    save_path = '\\'.join(save_path)
    experiment = file_path.split("\\")[-1]
    experiment = experiment.split(".csv")[0]

    return groups, metrics, syanpses

