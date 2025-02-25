
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from analyze_suite2p import suite2p_utility, detector_utility, plotting_utility, config_loader

config = config_loader.load_json_config_file()

def calculate_cell_freq(input_df):
    output_df = input_df.copy()
    output_df["SpikesCount"] = output_df["PeakTimes"].str.len()
    output_df["SpikesFreq"] = output_df["SpikesCount"] / ((input_df["Total Frames"] / config.general_settings.frame_rate)) #divide by total # of frames NOT framerate
    output_df['SpikesCV'] = output_df['PeakTimes'].apply(lambda x: pd.Series(x).std()) / output_df['SpikesFreq'] * 100
    return output_df

def calculate_cell_isi(input_df): #isi == interspike interval
    output_df = input_df.copy()
    output_df["SpikesDiff"] = output_df["PeakTimes"].apply(lambda x: list(pd.Series(x).diff().dropna()))
    output_df["DiffAvg"] = output_df["SpikesDiff"].apply(lambda x: pd.Series(x).mean())
    output_df["DiffMedian"] = output_df["SpikesDiff"].apply(lambda x: pd.Series(x).median())
    output_df["DiffCV"] = output_df["SpikesDiff"].apply(lambda x: pd.Series(x).std()) / output_df["DiffAvg"] * 100
    return output_df

#below I will need to accurately figure out how to integrate this in, I should meet with Marti Ritter next week to do so

def calculate_spike_amplitudes(input_df):
    output_df = input_df.copy()
    output_df["AvgAmplitude"] = output_df["Amplitudes"].apply(lambda x: pd.Series(x).mean())
    output_df["SpkAmpMedian"] = output_df["Amplitudes"].apply(lambda x: pd.Series(x).median())
    output_df["SpkAmpCV"] = output_df["Amplitudes"].apply(lambda x: pd.Series(x).std()) / output_df["AvgAmplitude"] * 100
    return output_df

def calculate_decay_fraction(row):
    if row["SpikesCount"] != 0:
        return row['DecayCount'] / row["SpikesCount"]
    else:
        return []
        

def calculate_decay_values(input_df):
    output_df = input_df.copy()
    output_df["DecayCount"] = output_df["DecayTimes"].apply(lambda arr: (len([x for x in arr if not pd.isna(x)])))
    output_df["DecayedFraction"] = output_df.apply(calculate_decay_fraction, axis =1)
    output_df["AvgDecayTime"] = output_df["DecayTimes"].apply(lambda x: pd.Series(x).dropna().mean())
    output_df["AvgDecayCV"] = output_df["DecayTimes"].apply(lambda x: pd.Series(x).dropna().std()) / output_df["AvgDecayTime"] * 100
    return output_df


def calculate_cell_stats(input_df, calculate_freq=True, calculate_isi=True, calculate_amplitudes=True, calculate_decays = True):
    output_df = input_df.copy()
    if calculate_freq:
        output_df = calculate_cell_freq(output_df)
    if calculate_isi:
        output_df = calculate_cell_isi(output_df)
    if calculate_amplitudes:
        output_df = calculate_spike_amplitudes(output_df)
    if calculate_decays:
        output_df = calculate_decay_values(output_df)
    return output_df

def translate_suite2p_dict_to_df(suite2p_dict):
    """this is the principle function in which we will create our .csv file structure; and where we will actually use
        our detector functions for spike detection and amplitude extraction"""
    def process_individual_synapse(deltaF):
        peaks = detector_utility.single_synapse_baseline_correction_and_peak_return(deltaF, return_peaks = True)
        amplitudes = detector_utility.single_synapse_baseline_correction_and_peak_return(deltaF, return_amplitudes=True)
        decay_times = detector_utility.single_synapse_baseline_correction_and_peak_return(deltaF, return_decay_time = True)
        peak_count = detector_utility.single_synapse_baseline_correction_and_peak_return(deltaF, return_peak_count=True)
        decay_frames = detector_utility.single_synapse_baseline_correction_and_peak_return(deltaF, return_decay_frames=True)
        return peaks, amplitudes, peak_count, decay_times, decay_frames

    results = []

    for idx, (is_used, deltaF) in enumerate(zip(suite2p_dict["IsUsed"],suite2p_dict["deltaF"])):
        if is_used:
            result = process_individual_synapse(deltaF)
        else:
            result = (np.array([]), np.array([]), 0, np.array([]), np.array([]))
        results.append(result)
    spikes_per_neuron, spike_amplitudes, peak_count, decay_times, decay_frames = zip(*results)
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     results = list(executor.map(lambda args: process_individual_synapse(*args), zip(suite2p_dict["F"], suite2p_dict["Fneu"])))
    # spikes_per_neuron, decay_points_after_peaks, spike_amplitudes, decay_times, peak_count = zip(*results)
#spikes_per_neuron from single_cell_peak_return OUTPUT = list of np.arrays        
    df = pd.DataFrame({"IsUsed": suite2p_dict["IsUsed"],
                       "Skew": suite2p_dict["stat"]["skew"],
                       "PeakTimes": spikes_per_neuron,
                       "PeakCount": peak_count, #TODO figure out if we can calculate all the coversions here before the pkl file
                       "Amplitudes": spike_amplitudes,
                        "DecayTimes": decay_times,
                       "DecayFrames": decay_frames,
                       "Total Frames": len(suite2p_dict["F"].T),
                       "Experimental Group": suite2p_dict['Group'],
                       "Replicate No.": suite2p_dict['sample'],
                       "File Name": suite2p_dict['file_name']
                       })
                       
    df.index.set_names("SynapseID", inplace=True)
    Img = plotting_utility.getImg(suite2p_dict["ops"])
    scatters, nid2idx, nid2idx_rejected, pixel2neuron, synapse_ID, nid2dx_dendrite, nid2idx_synapse = plotting_utility.getStats(suite2p_dict, Img.shape, df)
    
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
    processed_df = processed_df[processed_df['SpikesCount']>=2]

    return df, processed_df#, aggregate_stats

def translate_suite2p_outputs_to_csv(input_path, check_for_iscell=False, update_iscell = True):
    """This will create .csv files for each video loaded from out data fram function below.
        The structure will consist of columns that list: "Amplitudes": spike_amplitudes})
        
        col1: ROI #, col2: IsUsed (from iscell.npy); boolean, col3: Skew (from stats.npy); could be replaced with any 
        stat >> compactness, col3: spike frames (relative to input frames), col4: amplitude of each spike detected measured 
        from the baseline (the median of each trace)"""
    
    suite2p_outputs = suite2p_utility.get_all_suite2p_outputs_in_path(input_path, "samples", supress_printing=True)

    output_path = os.path.join(input_path,"csv_files")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    for suite2p_output in suite2p_outputs:
        output_directory = os.path.basename(suite2p_output)
        translated_path = os.path.join(output_path, f"{output_directory}.csv")
        processed_path = os.path.join(output_path, f"processed_{output_directory}.csv")
        
        suite2p_dict = suite2p_utility.load_suite2p_output(suite2p_output, config.general_settings.groups, input_path)
        
        raw_data, processed_data = translate_suite2p_dict_to_df(suite2p_dict)

        ops = suite2p_dict["ops"]
        Img = plotting_utility.getImg(ops)
        scatters, nid2idx, nid2idx_rejected, pixel2neuron, synapseID, nid2idx_dendrite, nid2idx_synapse = plotting_utility.getStats(suite2p_dict, Img.shape, raw_data, use_iscell=check_for_iscell)
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

        processed_data['Active_Synapses'] = len(synapseID)


        raw_data.to_csv(translated_path)
        processed_data.to_csv(processed_path)
        print(f"csvs created for {suite2p_output}")

        image_save_path = os.path.join(input_path, f"{suite2p_output}_plot.png") #TODO explore changing "input path" to "suite2p_output" to save the processing in the same 
        plotting_utility.dispPlot(Img, scatters, nid2idx, nid2idx_rejected, nid2idx_dendrite, nid2idx_synapse,
                                   pixel2neuron, suite2p_dict["F"], suite2p_dict["Fneu"], image_save_path, fill_ROIs=True)

    print(f"{len(suite2p_outputs)} .csv files were saved under {Path(config.general_settings.main_folder) / 'csv_files'}")

def create_experiment_summary(main_folder):
    csv_file_path = os.path.join(main_folder, 'csv_files')
    csv_files = list_all_files_of_type(csv_file_path, '.csv')
    processed_csvs = [file for file in csv_files if file.startswith('processed')]
    df_list = [pd.read_csv(os.path.join(csv_file_path, csv)) for csv in processed_csvs]
    merged_df = pd.concat(df_list, ignore_index=True)
    agg_columns = merged_df.select_dtypes(['float64', 'int'])
    aggregate_stats = agg_columns.groupby(merged_df['File Name']).agg(['mean', 'std', 'median']).reset_index()

# Include non-numeric columns in the final aggregated dataframe
    aggregate_stats['Experimental Group'] = merged_df.groupby('File Name')['Experimental Group'].first().values
    aggregate_stats['Replicate No.'] = merged_df.groupby('File Name')['Replicate No.'].first().values
    
    merged_df.to_csv(os.path.join(config.general_settings.main_folder, 'experiment_summary.csv'))
    return aggregate_stats, merged_df

def list_all_files_of_type(input_path, filetype):
    return [file for file in os.listdir(input_path) if file.endswith(filetype)]

def string_to_list_translator(input_string, strip_before_split="[ ]", split_on=" "):
    split_string = input_string.strip(strip_before_split).split(split_on)
    return list(filter(None, split_string))

def spike_list_translator(input_string):
    """This funciton is nested in the next. It is designed to convert the time stamp of each event into a time
        during the experiment (e.g. frame 2 = 1.1 seconds into the recording)"""
    string_list = string_to_list_translator(input_string)
    return np.array(string_list).astype(int) * config.general_settings.FRAME_INTERVAL

def amplitude_list_translator(input_string):
    amp_string_list = string_to_list_translator(input_string)
    amp_string_list = np.array(amp_string_list).astype(float)
    return (amp_string_list)


def decay_frame_list_translator(input_string):
     decay_frame_list = string_to_list_translator(input_string)
     return np.array(decay_frame_list).astype(float)*config.general_settings.FRAME_INTERVAL

def decay_time_list_translator(input_string):
    decay_time_list = string_to_list_translator(input_string) 
    return np.array(decay_time_list).astype(float)

def spike_df_iterator(input_path, return_name=True):
    for csv_file in list_all_files_of_type(input_path, "csv"):
        csv_path = os.path.join(input_path, csv_file)
        csv_df = pd.read_csv(csv_path, converters={"PeakTimes":spike_list_translator , "Amplitudes":amplitude_list_translator, 
        "DecayTimes": decay_time_list_translator, "DecayFrames": decay_frame_list_translator}, na_filter =False) #Remember to change 'Decaytimes; in the csv to DecayTimes
        yield csv_df, csv_file if return_name else csv_df

        
   #Again, binned stats are not necessary for synapses, but regardless, we can leave this here for now
def calculate_binned_stats(input_df):
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
    """This will convert .csv files into pickle files which behave like dataframes; but are faster and preserve CPU RAM"""
    csv_path = os.path.join(input_path, 'csv_files')
    output_path = os.path.join(input_path, 'pkl_files')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    for spike_df, file_name in spike_df_iterator(csv_path):
            processed_path = os.path.join(output_path, 
                                        f"{os.path.splitext(file_name)[0]}"
                                        f"Dur{int(config.general_settings.EXPERIMENT_DURATION)}s"
                                        f"Int{int(config.general_settings.FRAME_INTERVAL*1000)}ms"
                                        + ".pkl")
                
            if config.general_settings.FILTER_NEURONS:
                spike_df = spike_df[spike_df["IsUsed"]]
                
            processed_dict = {
                "cell_stats": calculate_cell_stats(spike_df),#, #consider removing binned_stats for now unless there becomes a need for synapse synchronization / congruence
                "binned_stats": calculate_binned_stats(spike_df)}
            

            pd.to_pickle(processed_dict, processed_path)

