
import pandas as pd
import numpy as np
import os
from configurations import *
from detector_utility import *
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
    peak_count = [single_synapse_baseline_correction_and_peak_return(f_trace, fneu_trace, return_peak_count = True)
                             for (f_trace, fneu_trace) in zip(suite2p_dict["F"], suite2p_dict["Fneu"])]
#spikes_per_neuron from single_cell_peak_return OUTPUT = list of np.arrays        
    df = pd.DataFrame({"IsUsed": suite2p_dict["IsUsed"],
                       "Skew": suite2p_dict["stat"]["skew"],
                       "PeakTimes": spikes_per_neuron,
                       "PeakCount": peak_count, #TODO figure out if we can calculate all the coversions here before the pkl file
                       "Amplitudes": spike_amplitudes,
                        "DecayTimes": decay_times,
                       "DecayFrames": decay_points_after_peaks,
                       "Total Frames": len(suite2p_dict["F"].T)})
                       
    df.index.set_names("SynapseID", inplace=True)
    df["IsUsed"] = False

    # df.fillna(0, inplace = True) potentially for decay time calculations
    return df

def translate_suite2p_outputs_to_csv(input_path, overwrite=False, check_for_iscell=False):
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

        ###TODO CHANGE ASAP

        # suite2p_dict = load_suite2p_output(suite2p_output, groups, input_path, use_iscell=check_for_iscell)
        # suite2p_dict = load_suite2p_output(suite2p_output, use_iscell=False)
        ops = suite2p_dict["ops"]
        Img = getImg(ops)
        scatters, nid2idx, nid2idx_rejected, pixel2neuron, synapseID = getStats(suite2p_dict["stat"], Img.shape, suite2p_df)
        synapse_key = set(synapseID)
        suite2p_df.loc[synapse_key, 'IsUsed'] = True

        suite2p_df['Active_Synapses'] = len(synapseID)
        suite2p_df.to_csv(translated_path)
        print(f"csv created for {suite2p_output}")

        image_save_path = os.path.join(input_path, f"{suite2p_output}_plot.png") #TODO explore changing "input path" to "suite2p_output" to save the processing in the same 
        dispPlot(Img, scatters, nid2idx, nid2idx_rejected, pixel2neuron, suite2p_dict["F"], suite2p_dict["Fneu"], image_save_path)

    print(f"{len(suite2p_outputs)} .csv files were saved under {main_folder+r'/csv_files'}")
