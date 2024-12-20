
import pandas as pd
import numpy as np
import os
from gui_config import gui_configurations as configurations
from analyze_suite2p import detector_utility
    #this is where all the detector functions will be used; at least initially
import concurrent.futures

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
    "ops": ["suite2p", "plane0", "ops.npy"],
    "deltaF": ["suite2p","plane0","deltaF.npy"]
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
        location = os.path.join(folder, *SUITE2P_STRUCTURE["stat"])
        if os.path.exists(location):
            continue
        if not os.path.isfile(os.path.join(folder, location)):
            return False
    return True


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
        amount = int(0.125*len(corrected_trace))
        middle = 0.5*len(corrected_trace)
        F_sample = (np.concatenate((corrected_trace[0:amount], corrected_trace[int(middle-amount/2):int(middle+amount/2)], 
                    corrected_trace[len(corrected_trace)-amount:len(corrected_trace)])))  #dynamically chooses beginning, middle, end 12.5%, changeable
        #TODO decide if mean, median or mode is best for deltaF calculations
        F_baseline = np.median(F_sample)
        deltaF.append((corrected_trace-F_baseline)/F_baseline)

    deltaF = np.array(deltaF)
    np.save(f"{savepath}/deltaF.npy", deltaF, allow_pickle=True)
    print(f"delta F calculated for {F_file[len(configurations.main_folder)+1:-21]}")
    csv_filename = f"{F_file[len(configurations.main_folder)+1:-21]}".replace("\\", "-") ## prevents backslahes being replaced in rest of code
    if not os.path.exists(configurations.main_folder + r'\csv_files_deltaF'): ## creates directory if it doesn't exist
        os.mkdir(configurations.main_folder + r'\csv_files_deltaF')
    np.savetxt(f"{configurations.main_folder}/csv_files_deltaF/{csv_filename}.csv", deltaF, delimiter=";") ### can be commented out if you don't want to save deltaF as .csv files (additionally to .npy)
    print(f"delta F traces saved as deltaF.npy under {savepath}\n")


def check_deltaF(folder_name_list):
    for folder in folder_name_list:
        location = os.path.join(folder, *SUITE2P_STRUCTURE["deltaF"])
        if os.path.exists(location):
            continue
        else:
            calculate_deltaF(location.replace("deltaF.npy","F.npy"))
            if os.path.exists(location):
                continue
            else:
                print("something went wrong, please calculate delta F manually by inserting the following code above: \n F_files = get_file_name_list(folder_path = configurations.main_folder, file_ending = 'F.npy') \n for file in F_files: calculate_deltaF(file)")


def get_all_suite2p_outputs_in_path(folder_path, file_ending, supress_printing = False): ## accounts for possible errors if deltaF files have been created before
    file_names = []
    other_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file_ending is "folders" and file.endswith(file_ending):
                    file_names.append(os.path.join(root, file)[:-21])
            elif file_ending is "F.npy" and not file.endswith('deltaF.npy'):
                file_names.append(os.path.join(root, file))
            else:
                if file.endswith(file_ending): other_files.append(os.pathjoin(root,file))
    if file_ending=="F.npy" or file_ending=="deltaF.npy" or file_ending=="predictions_deltaF.npy":
        if not supress_printing:
            print(f"{len(file_names)} {file_ending} files found:")
            print(file_names)
        return file_names
    elif file_ending=="folders":
        check_deltaF(file_names)  #checks if deltaf exists, else calculates it
        if not supress_printing:
            print(f"{len(file_names)} folders containing {file_ending} found:")
            print(file_names)
        return file_names
    else:
        print("Is the file ending spelled right?")
        return other_files
    

def load_suite2p_output(path, use_iscell=False):
    """here we define our suite2p dictionary from the SUITE2P_STRUCTURE...see above"""
    suite2p_dict = {
        "F": load_npy_array(os.path.join(path, *SUITE2P_STRUCTURE["F"])),
        "Fneu": load_npy_array(os.path.join(path, *SUITE2P_STRUCTURE["Fneu"])),
        "stat": load_npy_df(os.path.join(path, *SUITE2P_STRUCTURE["stat"]))[0].apply(pd.Series),
        "ops": load_npy_array(os.path.join(path, *SUITE2P_STRUCTURE["ops"])).item(),
        "iscell": load_npy_array(os.path.join(path, *SUITE2P_STRUCTURE["iscell"])),
        # "deltaF": load_npy_array(os.path.join(path, *SUITE2P_STRUCTURE["deltaF"]))
    }

    if use_iscell == False:
        suite2p_dict["IsUsed"] = [(suite2p_dict["stat"]["skew"] >= 1)] 

        suite2p_dict["IsUsed"] = pd.DataFrame(suite2p_dict["IsUsed"]).iloc[:,0:].values.T
        suite2p_dict["IsUsed"] = np.squeeze(suite2p_dict["IsUsed"])
    else:
        suite2p_dict["IsUsed"] = load_npy_df(os.path.join(path, *SUITE2P_STRUCTURE["iscell"]))[0].astype(bool)

    return suite2p_dict


def get_experimental_dates(main_folder):
    """returns a dictionary of all wells and the corresponding sample/replicate, the samples are sorted by date, everything sampled on the first date is then sample1, on the second date sample2, etc."""
    well_folders = get_all_suite2p_outputs_in_path(main_folder, "folders", supress_printing = True)
    date_list= []
    sample_dict = {}
    for well in well_folders:
        date_list.append(os.path.basename(well)[0:6]) ## append dates; should change if the date is not in the beginning of the file name usually [:6]
    distinct_dates = [i for i in set(date_list)]
    distinct_dates.sort(key=lambda x: int(x))
 
    for i1 in range(len(well_folders)):
        for i2, date in enumerate(distinct_dates):
            if date in well_folders[i1]: # if date in list
                sample_dict[well_folders[i1]]=f"sample_{i2+1}"
    return sample_dict


def translate_suite2p_dict_to_df(suite2p_dict):
    """this is the principle function in which we will create our .csv file structure; and where we will actually use
        our detector functions for spike detection and amplitude extraction"""
    def process_individual_synapse(f_trace, fneu_trace):
        peaks = detector_utility.single_synapse_baseline_correction_and_peak_return(f_trace, fneu_trace, return_peaks = True)
        amplitudes = detector_utility.single_synapse_baseline_correction_and_peak_return(f_trace, fneu_trace, return_amplitudes=True)
        decay_times = detector_utility.single_synapse_baseline_correction_and_peak_return(f_trace, fneu_trace, return_decay_times = True)
        peak_count = detector_utility.single_synapse_baseline_correction_and_peak_return(f_trace, fneu_trace, return_peak_count=True)
        decay_frames = detector_utility.single_synapse_baseline_correction_and_peak_return(f_trace, fneu_trace, return_decay_frames=True)
        return peaks, amplitudes, peak_count, decay_times, decay_frames

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda args: process_individual_synapse(*args), zip(suite2p_dict["F"], suite2p_dict["Fneu"])))
    spikes_per_neuron, decay_points_after_peaks, spike_amplitudes, decay_times, peak_count = zip(*results)
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


def translate_suite2p_outputs_to_csv(input_path, overwrite=False, check_for_iscell=False, update_iscell = True):
    """This will create .csv files for each video loaded from out data fram function below.
        The structure will consist of columns that list: "Amplitudes": spike_amplitudes})
        
        col1: ROI #, col2: IsUsed (from iscell.npy); boolean, col3: Skew (from stats.npy); could be replaced with any 
        stat >> compactness, col3: spike frames (relative to input frames), col4: amplitude of each spike detected measured 
        from the baseline (the median of each trace)"""
    
    suite2p_outputs = get_all_suite2p_outputs_in_path(input_path, "folders", supress_printing=True)

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

        ###TODO CHANGE ASAP to match somatic pipeline levels of flexibility
        # suite2p_dict = load_suite2p_output(suite2p_output, groups, input_path, use_iscell=check_for_iscell)
        # suite2p_dict = load_suite2p_output(suite2p_output, use_iscell=False)
        ops = suite2p_dict["ops"]
        Img = detector_utility.getImg(ops)
        scatters, nid2idx, nid2idx_rejected, pixel2neuron, synapseID = detector_utility.getStats(suite2p_dict["stat"], Img.shape, suite2p_df)
        iscell_path = os.path.join(suite2p_output, *SUITE2P_STRUCTURE['iscell'])
        if update_iscell:
            updated_iscell = suite2p_dict['iscell']
            for idx in nid2idx: 
                updated_iscell[idx, 0] = 1.0
            for idxr in nid2idx_rejected:
                updated_iscell[idxr,0] = 0.0
            np.save(iscell_path, updated_iscell)
            print(f"Updated iscell.npy saved for {suite2p_output}")
        synapse_key = set(synapseID)
        suite2p_df.loc[synapse_key, 'IsUsed'] = True

        suite2p_df['Active_Synapses'] = len(synapseID)
        suite2p_df.to_csv(translated_path)
        print(f"csv created for {suite2p_output}")

        image_save_path = os.path.join(input_path, f"{suite2p_output}_plot.png") #TODO explore changing "input path" to "suite2p_output" to save the processing in the same 
        detector_utility.dispPlot(Img, scatters, nid2idx, nid2idx_rejected, pixel2neuron, suite2p_dict["F"], suite2p_dict["Fneu"], image_save_path)

    print(f"{len(suite2p_outputs)} .csv files were saved under {configurations.main_folder+r'/csv_files'}")
