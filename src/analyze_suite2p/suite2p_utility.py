import pandas as pd
import numpy as np
import os
from analyze_suite2p import detector_utility, config_loader

config = config_loader.load_json_config_file()

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
    """
    Load NumPy .npy file as a Pandas DataFrame.
    
    Parameters:
    -----------
    npy_path: str-like
    Path object pointing to .npy file (e.g. F.npy / Fneu.npy)
    
    Returns:
    --------
    np.array(): load NumPy path with allow_pickle set to true.
    """
    return np.load(npy_path, allow_pickle=True) #functionally equivalent to np.load(npy_array) but iterable; w/ Pickle


def load_npy_df(npy_path):
    """
    Load NumPy .npy file as a Pandas DataFrame.
    
    Parameters:
    -----------
    npy_path: str-like
    Path object pointing to .npy file (e.g. F.npy / Fneu.npy)
    
    Returns:
    --------
    pd.DataFrame(np.array()): load NumPy path into a DataFrame with allow_pickle set to true.
    """
    return pd.DataFrame(np.load(npy_path, allow_pickle=True)) #load suite2p outputs as pandas dataframe


def load_npy_dict(npy_path):
    """
    Load NumPy .npy file as a dictionary.
    
    Parameters:
    -----------
    npy_path: str-like
    Path object pointing to .npy file (e.g. F.npy / Fneu.npy)
    
    Returns:
    --------
    np.load(npy_path, allow_pickle = True)[()] to load NumPy file as a dictionary
    """
    return np.load(npy_path, allow_pickle=True)[()] 


def check_for_suite2p_output(folder_name_list):
    """
    Verifies whether each folder in a list of folders contains suite2p-style output files.
    
    Parameters:
    -----------
    folder_name_list: list of strings
    List containing string paths to folder locations

    Returns:
    --------
    True: binary mask if each folder provided has suite2p generated files (checked with stat.npy)
    """
    for folder in folder_name_list:
        location = os.path.join(folder, *SUITE2P_STRUCTURE["stat"])
        if os.path.exists(location):
            continue
        if not os.path.isfile(os.path.join(folder, location)):
            return False
    return True


def check_deltaF(folder_name_list):
    """
    Checks each folder in a list of folders if F.npy and Fneu.npy have been normalized into a deltaF.npy file.
    
    Parameters:
    -----------
    folder_name_list: list of strings
    List containing string paths to folder locations with calcium imaging videos and suite2p-generated files

    Returns:
    --------
    IF no file deltaF.npy exists...run function detector_utility.calculate_deltaF() to automatically calculate and generate deltaF.npy file.
    """
    for folder in folder_name_list:
        location = os.path.join(folder, *SUITE2P_STRUCTURE["deltaF"])
        if os.path.exists(location):
            continue
        else:
            detector_utility.calculate_deltaF(location.replace("deltaF.npy","F.npy"))
            if os.path.exists(location):
                continue
            else:
                print("something went wrong, please calculate delta F manually by inserting the following code above: \n F_files = get_file_name_list(folder_path = configurations.main_folder, file_ending = 'F.npy') \n for file in F_files: calculate_deltaF(file)")


def get_all_suite2p_outputs_in_path(folder_path, file_ending, supress_printing = False): ## accounts for possible errors if deltaF files have been created before
    """
    Searches given parent folder for specific suite2p-generated files. Or subfolders containing recordings and suite2p analysis files.

    Parameters:
    -----------
    folder_path: str
    Parent folder path that contains experimental treatments and image subfolders with suite2p generated analysis run
    file_ending: str
    String indicating what file / folder should be searched for within the parent
                Accepted inputs: "F.npy", "deltaF.npy", "samples"
    suppress_printing: bool, optional
    Option to block printing for files / folders that were found and added to a list

    Returns:
    --------
    file_ending == "F.npy": returns all F.npy path locations as strings in a list
    file_ending == "deltaF.npy": returns all deltaF.npy path locations as strings in a list
    file_ending == "samples": returns all recording folders with both images and suite2p-generated analysis files
    """
    file_names = []
    other_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file_ending=="F.npy" and file.endswith(file_ending) and not file.endswith("deltaF.npy"):
                    file_names.append(os.path.join(root, file))
            elif file_ending=="deltaF.npy" and file.endswith(file_ending):
                    file_names.append(os.path.join(root, file))
            elif file_ending=="samples":
                if file.endswith("F.npy") and not file.endswith("deltaF.npy"):
                    file_names.append(os.path.join(root, file)[:-21])
            else:
                 if file.endswith(file_ending): other_files.append(os.path.join(root, file))
    if file_ending=="F.npy" or file_ending=="deltaF.npy":
        if not supress_printing:
            print(f"{len(file_names)} {file_ending} files found:")
            print(file_names)
        return file_names
    elif file_ending=="samples":
        check_deltaF(file_names)  #checks if deltaf exists, else calculates it
        if not supress_printing:
            print(f"{len(file_names)} folders containing {file_ending} found:")
            print(file_names)
        return file_names
    else:
        print("Is the file ending spelled right?")
        return other_files


def get_experimental_dates(main_folder):
    """
    Extract experiment dates from file names; chooses dates from the beginning of the file name.
    Creates a usable template for 'load_suite2p_output' function below.

    Parameters:
    -----------
    main_folder: str / Path
        Path to main folder (also found in config.json)

    Returns:
    --------
    sample_dict: dictionary
        Example dictionary used in function 'create_suite2p_df'
        Searches for unique dates at beginning of files (either 6 or 4 characters)
        Each unique date is assigned to a replicate number
    returns a dictionary of all folders and the corresponding sample/replicate, the samples are sorted by date, everything sampled on the first date is then sample1, on the second date sample2, etc.
    """
    image_folders = get_all_suite2p_outputs_in_path(main_folder, "samples", supress_printing = True)
    date_list= []
    sample_dict = {}
    for folder in image_folders:
        date_list.append(os.path.basename(folder)[0:6]) ## append dates; should change if the date is not in the beginning of the file name usually [:6]
    
    #Sanity check incase someone does not name files correctly
    i = 0
    for date in date_list:
        if type(date) != int:
            i=1
    if i == 1:
        date_list = []
        for folder in image_folders:
            date_list.append(os.path.basename(folder)[0:4])
    distinct_dates = [i for i in set(date_list)]
    distinct_dates.sort(key=lambda x: int(x))
 
    for i1 in range(len(image_folders)):
        for i2, date in enumerate(distinct_dates):
            if date in image_folders[i1]: # if date in list
                sample_dict[image_folders[i1]]=f"sample_{i2+1}"
    return sample_dict
    

def load_suite2p_output(data_folder, groups, main_folder, use_iscell = False):  ## creates a dictionary for the suite2p paths in the given data folder (e.g.: folder for folder_x)
    """
    Load all suite2p-generated analysis files for a given file into a single dictionary
    
    Parameters:
    data_folder: string-like
    Folder containing the suite2p-generated output folder 'suite2p' (suite2p\plane0\...)
    groups: list of strings
    Experimental conditions one level below main_folder
    Generally extracted from config.json and by running <python -m synapse_gui>
    main_folder: string-like
    Parent folder containing all experimental conditions and suite2p analysis folders for all files
    Generally extracted from config.json or by running <python -m synapse_gui>
    use_iscell: bool, optional
    Toggles using iscell.npy / suite2p ROI classifier to categorize ROIs
    Default: False --> User updates iscell based on activity during the first analysis
    any further reclassification done by the user is preserved by setting <use_iscell = True>
    
    Returns:
    --------
    suite2p_dict: dictionary
    Dictionary with all suite2p-generated files for a given recording as key:value pairs in a dictionary
    Can be compiled into dictionary of dictionaries with multiple image files each containing suite2p-generated analysis files nested within
    """
    suite2p_dict = {
        "F": load_npy_array(os.path.join(data_folder, *SUITE2P_STRUCTURE["F"]).replace('\\','/')),
        "Fneu": load_npy_array(os.path.join(data_folder, *SUITE2P_STRUCTURE["Fneu"]).replace('\\','/')),
        "stat": load_npy_df(os.path.join(data_folder, *SUITE2P_STRUCTURE["stat"]).replace('\\','/'))[0].apply(pd.Series),
        "ops": load_npy_array(os.path.join(data_folder, *SUITE2P_STRUCTURE["ops"]).replace('\\','/')).item(),
        "iscell": load_npy_array(os.path.join(data_folder, *SUITE2P_STRUCTURE["iscell"]).replace('\\','/')),
        "deltaF": load_npy_array(os.path.join(data_folder, *SUITE2P_STRUCTURE["deltaF"]).replace('\\','/'))
    }
#TODO need to update if not use_iscell to include dictionary items within my json dictionary
    if not use_iscell:
        suite2p_dict["IsUsed"] = [
            (suite2p_dict["stat"]["skew"] >= float(config.analysis_params.skew_threshold))]
        suite2p_dict["IsUsed"] = np.squeeze(suite2p_dict["IsUsed"])

    else:
        suite2p_dict["IsUsed"] = pd.DataFrame(suite2p_dict["iscell"]).iloc[:,0].values.T
        suite2p_dict["IsUsed"] = np.squeeze(suite2p_dict["iscell"])
        suite2p_dict['IsUsed'] = suite2p_dict['iscell'][:,0].astype(bool)

    if not groups:
        pass
        raise ValueError("The 'groups' list is empty. Please provide valid group names...")

    print(f"Data folder: {data_folder}")
    print(f"Groups: {groups}")
    print(f"Main folder: {main_folder}")
    found_group = False
    for group in groups:
        if (str(group)) in data_folder:
            group_name = os.path.basename(group.strip("\\/"))
            suite2p_dict["Group"] = group_name
            found_group = True
            print(f"Assigned Group: {suite2p_dict['Group']}")
    
    # debugging
    if "iscell" not in suite2p_dict:
        raise KeyError ("'IsUsed' was not defined correctly either")
    if "Group" not in suite2p_dict:
        raise KeyError("'Group' key not found in suite2p_dict.")
    if not found_group:
        raise KeyError(f"No group found in the data_folder path: {data_folder}")

    sample_dict = get_experimental_dates(main_folder) ## creates the sample number dict
   
    suite2p_dict["sample"] = sample_dict[data_folder]  ## gets the sample number for the corresponding folder folder from the sample dict
 
    suite2p_dict['file_name'] = data_folder
    return suite2p_dict
