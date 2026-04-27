import pandas as pd
import numpy as np
import os
from analyze_suite2p import detector_utility, config_loader

_DEFAULT_CONFIG = config_loader.load_json_config_file()
config = _DEFAULT_CONFIG

SUITE2P_STRUCTURE = {
    "F": ["suite2p", "plane0", "F.npy"],
    "Fneu": ["suite2p", "plane0", "Fneu.npy"],
    'spks': ["suite2p", "plane0", "spks.npy"],
    "stat": ["suite2p", "plane0", "stat.npy"],
    "iscell": ["suite2p", "plane0", "iscell.npy"],
    "ops": ["suite2p", "plane0", "ops.npy"],
    "deltaF": ["suite2p","plane0","deltaF.npy"]
}

def load_npy_array(npy_path):
    """
    Load a NumPy `.npy` file into a NumPy array.
    
    This function loads the `.npy` file located at the specified `npy_path` and returns it as a NumPy array.
    The `allow_pickle` option is set to `True` to allow loading pickled objects.
    
    Args:
    ----------
        npy_path (str or Path): The file path to the `.npy` file (e.g., `F.npy` or `Fneu.npy`).
    
    Returns:
    ----------
        numpy.ndarray: The loaded NumPy array from the `.npy` file.
    
    Example:
    ----------
        >>> load_npy_array('data/F.npy')
        array([1, 2, 3])
    """
    return np.load(npy_path, allow_pickle=True) #functionally equivalent to np.load(npy_array) but iterable; w/ Pickle


def load_npy_df(npy_path):
    """
    Load a NumPy `.npy` file as a Pandas DataFrame.
    
    This function loads the `.npy` file at the specified `npy_path` and converts it into a Pandas DataFrame.
    The `allow_pickle` option is set to `True` to allow loading pickled objects.
    
    Args:
    ----------
        npy_path (str or Path): The file path to the `.npy` file (e.g., `F.npy` or `Fneu.npy`).
    
    Returns:
    ----------
        pd.DataFrame: A Pandas DataFrame containing the loaded data from the `.npy` file.
    
    Example:
    ----------
        >>> load_npy_df('data/F.npy')
        DataFrame with shape (3, 3)
    """
    return pd.DataFrame(np.load(npy_path, allow_pickle=True)) #load suite2p outputs as pandas dataframe


def load_npy_dict(npy_path):
    """
    Load a NumPy `.npy` file as a dictionary.
    
    This function loads the `.npy` file at the specified `npy_path` and returns the contents as a dictionary.
    The `allow_pickle` option is set to `True` to allow loading pickled objects.
    
    Args:
    ----------
        npy_path (str or Path): The file path to the `.npy` file (e.g., `F.npy` or `Fneu.npy`).
    
    Returns:
    ----------
        dict: The loaded dictionary from the `.npy` file.
    
    Example:
    ----------
        >>> load_npy_dict('data/stat.npy')
        {'key1': value1, 'key2': value2}
    """
    return np.load(npy_path, allow_pickle=True)[()] 


def check_for_suite2p_output(folder_name_list):
    """
    Verifies whether each folder in a list of folders contains Suite2p-style output files.
    
    This function checks if the `stat.npy` file exists in each folder in the provided `folder_name_list`.
    If any folder does not contain the required output files, the function will return `False`.
    
    Args:
    ----------
        folder_name_list (list of str): A list of folder paths to check for Suite2p output files.
    
    Returns:
    ----------
        bool: `True` if all folders contain the required Suite2p files, `False` otherwise.
    
    Example:
    ----------
        >>> check_for_suite2p_output(['/path/to/folder1', '/path/to/folder2'])
        True
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
    Checks if `deltaF.npy` exists in each folder; if not, calculates and generates it.
    
    This function checks each folder in the `folder_name_list` to see if the `deltaF.npy` file exists. If it doesn't,
    the function will automatically calculate and generate `deltaF.npy` using the `detector_utility.calculate_deltaF` function.
    
    Args:
    ----------
        folder_name_list (list of str): A list of folder paths containing Suite2p-generated files.
    
    Returns:
    ----------
        None: If `deltaF.npy` is missing, it will be calculated and generated automatically.
    
    Example:
    ----------
        >>> check_deltaF(['/path/to/folder1', '/path/to/folder2'])
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
    Searches the given parent folder for specific Suite2p-generated files or subfolders containing recordings.
    
    This function recursively searches the `folder_path` for files matching the specified `file_ending` (e.g., `F.npy`,
    `deltaF.npy`, or `samples`). It can also return subfolders containing both image and Suite2p analysis files.
    
    Args:
    ----------
        folder_path (str or Path): The root folder path to search for Suite2p files.
        file_ending (str): The file type to search for. Accepted values: `F.npy`, `deltaF.npy`, `samples`.
        suppress_printing (bool, optional): Whether to suppress printing the found files/folders. Defaults to `False`.
    
    Returns:
    ----------
        list of str: A list of file or folder paths matching the specified `file_ending`.
    
    Example:
    ----------
        >>> get_all_suite2p_outputs_in_path('/path/to/data', 'F.npy')
        ['/path/to/data/subject1/suite2p/plane0/F.npy', '/path/to/data/subject2/suite2p/plane0/F.npy']
        >>> get_all_suite2p_outputs_in_path('/path/to/data', 'samples')
        ['/path/to/data/subject1', '/path/to/data/subject2']
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
                if file.endswith("stat.npy"):
                    file_names.append(os.path.join(root, file)[:-24])
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
    Extract experimental dates from folder names and assign replicates to each unique date.
    
    This function scans the folder names in `main_folder` and extracts the dates from the beginning of each folder name.
    It then assigns each unique date a corresponding replicate number (e.g., `sample1`, `sample2`).
    
    Args:
    ----------
        main_folder (str or Path): The path to the main folder containing subfolders with experiment data.
    
    Returns:
    ----------
        dict: A dictionary mapping each folder path to its corresponding sample/replicate number.
    
    Example:
    ----------
        >>> get_experimental_dates('/path/to/main_folder')
        {'/path/to/main_folder/experimental_condition/251030_file_image': 'sample_1', '/path/to/main_folder/experimental_condition/251126_file_image': 'sample_2'}
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
            date_list.append(os.path.basename(folder).split('_')[0])
    distinct_dates = [i for i in set(date_list)]
    distinct_dates.sort(key=lambda x: int(x))
 
    for i1 in range(len(image_folders)):
        for i2, date in enumerate(distinct_dates):
            if date in image_folders[i1]: # if date in list
                sample_dict[image_folders[i1]]=f"sample_{i2+1}"
    return sample_dict
    

def load_suite2p_output(data_folder, config, use_iscell = False):  ## creates a dictionary for the suite2p paths in the given data folder (e.g.: folder for folder_x)
    """
    Load all Suite2p output files for a given recording into a single dictionary.

    This includes fluorescence traces, neuropil signals, ROI statistics,
    Suite2p processing options, and classification arrays. Optionally replaces
    Suite2p's ``iscell.npy`` classification with user-defined skew-thresholding.

    Args:
    ----------
        data_folder : str or Path
            Path to the folder containing the Suite2p output directory.
        groups : list of str
            Names of experimental groups present inside ``main_folder``.
        main_folder : str or Path
            Root directory containing all experimental condition folders.
        use_iscell : bool, optional
            If ``True``, use Suite2p's ``iscell.npy`` array for ROI selection.
            If ``False`` (default), compute ``IsUsed`` via skewness thresholding.

    Returns:
    ----------
        dict
            Dictionary containing all Suite2p arrays and metadata associated with
            the recording, including assigned group and replicate label.
    Example:
    ----------
            >>> load_suite2p_output('/path/to/data_folder', config_dict['general_settings']['groups'], config_dict['general_settings']['main_folder'], use_iscell = False)
            {"F": [5,6,7,8...],
            "Fneu": [0,1,2,3...],
            "stat": {npix: [7], skew: [0.56], radius: 25,...}
            "ops": {dict}
            "iscell": 2D array [[1, 0.5602], [0, 0.1123]...],
            "deltaF": [0.25, 0.5, 0.67, 0.012,...],
            "IsUsed": [True, False, True, True, False, False, ...],
            "Group": 'Experimental_Treatment_Condition',
            "sample": 'Replicate01',
            "file_name": '202511_this_is_the_calcium_imaging_video_file_w_extension" 
            }
        
    """
    main_folder = str(config.general_settings.main_folder)
    groups = config.general_settings.groups
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
