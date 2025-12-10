import os
import shutil
from suite2p import run_s2p
from analyze_suite2p import analysis_utility, suite2p_utility, config_loader

_DEFAULT_CONFIG = config_loader.load_json_config_file()
config = _DEFAULT_CONFIG


def export_image_files_to_suite2p_format(parent_directory, file_ending= config.general_settings.data_extension):
    """
    Move image files into their own subfolders.
     
    This function takes image files with a given file extension and copies them into
    their own subfolder to give 1 image per subfolder where the image name is preserved
    in the folder name itself. This is necessary for Suite2p processing without the settings
    'look_one_level_down'.

    Args:
    -----------
    parent_directory: path / str
        Path-like object pointing to main folder to search through
    file_ending: str
        File type / file ending for image file
        example endings ('nd2', 'tif')

    Returns:
    --------
    Files organized in the following tree
        /path/to/parent_folder
        ├───experiment_condition_folder_1
        │   ├───image_folder_1
        |        ├───image_1
        │   ├───image_folder_2
        |        ├───image_2              
        ├───experiment_condition_folder_2
        │   ├───image_folder_1
        |        ├───image_1
        │   ├───image_folder_2
        |        ├───image_2
    """
    if not os.path.exists(parent_directory):
        print(f"Provided path does not exist: {parent_directory}")
        return
    
    # Process each directory within the parent directory
    for dir_name in os.listdir(parent_directory):
        dir_path = os.path.join(parent_directory, dir_name)
        if not os.path.isdir(dir_path):
            print(f"Skipping non-directory path: {dir_path}")
            continue

        # Processing each file within the directory
        for file in os.listdir(dir_path):
            if file.endswith(file_ending):
                name, _ = os.path.splitext(file)
                folder_path = os.path.join(dir_path, name)
                os.makedirs(folder_path, exist_ok=True)

                source = os.path.join(dir_path, file)
                destination = os.path.join(folder_path, file)

                try:
                    shutil.copy2(source, destination)
                    os.remove(source)
                    print(f"Processed and moved {file} to {folder_path}")
                except Exception as e:
                    print(f"Failed to process {file} due to {e}")
            else:
                print(f"Skipping non-{file_ending} file: {file}")

def get_all_image_folders_in_path(path):
    """
    Find all folders within a given path that contain exactly one `.nd2` file in their deepest subfolder.

    This function traverses the directory tree from the specified `path`, identifies all the folders that
    contain exactly one `.nd2` file in the deepest subfolder, and returns a list of those folders.

    Args:
        path (str): The root directory path to begin the search from. The function will walk through all
                    subdirectories starting from this path.

    Returns:
        list: A list of absolute paths to directories that contain exactly one `.nd2` file in their deepest
              subfolder. If no such directories are found, the list will be empty.

    Example:
        >>> get_all_image_folders_in_path("/home/user/images")
        ['/home/user/images/folder1', '/home/user/images/folder2']
    """

    def check_for_single_image_file_in_folder(current_path, file_ending = config.general_settings.data_extension):
        """
        Check if the specified directory contains exactly one `.nd2` file.

        This helper function scans a directory for files that match the specified `file_ending` (default is `.nd2`).
        It returns True if the directory contains exactly one such file, otherwise False.

        Args:
            current_path (str): The path of the directory to check.
            file_ending (str, optional): The file extension to look for. Default is `.nd2`.

        Returns:
            bool: True if the directory contains exactly one `.nd2` file, False otherwise.

        Example:
            >>> check_for_single_image_file_in_folder("/home/user/images/folder1")
            True
        """
        tiff_files = [file for file in os.listdir(current_path) if file.endswith(file_ending)]
        return len(tiff_files) == 1

    found_image_folders = []
    for current_path, directories, files in os.walk(path):
        # Check if current directory is a "deepest" directory (no subdirectories)
        if check_for_single_image_file_in_folder(current_path):
            #current_path = current_path.split("\\")[-2]
            found_image_folders.append(current_path)

    return found_image_folders


def process_files_with_suite2p(image_list, ops):
    """
    Process a list of image folders using the Suite2p pipeline.

    This function wraps Suite2p’s ``run_s2p`` function and applies a user-provided
    ``ops`` dictionary to each image folder. A temporary fast-disk directory is
    created if needed to store Suite2p-generated binary files.

    Args:
    -----
    image_list : list of str or Path
        List of folder paths containing images to be processed by Suite2p.
    ops : dict
        The Suite2p ``ops`` settings dictionary, typically loaded from ``ops.npy``.

    Notes:
    ------
    Each item in ``image_list`` is treated as a separate Suite2p input folder.
    Any exceptions raised during Suite2p processing are caught and reported,
    allowing the loop to continue.
    """
    for image_path in image_list:
        try:
                fast_disk_path = r'C:\BIN'
                if not os.path.exists(fast_disk_path):
                    os.makedirs(fast_disk_path)
                db = {
                'h5py': [], # a single h5 file path
                'h5py_key': 'data',
                'look_one_level_down': False, # whether to look in ALL subfolders when searching for images
                'data_path': [image_path], # a list of folders with images 
                                                    # (or folder of folders with images if look_one_level_down is True, or subfolders is not empty)
                                                    
                'subfolders': [], # choose subfolders of 'data_path' to look in (optional)
                'fast_disk': fast_disk_path, # string which specifies where the binary file will be stored (should be an SSD)
                }
        
                opsEnd = run_s2p(ops=ops, db=db)
        except (ValueError, AssertionError, IndexError, Exception) as e:
                print(f"Error processing {image_path}: {e}")

def main(config_file = None):
    """
    Run a full Suite2p preprocessing and analysis pipeline based on a configuration file.

    This function loads a JSON configuration, prepares Suite2p-compatible image
    folders, processes unprocessed recordings with Suite2p, translates Suite2p
    outputs to CSV, converts spike CSVs to pickles, and generates summary
    statistics for the experiment. A copy of the analysis configuration is saved
    as ``analysis_config.json`` inside the main experiment folder.

    Args:
    -----
    config_file : str or Path, optional
        Path to a JSON configuration file. If omitted, the default configuration
        from ``config_loader`` is used.

    Workflow:
    ---------
    1. Load configuration and ``ops.npy`` Suite2p settings.
    2. Export raw images into Suite2p format.
    3. Identify all image folders and detect existing Suite2p outputs.
    4. Run Suite2p on unprocessed folders (or all folders if overwrite is enabled).
    5. Convert Suite2p outputs to CSV and pickle formats.
    6. Generate experiment summary tables and statistical outputs.
    7. Save the analysis configuration used for reproducibility.

    Returns:
    --------
    None
        The function performs processing and file I/O but does not return a value.
    """
    import numpy as np
    global config  # <- important
    global config_dict
    if config_file is not None:
        config = config_loader.load_json_config_file(config_file)
        config_dict = config_loader.load_json_dict(config_file)

    else:
        config = config_loader.load_json_config_file()
        config_dict = config_loader.load_json_dict()

    main_folder = config.general_settings.main_folder
    data_extension = config.general_settings.data_extension
    ops_path = config.general_settings.ops_path
    ops = np.load(ops_path, allow_pickle=True).item()
    ops['frame_rate'] = config.general_settings.frame_rate
    ops['input_format'] = data_extension
    ops['max_iterations'] = 20
    export_image_files_to_suite2p_format(main_folder, file_ending = data_extension)
    image_folders = get_all_image_folders_in_path(main_folder)
    suite2p_samples = suite2p_utility.get_all_suite2p_outputs_in_path(config.general_settings.main_folder, file_ending="samples", supress_printing=True)
    unprocessed_files = []
    if config.analysis_params.overwrite_suite2p:
        process_files_with_suite2p(image_folders, ops)
    else:
        for image in image_folders:
            if image not in suite2p_samples:
                unprocessed_files.append(image)
    process_files_with_suite2p(unprocessed_files,ops)
    analysis_utility.translate_suite2p_outputs_to_csv(main_folder, check_for_iscell=config.analysis_params.use_suite2p_ROI_classifier, 
                                                      update_iscell = config.analysis_params.update_suite2p_iscell)
    try:
        analysis_utility.process_spike_csvs_to_pkl(main_folder)
    except KeyError as e:
        print("created pkl files from csv, but error occurred, please check manually")
    analysis_utility.create_experiment_summary(main_folder) 

    import json
    with open(os.path.join(main_folder, 'analysis_config.json'), 'w') as f:
        json.dump(config_dict, f, indent = 4)
    print(f"Analysis parameters saved in {main_folder} as analysis_config.json")
    analysis_utility.generate_synapse_counts_and_summary_stats(main_folder)
    from datetime import datetime

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    # analysis_utility.process_spike_csvs_to_pkl(main_folder, overwrite = True)



if __name__ == "__main__":
    main()
