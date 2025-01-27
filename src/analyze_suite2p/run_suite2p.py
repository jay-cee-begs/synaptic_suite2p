import os
import shutil
from suite2p import run_s2p
from analyze_suite2p import analysis_utility, suite2p_utility, config_loader

config = config_loader.load_json_config_file()



def export_image_files_to_suite2p_format(parent_directory, file_ending= config.general_settings.data_extension):
    """Export each image file (with variable file extension) into its own folder for suite2p processing, for all directories within a given parent directory."""
    
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
                    shutil.copy(source, destination)
                    os.remove(source)
                    print(f"Processed and moved {file} to {folder_path}")
                except Exception as e:
                    print(f"Failed to process {file} due to {e}")
            else:
                print(f"Skipping non-{file_ending} file: {file}")
#Loading in suite2p settings to begin processing

def get_all_image_folders_in_path(path):
    """
    Find all folders within a given path that contain exactly one .nd2 file in their deepest subfolder.
    
    Nested Function:
    - check_for_single_image_file_in_folder: Checks if a given directory contains exactly one .nd2 file.
    """

    def check_for_single_image_file_in_folder(current_path, file_ending = config.general_settings.data_extension):
        """
        Check if the specified path contains exactly one .nd2 file.
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


def process_files_with_suite2p(image_list):
        """
        Processes a list of image paths using the run_s2p function, applying specified configurations.

        Args:
        image_list (list of str): List of file paths to the images to be processed.
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
            
                 opsEnd = run_s2p(ops=config.general_settings.ops, db=db)
            except (ValueError, AssertionError, IndexError) as e:
                 print(f"Error processing {image_path}: {e}")

def main():
    config = config_loader.load_json_config_file()
    main_folder = config.general_settings.main_folder
    data_extension = config.general_settings.data_extension
    img_folders = get_all_image_folders_in_path(main_folder)
    if len(img_folders) == 0:
        export_image_files_to_suite2p_format(main_folder, file_ending = data_extension)
    image_folders = get_all_image_folders_in_path(main_folder)
    suite2p_samples = suite2p_utility.get_all_suite2p_outputs_in_path(config.general_settings.main_folder, file_ending="samples", supress_printing=True)
    if len(suite2p_samples) != len(image_folders):#TODO implement this or configurations.overwrite == False:
        process_files_with_suite2p(image_folders)
    analysis_utility.translate_suite2p_outputs_to_csv(main_folder, overwrite = True, check_for_iscell=False, update_iscell = True)
    analysis_utility.create_experiment_summary(config.general_settings.main_folder)
    # analysis_utility.process_spike_csvs_to_pkl(main_folder, overwrite = True)



if __name__ == "__main__":
    main()
