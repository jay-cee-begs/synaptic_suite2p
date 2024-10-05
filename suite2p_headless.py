import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from pathlib import Path
from PIL import Image
# !pip install nd2reader
from nd2reader import ND2Reader #only if converting to tiff
import shutil
# sys.path.insert(0, 'D:/users/JC/suite2p-0.14.0')
from suite2p import run_s2p

import configurations
from detector_utility import *
from suite2p_utility import *
from analysis_utility import *
from plotting_utility import *
#potential issue here in that configurations would need to 
#be accessed in both virtual environments if we define a directory here

BASE_DIR = configurations.main_folder

def getFilesWithExt(top_dir, files_ext):
    matches = []
    for root, dirnames, filenames in os.walk(str(top_dir)):
        for _dir in dirnames:
            matches += getFilesWithExt(_dir, files_ext)
        for filename in filenames:
            full_path = os.path.join(root, filename)
            if full_path.endswith(files_ext):
                matches.append(Path(os.path.join(root, filename)))
    return matches

def convertND2toTiff(fp_pathlib):
    
    print("Attempting to convert:", str(fp_pathlib))
    save_fp = tiffPathFromND2(fp_pathlib)
    save_dir = save_fp.parent
    print(f"Saving to: {save_fp} in dir: {save_dir}")
    save_dir.mkdir(parents=False, exist_ok = True)
    with ND2Reader(str(fp_pathlib)) as images:
        images_li=[]
        images.iter_axes='t'
        for idx in range(len(images)):
            images_li.append(Image.fromarray(np.array(images[idx])))
        images_li[0].save(save_fp, save_all=True, append_images=images_li[1:])
        print("Done converting")
        
def tiffPathFromND2(_file):
    return Path(f"{_file.parent}/{_file.stem}/{_file.stem}.tif")

def iterConvert():
    tiff_files = getFilesWithExt(BASE_DIR, ".tif")
    files_to_convert = [_file for _file in getFilesWithExt(BASE_DIR, ".nd2")
                       if tiffPathFromND2(_file) not in tiff_files]
    print("Files to convert:", files_to_convert)
    print("Total number of files to convert:", len(files_to_convert))
    for fp in tqdm.tqdm(files_to_convert):
        print(f"Processing {fp.name}.tif")
        convertND2toTiff(fp)

#iterConvert()



def export_image_files_to_suite2p_format(parent_directory, file_ending= configurations.data_extension):
    """Export .tif files each into its own folder for suite2p processing, for all directories within a given parent directory."""
    
    if not os.path.exists(parent_directory):
        print(f"Provided path does not exist: {parent_directory}")
        return
    
    # Process each directory within the parent directory
    for root, dirs, files in os.walk(parent_directory):
        # Processing each file within the directory
        for file in files:
            if file.endswith(file_ending):
                name, _ = os.path.splitext(file)
                folder_path = os.path.join(root, name)
                os.makedirs(folder_path, exist_ok=True)

                source = os.path.join(root, file)
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

    def check_for_single_image_file_in_folder(current_path, file_ending = configurations.data_extension):
        """
        Check if the specified path contains exactly one .nd2 file.
        """
        tiff_files = [file for file in os.listdir(current_path) if file.endswith(file_ending)]
        return len(tiff_files) == 1

    found_image_folders = []
    for current_path, directories, files in os.walk(path):
        # Check if current directory is a "deepest" directory (no subdirectories)
        if check_for_single_image_file_in_folder(current_path):
            found_image_folders.append(current_path)

    return found_image_folders

# Example Usage:
# image_folders = get_all_image_folders_in_path('/path/to/search')
# print(image_folders)

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
            
                 opsEnd = run_s2p(ops=configurations.ops, db=db)
            except (ValueError, AssertionError) as e:
                 print(f"Error processing {image_path}: {e}")

def main():
    main_folder = configurations.main_folder
    data_extension = configurations.data_extension
    img_folders = get_all_image_folders_in_path(main_folder)
    if len(img_folders) is 0:
        export_image_files_to_suite2p_format(main_folder, file_ending = data_extension)
    image_folders = get_all_image_folders_in_path(main_folder)
    process_files_with_suite2p(image_folders)
    #TODO make sure this part of the code is correct for the pipeline (easier said than done)
    translate_suite2p_outputs_to_csv(main_folder, overwrite = False)
    process_spike_csvs_to_pkl(main_folder, overwrite = True)


if __name__ == "__main__":
    main()


"""To Run:
activate suite2p
import run_suite2p 
if __name__ == "__main__":
    run_suite2p.main()
    """