import os
from pathlib import Path

def check_for_single_image_file_in_folder(current_path, file_ending):
    """Checks for number of files in each subfolder within current path
    
    Args:
    ========
    current_path: str
        path-like object telling where to search for files
    file_ending: str
        file extension to search for file type (e.g., tiff, tif, nd2)
        
    Returns:
    ========
    len(files): int
        Number of files within a current_path
    """
    files = [file for file in os.listdir(current_path) if file.endswith(file_ending)]
    return len(files)

def find_valid_folders(main_folder, ext):
    """Function to find folders with valid image files. 
       All folders are added to a list for processing with suite2p
       
       Args:
       ========
        main_folder: str
            path-like object for experiment folder containing all image files for processing
        ext: str
            string-like file extension (e.g., tiff, tif, nd2, mp4 bool
            Boolean check to check for multiple videos in the same subfolder
        
        Returns:
        ========
        valid: list
            list of folders containing image files (one file or multiple 
            optional: returns experiment folders with all images from one experimental treatment
                experiment folders can be processed in run_suite2p.py by export_image_files_to_suite2p_format
       """
    main_folder = Path(main_folder)

    if not main_folder.exists():
        raise ValueError("Main folder does not exist")

    valid = []

    for folder in main_folder.iterdir():
        if not folder.is_dir():
            continue

        if check_for_single_image_file_in_folder(folder, ext) >= 1:
            valid.append(folder.name)
        else:
            for sub in folder.iterdir():
                if sub.is_dir() and check_for_single_image_file_in_folder(sub, ext) == 1:
                    valid.append(folder.name)
                    break
    return valid


def build_exp_condition(folders):
    return {f: f for f in folders}

