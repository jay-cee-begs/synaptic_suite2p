import os
import numpy as np
import tqdm
from pathlib import Path
from PIL import Image
from nd2reader import ND2Reader #only if converting to tiff
import shutil
import sys
# sys.path.insert(0, 'D:/users/JC/suite2p-0.14.0')
from suite2p import run_s2p

from analyze_suite2p.config_loader import load_json_config_file, load_json_dict
from analyze_suite2p import suite2p_utility
_DEFAULT_CONFIG = load_json_config_file()
config = _DEFAULT_CONFIG


def getFilesWithExt(top_dir, files_ext):
    """
    Function to get all files with given data extension (nd2) in a directory

    Args:
    ----------
        top_dir : str / directory 
            Top / Parent directory containing image files
        files_ext : str
            File extension for given image files (e.g., 'tiff', 'nd2', 'tif')
            The function is optimized to find and convert ND2 files into tif files.
    
    Returns:
    ----------
        matches : list
            list of files matching files_ext within top_dir 
    """
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
    """
    Primary worker function to convert individual ND2 image file into tif file.
    The function uses the tiffPathFromND2 to create each new tiff path.
    The ND2 file is opened and re-saved as a tif file in a new subfolder with matching name.

    Args:
    ----------
        fp_pathlib : str / Path-like Object 
            File path leading to ND2 file to convert
        
    Returns:
    ----------

    Notes:
    ----------
        This function only works with a single image file. It is run, nested,
        within the function iterConvert found below. 
    """
    print("Attempting to convert:", str(fp_pathlib))
    save_fp = tiffPathFromND2(fp_pathlib)
    save_dir = save_fp.parent
    print(f"Saving to: {save_fp} in dir: {save_dir}")
    # save_dir.mkdir(parents=False, exist_ok = True)
    with ND2Reader(str(fp_pathlib)) as images:
        images_li=[]
        images.iter_axes='t'
        for idx in range(len(images)):
            images_li.append(Image.fromarray(np.array(images[idx])))
        images_li[0].save(save_fp, save_all=True, append_images=images_li[1:])
        print("Done converting")
        
def tiffPathFromND2(_file):
    """
    Iterator function to create file paths for converted tif files into subfolders.
    
    Args:
    ----------
        _file : str / Path object 
            File path leading to single tif file
    
    Returns:
    ----------
        Path(f"{_file.parent}/{_file.stem}/{_file.stem}.tif")
            pathlib object creating a subfolder with the name of the file stem
            _file.stem within the same directory leading to a .tif file of the same
            name that is going to be created.
    """
    return Path(f"{_file.parent}/{_file.stem}.tif")

def iterConvert(config):
    """
    Wrapper function to automate converting all ND2 files into tif files
    within a given directory.
    The function uses 'getFilesWithExt' to get ND2 files, 
    uses tiffPathFromND2 to create new file paths,
    and uses 'convertND2toTiff' to create tif files from ND2 files
    Args:
    ----------
        config : SimpleNameSpace dictionary 
            config.json file loaded as SimpleNameSpace dictionary
    
    Returns:
    ----------
        Nones
    """
    BASE_DIR = config.general_settings.main_folder

    tiff_files = getFilesWithExt(BASE_DIR, ".tif")
    files_to_convert = [_file for _file in getFilesWithExt(BASE_DIR, ".nd2")
                       if tiffPathFromND2(_file) not in tiff_files]
    print("Files to convert:", files_to_convert)
    print("Total number of files to convert:", len(files_to_convert))
    for fp in tqdm.tqdm(files_to_convert):
        print(f"Processing {fp.name}.tif")
        convertND2toTiff(fp)

