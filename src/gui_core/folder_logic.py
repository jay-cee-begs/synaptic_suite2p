import os
from pathlib import Path

def check_for_single_image_file_in_folder(current_path, file_ending):
    files = [file for file in os.listdir(current_path) if file.endswith(file_ending)]
    return len(files)

def find_valid_folders(main_folder, ext):
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

