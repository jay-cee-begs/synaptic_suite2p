import json
from pathlib import Path
from types import SimpleNamespace
import os


def load_json_config_file(config_path = None):
    """
    Load config.json synapse_gui configurations file as SimpleNamespace object

    Args:
    -----
    config_path: str, optional
        Path to JSON configurations file (config.json)
    
    Returns:
    --------
    SimpleNamespace JSON dictionary

    Example:
        >>> load_json_config_file()
        {config.general_settings.main_folder, config.general_settings.groups, 
        config.analysis_params.overwrite_suite2p, ...} 
    """
    if config_path is not None: 
        experiment = str(config_path).split('\\')[-1]
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)    
                print(f"Successfully loaded {experiment} json configurations file")
   
        except PermissionError as e:
            print(e)
            print("Config file path not given...trying to append `analysis_config.json` to the file path")
            with open(os.path.join(config_path, 'analysis_config.json'), 'r') as f:
                config_dict = json.load(f)
                print(f"Successfully loaded {experiment} json configurations file")

    else:
        script_dir = Path(__file__).resolve().parent  # Get current script directory (project/src/gui_config)
        default_path = str(script_dir).split('\\')[-1]
        json_filepath = (script_dir / "../../config/config.json").resolve()
        with open(json_filepath, 'r') as f:
            config_dict = json.load(f)
        print(f"Loading default {default_path} json configurations file")

    return json.loads(json.dumps(config_dict), object_hook=lambda d: SimpleNamespace(**d))


def load_json_dict(config_path = None):
    """
    Load config.json synapse_gui configurations file as a dictionary

    Args:
    -----
    config_path: str, optional
        Path to JSON configurations file (config.json)
    
    Returns:
    --------
    config_dict: dictionary

    Example:
        >>> load_json_config_file()
        {config['general_settings']['main_folder'], config['general_settings']['groups'], 
        config['analysis_params']['overwrite_suite2p'], ...} 
    """
    if config_path is not None: 
        experiment = str(config_path).split('\\')[-1]
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)    
                print(f"Successfully loaded {experiment} json configurations file")
   
        except PermissionError as e:
            print(e)
            print("Config file path not given...trying to append `analysis_config.json` to the file path")
            with open(os.path.join(config_path, 'analysis_config.json'), 'r') as f:
                config_dict = json.load(f)
                print(f"Successfully loaded {experiment} json configurations file")     
    else:
        script_dir = Path(__file__).resolve().parent  # Get current script directory (project/src/gui_config)
        default_path = str(script_dir).split('\\')[-1]
        json_filepath = (script_dir / "../../config/config.json").resolve()
        with open(json_filepath, 'r') as f:
            config_dict = json.load(f)
        print(f"Loading default {default_path} json configurations file")

    return config_dict