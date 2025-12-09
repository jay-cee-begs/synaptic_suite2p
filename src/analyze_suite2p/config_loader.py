import json
from pathlib import Path
from types import SimpleNamespace


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
        with open(config_path, 'r') as f:
            config_dict = json.load(f)       
    else:
        script_dir = Path(__file__).resolve().parent  # Get current script directory (project/src/gui_config)
        json_filepath = (script_dir / "../../config/config.json").resolve()
        with open(json_filepath, 'r') as f:
            config_dict = json.load(f)
    return json.loads(json.dumps(config_dict), object_hook=lambda d: SimpleNamespace(**d))

config = load_json_config_file()

def load_json_dict():
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
    script_dir = Path(__file__).resolve().parent  # Get current script directory (project/src/gui_config)
    json_filepath = (script_dir / "../../config/config.json").resolve()
    with open(json_filepath, 'r') as f:
        config_dict = json.load(f)
    return config_dict