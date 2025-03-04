import json
from pathlib import Path
from types import SimpleNamespace


def load_json_config_file(config_path = None):
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
    script_dir = Path(__file__).resolve().parent  # Get current script directory (project/src/gui_config)
    json_filepath = (script_dir / "../../config/config.json").resolve()
    with open(json_filepath, 'r') as f:
        config_dict = json.load(f)
    return config_dict