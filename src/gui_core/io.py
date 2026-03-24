import json
from pathlib import Path
from tkinter import messagebox

from gui_core.general_settings_model import GenSettings
from gui_core.analysis_model import AnalysisParams

def load_config():
        try:    
            script_dir = Path(__file__).resolve().parent  # Get current script directory (project/src/gui_config)
            config_file_path = (script_dir / "../../config/config.json").resolve()  # Navigate to config folder

            with open(config_file_path, 'r')as f:
                config_dict = json.load(f)
            
            config = GenSettings.from_dict(config_dict)
            return config
        except FileNotFoundError:
            messagebox.showerror("Error", "Configuration file not found. Starting with default settings.")
            return GenSettings()


def save_config(path, config: GenSettings):
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=1)