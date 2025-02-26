import numpy as np
import os
import sys
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
# from analyze_suite2p.config_loader import load_json_config_file
#TODO update configurations to JSON file and update code accordingly

class OpsEditor:
    def __init__(self, master):
        self.master = master
        self.master.title("Edit Analysis Parameters")

        # Load default operations
        
    # Define the parameters you want to allow editing, optionally have all be editable? CHANGE TO RELEVANT PARAMETERS
        self.editable_params = { #self.default_analysis_parameters
            'overwrite_suite2p': False,
            'skew_threshold': 1.0,
            'compactness_threshold': 1.4,
            "peak_detection_threshold": 4.5,
            'peak_count_threshold': 2,
            'Img_Overlay': 'max_proj',
            'use_suite2p_ROI_classifier': False,
            'update_suite2p_iscell': True,
            'return_decay_times': False,
        }

        self.vars = {}
        self.create_widgets()

    def load_configurations(self):
        script_dir = Path(__file__).resolve().parent  # Get current script directory (project/src/gui_config)
        config_file_path = (script_dir / "../../config/config.json").resolve()  # Navigate to config folder

        if Path(config_file_path).exists():
            with open(config_file_path, 'r')as f:
                return json.load(f)
        else:
            return self.editable_params
    
    def save_analysis_params(self):
        analysis_params = Path(__file__).resolve().parent / "../../config/analysis_params.json"
        updated_params = {}

        # Create input fields for each parameter
        for param, var in self.vars.items():
            if isinstance (var, tk.BooleanVar):
                updated_params[param] = var.get()
            elif isinstance(var, tk.StringVar):
                updated_params[param] = var.get()
            else:
                try:
                    updated_params[param] = float(var.get())
                except ValueError:
                    messagebox.showerror("Error", f"Invalid value for {param}")
        
        with open(analysis_params, 'w') as f:
            json.dump(updated_params, f, indent = 4)
        
        self.master.quit()

    def create_widgets(self):
        for idx, (param, value) in enumerate(self.editable_params.items()):
            tk.Label(self.master, text = param, font=("Arial", 12)).grid(row=idx, column=0, padx=10, pady=5, sticky='w')
            
            if isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                check_box = tk.Checkbutton(self.master, variable = var)
                check_box.grid(row=idx, column=1, padx=10,pady=5)

            elif param == "Img_Overlay":
                var = tk.StringVar(value=value)
                dropdown = ttk.Combobox(
                    self.master, textvariable=var, values=["max_proj", "meanImg"], state = 'readonly', width=20
                )
                dropdown.grid(row=idx, column = 1, padx=10, pady=5)
            # elif param == "peak_threshold":
            #     var = tk.IntVar(value=value)
            #     dropdown = ttk.Combobox(
            #         self.master, textvariable=var, values=[0,1,2,3,4,5], state = 'readonly', width=20
            #     )
            #     dropdown.grid(row=idx, column = 1, padx=10, pady=5)
            
            else:
                var = tk.StringVar(value=str(value))
                tk.Entry(self.master, textvariable=var, width=20).grid(row=idx, column=1, padx=10, pady=5)
            self.vars[param] = var

        tk.Button(self.master, text="Save Configurations", command=self.save_analysis_params).grid(row=len(self.editable_params), column=0, columnspan=2, pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    app = OpsEditor(root)
    root.mainloop()
