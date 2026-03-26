import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import filedialog
from pathlib import Path
import os
import subprocess
import threading 
import json

from gui_core.io import load_config, save_config
# from gui_core.folder_logic import find_valid_folders, build_exp_condition
from gui_core import folder_logic
from gui.ops_editor import OpsEditor
from gui_core.general_settings_model import GenSettings
from gui_core.analysis_model import AnalysisParams

import logging
from pathlib import Path
from tkinter import messagebox

# Set up logging to file and console
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigEditor:
    def __init__(self, master):
        self.master = master
        self.master.title("Synaptic suite2P Analysis Configurations Editor")

        # Load existing configurations
        self.config = load_config()
        logger.debug(f"CONFIG LOADED: {self.config}")
        # messagebox.showinfo("Debug", f"CONFIG LOADED: {self.config}")

        # Correct: GenSettings object is self.config
        self.gen_settings = self.config
        logger.debug(f"GEN SETTINGS: {vars(self.gen_settings)}")
        # messagebox.showinfo("Debug", f"GEN SETTINGS: {vars(self.gen_settings)}")

        # Initialize Tkinter variables
        self.selected_bat_file = tk.StringVar()  # Initialize selected_bat_file

        self.main_folder_var = tk.StringVar(value=self.gen_settings.main_folder)
        self.data_extension_var = tk.StringVar(value=self.gen_settings.data_extension)
        self.frame_rate_var = tk.IntVar(value=self.gen_settings.frame_rate)
        self.ops_path_var = tk.StringVar(value=self.gen_settings.ops_path)
        self.exp_dur_var = tk.IntVar(value=self.gen_settings.experiment_duration)
        self.bin_width_var = tk.IntVar(value=self.gen_settings.bin_width)

        self.groups = self.gen_settings.groups
        self.exp_condition = self.gen_settings.exp_condition

        # Debug Tk variable values
        logger.debug(f"Main folder var: {self.main_folder_var.get()}")
        logger.debug(f"Data extension var: {self.data_extension_var.get()}")
        logger.debug(f"Frame rate var: {self.frame_rate_var.get()}")
        logger.debug(f"Ops path var: {self.ops_path_var.get()}")
        logger.debug(f"Exp duration var: {self.exp_dur_var.get()}")
        logger.debug(f"Bin width var: {self.bin_width_var.get()}")
        # messagebox.showinfo("Debug", f"Tk Vars Loaded:\nMain folder: {self.main_folder_var.get()}\nFrame rate: {self.frame_rate_var.get()}")

        # Build GUI frames
        self.master.geometry("500x900")  # Set initial window size
        self.master.resizable(True, True)
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(fill="both", expand=True)

        # Main folder input
        tk.Label(self.main_frame, text="Experiment / Main Folder Path:").pack(anchor='w', padx=10, pady=5)
        tk.Entry(self.main_frame, textvariable=self.main_folder_var, width=50).pack(padx=10)
        tk.Button(self.main_frame, text="Browse", command=self.browse_folder).pack(padx=10, pady=5)

        # Data extension input
        tk.Label(self.main_frame, text="Data Extension:").pack(anchor='w', padx=10, pady=5)
        tk.Entry(self.main_frame, textvariable=self.data_extension_var).pack(padx=10)

        # Groups
        self.group_frame = tk.Frame(self.main_frame)
        self.group_frame.pack(padx=10, pady=5)
        tk.Label(self.group_frame, text="Adds all subfolders from the Experiment:").pack(side=tk.LEFT)
        tk.Button(self.group_frame, text="Add Experiment Conditions", command=self.add_group).pack(side=tk.LEFT)

        # Ops path
        tk.Label(self.main_frame, text="Suite2p configurations (ops.npy):").pack(anchor='w', padx=10, pady=5)
        ops_frame = tk.Frame(self.main_frame)
        ops_frame.pack(padx=10, pady=5)
        tk.Entry(ops_frame, textvariable=self.ops_path_var, width=40).pack(side=tk.LEFT)
        tk.Button(ops_frame, text="Browse", command=self.browse_ops_file).pack(side=tk.LEFT)
        # tk.Button(self.main_frame, text="Open Suite2p GUI", command=self.launch_suite2p_gui).pack(pady=5)
        # tk.Label(self.main_frame, text="Open Suite2p GUI to create a new ops file").pack(anchor='w', padx=30, pady=5)

        # Frame rate
        tk.Label(self.main_frame, text="Frame Rate:").pack(anchor='w', padx=10, pady=5)
        tk.Entry(self.main_frame, textvariable=self.frame_rate_var).pack(padx=10)

        # Experiment duration
        tk.Label(self.main_frame, text="Experiment Duration (seconds):").pack(anchor='w', padx=10, pady=5)
        tk.Entry(self.main_frame, textvariable=self.exp_dur_var).pack(padx=10)

        # Bin width
        tk.Label(self.main_frame, text="Network Bin Width (# frames):").pack(anchor='w', padx=10, pady=5)
        tk.Entry(self.main_frame, textvariable=self.bin_width_var).pack(padx=10)

        # Editable exp_condition
        self.exp_condition_frame = tk.Frame(self.main_frame)
        self.exp_condition_frame.pack(padx=10, pady=5)
        self.create_exp_condition_dict_entries(self.exp_condition_frame, " ", self.exp_condition)

        # Edit analysis params
        tk.Button(self.main_frame, text="Edit Analysis Parameters", command=self.open_ops).pack(pady=5)

        # Save button
        tk.Button(self.main_frame, text="Save Configurations", command=self.save).pack(pady=10)

        # Process button
        tk.Button(self.main_frame, text="Process", command=self.run_pipeline).pack(pady=10)

        # Final debug message
        logger.debug("GUI initialized successfully")
        # messagebox.showinfo("Debug", "GUI initialized successfully")


    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(-1 * (event.delta // 120), "units")   

    def open_ops(self):
        top = tk.Toplevel(self.master)
        OpsEditor(top, self.config)

    def launch_suite2p_gui(self):
        """Call the function to create new ops file"""
        current_dir = Path(__file__).parent
        scripts_dir = current_dir / "Scripts"
        bat_file = scripts_dir / "run_s2p_gui.bat"
        subprocess.call([str(bat_file)])  # Execute run_s2p_gui.bat


    
    def browse_ops_file(self):
        file_selected = filedialog.askopenfilename(filetypes=[("Ops Files", "*.npy")])
        if file_selected:
            self.ops_path_var.set(file_selected)

    def browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.main_folder_var.set(folder_selected)

    def edit_analysis_params(self):
        """Call the function to edit default ops"""
        current_dir = Path(__file__).parent
        scripts_dir = current_dir / "Scripts"
        bat_file = scripts_dir / "edit_analysis_params.bat"
        subprocess.call([str(bat_file)])  # Execute run_default_ops.bat
        self.merge_analysis_params()
    
    def merge_analysis_params(self):
        script_dir = Path(__file__).resolve().parent
        analysis_params_file = script_dir / "../../config/analysis_params.json"
        config_file_path = script_dir / "../../config/config.json"

        if Path(analysis_params_file).exists():
            with open(analysis_params_file, 'r') as f:
                analysis_params = json.load(f)
        
            if Path(config_file_path).exists():
                with open(config_file_path, 'r') as f:
                    config_data = json.load(f)
            else:
                config_data = {}
            
            config_data['analysis_params'] = analysis_params
            with open(config_file_path, 'w') as f:
                json.dump(config_data, f, indent=1)

            messagebox.showinfo("Success","Editable parameters updated! \n Merged with config.json was successful!")
        else:
            messagebox.showerror("Error", "No analysis parameters found;\n using default parameters")
            
            analysis_params = {
                "peak_count":self.peak_threshold,
                "skew": self.skew_threshold,
                "compact": self.compact_threshold,
                "overwrite_suite2p": self.overwrite_suite2p,
                "img_overlay": self.img_overlay,
                "use_suite2p_ROI_classifier": self.use_iscell,
            },
            if Path(config_file_path).exists():
                with open(config_file_path, 'r') as f:
                    config_data = json.load(f)
            else:
                config_data = {}
            
            config_data['analysis_params'] = analysis_params


           
    def add_group(self):
        folders = folder_logic.find_valid_folders(
            self.main_folder_var.get(), self.data_extension_var.get())
        self.config.groups = folders
        self.config.exp_condition = folder_logic.build_exp_condition(folders)

        print("Groups:", folders)

    def create_exp_condition_dict_entries(self, master, title, dictionary):
        """will allow you to edit dictionaries in the configurations file"""
        tk.Label(master, text=title).pack(anchor='w', padx=10, pady=5)
        self.dict_vars = {}
        for key, value in dictionary.items():
            frame = tk.Frame(master)
            frame.pack(padx=10, pady=5)
            tk.Label(frame, text="Key:").pack(side=tk.LEFT)
            key_var = tk.StringVar(value=key)
            value_var = tk.StringVar(value=value)
            self.dict_vars[key] = (key_var, value_var)
            tk.Entry(frame, textvariable=key_var, width=15).pack(side=tk.LEFT)
            tk.Label(frame, text="Value:").pack(side=tk.LEFT)
            tk.Entry(frame, textvariable=value_var, width=15).pack(side=tk.LEFT)


    def update_exp_condition_entries(self):
        """Update the entries in the exp_condition dictionary with the use of create_exp_condition_dict_entries"""
        for widget in self.exp_condition_frame.winfo_children():
            widget.destroy()  # Remove old entries
        self.create_exp_condition_dict_entries(self.exp_condition_frame, "exp_condition", self.exp_condition)    

    def save(self):
        self.gen_settings.main_folder = self.main_folder_var.get()
        self.gen_settings.data_extension = self.data_extension_var.get()
        self.gen_settings.frame_rate = self.frame_rate_var.get()
        self.gen_settings.ops_path = self.ops_path_var.get()
        self.gen_settings.bin_width = self.bin_width_var.get()
        self.gen_settings.experiment_duration = self.exp_dur_var.get()
        script_dir = Path(__file__).resolve().parent  # Get current script directory (project/src/gui_config)
        config_file_path = (script_dir / "../../config/config.json").resolve()
        save_config(config_file_path, self.config)
        print("Saved!")

    def create_process_button(self, parent_frame):
        tk.Button(parent_frame, text="Proceed", command=self.run_pipeline).pack(pady=5)

    def run_pipeline(self):  #Option to skip suite2p, will execute a different .bat then
        current_dir = Path(__file__).parent
        scripts_dir = os.path.join(current_dir, "Scripts") 
        bat_file = os.path.join(scripts_dir, "run_suite2p.bat")
        print(f"Executing {bat_file}")
        #subprocess.call([str(bat_file)])  # Execute sequence.bat
        threading.Thread(target=self.run_subprocess, args=(bat_file,)).start()
