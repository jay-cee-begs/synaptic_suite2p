import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import filedialog
from pathlib import Path
import os
import subprocess
import time
import threading 
import json
import numpy as np


class ConfigEditor:
    def __init__(self, master):
        self.master = master
        self.master.title("Synaptic suite2P Analysis Configurations Editor")
        self.master.geometry("450x750")  # Set initial window size

        # Create a canvas and a scrollbar
        self.canvas = tk.Canvas(master)
        self.scrollbar = tk.Scrollbar(master, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)


        # Configure the scrollbar #### Find a way to have the whole frame scrollable
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Link the scrollbar to the canvas
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack the canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Load existing configurations, needs an existing file to load from
        self.config = self.load_config()
        general_settings = self.config.get("general_settings", {})
        # analysis_params = {self.config.get("analysis_params", {})}

        self.selected_bat_file = tk.StringVar()  # Initialize selected_bat_file
        self.main_folder_var = tk.StringVar(value=general_settings.get('main_folder', ''))
        self.data_extension_var = tk.StringVar(value=general_settings.get('data_extension', ''))
        self.frame_rate_var = tk.IntVar(value=general_settings.get('frame_rate', 20))
        self.ops_path_var = tk.StringVar(value=general_settings.get('ops_path', ''))
        self.groups = self.config.get('groups', [])
        self.exp_condition = {}
        self.exp_dur_var = tk.IntVar(value=self.config.get("EXPERIMENT_DURATION", 180))
        self.bin_width_var = tk.IntVar(value=self.config.get("BIN_WIDTH", 5))

        # Main folder input
        tk.Label(self.scrollable_frame, text="Experiment / Main Folder Path:").pack(anchor='w', padx=10, pady=5)
        tk.Entry(self.scrollable_frame, textvariable=self.main_folder_var, width=50).pack(padx=10)
        
        # Button to open file explorer for selecting a folder
        tk.Button(self.scrollable_frame, text="Browse", command=self.browse_folder).pack(padx=10, pady=5)
        

        # Data extension input
        tk.Label(self.scrollable_frame, text="Data Extension:").pack(anchor='w', padx=10, pady=5)
        tk.Entry(self.scrollable_frame, textvariable=self.data_extension_var).pack(padx=10)

        
        # Group input
        self.group_frame = tk.Frame(self.scrollable_frame)
        self.group_frame.pack(padx=10, pady=5)
        tk.Label(self.group_frame, text="Adds all subfolders from the Experiment:").pack(side=tk.LEFT)
        tk.Button(self.group_frame, text="Add Experiment Conditions", command=self.add_group).pack(side=tk.LEFT)

        # Ops path input
        tk.Label(self.scrollable_frame, text="Suite2p configurations (ops.npy):").pack(anchor='w', padx=10, pady=5)
        #tk.Entry(self.scrollable_frame, textvariable=self.ops_path_var, width=50).pack(padx=10)
        
        # Option a: Insert file path
        ops_frame = tk.Frame(self.scrollable_frame)
        ops_frame.pack(padx=10, pady=5)
        tk.Entry(ops_frame, textvariable=self.ops_path_var, width=40).pack(side=tk.LEFT)
        tk.Button(ops_frame, text="Browse", command=self.browse_ops_file).pack(side=tk.LEFT)

        # Option c: Create new ops file
        tk.Button(self.scrollable_frame, text="Open Suite2p GUI", command=self.launch_suite2p_gui).pack(pady=5)
       
        tk.Label(self.scrollable_frame, text="Open Suite2p GUI to create a new ops file").pack(anchor='w', padx=30, pady=5)
        # Frame rate input
        tk.Label(self.scrollable_frame, text="Frame Rate:").pack(anchor='w', padx=10, pady=5)
        tk.Entry(self.scrollable_frame, textvariable=self.frame_rate_var).pack(padx=10)

        # EXPERIMENT_DURATION input
        tk.Label(self.scrollable_frame, text = "Experiment Duration (seconds):").pack(anchor='w', padx=10, pady=5)
        tk.Entry(self.scrollable_frame, textvariable=self.exp_dur_var).pack(padx=10)

        # BIN_WIDTH input
        tk.Label(self.scrollable_frame, text = "Network Bin Width (# frames):").pack(anchor='w', padx=10, pady=5)
        tk.Entry(self.scrollable_frame, textvariable=self.bin_width_var).pack(padx=10)

        # Editable exp_condition
        self.exp_condition_frame = tk.Frame(self.scrollable_frame)
        self.exp_condition_frame.pack(padx=10, pady=5)
        self.create_exp_condition_dict_entries(self.exp_condition_frame, " ", self.exp_condition)

        # Edit analysis_params.json
        tk.Button(self.scrollable_frame, text="Edit Analysis Parameters", command=self.edit_analysis_params).pack(pady=5)

        # Save button
        tk.Button(self.scrollable_frame, text="Save Configurations", command=self.save_config).pack(pady=10)

        # Processing button
        tk.Button(self.scrollable_frame, text="Process", command=self.proceed).pack(pady=10)


################ Functions AREA ################    put in seperate file eventually
               
    def setup_ui(self):
        # Setup the UI components in here in the future
        # order is the order of appearance in the gui
        tk.Button(self.scrollable_frame, text="Save Configurations", command=self.save_config).pack(pady=10)
        self.create_process_button()

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(-1 * (event.delta // 120), "units")   

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


    def load_config(self):
        try:    
            script_dir = Path(__file__).resolve().parent  # Get current script directory (project/src/gui_config)
            config_file_path = (script_dir / "../../config/config.json").resolve()  # Navigate to config folder

            with open(config_file_path, 'r')as f:
                config = json.load(f)
        except FileNotFoundError:
            messagebox.showerror("Error", "Configuration file not found. Starting with default settings.")
            return {}
        return config

    def add_group(self):
        self.groups.clear()
        main_folder = self.main_folder_var.get().strip()
        if not Path(main_folder).exists():
            messagebox.showerror("Error", "Main folder does not exist.")
            return
        
        file_ending = self.data_extension_var.get().strip()  # Get the specified file extension

        def check_for_single_image_file_in_folder(current_path, file_ending):
            """
            Check if the specified path contains exactly one file with the given extension.
            """
            files = [file for file in os.listdir(current_path) if file.endswith(file_ending)]
            return len(files)
        all_folders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]
        excluded_substrings = []
        unique_folders = [folder for folder in all_folders if not any(excluded in folder for excluded in excluded_substrings)]

        file_ending = self.data_extension_var.get().strip()  # Get the specified file extension

        valid_folders = []  # To hold valid folders
        
        for folder_name in unique_folders:
            current_folder_path = os.path.join(main_folder, folder_name)
            if check_for_single_image_file_in_folder(current_folder_path, file_ending) >= 1:
                valid_folders.append(folder_name)
                
            else:
                # Check if any subfolder has exactly one file with the specified extension
                subfolders = [f for f in os.listdir(current_folder_path) if os.path.isdir(os.path.join(current_folder_path, f))]
                for subfolder in subfolders:
                    subfolder_path = os.path.join(current_folder_path, subfolder)
                    if check_for_single_image_file_in_folder(subfolder_path, file_ending) == 1:
                        valid_folders.append(folder_name)
                        break  # No need to check other subfolders if one matches

        for folder_name in valid_folders:
            group_path = f"\\{folder_name}" if not folder_name.startswith("\\") else folder_name
            
            if folder_name not in self.exp_condition:
                self.exp_condition[folder_name] = f"{folder_name}" #populates it with the folder name for the user to change?
            
            if group_path not in self.groups:
                self.groups.append(group_path)

        self.update_exp_condition_entries()
        
        if valid_folders:
            messagebox.showinfo("Groups Added", f"Added Groups: {', '.join(valid_folders)}")
        else:
            messagebox.showinfo("No Groups Added", "No (sub-)folders with one or more files matching the specified extension were found.")
            

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
        
    def save_config(self):
        main_folder = str(Path(self.main_folder_var.get().strip()).resolve())
        data_extension = self.data_extension_var.get().strip()
        frame_rate = self.frame_rate_var.get()
        ops_path = str(Path(self.ops_path_var.get().strip()).resolve())
        BIN_WIDTH = self.bin_width_var.get()
        EXPERIMENT_DURATION = self.exp_dur_var.get()

        if not Path(main_folder).exists():
            messagebox.showerror("Error", "Main folder does not exist.")
            return

        script_dir = Path(__file__).resolve().parent  # Get current script directory (project/src/gui_config)
        json_filepath = (script_dir / "../../config/config.json").resolve()  # Navigate to config folder
        analysis_params_path = (script_dir / "../../config/analysis_params.json")
        if analysis_params_path.exists():
            with open(analysis_params_path, 'r') as f:
                analysis_params = json.load(f)
        else:
            analysis_params = {'overwritesuite2p': False,
            'skew_threshold': 1.0,
            'compactness_threshold': 1.4, #TODO implement cutoff / filter to rule out compact failing ROIs
            "peak_detection_threshold": 4.5,
            'peak_count_threshold': 2,
            'Img_Overlay': 'max_proj',
            'use_suite2p_ROI_classifier': False,
            'update_suite2p_iscell': True,
            'return_decay_times': False,}

        config_data = {
            "general_settings":{
                "main_folder": main_folder,
                "groups": [str(Path(main_folder) / condition) for condition in self.dict_vars.keys()],
                "group_number": len(self.groups),
                "exp_condition": {key_var.get(): value_var.get() for key_var, (key_var, value_var) in self.dict_vars.items()},
                "data_extension": data_extension,
                "frame_rate": frame_rate,
                "ops_path": ops_path,
                "BIN_WIDTH": BIN_WIDTH,
                "EXPERIMENT_DURATION": EXPERIMENT_DURATION,
                "FRAME_INTERVAL": 1 / float(frame_rate),
                "FILTER_NEURONS": True,
            },
            "analysis_params": analysis_params
        }
        with open(json_filepath, 'w') as json_file:
            json.dump(config_data, json_file, indent=1)
        messagebox.showinfo("Success", "Configurations saved successfully.")

    def get_current_dir(self):
        return self.current_dir 
    
    def move_up(self, levels = 1):
        new_dir = self.current_dir

    def show_log_window(self, log_file):
        log_window = tk.Toplevel(self.master)
        log_window.title("Process Log")

        with open(log_file, "r") as f:
            log_content = f.read()

        text_widget = tk.Text(log_window, wrap="word")
        text_widget.insert("1.0", log_content)
        text_widget.config(state=tk.DISABLED)  # Make the text widget read-only
        text_widget.pack(expand=True, fill="both")

        tk.Button(log_window, text="Close", command=log_window.destroy).pack(pady=5)

    def create_process_button(self, parent_frame):
        tk.Button(parent_frame, text="Process", command=self.proceed).pack(pady=5)

    def proceed(self):  #Option to skip suite2p, will execute a different .bat then
        current_dir = Path(__file__).parent
        scripts_dir = os.path.join(current_dir, "Scripts") 
        bat_file = os.path.join(scripts_dir, "run_suite2p.bat")
        print(f"Executing {bat_file}")
        #subprocess.call([str(bat_file)])  # Execute sequence.bat
        threading.Thread(target=self.run_subprocess, args=(bat_file,)).start()

    def run_subprocess(self, bat_file):
        subprocess.call([str(bat_file)])  # Execute sequence.bat
        # Redirect the terminal output to a text file, seperate function to reduce interference with the process bar
        scripts_dir = Path(bat_file).parent
        log_file = scripts_dir / "process_log.txt"
        with open(log_file, "w") as f:
            process = subprocess.Popen([str(bat_file)], stdout=f, stderr=subprocess.STDOUT)
            process.wait()

        # Display the log file content in a new GUI window
        self.show_log_window(log_file)

    def show_ops_options(self):
        ops_window = tk.Toplevel(self.master)
        ops_window.title("Select Ops File Option")

        tk.Label(ops_window, text="Choose how to obtain the .ops file:").pack(padx=10, pady=10)

        # Option a: Insert file path
        tk.Label(ops_window, text="Insert Ops File Path:").pack(padx=10, pady=5)
        ops_path_entry = tk.Entry(ops_window, width=50)
        ops_path_entry.pack(padx=10, pady=5)
        
        def set_ops_path():
            self.ops_path_var.set(ops_path_entry.get())
            ops_window.destroy()

        tk.Button(ops_window, text="Set Ops Path", command=set_ops_path).pack(pady=5)

        # Option b: Edit default ops
        tk.Button(ops_window, text="Edit Default Ops", command=self.default_ops_suite2p).pack(pady=5)

        # Option c: Run Suite2P GUI
        tk.Button(ops_window, text="Run Suite2P", command=self.run_suite2p).pack(pady=5)

    def default_ops_suite2p(self):
        # Placeholder for the default ops function
        messagebox.showinfo("Default Ops", "Running default_ops_suite2p... (implement this function)")

    def run_suite2p(self):
        # Placeholder for running the Suite2P GUI
        messagebox.showinfo("Suite2P GUI", "Running Suite2P GUI... (implement this function)")

###### Progress bar #####



    def process_files(self):
        if not self.skip_suite2p_var.get():
            file_count = self.count_files_with_ending()
            if file_count > 0:
                self.show_progress_bar(file_count)
            else:
                messagebox.showerror("Error", "No files found with the specified file ending.")
        else:
            messagebox.showinfo("Info", "Skipping Suite2p processing.")

    def count_files_with_ending(self):
        main_folder = self.main_folder_var.get().strip()
        file_ending = self.data_extension_var.get().strip()
        file_count = 0

        for root, dirs, files in os.walk(main_folder):
            for file in files:
                if file.endswith(file_ending):
                    file_count += 1

        return file_count

    def show_progress_bar(self, file_count):
        progress_window = tk.Toplevel(self.scrollable_frame)
        progress_window.title("Processing Files")

        tk.Label(progress_window, text="Processing files...").pack(pady=10)

        progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate')
        progress_bar.pack(pady=10)

        estimated_time = 55 * 60  # 55 minutes for 24 files
        time_per_file = estimated_time / 24
        total_time = time_per_file * file_count

        def update_progress():
            for i in range(file_count):
                time.sleep(time_per_file)
                progress_bar['value'] += (100 / file_count)
                progress_window.update_idletasks()

            progress_window.destroy()
            messagebox.showinfo("Info", "Processing completed.")

        threading.Thread(target=update_progress).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = ConfigEditor(root)
    root.mainloop()
