import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from pathlib import Path
import os
import subprocess

class ConfigEditor:
    def __init__(self, master):
        self.master = master
        self.master.title("Ultimate Suite2P + Cascade Configuration Editor")
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
        self.config = self.load_config("gui_configurations.py")
        
        # if 'parameters' not in self.config:
        #     self.config['parameters'] = {
        #         'feature': ['Active_Neuron_Proportion'],  # Default features
        #         'stat_test': 't-test',                   # Default stat_test
        #         'type': 'box',                           # Default plot type
        #         'legend': 'auto',                        # Default legend option
        #         # Add other default parameters as needed
        #     }

        self.main_folder_var = tk.StringVar(value=self.config.get('main_folder', ''))
        self.data_extension_var = tk.StringVar(value=self.config.get('data_extension', ''))
        self.frame_rate_var = tk.IntVar(value=self.config.get('frame_rate', 20))
        self.ops_path_var = tk.StringVar(value=self.config.get('ops_path', ''))
        self.csc_path_var = tk.StringVar(value=self.config.get('cascade_file_path', ''))
        self.groups = self.config.get('groups', [])
        self.exp_condition = {key: value for key, value in self.config.get('exp_condition', {}).items()}
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

        #cascade path input
        self.csc_frame = tk.Frame(self.scrollable_frame)
        self.csc_frame.pack(padx=10, pady=5)
        tk.Label(self.csc_frame, text="Only Change when Cascade installation changed:").pack(side=tk.LEFT)
        tk.Entry(self.csc_frame, textvariable=self.csc_path_var, width=40).pack(side=tk.LEFT)
       
        # Ops path input
        tk.Label(self.scrollable_frame, text="Ops Path Options:").pack(anchor='w', padx=10, pady=5)
        #tk.Entry(self.scrollable_frame, textvariable=self.ops_path_var, width=50).pack(padx=10)
        
        # Option a: Insert file path
        ops_frame = tk.Frame(self.scrollable_frame)
        ops_frame.pack(padx=10, pady=5)
        tk.Entry(ops_frame, textvariable=self.ops_path_var, width=40).pack(side=tk.LEFT)
        tk.Button(ops_frame, text="Browse", command=self.browse_ops_file).pack(side=tk.LEFT)

        # Option b: Edit default ops
        tk.Button(self.scrollable_frame, text="Edit Default Ops", command=self.edit_default_ops).pack(pady=5)

        # Option c: Create new ops file
        tk.Button(self.scrollable_frame, text="Create New Ops File (WIP)", command=self.create_new_ops_file).pack(pady=5)
        tk.Label(self.scrollable_frame, text="Press any key in terminal when GUI is stuck").pack(anchor='w', padx=10, pady=5)
        # Frame rate input
        tk.Label(self.scrollable_frame, text="Frame Rate:").pack(anchor='w', padx=10, pady=5)
        tk.Entry(self.scrollable_frame, textvariable=self.frame_rate_var).pack(padx=10)

        # EXPERIMENT_DURATION input
        tk.Label(self.scrollable_frame, text = "Experiment Duration (seconds):").pack(anchor='w', padx=10, pady=5)
        tk.Entry(self.scrollable_frame, textvariable=self.exp_dur_var).pack(padx=10)

        # BIN_WIDTH input
        tk.Label(self.scrollable_frame, text = "Network Bin Width (# frames):").pack(anchor='w', padx=10, pady=5)
        tk.Entry(self.scrollable_frame, textvariable=self.bin_width_var).pack(padx=10)



        # TimePoints input
        tk.Label(self.scrollable_frame, text="In case you need to rename your Baseconditions:").pack(anchor='w')
        tk.Label(self.scrollable_frame, text="Left: Insert the name you assigned your timepoint in the recording").pack(anchor='w')
        tk.Label(self.scrollable_frame, text="Right: your desired name").pack(anchor='w')
        self.timepoint_frame = tk.Frame(self.scrollable_frame)
        self.timepoint_frame.pack(padx=10, pady=5)
        self.timepoint_key_var = tk.StringVar()
        self.timepoint_value_var = tk.StringVar()
        tk.Entry(self.timepoint_frame, textvariable=self.timepoint_key_var, width=20).pack(side=tk.LEFT)
        tk.Entry(self.timepoint_frame, textvariable=self.timepoint_value_var, width=20).pack(side=tk.LEFT)
        tk.Label(self.scrollable_frame, text="Press 'Add TimePoint' for each").pack(anchor='w')

        # Editable exp_condition
        tk.Label(self.scrollable_frame, text="Same goes for your Groups, dont leave the brackets empty:").pack(anchor='w')
        tk.Label(self.scrollable_frame, text="(In case your structure looks like 'TimePoint_Condition' you can remove 'TimePoint_' )").pack(anchor='w')
        self.exp_condition_frame = tk.Frame(self.scrollable_frame)
        self.exp_condition_frame.pack(padx=10, pady=5)
        self.create_dict_entries(self.exp_condition_frame, " ", self.exp_condition)

        # Editable parameters
        self.parameters_frame = tk.Frame(self.scrollable_frame)
        self.parameters_frame.pack(padx=10, pady=5)
        # self.create_parameters_entries()

        # Editable pairs
        tk.Label(self.scrollable_frame, text="Pairs for the stat test (input as (Group1, GroupA), (Group2, GroupB), etc:").pack(anchor='w', padx=10, pady=5)
        self.pairs_var = tk.StringVar(value=", ".join([f"{pair}" for pair in self.config.get('pairs', [])]))
        tk.Entry(self.scrollable_frame, textvariable=self.pairs_var, width=50).pack(padx=10)


        # Save button
        tk.Button(self.scrollable_frame, text="Save Configurations", command=self.save_config).pack(pady=10)

        # Skip Suite2P option
        self.skip_suite2p_var = tk.BooleanVar()
        tk.Checkbutton(self.scrollable_frame, text="Skip Suite2P, no need to run it twice :)", variable=self.skip_suite2p_var).pack(anchor='w', padx=10, pady=5)

        # Use native suite2p ROI classifications
        self.skip_iscell_var = tk.BooleanVar()
        tk.Checkbutton(self.scrollable_frame, text = "Use iscell.npy", variable=self.skip_iscell_var).pack(anchor='w', padx = 10, pady = 5)

        # Processing button
        tk.Button(self.scrollable_frame, text="Process", command=self.proceed).pack(pady=10)

        # # Initialize empty TimePoints dictionary
        self.timepoints = {}


################ Functions AREA ################    put in seperate file eventually
               
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(-1 * (event.delta // 120), "units")   

    def edit_default_ops(self):
        """Call the function to edit default ops"""
        current_dir = Path(__file__).parent
        scripts_dir = current_dir / "Scripts"
        bat_file = scripts_dir / "run_default_ops.bat"
        subprocess.call([str(bat_file)])  # Execute run_default_ops.bat

    def create_new_ops_file(self):
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


    def load_config(self, filepath):
        config = {}
        try:
            # Get the directory of the current script
            script_dir = os.getcwd()
            # Construct the absolute path to the configuration file
            abs_filepath = os.path.join(script_dir, filepath)
            
            with open(abs_filepath) as f:
                exec(f.read(), config)
        except FileNotFoundError:
            messagebox.showerror("Error", "Configuration file not found. Starting with default settings.")
            return {}
        return config

    def add_group(self):
        self.groups.clear()
        main_folder = self.main_folder_var.get().strip()
        if not os.path.exists(main_folder):
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
        
    #     # After adding groups, update timepoints using unique prefixes
    #     unique_prefixes = self.get_unique_prefixes(prefix_length=3)
    #     for prefix in unique_prefixes:
    #         if prefix not in self.timepoints:
    #             self.timepoints[prefix] = prefix  # Set the key-value as the prefix itself

    #     # Update timepoint entries UI
    #     self.update_timepoint_entries()

    #     if valid_folders:
    #         messagebox.showinfo("Groups Added", f"Added Groups: {', '.join(valid_folders)}")
    #     else:
    #         messagebox.showinfo("No Groups Added", "No (sub-)folders with one or more files matching the specified extension were found.")
            
            
    # def update_timepoint_entries(self):
    #     """Update the entries in the timepoints dictionary with the current keys and values."""
    #     # Clear previous timepoint entries
    #     for widget in self.timepoint_frame.winfo_children():
    #         widget.destroy()

    #     # Create new timepoint entries for each key-value pair in the timepoints dictionary
    #     for key, value in self.timepoints.items():
    #         frame = tk.Frame(self.timepoint_frame)
    #         frame.pack(padx=10, pady=5)
    #         tk.Label(frame, text="Key:").pack(side=tk.LEFT)
    #         key_var = tk.StringVar(value=key)
    #         value_var = tk.StringVar(value=value)
    #         self.timepoints[key] = value_var  # Store the variable to the dictionary
    #         tk.Entry(frame, textvariable=key_var, width=20).pack(side=tk.LEFT)
    #         tk.Label(frame, text="Value:").pack(side=tk.LEFT)
    #         tk.Entry(frame, textvariable=value_var, width=20).pack(side=tk.LEFT)               
            

    def create_dict_entries(self, master, title, dictionary):
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
        """Update the entries in the exp_condition dictionary with the use of create_dict_entries"""
        for widget in self.exp_condition_frame.winfo_children():
            widget.destroy()  # Remove old entries
        self.create_dict_entries(self.exp_condition_frame, "exp_condition", self.exp_condition)

    # def create_parameters_entries(self):
    #     """Create entries for the parameters dictionary, contains lists for the various dropdown options"""
    #     self.parameters_vars = {}
    #     # List of selectable values for 'stat_test'
    #     stat_test_options = [
    #         "t-test_ind", "t-test_welch",   "t-test_paired", "Mann-Whitney", "Mann-Whitney-gt", "Mann-Whitney-ls", "Wilcoxon", "Kruskal", "Brunner-Munzel"]
        
    #     # List of selectable values for 'type'
    #     type_options = [
    #         "strip", "swarm", "box", "violin", 
    #         "boxen", "point", "bar", "count"]
        
    #     # List of selectable values for 'legend'
    #     legend_options = ["auto", "inside", "false"]

    #     # Feature selection from CSV file
    #     self.load_features_from_csv()
        
    #     # Ensure 'feature' key exists in parameters
    #     parameters = self.config.get('parameters', {})
    #     if 'feature' not in parameters:
    #         parameters['feature'] = ['Active_Neuron_Proportion']  # Default feature


    #     for key, value in self.config.get('parameters', {}).items():
    #         frame = tk.Frame(self.parameters_frame)
    #         frame.pack(pady=5)
    #         tk.Label(frame, text=key).pack(side=tk.LEFT)
            
    #         var = tk.StringVar(value=value)
    #         self.parameters_vars[key] = var
            
    #         if key == 'stat_test':
    #             dropdown = tk.OptionMenu(frame, var, *stat_test_options)
    #             dropdown.pack(side=tk.LEFT)
    #         elif key == 'type':
    #             dropdown = tk.OptionMenu(frame, var, *type_options)
    #             dropdown.pack(side=tk.LEFT)
    #         elif key == 'legend':
    #             dropdown = tk.OptionMenu(frame, var, *legend_options)
    #             dropdown.pack(side=tk.LEFT)
    #         elif key == 'feature':
    #             # Use Listbox for multiple feature selection
    #             feature_listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, height=6, width=40)
    #             for feature in self.features_list:
    #                 feature_listbox.insert(tk.END, feature)
    #             feature_listbox.pack(side=tk.LEFT)
    #             # Store reference to feature_listbox as an instance variable
    #             self.feature_listbox = feature_listbox

    #         elif key == 'testby':                
    #             continue  # Skip 'testby' as it is a list
    #         else:
    #             tk.Entry(frame, textvariable=var, width=20).pack(side=tk.LEFT)

    # def load_features_from_csv(self):
    #     """Load column names from the 'new_experiment_summary.csv' for feature selection"""
    #     main_folder = self.main_folder_var.get().strip()
    #     if not os.path.exists(main_folder):
    #         messagebox.showerror("Error", "Main folder does not exist.")
    #         self.features_list = ['Active_Neuron_Proportion']
    #         return

    #     csv_file_path = os.path.join(main_folder, 'new_experiment_summary.csv')
    #     if not os.path.exists(csv_file_path):
    #         messagebox.showerror("Error", f"File {csv_file_path} not found.")
    #         self.features_list = ['Active_Neuron_Proportion']
    #         return
        
    #     # Read the CSV file to get the columns
    #     try:
    #         import pandas as pd
    #         df = pd.read_csv(csv_file_path)
    #         # Exclude certain columns from the features list
    #         excluded_columns = ['Prediction_File', 'Group', 'Time_Point']  # Add the columns you want to exclude
    #         self.features_list = [col for col in df.columns if col not in excluded_columns]
                  
    #     except Exception as e:
    #         messagebox.showerror("Error", f"Failed to read the CSV file: {str(e)}")
    #         self.features_list = ['Active_Neuron_Proportion']
    #         return

    # ####copied from functions_data_transformation.py to get the TimePoints names before csv creation
    # def get_unique_prefixes(self, prefix_length=3):
    #     """Get unique prefixes from the group names, corresponding to the prefixes chosen by cascade"""
    #     prefixes = set()
    #     for name in self.groups:
    #                 # Normalize the path by using os.path.normpath, which handles both slashes
    #         normalized_path = os.path.normpath(name)
            
    #         # Remove drive letter and leading directories by splitting and taking the last part
    #         last_part = normalized_path.split(os.sep)[-1]  # Get the last part of the path
            
    #         # Now, get the prefix based on the desired length (e.g., first 3 characters)
    #         prefix = last_part[:prefix_length]
    #         prefixes.add(prefix)
    #     return prefixes


        
        

    def csc_path(self):
        """Call this function to get or set the path to the cascade file"""
        csc_path = self.csc_path_var.get().strip()

        # Check if the path is already in the configuration
        if 'cascade_file_path' in self.config:
            csc_path = self.config['cascade_file_path']
            self.csc_path_var.set(csc_path)
        else:
            self.config['cascade_file_path'] = csc_path
        
        return csc_path
    
    # def reload_features_listbox(self):
    #     """Reload the feature listbox to reflect updated feature list from config."""
    #     # First, clear the listbox
    #     self.feature_listbox.delete(0, tk.END)
        
    #     # Add the updated features
    #     for feature in self.features_list:
    #         self.feature_listbox.insert(tk.END, feature)

    def reload_config(self):
        """Reload the configuration file to refresh the GUI."""
        self.config = self.load_config("gui_configurations.py")  # Reload the configuration file
        # Update the GUI variables with the new values from the config
        self.main_folder_var.set(self.config.get('main_folder', ''))
        self.data_extension_var.set(self.config.get('data_extension', ''))
        self.frame_rate_var.set(self.config.get('frame_rate', 20))
        self.exp_dur_var.set(self.config.get('EXPERIMENT_DURATION', 180))
        self.ops_path_var.set(self.config.get('ops_path', ''))
        self.csc_path_var.set(self.config.get('cascade_file_path', ''))
        self.groups = self.config.get('groups', [])
        self.exp_condition = {key: value for key, value in self.config.get('exp_condition', {}).items()}
        self.timepoints = self.config.get('TimePoints', {})
        
        # Update the GUI components to reflect the new values
        self.update_exp_condition_entries()
        # self.create_parameters_entries()
        self.reload_features_listbox()
        # Optionally, you can also refresh other specific widgets or labels here.
        messagebox.showinfo("Config Reloaded", "Configuration file has been reloaded successfully.")
    
    def save_config(self):
        main_folder = self.main_folder_var.get().strip()
        data_extension = self.data_extension_var.get().strip()
        frame_rate = self.frame_rate_var.get()
        ops_path = self.ops_path_var.get().strip()
        csc_path = self.csc_path_var.get().strip()
        BIN_WIDTH = self.bin_width_var.get()
        EXPERIMENT_DURATION = self.exp_dur_var.get()

        if not os.path.exists(main_folder):
            messagebox.showerror("Error", "Main folder does not exist.")
            return

        exp_condition = {key_var.get(): value_var.get() for key_var, (key_var, value_var) in self.dict_vars.items()} ### ????????????? is this still needed?? 

        pairs_input = self.pairs_var.get().strip()

        # Get selected features as a string in brackets
        # selected_features = [self.feature_listbox.get(i) for i in self.feature_listbox.curselection()]

        # selected_features = ", ".join([f"'{feature}'" for feature in selected_features])
        
        # Construct the absolute path to the configuration file, saving uses the same logic as loading now
        script_dir = os.path.dirname(__file__)
        config_filepath = os.path.join(script_dir, 'gui_configurations.py')

        # if selected_features:
        #     selected_features = f"[{selected_features}]"
        # else:
        #     selected_features = "['Active_Neuron_Proportion', 'Total_Estimated_Spikes_proportion_scaled']"
        # #clearing the parameters dictionary before adding the new values
        # self.config['parameters']['feature'] = selected_features
        with open(config_filepath, 'w') as f:
            f.write('import numpy as np \n')
            f.write(f"main_folder = r'{main_folder}'\n")
            for i, group in enumerate(self.groups, start=1):
                f.write(f"group{i} = main_folder + r'{group}'\n")
            f.write(f"group_number = {len(self.groups)}\n")
            f.write(f"data_extension = '{data_extension}'\n")
            f.write(f"frame_rate = {frame_rate}\n")
            f.write(f"cascade_file_path = r'{csc_path}'\n")
            f.write(f"ops_path = r'{ops_path}'\n")
            f.write("ops = np.load(ops_path, allow_pickle=True).item()\n")
            f.write("ops['frame_rate'] = frame_rate\n")
            f.write("ops['input_format'] = data_extension\n")
            f.write(f"BIN_WIDTH = {BIN_WIDTH}\n")
            f.write(f"EXPERIMENT_DURATION = {EXPERIMENT_DURATION}\n")
            f.write("FRAME_INTERVAL = 1 / frame_rate\n")
            f.write("FILTER_NEURONS = True\n")


            f.write("exp_condition = {\n")
            for key, (key_var, value_var) in self.dict_vars.items():
                f.write(f"    '{key_var.get()}': '{value_var.get()}',\n")
            f.write("}\n")

            f.write("## Additional configurations\n")
            f.write("nb_neurons = 16\n")
            f.write('model_name = "Global_EXC_10Hz_smoothing200ms"\n')
            f.write("FILTER_NEURONS = True\n")
            f.write("groups = []\n")
            f.write("for n in range(group_number):\n")
            f.write("    group_name = f\"group{n + 1}\"\n")
            f.write("    if group_name in locals():\n")
            f.write("        groups.append(locals()[group_name])\n")
        messagebox.showinfo("Success", "Configurations saved successfully.")

        #reload the gui
        # self.reload_config()

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

    def proceed(self):  #Option to skip suite2p, will execute a different .bat then
        current_dir = Path(__file__).parent
        scripts_dir = current_dir / "Scripts" 
        if self.skip_suite2p_var.get():
            bat_file = scripts_dir / "run_plots.bat"
        else:
            bat_file = scripts_dir / "run_sequence.bat"
            
        print(f"Executing {bat_file}")
        subprocess.call([str(bat_file)])  # Execute sequence.bat
        # Redirect the terminal output to a text file
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

if __name__ == "__main__":
    root = tk.Tk()
    app = ConfigEditor(root)
    root.mainloop()
