import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import os
import numpy as np

class ConfigEditor:
    def __init__(self, master):
        # Load existing configurations
        self.master = master
        self.master.title("GUI Configurations Editor")
        self.master.geometry("700x900")  # Set initial window size

        # Create a canvas and a scrollbar
        self.canvas = tk.Canvas(master)
        self.scrollbar = tk.Scrollbar(master, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        # Configure the scrollbar
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

        # Load existing configurations
        self.config = self.load_config("gui_configurations.py")

        self.main_folder_var = tk.StringVar(value=self.config.get('main_folder', ''))
        self.data_extension_var = tk.StringVar(value=self.config.get('data_extension', ''))
        self.frame_rate_var = tk.IntVar(value=self.config.get('frame_rate', 0))
        self.ops_path_var = tk.StringVar(value=self.config.get('ops_path', ''))
        self.groups = self.config.get('groups', [])
        self.groups22 = {key: value for key, value in self.config.get('Groups22', {}).items()}

        # Main folder input
        tk.Label(self.scrollable_frame, text="Experiment / Main Folder Path:").pack(anchor='w', padx=10, pady=5)
        tk.Entry(self.scrollable_frame, textvariable=self.main_folder_var, width=50).pack(padx=10)

        # Button to open file explorer for selecting a folder
        tk.Button(self.scrollable_frame, text="Browse", command=self.browse_folder).pack(padx=10, pady=5)

        # Group input
        self.group_frame = tk.Frame(self.scrollable_frame)
        self.group_frame.pack(padx=10, pady=5)
        tk.Label(self.group_frame, text="Adds all subfolders from the Experiment:").pack(side=tk.LEFT)
        #self.group_entry = tk.Entry(self.group_frame, width=50)
        #self.group_entry.pack(side=tk.LEFT)
        tk.Button(self.group_frame, text="Add Group", command=self.add_group).pack(side=tk.LEFT)

        # Data extension input
        tk.Label(self.scrollable_frame, text="Data Extension:").pack(anchor='w', padx=10, pady=5)
        tk.Entry(self.scrollable_frame, textvariable=self.data_extension_var).pack(padx=10)

        # Frame rate input
        tk.Label(self.scrollable_frame, text="Frame Rate:").pack(anchor='w', padx=10, pady=5)
        tk.Entry(self.scrollable_frame, textvariable=self.frame_rate_var).pack(padx=10)

        # Ops path input
        tk.Label(self.scrollable_frame, text="Select Ops Path:").pack(anchor='w', padx=10, pady=5)
        tk.Entry(self.scrollable_frame, textvariable=self.ops_path_var, width=50).pack(padx=10)

        # TimePoints input
        self.timepoint_frame = tk.Frame(self.scrollable_frame)
        self.timepoint_frame.pack(padx=10, pady=5)
        tk.Label(self.scrollable_frame, text="In case you need to rename your Baseconditions:").pack(anchor='w')
        self.timepoint_key_var = tk.StringVar()
        self.timepoint_value_var = tk.StringVar()
        tk.Entry(self.timepoint_frame, textvariable=self.timepoint_key_var, width=20).pack(side=tk.LEFT)
        tk.Entry(self.timepoint_frame, textvariable=self.timepoint_value_var, width=20).pack(side=tk.LEFT)
        tk.Button(self.scrollable_frame, text="Add TimePoint", command=self.add_timepoint).pack(padx=10)

        # Editable Groups22
        self.groups22_frame = tk.Frame(self.scrollable_frame)
        self.groups22_frame.pack(padx=10, pady=5)
        self.create_dict_entries(self.scrollable_frame, "Groups22", self.groups22)

        # Editable pairs
        tk.Label(self.scrollable_frame, text="Pairs for the stat test (input as: 'key1:value1, key2:value2'):").pack(anchor='w', padx=10, pady=5)
        self.pairs_var = tk.StringVar(value=", ".join([f"{pair}" for pair in self.config.get('pairs', [])]))
        tk.Entry(self.scrollable_frame, textvariable=self.pairs_var, width=50).pack(padx=10)

        # Editable parameters
        self.parameters_frame = tk.Frame(self.scrollable_frame)
        self.parameters_frame.pack(padx=10, pady=5)
        self.create_parameters_entries()

        # Save button
        tk.Button(self.scrollable_frame, text="Save Configurations", command=self.save_config).pack(pady=10)

        # Initialize empty TimePoints dictionary
        self.timepoints = {}

    def browse_folder(self):
        """Open a file dialog to select a folder and set the main folder path."""
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.main_folder_var.set(folder_selected)

    def load_config(self, filepath):
        """Load configurations from a file."""
        config = {}
        try:
            with open(filepath) as f:
                exec(f.read(), config)
        except FileNotFoundError:
            messagebox.showerror("Error", "Configuration file not found. Starting with default settings.")
            return {}
        return config

    def add_group(self):
        """Add all subfolders in the main folder, excluding certain names."""
        main_folder = self.main_folder_var.get().strip()  # Get the main folder path from the input

        if not os.path.exists(main_folder):
            messagebox.showerror("Error", "Main folder does not exist.")
            return

        # List all directories in the main folder
        all_folders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]

        # Filter out folders with specific substrings
        excluded_substrings = ['csv_files', 'pkl_files', 'csv_files_deltaF']
        unique_folders = [folder for folder in all_folders if not any(excluded in folder for excluded in excluded_substrings)]

        for folder_name in unique_folders:
            group_path = f"\\{folder_name}" if not folder_name.startswith("\\") else folder_name
            
            # Add to Groups22 if not already present
            if folder_name not in self.groups22:
                self.groups22[folder_name] = ''
            
            # Prevent duplicates in groups
            if group_path not in self.groups:
                self.groups.append(group_path)

        self.update_groups22_entries()
        messagebox.showinfo("Groups Added", f"Added Groups: {', '.join(unique_folders)}")


    def add_timepoint(self):
        key = self.timepoint_key_var.get().strip()
        value = self.timepoint_value_var.get().strip()
        if key and value:
            self.timepoints[key] = value
            self.timepoint_key_var.set('')  # Clear input
            self.timepoint_value_var.set('')  # Clear input
            messagebox.showinfo("TimePoint Added", f"Added TimePoint: {key} -> {value}")
        else:
            messagebox.showwarning("Input Error", "Please enter both key and value for TimePoint.")

    def create_dict_entries(self, master, title, dictionary):
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

    def update_groups22_entries(self):
        for widget in self.groups22_frame.winfo_children():
            widget.destroy()  # Remove old entries
        self.create_dict_entries(self.groups22_frame, "Groups22", self.groups22)

    def create_parameters_entries(self):
        self.parameters_vars = {}
        
        # List of selectable values for 'stat_test'
        stat_test_options = [
            "t-test_ind", "t-test_welch", "t-test_paired", 
            "Mann-Whitney", "Mann-Whitney-gt", "Mann-Whitney-ls", 
            "Levene", "Wilcoxon", "Kruskal", "Brunner-Munzel"
        ]
        
        # List of selectable values for 'type'
        type_options = [
            "strip", "swarm", "box", "violin", 
            "boxen", "point", "bar", "count"
        ]
        
        # List of selectable values for 'legend'
        legend_options = ["auto", "inside", "false"]
        
        for key, value in self.config.get('parameters', {}).items():
            frame = tk.Frame(self.parameters_frame)
            frame.pack(pady=5)
            tk.Label(frame, text=key).pack(side=tk.LEFT)
            
            var = tk.StringVar(value=value)
            self.parameters_vars[key] = var
            
            if key == 'stat_test':
                # Create a dropdown menu for 'stat_test'
                dropdown = tk.OptionMenu(frame, var, *stat_test_options)
                dropdown.pack(side=tk.LEFT)
            elif key == 'type':
                # Create a dropdown menu for 'type'
                dropdown = tk.OptionMenu(frame, var, *type_options)
                dropdown.pack(side=tk.LEFT)
            elif key == 'legend':
                # Create a dropdown menu for 'legend'
                dropdown = tk.OptionMenu(frame, var, *legend_options)
                dropdown.pack(side=tk.LEFT)
            elif key == 'pairs':
                tk.Entry(frame, textvariable=var, width=20, state='readonly').pack(side=tk.LEFT)  # Make pairs non-editable
            else:
                tk.Entry(frame, textvariable=var, width=20).pack(side=tk.LEFT)

    def save_config(self):
        main_folder = self.main_folder_var.get().strip()
        data_extension = self.data_extension_var.get().strip()
        frame_rate = self.frame_rate_var.get()
        ops_path = self.ops_path_var.get().strip()

        # Ensure the main folder exists
        if not os.path.exists(main_folder):
            messagebox.showerror("Error", "Main folder does not exist.")
            return

        # Prepare TimePoints and Groups22
        groups22 = {key_var.get(): value_var.get() for key_var, (key_var, value_var) in self.dict_vars.items()}

        # Get pairs from user input
        pairs_input = self.pairs_var.get().strip()

        # Write configurations back to the file
        with open('gui_configurations.py', 'w') as f:
            f.write(f"main_folder = r'{main_folder}'\n")
            for i, group in enumerate(self.groups, start=1):
                f.write(f"group{i} = main_folder + r'{group}'\n")
            f.write(f"group_number = {len(self.groups)}\n")
            f.write(f"data_extension = '{data_extension}'\n")
            f.write(f"frame_rate = {frame_rate}\n")
            f.write(f"ops_path = r'{ops_path}'\n")
            
            # Write TimePoints
            f.write("TimePoints = {\n")
            for key, value in self.timepoints.items():
                f.write(f"    '{key}': '{value}',\n")
            f.write("}\n")

            # Write Groups22
            f.write("Groups22 = {\n")
            for key, (key_var, value_var) in self.dict_vars.items():
                f.write(f"    '{key_var.get()}': '{value_var.get()}',\n")
            f.write("}\n")

            # Write pairs as a single string
            f.write(f"pairs = [ {pairs_input} ]\n")

            # Write parameters, including testby with pairs
            f.write("parameters = {\n")
            f.write(f"    'testby': pairs,\n")  # Set 'testby' to the string 'pairs'
            for key, var in self.parameters_vars.items():
                if key != 'testby':  # Exclude 'testby' from user input
                    f.write(f"    '{key}': '{var.get()}',\n")
            f.write("}\n")

            # Append the new block of code
            f.write("## plot a set of nb_neurons randomly chosen neuronal traces (first seconds)\n")
            f.write("nb_neurons = 16 ## maybe put directly into cascade_this???\n\n")
            f.write('model_name = "Global_EXC_10Hz_smoothing200ms"\n')
            f.write('## select fitting model from list (created in cascada code) ##\n')
            f.write('## list still in CASCADE code, maybe add here##\n\n')
            f.write("EXPERIMENT_DURATION = 60\n")
            f.write("FRAME_INTERVAL = 1 / frame_rate\n")
            f.write("BIN_WIDTH = 20  # SET TO APPROX 200ms\n")
            f.write("FILTER_NEURONS = True\n\n")
            f.write("groups = []\n")
            f.write("for n in range(group_number):\n")
            f.write("    group_name = f\"group{n+1}\"\n")
            f.write("    if group_name in locals():\n")
            f.write("        groups.append(locals()[group_name])\n\n")

            messagebox.showinfo("Success", "Configurations saved successfully.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ConfigEditor(root)
    root.mainloop()
