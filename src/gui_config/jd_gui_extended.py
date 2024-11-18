import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from pathlib import Path
import os
import subprocess

class TwoColumnFrame(tk.Frame):
    def __init__(self,master):
        super().__init__(master)

        #first column
        self.left_frame = tk.Frame(self)
        self.left_frame.pack(side="left", fill = "both", expand=True, padx=10)

        self.right_frame = tk.Frame(self)
        self.right_frame.pack(side = "right", fill = "both", expand = True, padx=10)



class ConfigEditor:
    def __init__(self, master):
        self.master = master
        self.master.title("Synaptic Suite2P Pipeline Configuration Editor")
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

        self.main_folder_var = tk.StringVar(value=self.config.get('main_folder', ''))
        self.data_extension_var = tk.StringVar(value=self.config.get('data_extension', ''))
        self.frame_rate_var = tk.IntVar(value=self.config.get('frame_rate', 10))
        self.ops_path_var = tk.StringVar(value=self.config.get('ops_path', ''))
        self.groups = self.config.get('groups', [])
        self.groups22 = {key: value for key, value in self.config.get('Experimental Conditions', {}).items()}

        # Main folder input
        tk.Label(self.scrollable_frame, text="Experiment Folder Path:").pack(anchor='w', padx=10, pady=5)
        tk.Entry(self.scrollable_frame, textvariable=self.main_folder_var, width=50).pack(padx=10)
        # Button to open file explorer for selecting a folder
        tk.Button(self.scrollable_frame, text="Browse", command=self.browse_folder).pack(padx=10, pady=5)
        
        # Data extension input
        tk.Label(self.scrollable_frame, text="Data Extension:").pack(anchor='w', padx=10, pady=5)
        tk.Entry(self.scrollable_frame, textvariable=self.data_extension_var).pack(padx=10)
        #intermediate save button, to save the configurations before adding the groups
        tk.Button(self.scrollable_frame, text="Save Configurations (optional, in case you changed the data extension)", command=self.save_config).pack(pady=10)
        
        # Group input
        self.group_frame = tk.Frame(self.scrollable_frame)
        self.group_frame.pack(padx=10, pady=5)
        tk.Label(self.group_frame, text="Adds all subfolders from the Experiment:").pack(side=tk.LEFT)
        tk.Button(self.group_frame, text="Add Group", command=self.add_group).pack(side=tk.LEFT)


       
        # Ops path input
        tk.Label(self.scrollable_frame, text="Ops Path Options:").pack(anchor='w', padx=10, pady=5)
        tk.Entry(self.scrollable_frame, textvariable=self.ops_path_var, width=50).pack(padx=10)
        
        # Option a: Insert file path
        ops_frame = tk.Frame(self.scrollable_frame)
        ops_frame.pack(padx=10, pady=5)
        tk.Button(ops_frame, text="Browse", command=self.browse_ops_file).pack(side=tk.LEFT)

        # Option b: Edit default ops
        tk.Button(self.scrollable_frame, text="Edit Default Ops", command=self.edit_default_ops).pack(pady=5)

        # Option c: Create new ops file
        tk.Button(self.scrollable_frame, text="Create New Ops File", command=self.create_new_ops_file).pack(pady=5)
        tk.Label(self.scrollable_frame, text="Press any key in terminal when GUI is stuck").pack(anchor='w', padx=10, pady=5)
        # Frame rate input
        tk.Label(self.scrollable_frame, text="Frame Rate:").pack(anchor='w', padx=10, pady=5)
        tk.Entry(self.scrollable_frame, textvariable=self.frame_rate_var).pack(padx=10)

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
        tk.Button(self.scrollable_frame, text="Add TimePoint", command=self.add_timepoint).pack(padx=10)

        # Editable Groups22
        tk.Label(self.scrollable_frame, text="Same goes for your Groups:").pack(anchor='w')
        tk.Label(self.scrollable_frame, text="(In case your structure looks like 'TimePoint_Condition' you can remove 'TimePoint_' )").pack(anchor='w')
        self.groups22_frame = tk.Frame(self.scrollable_frame)
        self.groups22_frame.pack(padx=10, pady=5)
        self.create_dict_entries(self.groups22_frame, "Groups22", self.groups22)

        # Editable parameters
        self.parameters_frame = tk.Frame(self.scrollable_frame)
        self.parameters_frame.pack(padx=10, pady=5)
        self.create_parameters_entries()

        # Editable pairs
        tk.Label(self.scrollable_frame, text="Pairs for the stat test (input as (Group1, GroupA), (Group2, GroupB), etc:").pack(anchor='w', padx=10, pady=5)
        self.pairs_var = tk.StringVar(value=", ".join([f"{pair}" for pair in self.config.get('pairs', [])]))
        tk.Entry(self.scrollable_frame, textvariable=self.pairs_var, width=50).pack(padx=10)


        # Save button
        tk.Button(self.scrollable_frame, text="Save Configurations", command=self.save_config).pack(pady=10)

        # Skip Suite2P option
        self.skip_suite2p_var = tk.BooleanVar()
        tk.Checkbutton(self.scrollable_frame, text="Skip Suite2P", variable=self.skip_suite2p_var).pack(anchor='w', padx=10, pady=5)
        self.skip_iscell_var = tk.BooleanVar()
        tk.Checkbutton(self.scrollable_frame, text = "Use iscell.npy", variable=self.skip_iscell_var).pack(anchor='w', padx = 10, pady = 5)
        # Processing button
        tk.Button(self.scrollable_frame, text="Process", command=self.proceed).pack(pady=10)

        # Initialize empty TimePoints dictionary
        self.timepoints = {}

################ Functions AREA ################    put in seperate file eventually
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(-1 * (event.delta // 120), "units")
    def edit_default_ops(self):
        """Call the function to edit default ops"""
        base_path = os.path.dirname(os.path.abspath(__file__))
        batch_file_path = os.path.join(base_path, "Scripts", "run_default_ops.bat")
        subprocess.call([batch_file_path])  # Execute run_ops.bat

    def create_new_ops_file(self):
        """Call the function to create new ops file"""
        base_path = os.path.dirname(os.path.abspath(__file__))
        batch_file_path = os.path.join(base_path, "Scripts", "run_s2p_gui.bat")
        subprocess.call([batch_file_path]) # Execute run_s2p_gui.bat

    def browse_ops_file(self):
        file_selected = filedialog.askopenfilename(filetypes=[("NumPy Files", "*.npy")])
        if file_selected:
            self.ops_path_var.set(file_selected)

    def browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.main_folder_var.set(folder_selected)

    def load_config(self, filepath):
        config = {}
        try:
            with open(filepath) as f:
                exec(f.read(), config)
        except FileNotFoundError:
            messagebox.showerror("Error", "Configuration file not found. Starting with default settings.")
            return {}
        return config


    def add_group(self):
        main_folder = self.main_folder_var.get().strip()
        if not os.path.exists(main_folder):
            messagebox.showerror("Error", "Main folder does not exist.")
            return
        def check_for_single_image_file_in_folder(current_path, file_ending):
            """
            Check if the specified path contains exactly one file with the given extension.
            """
            files = [file for file in os.listdir(current_path) if file.endswith(file_ending)]
            return len(files) == 1
        
        all_folders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]
        excluded_substrings = []
        unique_folders = [folder for folder in all_folders if not any(excluded in folder for excluded in excluded_substrings)]

        file_ending = self.data_extension_var.get().strip()  # Get the specified file extension

        valid_folders = []  # To hold valid folders

        for folder_name in unique_folders:
            current_folder_path = os.path.join(main_folder, folder_name)
            
            # Check if any subfolder has exactly one file with the specified extension
            subfolders = [f for f in os.listdir(current_folder_path) if os.path.isdir(os.path.join(current_folder_path, f))]
            for subfolder in subfolders:
                subfolder_path = os.path.join(current_folder_path, subfolder)
                if check_for_single_image_file_in_folder(subfolder_path, file_ending):
                    valid_folders.append(folder_name)
                    break  # No need to check other subfolders if one matches

        for folder_name in valid_folders:
            group_path = f"\\{folder_name}" if not folder_name.startswith("\\") else folder_name
            
            if folder_name not in self.groups22:
                self.groups22[folder_name] = ''
            
            if group_path not in self.groups:
                self.groups.append(group_path)

        self.update_groups22_entries()

        if valid_folders:
            messagebox.showinfo("Groups Added", f"Added Groups: {', '.join(valid_folders)}")
        else:
            messagebox.showinfo("No Groups Added", "No folders with a single file matching the specified extension were found.")


    def add_timepoint(self):
        """call this function to change a/each timepoint name"""
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


    def update_groups22_entries(self):
        """Update the entries in the Groups22 dictionary with the use of create_dict_entries"""
        for widget in self.groups22_frame.winfo_children():
            widget.destroy()  # Remove old entries
        self.create_dict_entries(self.groups22_frame, "Groups22", self.groups22)


    def create_parameters_entries(self):
        """Create entries for the parameters dictionary, contains lists for the various dropdown options"""
        self.parameters_vars = {}
        # List of selectable values for 'stat_test'
        stat_test_options = [
            "t-test", "Mann-Whitney", "Wilcoxon", "Kruskal", "Brunner-Munzel"]
        
        # List of selectable values for 'type'
        type_options = [
            "strip", "swarm", "box", "violin", 
            "boxen", "point", "bar", "count"]
        
        # List of selectable values for 'legend'
        legend_options = ["auto", "inside", "false"]

        for key, value in self.config.get('parameters', {}).items():
            frame = tk.Frame(self.parameters_frame)
            frame.pack(pady=5)
            tk.Label(frame, text=key).pack(side=tk.LEFT)
            
            var = tk.StringVar(value=value)
            self.parameters_vars[key] = var
            
            if key == 'stat_test':
                dropdown = tk.OptionMenu(frame, var, *stat_test_options)
                dropdown.pack(side=tk.LEFT)
            elif key == 'type':
                dropdown = tk.OptionMenu(frame, var, *type_options)
                dropdown.pack(side=tk.LEFT)
            elif key == 'legend':
                dropdown = tk.OptionMenu(frame, var, *legend_options)
                dropdown.pack(side=tk.LEFT)
            elif key == 'testby':                
                continue  # Skip 'testby' as it is a list
            else:
                tk.Entry(frame, textvariable=var, width=20).pack(side=tk.LEFT)


    def save_config(self):
        main_folder = self.main_folder_var.get().strip()
        data_extension = self.data_extension_var.get().strip()
        frame_rate = self.frame_rate_var.get()
        ops_path = self.ops_path_var.get().strip()

        if not os.path.exists(main_folder):
            messagebox.showerror("Error", "Main folder does not exist.")
            return

        groups22 = {key_var.get(): value_var.get() for key_var, (key_var, value_var) in self.dict_vars.items()} ### ????????????? is this still needed?? 

        pairs_input = self.pairs_var.get().strip()

        with open('gui_configurations.py', 'w') as f:
            f.write(f"main_folder = r'{main_folder}'\n")
            for i, group in enumerate(self.groups, start=1):
                f.write(f"group{i} = main_folder + r'{group}'\n")
            f.write(f"group_number = {len(self.groups)}\n")
            f.write(f"data_extension = '{data_extension}'\n")
            f.write(f"frame_rate = {frame_rate}\n")
            f.write(f"ops_path = r'{ops_path}'\n")

            f.write("TimePoints = {\n")
            for key, value in self.timepoints.items():
                f.write(f"    '{key}': '{value}',\n")
            f.write("}\n")

            f.write("Groups22 = {\n")
            for key, (key_var, value_var) in self.dict_vars.items():
                f.write(f"    '{key_var.get()}': '{value_var.get()}',\n")
            f.write("}\n")

            f.write(f"pairs = [ {pairs_input} ]\n")

            f.write("parameters = {\n")
            f.write(f"    'testby': pairs,\n") # Add 'testby' to the parameters, assigns the pairs value to it, this is not user-editable
            for key, var in self.parameters_vars.items():
                if key != 'testby':  # Exclude 'testby' from user input
                    f.write(f"    '{key}': '{var.get()}',\n")
            f.write("}\n")
            #### Add addtionals here, maybe make them editable in the gui as well
            f.write("## Additional configurations\n")
            f.write("nb_neurons = 16\n")
            f.write('model_name = "Global_EXC_10Hz_smoothing200ms"\n')
            f.write("EXPERIMENT_DURATION = 60\n")
            f.write("FRAME_INTERVAL = 1 / frame_rate\n")
            f.write("BIN_WIDTH = 20\n")
            f.write("FILTER_NEURONS = True\n")
            f.write("groups = []\n")
            f.write("for n in range(group_number):\n")
            f.write("    group_name = f\"group{n + 1}\"\n")
            f.write("    groups.append(eval(group_name))\n")
            f.write("for name, value in Groups22.items():\n")
            f.write("    # Add your logic to handle Groups22\n")
            f.write("    pass\n")

        messagebox.showinfo("Success", "Configurations saved successfully.")

    def proceed(self):  #Option to skip suite2p, will execute a different .bat then 
        base_path = os.path.dirname(os.path.abspath(__file__))
        if self.skip_suite2p_var.get():
            batch_file_path = os.path.join(base_path, "..", "gui_config", "Scripts", "run_plots.bat")
            subprocess.call([batch_file_path])  # Execute run_plots.bat
        else:
            batch_file_path = os.path.join(base_path, "..", "analyze_suite2p", "Scripts", "analyze_suite2p.bat")
            subprocess.call([batch_file_path])  # Execute sequence.bat
        if self.skip_iscell_var.get():
            batch_file_path = os.path.join(base_path,  "..", "gui_config", "Scripts", "jd_default_gui.bat")
        else:
            print("successfully made it through processing)")
            # batch_file_path = os.path.join(base_path,  "..", "gui_config", "Scripts", "run_s2p_gui.bat")


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
