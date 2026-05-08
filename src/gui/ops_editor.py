import tkinter as tk
import tkinter as tk
from tkinter import ttk
from gui_core.analysis_model import AnalysisParams
from gui_core.multivid_reg_model import MultiVid_Reg_Params

class OpsEditor:
    def __init__(self, master, config):
        self.master = master
        self.config = config
        self.params = config.analysis_params

        self.vars = {}
        self.create_widgets()

    def create_widgets(self):
        for idx, (param, value) in enumerate(self.params.to_dict().items()):
            tk.Label(self.master, text=param).grid(row=idx, column=0)

            if isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                tk.Checkbutton(self.master, variable=var).grid(row=idx, column=1)

            elif param == "Img_Overlay":
                var = tk.StringVar(value=value)
                ttk.Combobox(
                    self.master,
                    textvariable=var,
                    values=["max_proj", "meanImg"],
                    state="readonly"
                ).grid(row=idx, column=1)
                        
            elif param == "baseline_correction":
                var = tk.StringVar(value=value)
                ttk.Combobox(
                    self.master,
                    textvariable=var,
                    values=["airPLS", "rolling_median"],
                    state="readonly"
                ).grid(row=idx, column=1)
                        

            else:
                var = tk.StringVar(value=str(value))
                tk.Entry(self.master, textvariable=var).grid(row=idx, column=1)

            self.vars[param] = var

        tk.Button(self.master, text="Save", command=self.save).grid(row=len(self.vars), column=0)

    def save(self):
        updated = {}

        for key, var in self.vars.items():
            val = var.get()

            if isinstance(var, tk.BooleanVar):
                updated[key] = val
            else:
                try:
                    updated[key] = float(val) if "." in val else int(val)
                except:
                    updated[key] = val

        self.config.analysis_params = AnalysisParams.from_dict(updated)
        self.master.destroy()


class MultiVidEditor:
    def __init__(self, master, config):
        self.master = master
        self.config = config
        self.params = config.multivid_params
        
        self.vars = {}
        self.length_frame = None
        self.length_vars = []

        self.create_widgets()

    
    def create_widgets(self):
        for idx, (param, value) in enumerate(self.params.to_dict().items()):
            if param in ['unequal_treatment_lengths','treatment_length_units']:
                continue
            tk.Label(self.master, text=param).grid(row=idx, column=0)
            
            if param == 'Treatment_No':
                var = tk.IntVar(value = value)

                tk.Spinbox(
                    self.master,
                    from_ = 1,
                    to = 20,
                    textvariable=var,
                    width = 5,
                ).grid(row = idx, column = 1)

                var.trace_add("write", self.update_treatment_length_inputs)

            elif isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                tk.Checkbutton(self.master, variable=var).grid(row=idx, column=1)

                if param == 'equal_baseline_and_treatments':
                    var.trace_add("write", self.update_treatment_length_inputs)

            else:
                var = tk.StringVar(value = str(value))
                tk.Entry(self.master, textvariable=var).grid(row = idx, column = 1)
            
            self.vars[param] = var
        
        self.length_frame = tk.Frame(self.master)
        self.length_frame.grid(row = 20, column = 0, columnspan=2, pady = 10)
        if not self.vars['equal_baseline_and_treatments'].get():
            self.update_treatment_length_inputs()
        
        tk.Button(self.master, text = "Save", command = self.save).grid(row = 30, column = 0)

    def update_treatment_length_inputs(self, *args):

        if self.vars['equal_baseline_and_treatments'].get():
            self.length_frame.grid_remove()
            return
        else:
            self.length_frame.grid()
            
        for widget in self.length_frame.winfo_children():
            widget.destroy()
        
        self.length_vars = []
        
        tk.Label(self.length_frame,
                 text = "Treatment length units"
                 ).grid(row=0, column=0)
        
        self.length_unit_var = tk.StringVar(
            value = self.params.treatment_length_units
        )
        
        ttk.Combobox(
            self.length_frame,
            textvariable=self.length_unit_var,
            values=["seconds", "frames"],
            state="readonly",
            width = 10
        ).grid(row=0, column=1)

        try:
            num_treatments = int(self.vars["Treatment_No"].get())
        except:
            return
        
        for i in range(1, num_treatments + 2):
            label = "Baseline length" if i == 1 else f"Treatment {i-1} length"
        
            tk.Label(self.length_frame, text = label).grid(row = i, column = 0)

            var = tk.StringVar()

            if i < len(self.params.unequal_treatment_lengths):
                var.set(str(self.params.unequal_treatment_lengths[i]))

            tk.Entry(
                self.length_frame,
                textvariable=var,
                width = 10
            ).grid(row = i, column = 1)

            self.length_vars.append(var)

    def save(self):
        updated = {}

        for key, var in self.vars.items():
            val = var.get()
            if isinstance(var, tk.BooleanVar):
                updated[key] = val
            else:
                try:
                    updated[key] = float(val) if "." in val else int(val)
                except:
                    updated[key] = val
        if not self.vars['equal_baseline_and_treatments'].get():
            for var in self.length_vars:
                updated['unequal_treatment_lengths'] = [float(var.get())]

            updated['treatment_length_units'] = self.length_unit_var.get()
        else:
            updated['unequal_treatment_lengths'] = []
        

        self.config.multivid_params = MultiVid_Reg_Params.from_dict(updated)
        self.master.destroy()