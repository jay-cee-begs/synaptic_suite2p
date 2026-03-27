import tkinter as tk
import tkinter as tk
from tkinter import ttk
from gui_core.analysis_model import AnalysisParams

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