import numpy as np
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from suite2p import default_ops

class OpsEditor:
    def __init__(self, master):
        self.master = master
        self.master.title("Edit Operations")

        # Load default operations
        self.ops = default_ops()
        
    # Define the parameters you want to allow editing, optionally have all be editable? CHANGE TO RELEVANT PARAMETERS
        self.editable_params = {
            '1Photon_Registration': self.ops.get('1Preg', False),

            "Functional ROI Detection": self.ops.get('sparse_mode', 1),
            'Threshold': self.ops.get('threshold_scaling', 1.0),
            'Denoise': self.ops.get('denoise', 1),

            "Anatomical ROI Detection": self.ops.get('anatomical_only', 1),
            'ROI Diameter': self.ops.get('diameter', 16),
            'CellPose_Threshold': self.ops.get('cellprob_threshold', 0.5),
            'Flow_Threshold': self.ops.get('flow_threshold', 0.25),
            
        }

        self.vars = {}

        # Create input fields for each parameter
        for param, value in self.editable_params.items():
            frame = tk.Frame(self.master)
            frame.pack(pady=5)
            tk.Label(frame, text=param).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(value))
            self.vars[param] = var
            tk.Entry(frame, textvariable=var, width=10).pack(side=tk.LEFT)

        # Display current save location
        self.save_location_label = tk.Label(self.master, text=f"Current save location: ../suite2p/ops/ops_1P.npy")
        self.save_location_label.pack(pady=5)

        # Save button
        self.save_button = tk.Button(self.master, text="Save", command=self.save_ops)
        self.save_button.pack(pady=10)

    def save_ops(self):
        # Get the modified values
        for param in self.vars:
            if param in self.ops:
                value = self.vars[param].get()
                if value.lower() in ['true', 'false']:
                    self.ops[param] = value.lower() == 'true'
                else:
                    self.ops[param] = float(value) if value.replace('.', '', 1).isdigit() else value

        # Print all parameters to terminal for debugging
        print("Current Operations Parameters:")
        for param, value in self.ops.items():
            print(f"{param}: {value}")

        # Open file dialog for saving the modified ops
        save_path = filedialog.asksaveasfilename(
            defaultextension=".npy",
            filetypes=[("NumPy files", "*.npy")],
            title="Choose save location"
        )

        if save_path:
            np.save(save_path, self.ops)
            messagebox.showinfo("Success", f"Saved operations to: {save_path}")
            self.master.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = OpsEditor(root)
    root.mainloop()
