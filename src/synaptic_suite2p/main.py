import tkinter as tk
from tkinter import messagebox, filedialog
import os
import threading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def launch_analyze_suite2p():
    print("Launching suite2p analysis")
    batch_path = os.path.join(SCRIPT_DIR, "..", "analyze_suite2p", "Scripts", "analyze_suite2p.bat")
    batch_file_path = os.path.normpath(batch_path)
        # os.system(batch_path)
def launch_suite2p_gui():
    print("Launching suite2p GUI...")
    batch_path = os.path.join(SCRIPT_DIR, "..", "gui_config", "Scripts", "run_s2p_gui.bat")
    batch_file_path = os.path.normpath(batch_path)
    
    # os.system("src\gui_config\Scripts\run_s2p_gui.bat")

def launch_basic_user_settings():
    print("Launching simple settings...")
    batch_path = os.path.join(SCRIPT_DIR, "..", "gui_config", "Scripts", "run_default_ops.bat")
    batch_file_path = os.path.normpath(batch_path)
    
    # os.system("src\gui_config\Scripts\run_default_ops.bat")

def launch_main_user_GUI():
    print("Launching Pipeline User interface...")
    batch_path = os.path.join(SCRIPT_DIR, "..", "gui_config", "Scripts", "run_s2p_gui.bat")
    batch_file_path = os.path.normpath(batch_path)
    
    # os.system("src\gui_config\jd_gui_extended.py")

def main():
    root = tk.Tk()
    root.title("Synaptic Suite2p")
    root.geometry("250x300") 

    tk.Button(root, text = "Run suite2p pipeline", command = launch_main_user_GUI).pack()
    tk.Button(root, text = "Open suite2p GUI", command = launch_suite2p_gui).pack()
    tk.Button(root, text = 'Exit', command = root.quit).pack()
    root.mainloop()

if __name__ == "__main__":
    main()