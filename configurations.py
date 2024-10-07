### CONFIGURATIONS ###

main_folder = r'C:\240918_synapse_img'
# group1 = r'C:\temp_processing\synapse_pipeline'
# group_number = 1
# groups = []
# for n in range(group_number):
#     group_name = f"group{n+1}"
#     if group_name in locals():
#         groups.append(locals()[group_name])

import numpy as np
data_extension = "nd2"
frame_rate = 20
EXPERIMENT_DURATION = 180 #needs to be changed
#total number of seconds that the experiment lasts (tends to be flexible so we will have to figure out how to set this)
FRAME_INTERVAL = 0.05
#frame_interval is calculated as 1 / frame_rate
#we are not doing binned experiments so this is not necessary; again, will comment out later, but for now too scared of bugs
BIN_WIDTH = 10
#FILTER_NEURONS applies strictly to 'iscell.npy' file; in most instances, we will use all ROIs anyway, but keep true for clarity
FILTER_NEURONS = True

ops_path = r'C:\Users\jcbegs\python3\suite2p_ops\2024.04.24_20fps_func_not_conn_40x_wide_ops.npy'
ops = np.load(ops_path, allow_pickle = True).item()
ops["input_format"] = data_extension
ops["threshold_scaling"] = 1.0
ops["fs"] = frame_rate



BASE_DIR = main_folder
#left over nomenclature from converting ND2 to tif
