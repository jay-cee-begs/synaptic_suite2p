import numpy as np 
main_folder = r'C:\LSD__Data_40x_synapse'
group1 = main_folder + r'\LSD_sample'
group2 = main_folder + r'\PBS_sample'
group_number = 2
data_extension = 'nd2'
frame_rate = 20
ops_path = r'C:\Users\jcbegs\python3\suite2p_ops\2024.04.24_20fps_func_not_conn_40x_wide_ops.npy'
ops = np.load(ops_path, allow_pickle=True).item()
ops['frame_rate'] = frame_rate
ops['input_format'] = data_extension
BIN_WIDTH = 5
EXPERIMENT_DURATION = 180
FRAME_INTERVAL = 1 / frame_rate
FILTER_NEURONS = True
exp_condition = {
    'LSD_sample': 'LSD_sample',
    'PBS_sample': 'PBS_sample',
}
## Additional configurations
groups = []
for n in range(group_number):
    group_name = f"group{n + 1}"
    if group_name in locals():
        groups.append(locals()[group_name])
