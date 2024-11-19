import numpy as np
main_folder = r'C:/LSD__Data_40x_synapse'
group1 = main_folder + r'\LSD_sample'
group2 = main_folder + r'\PBS_sample'

group_number = 2
data_extension = 'nd2'
frame_rate = 20
ops_path = r'C:\Users\jcbegs\python3\suite2p_ops\2024.04.24_20fps_func_not_conn_40x_wide_ops.npy'
ops = np.load(r'C:\Users\jcbegs\python3\suite2p_ops\2024.04.24_20fps_func_not_conn_40x_wide_ops.npy', allow_pickle = True).item()
ops['input_format'] = 'nd2'
ops['frame_rate'] = 20
BIN_WIDTH = 5
EXPERIMENT_DURATION = 180
FRAME_INTERVAL = 1 / frame_rate
FILTER_NEURONS = True
TimePoints = {
}
exp_groups = {
    'LSD_sample': 'treatment',
    'PBS_sample': 'baseline',
}
pairs = [ ('PBS', 'LSD') ]
parameters = {
    'testby': pairs,
    'x': 'Group',
    'feature': 'SpikesFreq',
    'type': 'violin',
    'plotby': 'Group',
    'y_label': 'SpikesFreq',
    'x_label': '',
    'stat_test': 'Mann-Whitney',
    'legend': 'auto',
    'location': 'outside',
    'palette': 'viridis',
}
## Additional configurations
groups = []
for n in range(group_number):
    group_name = f"group{n + 1}"
    groups.append(eval(group_name))
for name, value in exp_groups.items():
    # Add your logic to handle Groups22
    pass
