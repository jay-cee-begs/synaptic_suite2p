import numpy as np
main_folder = r'C:/LSD__Data_40x_synapse'
group1 = main_folder + r'C:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapse\LSD_sample'
group2 = main_folder + r'C:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapse\PBS_sample'
group3 = main_folder + r'C:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapse\LSD_sample'
group4 = main_folder + r'C:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapse\PBS_sample'
group5 = main_folder + r'C:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapse\LSD_sample'
group6 = main_folder + r'C:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapse\PBS_sample'
group7 = main_folder + r'C:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapse\LSD_sample'
group8 = main_folder + r'C:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapse\PBS_sample'
group9 = main_folder + r'C:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapse\LSD_sample'
group10 = main_folder + r'C:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapse\PBS_sample'
group11 = main_folder + r'C:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapse\LSD_sample'
group12 = main_folder + r'C:/LSD__Data_40x_synapseC:/LSD__Data_40x_synapse\PBS_sample'
group13 = main_folder + r'C:/LSD__Data_40x_synapse\LSD_sample'
group14 = main_folder + r'C:/LSD__Data_40x_synapse\PBS_sample'
group15 = main_folder + r'\LSD_sample'
group16 = main_folder + r'\PBS_sample'
group_number = 16
data_extension = 'nd2'
frame_rate = 20
ops_path = r'C:\Users\jcbegs\python3\suite2p_ops\2024.04.24_20fps_func_not_conn_40x_wide_ops.npy'
ops = np.load(r'C:\Users\jcbegs\python3\suite2p_ops\2024.04.24_20fps_func_not_conn_40x_wide_ops.npy', allow_pickle = True).item()
ops['input_format'] = 'nd2'
ops['frame_rate'] = 20
BIN_WIDTH = r'0'
EXPERIMENT_DURATION = r'180'
FRAME_INTERVAL = 1 / frame_rate
FILTER_NEURONS = True
TimePoints = {
}
Groups22 = {
    'LSD_sample': 'LSD',
    'PBS_sample': 'PBS',
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
    # Add your logic to handle exp_groups
    pass
