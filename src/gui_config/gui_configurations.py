main_folder = r'F:\JC_calcium_imaging\001_synapse_ca_imaging\glycine\240207_pbs_glycine'
group1 = main_folder + r'F:\JC_calcium_imaging\001_synapse_ca_imaging\glycine\240207_pbs_glycine\Gly'
group2 = main_folder + r'F:\JC_calcium_imaging\001_synapse_ca_imaging\glycine\240207_pbs_glycine\PBS'
group_number = 2
data_extension = 'nd2'
frame_rate = 20
ops_path = r'C:\Users\jcbegs\python3\suite2p_ops\2024.04.24_20fps_func_not_conn_40x_wide_ops.npy'
TimePoints = {
}
Groups22 = {
}
pairs = [ ('PBS', 'Glycine') ]
parameters = {
    'testby': pairs,
    'x': 'Group',
    'feature': 'SpikesFreq',
    'type': 'violin',
    'plotby': 'Time_Point',
    'y_label': 'SpikesFreq',
    'x_label': '',
    'stat_test': 'Mann-Whitney',
    'legend': 'auto',
    'location': 'outside',
    'palette': 'viridis',
}
## Additional configurations
nb_neurons = 16
model_name = "Global_EXC_10Hz_smoothing200ms"
EXPERIMENT_DURATION = 60
FRAME_INTERVAL = 1 / frame_rate
BIN_WIDTH = 20
FILTER_NEURONS = True
groups = []
for n in range(group_number):
    group_name = f"group{n + 1}"
    groups.append(eval(group_name))
for name, value in Groups22.items():
    # Add your logic to handle Groups22
    pass
