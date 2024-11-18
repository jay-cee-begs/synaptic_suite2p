import numpy as np 

main_folder = r'D:\users\JC\pipeline\008-JC Synapse Imaging\240207 NMDA pt AB 50k glass DIV20\pbs40X CONTROL'
group1 = main_folder + r'\Gly'
group2 = main_folder + r'\PBS'

group_number = 2

data_extension = 'nd2'
frame_rate = 20
ops_path = r'D:\users\JC\suite2p_ops_n_classifiers\2024.04.24_20fps_func_not_conn_40x_wide_ops.npy'
ops = np.load(ops_path, allow_pickle=True).item()
ops['input_format'] = data_extension
ops['fs'] = frame_rate
TimePoints = {
}
Experimental_Groups = {
    'Glycine': '',
    'PBS': '',

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
frame_rate = 20
EXPERIMENT_DURATION = 180
FRAME_INTERVAL = 1 / frame_rate
BIN_WIDTH = 5
FILTER_NEURONS = True
groups = []
for n in range(group_number):
    group_name = f"group{n + 1}"
    groups.append(eval(group_name))
for name, value in Experimental_Groups.items():
    # Add your logic to handle Experimental_Groups
    pass
