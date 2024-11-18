main_folder = r'C:/Users/Justus/data/test_data_binned'
group1 = main_folder + r'\ACSF_baseline'
group2 = main_folder + r'\ACSF_GBZ'
group3 = main_folder + r'\NaOH_baseline'
group4 = main_folder + r'\NaOH_GBZ'
group5 = main_folder + r'\NBA_baseline'
group6 = main_folder + r'\NBA_GBZ'
group_number = 6

data_extension = 'nd2'
ops_path = r'D:\users\JC\suite2p_ops_n_classifiers\2024.04.24_20fps_func_not_conn_40x_wide_ops.npy'

TimePoints = {
}
Experimental_Groups = {
    'ACSF_baseline': '',
    'ACSF_GBZ': '',
    'NaOH_baseline': '',
    'NaOH_GBZ': '',
    'NBA_baseline': '',
    'NBA_GBZ': '',
}
pairs = [ ('Baseline', 'GBZ'), ('Baseline', 'TTX') ]
parameters = {
    'testby': pairs,
    'x': 'Group',
    'feature': 'Total_Estimated_Spikes',
    'type': 'violin',
    'plotby': 'Time_Point',
    'y_label': 'y label',
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
