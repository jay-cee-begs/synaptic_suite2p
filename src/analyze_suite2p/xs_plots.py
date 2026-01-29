from analyze_suite2p import config_loader, suite2p_utility, analysis_utility, plotting_utility, detector_utility
import os
import pandas as pd
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ["Arial"]
def prep_syn_over_time_df(experiment_folder):
    
    try:
        experiment = experiment_folder.split('\\')[-1]
    except '.' in experiment:
        print(f"Experiment is {experiment} and does not fit data structure") 

    csv_file = experiment + "_synapse_average.csv"
    df = pd.read_csv(os.path.join(experiment_folder, csv_file)).dropna()
    
    df['img_time'] = df['Unnamed: 0'].astype(str).apply(lambda x: x.split("00")[-1])
    try:
        df['img_time'] = df['img_time'].astype(int).apply(lambda x: x*3)
    except ValueError as e:
        print(e)
        df['img_time'] = df['Unnamed: 0'].astype(str).apply(lambda x: x.split("_r0")[-1])
        df['img_time'] = df['img_time'].astype(int).apply(lambda x: x*3)
    print(df['img_time'].unique())
    df["File_Name"] = df['Unnamed: 0'].astype(str)
    df['exp_date'] = df["File_Name"].apply(lambda x: x.split('\\')[-1])
    df['exp_date'] = df['exp_date'].apply(lambda x: x.split("_")[0])
    df = df.drop('Unnamed: 0', axis = 1)
    df = df.sort_values(by = ["Experimental_Group", "File_Name", "img_time"])

    print(df["Experimental_Group"].unique())

    return df


df = prep_syn_over_time_df(r'E:\washing_acute_repeats\z-score')

df = df.sort_values(by = ["Experimental_Group", "File_Name", "img_time"])


base = df[df['Experimental_Group'] == 'z-base']
PDBu = df[df['Experimental_Group'] == 'PDBu']
APV = df[df['Experimental_Group'] == 'APV']

PDBu_merge = PDBu.merge(base, on =['img_time','exp_date'], suffixes= ('_treat', '_base'))

PDBu_merge['active_proportion'] = PDBu_merge['dendrite_ROI_treat'].astype('float') / PDBu_merge['dendrite_ROI_base'].astype('float')

APV = df[df['Experimental_Group'] == 'APV']
Ket_base = df[df['Experimental_Group'] == 'Ket_base']

APV_merge = APV.merge(base, on =['img_time','exp_date'], suffixes= ('_treat', '_base'))

APV_merge['active_proportion'] = APV_merge['dendrite_ROI_treat'].astype('float') / APV_merge['dendrite_ROI_base'].astype('float')
import seaborn as sns
import numpy as np
colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', 
                '#0072B2', '#D55E00', '#CC79A7', '#999999', '#000000', '#999933']

concat_df = [PDBu_merge, APV_merge, Ket_merged, mem_merged, gly_merged, nbqx_merged]
concat_df = [ APV_merge, Ket_merged, mem_merged]

all_exp = pd.concat(concat_df)
print(all_exp['active_proportion'].describe())
print(all_exp.head())
# all_exp['active_proportion'] = 1- abs(all_exp['active_proportion'])
plt.figure(figsize=(10,8))
sns.lineplot(all_exp, x = 'img_time', y = 'active_proportion', hue = "Experimental_Group_treat",alpha = 1, errorbar="se")
plt.xlabel("Time Since Application [min]")
plt.ylabel("Synapses relative to baseline")

plt.ylim(0, 2)
plt.axhline(1, linestyle = '--', color = 'black')
plt.legend()
# plt.tight_layout()
# plt.gca().invert_xaxis()  # Invert the x-axis
from pathlib import Path
# plt.savefig(os.path.join(r"C:\Users\jcbegs\2025-synapse_paper", 'synapses_over_time.svg'))
plt.show()
