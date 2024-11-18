import os
import sys

import pandas as pd
import numpy as np
folder = r'C:\Users\jcbeg\Desktop\004 - Cell Profiler TroubleShooting\sults'

def merge_cellprofiler_csvs(folder, summary_file_name):
    img_data = pd.read_csv(folder + r'\GCaMP6f_SkeletonizationImageArea.csv')
    skele_data = pd.read_csv(folder + r'\GCaMP6f_SkeletonizationImage.csv')

    merged_df = skele_data.merge(img_data, on="FileName_Originals")

    merged_df['FileName_Originals'] = merged_df['FileName_Originals'].str[4:-4]
    merged_df['Total Area'] = merged_df['AreaShape_BoundingBoxMaximum_Y'] * merged_df['AreaShape_BoundingBoxMaximum_X']
    merged_df['Normalized Skeleton Coverage'] = merged_df['AreaOccupied_AreaOccupied_Neurites_Skeleton']
    merged_df['Normalized Neurite Area'] = merged_df['AreaOccupied_AreaOccupied_Thresholded_Neurites']
    merged_df = merged_df[['Normalized Neurite Area', "Normalized Skeleton Coverage", 'FileName_Originals']]
    sorted_skeletons = merged_df.sort_values(by="FileName_Originals").reset_index(drop=True)
    # print(sorted_skeletons)
    
    stats = pd.read_csv(folder + '\\' + summary_file_name)
    stats['FileName_Originals'] = stats['file_name'].apply(lambda x: "".join(x.split("Dur")[0]))
    stats = stats.drop(columns = 'Unnamed: 0')
    sorted_experiment_stats = stats.sort_values(by='file_name').reset_index(drop=True)
    # print(sorted_experiment_stats)
    full_df = sorted_experiment_stats.merge(sorted_skeletons)
    #Troubleshooting
    # for i in range(len(full_df)):
    #     idx1 = full_df['file_name'].iloc[i]
    #     idx2 = full_df['FileName_Originals'].iloc[i]
    #     if idx1 != idx2:
    #         print(idx1)
    full_df['area_normalized_synapses'] = full_df['synapse_count'] / full_df['Normalized Neurite Area']
    full_df['skeleton_normalized_synapses'] = full_df['synapse_count'] / full_df["Normalized Skeleton Coverage"]

    return full_df

if __name__ == "__main__":
    df = merge_cellprofiler_csvs(folder)
    print(df)