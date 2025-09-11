import os
import sys

import pandas as pd
import numpy as np
# from analyze_suite2p.config_loader import load_json_config_file
folder = r'G:\washing_acute_repeats\Ketamine_Mem_DMSO_low_conc'
experiment = folder.split('\\')[-1]
File_Name = str(experiment) + "_experiment_summary.csv"


def load_experiment_csv(experiment_folder):
    import os
    from analyze_suite2p import config_loader
    config = config_loader.load_json_config_file(os.path.join(experiment_folder, 'analysis_config.json'))
    file_path = os.path.join(config.general_settings.main_folder,f"{str(config.general_settings.main_folder.split('\\')[-1])}_experiment_summary.csv")
    data = pd.read_csv(file_path)
    # synapses = data[["Experimental_Group", 
    #                  "File_Name", 
    #                  'synapse_ROI', 
    #                  "dendrite_ROI",
    #                  "total_ROIs",
    #                  "SpikesFreq",
    #                  "AvgAmplitude",
    #                  "AvgDecayTime" ]].drop_duplicates()
    # syanpses = synapses.groupby("File_Name").agg({"Experimental_Group": "first",
    #                                           "synapse_ROI":["mean"],
    #                                           "dendrite_ROI": ["mean"],
    #                                           "total_ROIs": ["mean"],
    #                                           "SpikesFreq": ["mean"],
    #                                           "AvgAmplitude": ["mean"],
    #                                           "AvgDecayTime": ["mean"]})
    synapses = data[["Experimental_Group", 
                     "Replicate_No.",
                     "File_Name",
                     "SpikesFreq", 
                     'synapse_ROI', 
                     "dendrite_ROI",
                     "total_ROIs"]].dropna()
    syanpses = synapses.groupby("File_Name").agg({"Experimental_Group": "first",
                                                  "Replicate_No.": "first",
                                                  "synapse_ROI":["mean"],
                                                  "dendrite_ROI": ["mean"],
                                                  "total_ROIs": ["mean"],
                                                  "SpikesFreq": ["mean"]})
    groups = synapses["Experimental_Group"].unique().tolist()
    mapped_groups = synapses["Experimental_Group"]
    # Metrics and grouping
    metrics = [
        "synapse_ROI",
        "dendrite_ROI",
        "total_ROIs",
        "SpikesFreq",
    #     "AvgAmplitude",
    #     "AvgDecayTime"
    ]
    save_path = file_path.split("\\")[0:-1]
    save_path = '\\'.join(save_path)
    experiment = file_path.split("\\")[-1]
    experiment = experiment.split(".csv")[0]

    return groups, metrics, syanpses

def merge_cellprofiler_csvs(folder, summary_File_Name = f"{experiment}_experiment_summary.csv"):
    # img_data = pd.read_csv(os.path.join(folder, 'CellProfiler', 'GCaMP6f_SkeletonizationImageArea.csv'))
    skele_data = pd.read_csv(os.path.join(folder, 'CellProfiler', 'GCaMP6f_new_skeleImage.csv'))
    pix2micron = 3.0769
    merged_df = skele_data.sort_values(by=['FileName_Originals']).reset_index(drop=True)

    merged_df['FileName_Originals'] = merged_df['FileName_Originals'].apply(lambda x: x.split("_")[1:])
    merged_df['FileName_Originals'] = merged_df['FileName_Originals'].apply(lambda x: "_".join(x))
    merged_df['FileName_Originals'] = merged_df['FileName_Originals'].str[0:-15]
    # merged_df['FileName_Originals'] = merged_df['FileName_Originals'].str.split("_Projection")[0]
    # merged_df['Total Area'] = merged_df['AreaShape_BoundingBoxMaximum_Y'] * merged_df['AreaShape_BoundingBoxMaximum_X']
    # merged_df['Total Area um'] = (merged_df['AreaShape_BoundingBoxMaximum_Y']/pix2micron) * (merged_df['AreaShape_BoundingBoxMaximum_X']/pix2micron)

    merged_df['Normalized Skeleton Coverage'] = merged_df['AreaOccupied_AreaOccupied_Skeleton_no_enhance']
    merged_df["um Skeleton Coverage"] = merged_df['AreaOccupied_AreaOccupied_Skeleton_no_enhance'] / pix2micron
    merged_df['Normalized Neurite Area'] = merged_df['AreaOccupied_AreaOccupied_Thresholded_neurites_no_enhance']
    merged_df['um Neurite Area'] = merged_df['AreaOccupied_AreaOccupied_Thresholded_neurites_no_enhance'] / pix2micron

    merged_df = merged_df[['Normalized Neurite Area','um Neurite Area', "Normalized Skeleton Coverage",'um Skeleton Coverage', 'FileName_Originals']]
    sorted_skeletons = merged_df.sort_values(by="FileName_Originals").reset_index(drop=True)
    stats = pd.read_csv(os.path.join(folder, f'{experiment}_synapse_average.csv')).dropna()

    stats['FileName_Originals'] = stats['Unnamed: 0'].astype("str")

    stats['FileName_Originals'] = stats['FileName_Originals'].apply(lambda x: "".join(x.split("\\")[-1]))

    stats = stats.drop(columns = ['Unnamed: 0'])
    print(stats.columns)
    sorted_experiment_stats = stats.sort_values(by='FileName_Originals').reset_index(drop=True)

    print(sorted_experiment_stats.columns)
    print(sorted_skeletons.columns)
    full_df = sorted_skeletons.merge(sorted_experiment_stats, on='FileName_Originals', how = 'right')
    full_df

    print(sorted_skeletons["FileName_Originals"].head())
    print(sorted_experiment_stats["FileName_Originals"].head())

    # How many exact matches?
    overlap = set(sorted_skeletons["FileName_Originals"]) & set(sorted_experiment_stats["FileName_Originals"])
    print("Number of matching keys:", len(overlap))
    print("Example overlaps:", list(overlap)[:10])

    # Which keys are missing?
    print("Only in skeletons:", set(sorted_skeletons["FileName_Originals"]) - set(sorted_experiment_stats["FileName_Originals"]))
    print("Only in stats:", set(sorted_experiment_stats["FileName_Originals"]) - set(sorted_skeletons["FileName_Originals"]))

    full_df['Normalized Skeleton Coverage'] = full_df['Normalized Skeleton Coverage'].astype('float') 
    full_df['Normalized Neurite Area'] = full_df['Normalized Neurite Area'].astype('float') 
    full_df['synapse_ROI'] = full_df['synapse_ROI'].astype('float') 

    full_df['area_normalized_synapses'] = full_df['synapse_ROI'] / full_df['Normalized Neurite Area']
    full_df['skeleton_normalized_synapses'] = full_df['synapse_ROI'] / full_df["Normalized Skeleton Coverage"]
    full_df['normalized_synapses'] = full_df['synapse_ROI'] / (full_df['um Skeleton Coverage'] / 10)

    return full_df

def main():
    groups, metrics, synapses_csv = load_experiment_csv(folder)
    synapses_csv.to_csv(os.path.join(folder, f'{experiment}_synapse_average.csv'))
    df = merge_cellprofiler_csvs(folder)
    df.to_csv(os.path.join(folder, f'{experiment}_synapse_normalized_data.csv'))
    return df

if __name__ == "__main__":
    groups, metrics, synapses_csv = load_experiment_csv(folder)
    synapses_csv.to_csv(os.path.join(folder, f'{experiment}_synapse_average.csv'))
    df = merge_cellprofiler_csvs(folder)
    df.to_csv(os.path.join(folder, f'{experiment}_synapse_normalized_data.csv'))