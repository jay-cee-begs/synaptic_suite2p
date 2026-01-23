import os
import sys

import pandas as pd
import numpy as np
# from analyze_suite2p.config_loader import load_json_config_file



def load_experiment_csv(experiment_folder):
    import os
    from analyze_suite2p import config_loader
    config = config_loader.load_json_config_file(os.path.join(experiment_folder, 'analysis_config.json'))
    csv_file = str(config.general_settings.main_folder.split('\\')[-1]) + "_experiment_summary.csv"
    file_path = os.path.join(config.general_settings.main_folder,csv_file)
    data = pd.read_csv(file_path)
    # synapses = data[["Experimental_Group", 
    #                  "File_Name", 
    #                  'synapse_ROI', 
    #                  "dendrite_ROI",
    #                  "total_ROIs",
    #                  "SpikesFreq",
    #                  "AvgAmplitude",
    #                  "AvgDecayTime" ]].drop_duplicates()

    synapses = data[["Experimental_Group", 
                     "Replicate_No.",
                     "File_Name",
                     "SpikesFreq",
                      
                     'synapse_ROI', 
                     "dendrite_ROI",
                     "total_ROIs"]].dropna()
    groups = synapses["Experimental_Group"].unique().tolist()

    synapses = synapses.groupby("File_Name").agg({"Experimental_Group": "first",
                                                  "Replicate_No.": "first",
                                                  "synapse_ROI":["mean"],
                                                  "dendrite_ROI": ["mean"],
                                                  "total_ROIs": ["mean"],
                                                  "SpikesFreq": ["mean"]})
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

    return groups, metrics, synapses

def merge_cellprofiler_csvs_without_fuzzy_match(folder):
    # img_data = pd.read_csv(os.path.join(folder, 'CellProfiler', 'GCaMP6f_SkeletonizationImageArea.csv'))
    skele_data = pd.read_csv(os.path.join(folder, 'CellProfiler', 'GCaMP6f_new_skeleImage.csv'))
    pix2micron = 3.0769
    merged_df = skele_data.sort_values(by=['FileName_Originals']).reset_index(drop=True)

    merged_df['FileName_Originals'] = merged_df['FileName_Originals'].apply(lambda x: x.split("_")[1:])
    merged_df['FileName_Originals'] = merged_df["FileName_Originals"].apply(lambda x: "_".join(x))
    merged_df['FileName_Originals'] = merged_df['FileName_Originals'].apply(lambda x: x.split("_Projection.tif")[0])
    merged_df['Normalized Skeleton Coverage'] = merged_df['AreaOccupied_AreaOccupied_Skeleton_no_enhance']
    merged_df["um Skeleton Coverage"] = merged_df['AreaOccupied_AreaOccupied_Skeleton_no_enhance'] / pix2micron
    merged_df['Normalized Neurite Area'] = merged_df['AreaOccupied_AreaOccupied_Thresholded_neurites_no_enhance']
    merged_df['um Neurite Area'] = merged_df['AreaOccupied_AreaOccupied_Thresholded_neurites_no_enhance'] / pix2micron

    merged_df = merged_df[['Normalized Neurite Area','um Neurite Area', "Normalized Skeleton Coverage",'um Skeleton Coverage', 'FileName_Originals']]
    sorted_skeletons = merged_df.sort_values(by="FileName_Originals").reset_index(drop=True)
    experiment = folder.split('\\')[-1]
    stats = pd.read_csv(os.path.join(folder, f'{experiment}_synapse_average.csv')).dropna()

    stats['FileName_Originals'] = stats['Unnamed: 0'].astype("str")

    stats['FileName_Originals'] = stats['FileName_Originals'].apply(lambda x: "".join(x.split("\\")[-1]))
    try:
        stats = stats.drop(columns = ['Unnamed: 0', 'SpikesDiff'])
    except KeyError as e:
        stats = stats.drop(columns = 'Unnamed: 0')
    print(stats.columns)
    # sorted_experiment_stats = stats.sort_values(by=['Experimental_Group','FileName_Originals']).reset_index(drop=True)
    avg_experiment_stats = stats.groupby(['Experimental_Group', 'Replicate_No.', 'FileName_Originals']).agg('mean')
    sorted_avg_experiment_stats = avg_experiment_stats.sort_values(by = ['Experimental_Group', "FileName_Originals"]).reset_index()
    print(sorted_avg_experiment_stats.columns)
    print(sorted_skeletons.columns)
    full_df = sorted_skeletons.merge(sorted_avg_experiment_stats, on='FileName_Originals', how = 'right')
    full_df

    print(sorted_skeletons["FileName_Originals"].head())
    print(sorted_avg_experiment_stats["FileName_Originals"].head())

    # How many exact matches?
    overlap = set(sorted_skeletons["FileName_Originals"]) & set(sorted_avg_experiment_stats["FileName_Originals"])
    print("Number of matching keys:", len(overlap))
    print("Example overlaps:", list(overlap)[:10])

    # Which keys are missing?
    print("Only in skeletons:", set(sorted_skeletons["FileName_Originals"]) - set(sorted_avg_experiment_stats["FileName_Originals"]))
    print("Only in stats:", set(sorted_avg_experiment_stats["FileName_Originals"]) - set(sorted_skeletons["FileName_Originals"]))
    skele_only = set(sorted_skeletons["FileName_Originals"]) - set(sorted_avg_experiment_stats["FileName_Originals"])
    stat_only = set(sorted_avg_experiment_stats["FileName_Originals"]) - set(sorted_skeletons["FileName_Originals"])
    full_df['Normalized Skeleton Coverage'] = full_df['Normalized Skeleton Coverage'].astype('float') 
    full_df['Normalized Neurite Area'] = full_df['Normalized Neurite Area'].astype('float') 
    full_df['synapse_ROI'] = full_df['synapse_ROI'].astype('float') 

    full_df['area_normalized_synapses'] = full_df['synapse_ROI'] / full_df['Normalized Neurite Area']
    full_df['skeleton_normalized_synapses'] = full_df['synapse_ROI'] / full_df["Normalized Skeleton Coverage"]
    full_df['normalized_synapses'] = full_df['synapse_ROI'] / (full_df['um Skeleton Coverage'] / 10)

    return full_df, skele_only, stat_only

def normalize_synapse_to_skeletons_safe_match(experiment_folder, fuzzy_threshold):
    from fuzzywuzzy import process
    global config
    global config_dict
    from analyze_suite2p import config_loader
    config = config_loader.load_json_config_file(os.path.join(experiment_folder, 'analysis_config.json'))

    pix2micron = 3.0769
    experiment = str(experiment_folder).split('\\')[-1]
    try:
        exp_df = pd.read_csv(os.path.join(config.general_settings.main_folder, f'{experiment}_synapse_average.csv'))
        exp_df = exp_df.dropna().reset_index()
        exp_df["FileName_Originals"] = exp_df['Unnamed: 0']
        exp_df = exp_df.drop('Unnamed: 0', axis = 1)
    except FileNotFoundError as e:
        print("synapse summary file does not exists...calculating now")
        groups, metrics, synapses = load_experiment_csv(experiment_folder)
        synapses.to_csv(os.path.join(experiment_folder, f'{experiment}_synapse_average.csv'))

    exp_df = pd.read_csv(os.path.join(experiment_folder, f'{experiment}_synapse_average.csv'))
    exp_df = exp_df.dropna().reset_index()
    exp_df["FileName_Originals"] = exp_df['Unnamed: 0']
    exp_df = exp_df.drop('Unnamed: 0', axis = 1)
    exp_df["FileName_Originals"] = exp_df["FileName_Originals"].apply(lambda x: x.split('\\')[-1])

    sorted_experiment_stats = exp_df.sort_values(by=['FileName_Originals', "Experimental_Group"]).reset_index(drop=True)


    CellProfilerSkeletons = pd.read_csv(os.path.join(experiment_folder, 'CellProfiler', 'GCaMP6f_new_skeleImage.csv'))
    CellProfilerSkeletons['Experimental_Group'] =CellProfilerSkeletons['FileName_Originals'].apply(lambda x: x.split('_2')[0])

    CellProfilerSkeletons['FileName_Originals'] = CellProfilerSkeletons['FileName_Originals'].apply(lambda x: x.split('_2')[1:])
    CellProfilerSkeletons['FileName_Originals'] =CellProfilerSkeletons['FileName_Originals'].apply(lambda x: "2" + "_2".join(x))
    CellProfilerSkeletons['FileName_Originals'] = CellProfilerSkeletons['FileName_Originals'].apply(lambda x: x.split("_Projection")[0])
    CellProfilerSkeletons['Normalized Skeleton Coverage'] = CellProfilerSkeletons['AreaOccupied_AreaOccupied_Skeleton_no_enhance']
    CellProfilerSkeletons["um Skeleton Coverage"] = CellProfilerSkeletons['AreaOccupied_AreaOccupied_Skeleton_no_enhance'] / pix2micron
    CellProfilerSkeletons['Normalized Neurite Area'] = CellProfilerSkeletons['AreaOccupied_AreaOccupied_Thresholded_neurites_no_enhance']
    CellProfilerSkeletons['um Neurite Area'] = CellProfilerSkeletons['AreaOccupied_AreaOccupied_Thresholded_neurites_no_enhance'] / pix2micron
    #Add Experimental_Group column for sorting
    print(CellProfilerSkeletons.Experimental_Group.unique())
    sorted_CellProfilerSkeletons = CellProfilerSkeletons.sort_values(by=['FileName_Originals', 'Experimental_Group']).reset_index(drop=True)

    sorted_CellProfilerSkeletons = sorted_CellProfilerSkeletons[['Normalized Neurite Area','um Neurite Area', 
                           "Normalized Skeleton Coverage", 'um Skeleton Coverage',
                            'FileName_Originals', 'Experimental_Group']]
    
    def fuzzy_matched_files(df1, df2, group_col = "Experimental_Group", 
                            match_col = 'FileName_Originals', 
                            threshold = fuzzy_threshold):

        mapping = {}

        df1['combined_key'] = (
            df1[group_col].astype(str) + " | " + df1[match_col].astype(str)
            ).to_list()
        
        df2['combined_key'] = (
            df2[group_col].astype(str) + " | " + df2[match_col].astype(str)
            ).to_list()
        
        for group in df2[group_col].unique():
            df1_sub = df1[df1[group_col] == group]
            df2_sub = df2[df2[group_col] == group]

            if df1_sub.empty:
                print("No Groups Found")
                break
            
            df1_filenames = df1_sub[match_col].astype(str).tolist()

            for _, row in df2_sub.iterrows():
                #key = f'{group} | {row[match_col], df1_filenames}'
                key = f"{group} | {row[match_col]}"

                match = process.extractOne(row[match_col], df1_filenames)

                if match and match[1] >= threshold:
                    matched_filename = match[0]
                    mapping[key] = f"{group} | {matched_filename}"
                else:
                    mapping[key] = None

        
        return df1, df2, mapping
    
    sorted_experiment_stats,sorted_CellProfilerSkeletons, file_map = fuzzy_matched_files(df1=sorted_experiment_stats,
                          df2 = sorted_CellProfilerSkeletons,
                          group_col = "Experimental_Group",
                          match_col = "FileName_Originals",
                          threshold = fuzzy_threshold)
    
    sorted_CellProfilerSkeletons["Matched_Key"] = (
        sorted_CellProfilerSkeletons["combined_key"].map(file_map)
    )

    merged_df = sorted_experiment_stats.merge(
        sorted_CellProfilerSkeletons,
        left_on="combined_key",
        right_on="Matched_Key",
        how="outer"
    )
    missing = merged_df[merged_df["Matched_Key"].isna()]

    print(f"Unmatched rows: {len(missing)}")
    print(missing[["combined_key_x", "combined_key_y"]].head(20))

    merged_df = merged_df.dropna()
    merged_df['synapse_ROI'] = merged_df['synapse_ROI'].astype(float)
    merged_df['dendrite_ROI'] = merged_df['dendrite_ROI'].astype(float)
    merged_df['total_ROIs'] = merged_df['total_ROIs'].astype(float)


    merged_df['synapses_per_10um'] = (merged_df['synapse_ROI'] / merged_df["um Skeleton Coverage"]) * 10
    merged_df['dendrites_per_10um'] = (merged_df['dendrite_ROI'] / merged_df["um Skeleton Coverage"]) * 10
    merged_df['total_per_10um'] = (merged_df['total_ROIs'] / merged_df["um Skeleton Coverage"]) * 10

    
    return merged_df, missing


def main(folder):
    experiment = folder.split('\\')[-1]

    groups, metrics, synapses_csv = load_experiment_csv(folder)
    synapses_csv.to_csv(os.path.join(folder, f'{experiment}_synapse_average.csv'))
    df = merge_cellprofiler_csvs_without_fuzzy_match(folder)
    df.to_csv(os.path.join(folder, f'{experiment}_synapse_normalized_data.csv'))
    return df

