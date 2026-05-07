# Synaptic suite2p README.md

An automated synaptic calcium imaging detection pipeline based on Suite2p optimized for primary neuronal cell cultures and widefield microscopy

This code takes the MouseLand suite2p software (https://github.com/MouseLand/suite2p), a calcium imaging ROI detector and fluorescence extractor, and applies it to detecting primarily NMDAR-mediated synaptic puncta and calcium transients in 0mM Mg2+ solutions with TTX. 

The current pipeline has only been tested on Windows; Linux compatibility is not compatible with the current GUI

## Setup and Installation 
You will need to use a python interpretter (either Anaconda, miniforge, python.exe version 3.9, etc.)
    Newer versions of Python should also be viable for analysis, but these have not been tested thoroughly.

To start you will need to create a fork of this repository (synaptic suite2p) to your current machine through GitHub or by downloading the source code as a zip file above. 


### INSTALLING SUITE2P

1. Make a fork of this online repository to your personal Github; set up to track this repo to get all future updates and improvements.

2. Make a local copy of the synaptic suite2p repository on your computer saved in \Documents\Github\synaptic_suite2p
    This is handled automatically be the GitHub Desktop app, or example.

3. Navigate to the local directory where your clone of the repository is saved. Do this in an anaconda or miniforge terminal window using the `cd` command and the path to the copied repository `cd path\to\synaptic_suite2p`

4. Create a virtual environment in anaconda / miniforge for suite2p using python 3.9 by running the command `conda create -n suite2p python=3.9`

5. To confirm suite2p installed correctly, follow Mouseland's guidelines and run the command `python -m pip install suite2p[gui]` to make sure you have access to the suite2p user interface. The version used for this project was suite2p==0.14.0

6. Lastly, please run `pip install -e .` from the main project folder of synaptic_suite2p `path\to\synaptic_suite2p`


**NOTE**: some additional packages will need to be installed (e.g.`pip install seaborn BaselineRemoval pynapple` image software-specific  packages such as `nd2` might also nee to be installed for processing Nikon microscope image files)

# Workflow

* Prior to starting you should organize your data into the following experiment structure
```
    \path\to\experiment\folder
    ├───experiment_condition_folder_1
    │   ├───image_or_image_folder_1
    │   ├───image_or_image_folder_2
    │   ├───image_or_image_folder_3
    │       
    ├───experiment_condition_folder_2
    │   ├───image_or_image_folder_1
    │   ├───image_or_image_folder_2
    │   └───image_or_image_folder_3
    │       
    ├───experiment_condition_folder_3
    │   ├───image_or_image_folder_1
    │   ├───image_or_image_folder_2
    │   └───image_or_image_folder_3
    │       
    ├───experiment_condition_folder_4
    │   ├───image_or_image_folder_1
    │   ├───image_or_image_folder_2
    │   ├───image_or_image_folder_3
```
* Both image files and folders with images are acceptable. The code will look inside an experiment folder for experimental conditions. These conditions should contain image fies of a particular type (e.g., tiff, nd2) that are unsorted or pre-sorted into subfolders (if multiple images exist for similar regions). When the experiment_condition folder contains multiple images, the code will automatically sort each image file into its own folder so that it can be processed individually by Suite2p. 


1. AFTER THE INSTALL: Please open up batch files using Visual Studio Code or another variant to look at windows batch files.

2. For every `CALL` `path_to_conda_activate.bat`, please update the path to your own base conda path. you can find this path by running `conda env list` and replacing everything before `\Scripts\activate.bat` This will only need to be done the first time openning the analysis pipeline.

3. The GUI will now run the analysis pipeline correctly.
To launch the gui either double click `run_analysis_gui.bat` found in `synaptic_suite2p\src\gui_config\Scripts` or navigate to the gui_config folder using `cd path\to\GitHub\folder\synaptic_suite2p\src\gui` followed by `python -m run_gui`

4. After launching the GUI change the `Experiment / Main Folder Path:` using the `Browse` button or by manually typing in the folder containing all imaging files already presorted into experimental treatments (and potentially similar regions if running registration and comparing the same synapses over time)

5. Enter the `Data Extension` for your imaging files without `.` (e.g. `tif`, `tiff`, or `nd2`) and click `Add Experiment Conditions` to automatically add the subfolders containing images as experimental groups. The Experiment Conditions will automatically populate a dictionary visible lower in the GUI. 

* It is essential that the '.' is not in the data extension / file ending so Suite2p will run and process the data appropriately!

6. In the `Suite2p settings (ops.npy):` Browse or manually enter your own suite2p detection settings (`.npy` file). Example Suite2p_ops files are provided in the repository; however, custom settings files can be generated and checked for accuracy using the Suite2p GUI. 

7. Enter the `Frame Rate:` of your images (in frames per second)<br>
Enter the `Experiment Duration` in number of seconds<br>
Enter the `Network Bin Width` in numbers of frames to group for synchronous synapse detection (default 5 frames)

8. `Edit Analysis Parameters` to change the analysis_params section of the `config.json` file to adjust post-processing analysis settings for suite2p data
```
Editable Analysis Parameters
    overwite_suite2p: boolean 
        Allows pipeline to overwrite pre-existing suite2p files and deltaF files

    multivid_processing: boolean
        Tells the pipeline that multiple images exist per region, per subfolder.
        IF this is selected, it will automatically launch a Pop-up GUI for these settings later. 

    use_suite2p_ROI_classifier: boolean
        Utilizes Suite2p in-built classifier
        *This is not recommended on a first pass of the data, only after manual curation. 
        *Users can make their own classifiers for synaptic calcium imaging data if desired. 

    update_suite2p_iscell: boolean 
        Update the Suite2p `iscell.npy` file for filtering "real" and "noise" ROIs from one another. 
        1 = "cells" or included ROIs and 0 = "non-cells" or excluded ROIs

    Img_Overlay: str ("max_proj" or "meanImg") 
        choice of max projection (max_proj) or mean image (meanImg) as a base for overlaying synaptic ROIs

    skew_threshold: float
        minimum fluorescence skew required for ROI to be considered real 
        (default = 1.0)

    compactness_threshold: float
        How circular an ROI needs to be to be subclassified as a dendritic ROI.
        default = 1.4; to turn off: compactness_threshold > 10 

    baseline_correction: str ("airPLS" or "rolling_median")
        User choice of baseline correction between airPLS algorithm from NMR analysis or rolling median. These functions are used to general dF / F0 files

    lambda_window: int
        The lambda_ value for optmizing airPLS correction (recommended value: ~100; values closer to 10 make the baseline perfectly flat)
        The _window value for number of frames to calculate rolling median over (recommended value: ~100-500)

    MAD_baseline_filter_threshold: float
        Number of MAD-estimated standard deviations above MAD to use as a cutoff for isolating baseline / noise frames

    peak_detection_threshold: float
        Number of standard deviations above gaussian fit to noise to set as the threshold for peak detection. (default: 4.5 SD)
    
    peak_count_threshold: minimum number of calcium peaks per trace
        default: 1
    
Multivid_Registration_Params
        saved in config.json as general_settings['multivid_params'] or as general namespace object general_settings.multivid_params
    
    Treatment_No: Number of treatments / Videos (including baseline)
        Interactive to control for number of videos within a given subfolder
        e.g., Treatment_No = 3; baseline -> treatment1 -> treatment2
    
    equal_baseline_and_treatments: boolean
        Are all videos (e.g., baseline and treatments the same number of frames?)
        
        IF not selected, other settings display listed below

        Treatment length units: "seconds" or "frames"
            Units for the length of each video provided. Seconds and Frames for each video are possible units
        
        Video Lengths: user input
            Video lengths for Baseline and Treatment 1, Treatment 2, etc. 

        
```

analysis_params and multivid_params are written as dictionaries within the config.json along with the general_settings dictionary containing the main experiment information. 
Settings are saved each time a Pop-up window is closed. 

10. ***Click*** `Save Configurations` to update the configurations file (config\config.json) for analysis; these parameters can also be changed manually by the user
    Alternatively, settings can be saved when the pipeline is run below using `Process`

11. Click `Process` to run the pipeline 

At the end of the processing, there will be summary files in the experimental_condition folders

Each experimental condition folder will have ROIs circled overlayed on a meanImg or max_proj depending on which is chosen in the analysis_params settings
<br> 
In the Experiment / Main Folder: 
    Summary statistics are exported in csv file format in the file `Experiment Summary.csv`
<br>
    Individual file analyses in csv format will be saved in `experiment_folder\csv_files`

    Pickle files are saved in `experiment_folder\csv_files` these contain all of the raw data and processed data from csvs
    
CONTENTS:
```
notebook_scripts/
    run_pipeline: example for updating config.json settings without the user interface (GUI) and for seeing which functions run in the process

src/
        main code of the repository
    plotting_utility:  for generating plots from processed data and performing basic statistical tests
    
    suite2p_utility: unpacking suite2p outputs
        
    detector_utility: peak detection and fluorescence nromalization

    analysis_utility: converting spikes and amplitudes into measurable statistics 

    run_suite2p: transfers image files into individual folders to run with suite2p and suite2p to run without GUI
    
    config_loader: functions for loading JSON config files as namespace objects or dictionaries

    export_cellprofiler_skeletons: functions to calculate synapses per image file and map them to CellProfiler pipeline outputs
        *NOTE These likely will not work directly with multivid_processing and are likely not necessary since synapses are tracked across conditions. 

gui/
        Code to run gui and automated processing
    Scripts/
        .bat files that can be used to run the pipeline from the GUI
        run_plots.bat:
        run_s2p_gui.bat:
        run_suite2p.bat:
        run_user_GUI.bat:
        syn2p_setup.bat:

    analysis_params: establishes post-processing analysis parameters in the GUI
    run_gui: code to run the gui; called using `python -m run_gui`
    config_editor: ConfigEditor class for populating GUI 
    ops_editor: AnalysisParams class for loading special analysis settings

gui_core/
        backend functions and files for running the GUI smoothly
    analysis_model: Default AnalysisParams 
    multivid_reg: Default MultiVidEditor settings
    folder_logic: Functions to load files and define experimental groups for establishing post-processing rules and organization structures
    general_settings_model: GenSettings class containing default settings for config.json general_settings
    io: Functions for saving and loading config.json files for the GUI
        io functions only are used in the GUI and not elsewhere in the pipeline

suite2p_ops_files/
        Example ops.npy files for suite2p settings
    synaptic_suite2p_w_TTX_multivid_registration.npy: suite2p settings for detecting synapses across multiple recordings
    synaptic_suite2p_w_TTX.npy: suite2p settings for detecting synapses in individual recordings

R_analysis/
        Code to run mixed-effect models and bootstrapping
    glmm_effect_functions.R: function for running GLMM_TMB R package on processed synapse data

    synapse_effect_bootstraps.R: function for running clustered, stratified bootstrap resampling with replacement on processed syanpse data

neurite_normalization/
        CellProfiler and FIJI macros for determining neurite coverage
    Cell_Profiler_Projections_Dendrite_Coverage.ijm: ImageJ macro for automatically generating Average projection images
        
    mask_neurons_and_skeletonize_neurites.cpproj: CellProfiler pipeline for removing soma signal and skeletonizing neurites from average projection images

    MAX projection minus MIN projection.ijm: ImageJ macro to generate max minus min projection images as seen in figures

config/
    analysis_params.json: JSON file containing outputs for the current experiment from analysis_params.py
    config.json: JSON file containing all information for running the current experiment (including analysis_params.json)

.gitattributes
.gitignore
    files for git not to track
LICENSE
README.md

```
## Configuration

You can modify the behavior of the application by creating and editing `config/user_config.json`. 
Here are some example configuration options:

```json
{
 "general_settings": {
  "main_folder": "E:/concat_z-score",
  "data_extension": "nd2",
  "frame_rate": 20,
  "ops_path": "C:/Users/jcbeg/Documents/GitHub/synaptic_suite2p/suite2p_ops_files/synaptic_suite2p_w_TTX_multivid_reg.npy",
  "groups": [
   "replicate01",
   "replicate02",
   "replicate03"
  ],
  "exp_condition": {
   "replicate01": "replicate01",
   "replicate02": "replicate02",
   "replicate03": "replicate03"
  },
  "BIN_WIDTH": 5,
  "EXPERIMENT_DURATION": 540
 },
 "analysis_params": {
  "overwrite_suite2p": false,
  "multivid_processing": true,
  "use_suite2p_ROI_classifier": false,
  "update_suite2p_iscell": true,
  "Img_Overlay": "max_proj",
  "return_decay_times": true,
  "skew_threshold": 1.0,
  "compactness_threshold": 1.4,
  "peak_detection_threshold": 4.5,
  "peak_count_threshold": 1
 },
 "multivid_params": {
  "Treatment_No": 2,
  "equal_baseline_and_treatments": true,
  "unequal_treatment_lengths": [],
  "treatment_length_units": "frames"
 }
}
```