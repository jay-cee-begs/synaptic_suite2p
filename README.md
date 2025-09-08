# Synaptic suite2p README.md

A semi-automated calcium imaging detection (using suite2p) and deconvolution (using cascade) pipeline for primary neuronal culture calcium imaging using widefield microscopy

This code takes the MouseLand suite2p software (https://github.com/MouseLand/suite2p), a calcium imaging ROI detector and fluorescence extractor, and applies it to detecting NMDA synaptic puncta in 0mM Mg2+ solutions with TTX. 

## Setup and Installation 
You will need to use a python interpretter (either Anaconda, miniforge, python.exe version 3.9, etc)

To start you will need to create a fork of this repository (synaptic suite2p) to your current machine through github or by downloading the source code as a zip file. 
NOTE: likely would be _easier_ to just make a package with the src code on PyPi

Alternatively you should download a zip file of the code and save it somewhere you can access easily

### INSTALLING SUITE2P

1. Make a fork of this online repository to your personal Github; set up to track this repo to get all future updates

2. Make a local copy of the synaptic suite2p repository on your computer saved in \Documents\Github\synaptic_suite2p

3. Navigate to the local directory where your clone of the repository is saved. Do this in an anaconda or miniforge terminal window using the `cd` command and the path to the copied repository

4. Create a virtual environment in anaconda / miniforge for suite2p using python 3.9 by running the command `conda create -n suite2p python=3.9`

5. To confirm suite2p installed correctly, follow Mouseland's guidelines and run the command `python -m pip install suite2p[gui]` to make sure you have access to the suite2p user interface. The version used for this project was suite2p==0.14.0

6. Lastly, please run `pip install -e .` from the main project folder of synaptic_suite2p


**NOTE**: some additional packages will need to be installed (e.g.`pip install nd2 seaborn BaselineRemoval`)

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
* Both image files and folders with images are acceptable. The code will look for image folders containing single image files; if it finds none, it will move all image files of a particular type (e.g. nd2 / tif) into folders of the same name automatically


1. AFTER THE INSTALL: Please open up batch files using Visual Studio Code or another variant to look at windows batch files

2. For every `CALL` `path_to_conda_activate.bat`, please update the path to your own base conda path. you can find this path by running `conda env list` and replacing everything before `\Scripts\activate.bat` This will only need to be done the first time openning the analysis pipeline

3. The GUI will now run the analysis pipeline correctly.
To launch the gui either double click `run_analysis_gui.bat` found in `synaptic_suite2p\src\gui_config\Scripts` or navigate to the gui_config folder using `cd .\synaptic_suite2p\src\gui_config` followed by `python -m synapse_gui`

4. After launching the synapse_gui change the `experiment / main_folder` using the `Browse` button (the folder containing experimental conditions as subfolders)

5. Enter the file extension for your image type without `.` (e.g. `tif`, `tiff`, or `nd2`) and click `Add Experiment Conditions` to automatically add the subfolders containing images as experimental groups

6. Find your own suite2p detection settings (`.npy` file) using `Browse` or `Open Suite2p GUI` using the suite2p GUI
NOTE: please wait for the GUI to launch, it will take some time and then open the settings for testing using `CTRL+R` or File -> Run_Suite2p

7. Enter the frame rate of your images using `Frame Rate:`<br>
Enter the `Experiment Duration` in number of seconds<br>
Enter the `Network Bin Width` in numbers of frames to group for synchronous synapse detection (default 5 frames)

8. `Edit Analysis Parameters` to change the analysis_params.json file and modify analysis settings for post-processing of suite2p data
```
Editable Analysis Parameters
    overwrite_csv: true / false (overwrites csv files containing synapse analysis)
    
    skew_threshold: minimum fluorescence skew required for ROI to be considered real 
        (default = 1)
    
    peak_detection_threshold: Number of standard deviations above gaussian fit to noise to set as the threshold for peak detection
        (default: 4.5 SD); for more information please see peak_detection_threshold.ipynb

    peak_count_threshold: minimum number of calcium peaks per trace for ROI to be counted towards analysis

    Img_Overlay: choice of max projection (max_proj) or mean img (meanImg) as a base for overlaying synaptic ROIs

    use_suite2p_ROI_classifier: true / false (use built-in suite2p classifier NOT RECOMMENDED)

    update_suite2p_iscell: true / false (update iscell.npy for visualizing ROIs considered "cells" -true- or "non-cells" -false-)
    
        
```

analysis_params are written to their own file automatically when the window is closed

10. ***Click*** `Save Configurations` to update the configurations file (config\config.json) for analysis; these parameters can also be changed manually by the user

11. Click Process to run 

At the end of the processing, there will be summary files in each of the image folders

Each experimental condition folder will have ROIs circled overlayed on a meanImg or max_proj depending on which is chosen in the analysis_params settings
<br> 
Summary statistics are exported in csv file format in the file `Experiment Summary.csv`
<br>
Individual file analyses in csv format will be saved in `experiment_folder\csv_files`

CONTENTS:
```
notebook_scripts/
    test_pipline: for running suite2p in its current state, including user inputs
src/
    main code of the repository
    init: marks code as source code
    utility:
        plotting: for generating plots and doing statistical tests
        suite2p: unpacking suite2p outputs
        detector: detect events from suite2p: extracted traces
        analysis: converting spikes and amplitudes into measurable statistics 
    run_functions:
        suite2p_headless: allows suite2p to run without GUI
    configurations:
        not tracked by git, constants for your experiments
suite2p/
    pyproject.toml: 
        dependencies for this project (work in progress)
    export_cell_profiler_skeletons.py
        code for skeletonization normalization
    tau_sandbox
        trace plotting sandbox
    .gitignore
        files for git not to track
    LICENSE
    README.md
tests/
    yet to be implemented

config/
    configurations: User defined constants that will remain here

```
## Configuration

You can modify the behavior of the application by creating and editing `config/user_config.json`. 
Here are some example configuration options:

```json
{
 "general_settings": {
  "main_folder": "C:\\synapse_groundtruth",
  "groups": [
   "C:\\synapse_groundtruth\\202501_cropped_ground_truth2",
   "C:\\synapse_groundtruth\\202501_ground_truth3",
   "C:\\synapse_groundtruth\\202501_ground_truth_PBS"
  ],
  "group_number": 3,
  "exp_condition": {
   "202501_cropped_ground_truth2": "202501_cropped_ground_truth2",
   "202501_ground_truth3": "202501_ground_truth3",
   "202501_ground_truth_PBS": "202501_ground_truth_PBS"
  },
  "data_extension": "tif", //or "nd2" or "tiff";
  "frame_rate": 10,
  "ops_path": "path\\to\\suite2p\\settings\\file\\ops.npy",
  "BIN_WIDTH": 5,
  "EXPERIMENT_DURATION": 180,
  "FRAME_INTERVAL": 0.1,
  "FILTER_NEURONS": true
 },
 "analysis_params": {
  "overwrite_csv": true,
  "overwrite_pkl": false,
  "skew_threshold": "1.0",
  "compactness_threshold": "1.4",
  "peak_detection_threshold": "4.5",
  "peak_count_threshold": "2",
  "Img_Overlay": "max_proj",
  "use_suite2p_ROI_classifier": false,
  "update_suite2p_iscell": true,
  "return_decay_times": false
 }
}
