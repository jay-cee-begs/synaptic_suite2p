
# Synaptic suite2p README.md

This code takes the MouseLand suite2p software (https://github.com/MouseLand/suite2p), a calcium imaging ROI detector and fluorescence extractor, and applies it to detecting NMDA synaptic puncta in 0mM Mg2+ solutions with TTX.

## Part 1: creating an environment for suite2p

First please download the project code or create a fork on GitHub so that you can access the code locally on your system. 

Next, follow the instructions on the suite2p Github for creating a suite2p virtual environment and installation of the suite2p software (installation of suite2p with GUI functionality is recommended)
-(python -m) pip install suite2p[gui]

Additionally, you will need to install BaselineRemoval, pynapple, seaborn, (probably something else with pip)

If you are working with non-tiff files, you will need to install packages to allow for suite2p to access and read these files (e.g. nd2 files: pip install nd2)


## Part 2: Installing synaptic_suite2p source code as editable installation

After creating your synaptic_suite2p environment, please navigate the head of the directory (synaptic_suite2p) using cd "PATH_TO_DIRECTORY"; if your project folder is on a different drive than your conda installation; switch to the drive first (i.e., 'D:' 'C:', etc.)

activate the synaptic_suite2p virtual environment by running $conda activate suite2p (or whatever you named this)

once in the directory, please run $pip install -e . (this will automatically install all the source code in the src folder to your environment)

From this point on we can now access all the code and go through examples in the example_data folders

once inside you can activate the environment and run the code on the test data ***YET TO BE PROVIDED

## Workflow

The code is designed to run through an interactive GUI. In order for this to work, please navigate to the "Scripts" folder in src\gui_config\Scripts and manually change the file string to the location of your own conda / anaconda / miniforge / python installation

we will then navigate to src\gui_config and run python -m synpase_gui to launch the user interface

Prior to analysis your data should be organized in the following format for the pipeline to work

experiment_folder/
    experimental_condition1:
        individual image files
    experimental_condition2:
        individual image files
    experimental_condition3:
        individual image files


In the user interface, we will browse for an experiment folder to analyze (main_folder)
Then we will enter the file extension of our image files without the '.' (e.g., tif, nd2)
We can then click "Add Experiment Conditions" to automatically add the different experimental_condition subfolders to the analysis

The pipeline will keep track of which image files come from which folder. 

We will also need to specify the suite2p settings that we want to use. The settings file used in the paper is included in the GUI and can be a good base for other users to base their own settings off of. 
To create or edit the suite2p settings; please select "Open Suite2p GUI"

We will then specify the frame_rate and experiment_duration for the image files provided

Lastly, we will update the configurations with "Save Configurations" to update the local gui_configurations file

Suite2p can now be run on synaptic imaging data with the "Process" button.
For post-analysis, if you would like to use suite2p's ROI classification, make sure "Use iscell.npy" is checked (NOT RECOMMENDED)



CONTENTS:

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


## Configuration

You can modify the behavior of the application by creating and editing `config/user_config.yaml`. 
Here are some example configuration options:

```yaml
data_path: "/path/to/data"
output_path: "/path/to/output"
processing_options:
  filter_type: "median"  # Options: median, gaussian, etc.
  filter_size: 5         # Size of the filter window
model:
  type: "random_forest"  # Machine learning model type
  parameters:
    n_estimators: 100    # Number of estimators for Random Forest
    max_depth: 10        # Maximum depth of the trees
