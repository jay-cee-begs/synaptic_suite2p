Synaptic suite2p README.md

This code takes the MouseLand suite2p software (https://github.com/MouseLand/suite2p), an automated calcium imaging ROI detector and fluorescence extractor, and applies it to a new method of automatically detecting in vitro NMDA synaptic calcium influx.

Part 1: creating an environment for suite2p

First please download the project code or create a fork on GitHub so that you can access the code locally on your system. 

Option 1 (Anaconda): In an anaconda prompt, you will need to create a conda virtual environment for this project with python3.8
The environment and all of its dependencies are stored in a requirements.txt file. 

To create an environment using these requirements, enter $conda env create -n synaptic_suite2p python=3.8$. This is recommended as MouseLand's suite2p installation lacks key installables such as chardet and PyYAML which are needed to install the source code for this project (see below).

to install synaptic_suite2p requirements run $conda activate suite2p$ followed by $pip install -r requirements.txt

*For conda installations, the name does not really matter. -n = name, -r = requires

If you would like to follow the original author's recommendations, p go to suite2p's README.md file for a full explanation of how to download their software from them and set up a virtual environment for suite2p ((https://github.com/MouseLand/suite2p).
You will still need to run some commands to get all the appropriate packages (e.g., $conda install chardet pyyaml) 
It is RECOMMENDED that you in 

Installing synaptic_suite2p source code as editable installation

After creating your synaptic_suite2p environment, please navigate the head of the directory (synaptic_suite2p) using cd "PATH_TO_DIRECTORY"; if your project folder is on a different drive than your conda installation; switch to the drive first (i.e., 'D:' 'C:', etc.)

activate the synaptic_suite2p virtual environment by running $conda activate suite2p (or whatever you named this)

once in the directory, please run $pip install -e . (this will automatically install all the source code in the src folder to your environment)

From this point on we can now access all the code and go through examples in the example_data folders

once inside you can activate the environment and run the code on the test data ***YET TO BE PROVIDED

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


To install the src
