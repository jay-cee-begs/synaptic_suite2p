Synaptic suite2p README.md

This code takes the MouseLand suite2p software, an automated calcium imaging ROI detector, and applies it to a new method of mass detecting synapses with  adaptation of 2-photon in vivo suite2p for in vitro NMDA synaptic calcium imaging

Part 1: creating a repository

for this you will need to have a conda virtual environment (suite2p) created with python3.8 from the provided requirements.txt file
alternatively, you can use the suite2p_env.yml file


To create a conda env, simply use $conda create -n suite2p python=3.8 -r suite2p_requirements.txt
can also be done outside of anaconda if you want open sourced code, then you can download python and custom install it so it runs only in environments
https://www.youtube.com/watch?v=28eLP22SMTA&t=11s
-r requirements.txt


once inside you can activate the environment and run the code on the test data

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