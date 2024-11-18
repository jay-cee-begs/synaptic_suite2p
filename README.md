Synaptic suite2p README.md

An adaptation of 2-photon in vivo suite2p for in vitro NMDA synaptic calcium imaging

Part 1: creating a repository

for this you will need to have a conda virtual environment (suite2p) created with python3.8 from the provided requirements.txt file
alternatively, you can use the suite2p_env.yml file


To create a conda env, simply use $conda create -n suite2p python=3.8 -r requirements.txt
can also be done outside of anaconda if you want open sourced code, then you can download python and custom install it so it runs only in environments
https://www.youtube.com/watch?v=28eLP22SMTA&t=11s
-r requirements.txt


once inside you can activate the environment and run the code on the test data

CONTENTS:
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

Everything can be run in the provided example
jupyter notebook (pipeline_testing.ipynb)