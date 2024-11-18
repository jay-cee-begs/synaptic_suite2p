@echo off

:: Activate the first virtual environment, evt. use Activate.ps1

CALL "C:\miniforge3\Scripts\activate.bat" suite2p

:: Run the default ops script

python -m jd_default_ops

:: Deactivate the first environment
CALL conda deactivate