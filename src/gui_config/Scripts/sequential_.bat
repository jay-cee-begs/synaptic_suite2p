@echo off

:: Activate the first virtual environment, evt. use Activate.ps1

CALL "C:\miniforge3\envs\suite2p\Scripts\activate.bat" suite2p

:: Run the first script
python -m suite2p_headless

:: Deactivate the first environment
CALL conda deactivate





:: keep terminal open 
pause