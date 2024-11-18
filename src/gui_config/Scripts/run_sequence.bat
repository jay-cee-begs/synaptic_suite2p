@echo off

:: Activate the first virtual environment, evt. use Activate.ps1

CALL "C:\miniforge3\envs\suite2p\Scripts\activate.bat" suite2p

:: Run the first script
python -m syn_process

:: Deactivate the first environment
CALL conda deactivate
pause
:: Activate the third virtual environment
CALL "C:\Users\Justus\Anaconda3\Scripts\activate.bat" data_env

:: Run the third script 
python -m plotting_constants



:: keep terminal open 
pause