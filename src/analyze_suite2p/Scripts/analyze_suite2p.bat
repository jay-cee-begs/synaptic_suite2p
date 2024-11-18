@echo off

CALL "C:\miniforge3\Scripts\activate.bat" suite2p

SET SCRIPT_DIR=%~dp0
SET PROJECT_DIR=%SCRIPT_DIR%\..\..
SET PYTHONPATH=%PROJECT_DIR%\src;%PYTHONPATH%

conda info --envs

python -m run_suite2p

CALL conda deactivate