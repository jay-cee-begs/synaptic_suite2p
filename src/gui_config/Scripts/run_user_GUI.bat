@echo off

:: Activate the first virtual environment, evt. use Activate.ps1

CALL "C:\miniforge3\Scripts\activate.bat" suite2p

@echo off

SET SCRIPT_DIR=%~dp0
SET PROJECT_DIR=%SCRIPT_DIR%\..

cd /d %PROJECT_DIR%

python -m jd_extended_gui


pause

:: Run the default ops script

python -m jd_default_ops

:: Deactivate the first environment
CALL conda deactivate