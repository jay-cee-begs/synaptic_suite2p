@echo off

SET SCRIPT_DIR=%~dp0
SET PROJECT_DIR=%SCRIPT_DIR%\..

cd /d %PROJECT_DIR%
CALL "C:\miniforge3\Scripts\activate.bat" suite2p

python -m run_suite2p


pause
