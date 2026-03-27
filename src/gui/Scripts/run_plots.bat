@echo off

:: Activate the data_env virtual environment
CALL "C:\miniforge3\Scripts\activate.bat" data_env


python -c "import sys; print('Python executable:', sys.executable); print('sys.path:', sys.path)"


set script_dir=%~dp0

set src_dir=%script_dir%..\..\


set src_dir = %script_dir%..\..\src


cd "%src_dir%\plotting"



:: Run the plotting script 
python -m plotting_constants

:: keep terminal open 