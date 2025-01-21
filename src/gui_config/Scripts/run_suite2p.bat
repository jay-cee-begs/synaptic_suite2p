@echo off

:: Activate the first virtual environment, evt. use Activate.ps1

CALL "C:\Users\jcbeg\pyenv\syn_suite2p\Scripts\activate.bat"

:: Set the script and source directories
set script_dir=%~dp0
set src_dir=%script_dir%..\..

python "%src_dir%\analyze_suite2p\run_suite2p.py"


:: Change directory to the analyze_suite2p folder


:: Run the default ops script


