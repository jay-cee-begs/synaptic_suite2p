@echo off

:: Activate the s2p environment
CALL "C:\Users\jcbeg\pyenv\suite2p\Scripts\activate.bat" 

:: Run the first script

set script_dir=%~dp0

@REM set src_dir=%script_dir%..\..\


cd "%script_dir%..\"


python -m analysis_params