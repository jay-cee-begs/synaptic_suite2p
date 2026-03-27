@echo off

set /p python_path="Enter the full path to your Python Interpreter (e.g. "C:\Users\jcbegs\miniforge3\Scripts\activate.bat"): "
setx MY_PYTHON_ENV "%python_path%"

echo Python Interpreter set...Please restart your command prompt