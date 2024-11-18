@echo off

:: Activate the s2p environment
CALL "C:\Users\Justus\Anaconda3\Scripts\activate.bat" suite2p

python -m suite2p

:: Deactivate the environment
CALL conda deactivate

pause