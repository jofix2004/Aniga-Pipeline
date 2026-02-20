@echo off
echo =======================================================
echo Aniga3 Chapter Processor Setup and Runner
echo =======================================================

echo Checking dependencies...
pip install -r requirements.txt

echo.
echo Starting Server...
python aniga3_server.py
pause
