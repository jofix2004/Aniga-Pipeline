@echo off
echo =======================================================
echo Aniga Local Manager Setup and Runner
echo =======================================================

echo Checking dependencies...
pip install -r requirements.txt

echo.
echo Starting Server...
python local_app.py
pause
