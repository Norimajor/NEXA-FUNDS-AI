@echo off
cd /d "%~dp0"
set HOST=127.0.0.1
set PORT=8001
set ALLOW_LOCAL_ML_PREDICTIONS=true
call .\.venv\Scripts\activate.bat
python main.py
if errorlevel 1 (
  echo.
  echo NEXA BTC ML service failed to start.
  echo Check the Python environment and port 8001.
  pause
)