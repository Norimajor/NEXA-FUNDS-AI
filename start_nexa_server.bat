@echo off
cd /d "%~dp0"
set "HOST=127.0.0.1"
set "PORT=4000"
set "ALLOW_LOCAL_ML_PREDICTIONS=true"
set "NEXA_AI_API_URL=http://127.0.0.1:8001"
set "NEXA_AI_TIMEOUT_MS=12000"
set "NEXA_H1_MODEL_PATH=models\pooled_h1_20260929.joblib"
set "NEXA_M15_MODEL_PATH=models\pooled_m15_20260929.joblib"
set "NEXAFUNDS_DIR=%USERPROFILE%\nexafunds"
curl.exe -sS --max-time 2 -o nul http://127.0.0.1:4000/api/__gateway_probe__ >nul 2>&1
if not errorlevel 1 goto gateway_ready
if not exist "%NEXAFUNDS_DIR%\package.json" goto gateway_missing
start "NexaFunds Gateway" /D "%NEXAFUNDS_DIR%" cmd /k "npm start"
goto check_python
:gateway_missing
  echo NexaFunds gateway project not found at "%NEXAFUNDS_DIR%".
  echo The Python ML service will still start, but the EA gateway on port 4000 will be unavailable.
goto check_python
:gateway_ready
echo NexaFunds gateway is already healthy on port 4000.
:check_python
curl.exe -fsS --max-time 2 http://127.0.0.1:8001/health >nul 2>&1
if not errorlevel 1 goto python_ready
set "PORT=8001"
call .\.venv\Scripts\activate.bat
python main.py
if errorlevel 1 (
  echo.
  echo NEXA Python ML service failed to start.
  echo Check the Python environment and dependencies.
  pause
)
goto end
:python_ready
echo NEXA Python ML service is already healthy on port 8001.
:end
