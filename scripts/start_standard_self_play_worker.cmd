@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

if exist "%SCRIPT_DIR%..\.venv\Scripts\python.exe" (
  "%SCRIPT_DIR%..\.venv\Scripts\python.exe" "%SCRIPT_DIR%start_standard_self_play_workers.py" %*
  exit /b %errorlevel%
)

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 "%SCRIPT_DIR%start_standard_self_play_workers.py" %*
  exit /b %errorlevel%
)

python "%SCRIPT_DIR%start_standard_self_play_workers.py" %*
