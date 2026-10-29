@echo off
setlocal
cd /d "%~dp0"
if not exist "env\Scripts\python.exe" (
    echo ERROR: env\Scripts\python.exe was not found.
    pause
    exit /b 1
)
call "env\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Could not activate the project environment.
    pause
    exit /b 1
)
set "PYTHONIOENCODING=utf-8"
"env\Scripts\python.exe" -u run_services.py %*
if errorlevel 1 pause
endlocal
