@echo off
setlocal
cd /d "%~dp0"
if /I "%~1"=="--docker" (
    if not exist "env\Scripts\python.exe" (
        py -3.12 deployment\container\bootstrap_inference.py
        if errorlevel 1 goto setup_failed
    ) else if exist "env\.hybrid_install_pending" (
        "env\Scripts\python.exe" deployment\container\bootstrap_inference.py
        if errorlevel 1 goto setup_failed
    )
)
if not exist "env\Scripts\python.exe" (
    echo ERROR: Create env and install requirements.txt for native mode, or use --docker for hybrid setup.
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
exit /b

:setup_failed
echo ERROR: Check Python 3.12, prepared model files and the installation error above.
pause
exit /b 1
