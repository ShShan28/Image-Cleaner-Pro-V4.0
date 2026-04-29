@echo off
pushd "%~dp0"
echo Starting Image Cleaner Pro...

REM Disable Streamlit telemetry and onboarding
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_SERVER_HEADLESS=true

REM Check if venv exists and is valid for this PC
set "REBUILD_VENV=0"
if not exist "venv\Scripts\python.exe" (
    set "REBUILD_VENV=1"
) else (
    "venv\Scripts\python.exe" --version >nul 2>&1
    if errorlevel 1 (
        echo [INFO] Existing environment is broken or from another PC. Cleaning up...
        rmdir /s /q "venv"
        set "REBUILD_VENV=1"
    )
)

if "%REBUILD_VENV%"=="1" (
    echo [INFO] Running auto-setup...
    call setup.bat
    if errorlevel 1 (
        echo [ERROR] Setup failed. Please ensure Python is installed and you have internet access.
        pause
        exit /b
    )
)

REM Run using python module to bypass PATH issues
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" -m streamlit run "backend\app.py"
) else (
    echo [ERROR] Virtual environment not found. Please run setup.bat first.
    pause
)

popd
pause
