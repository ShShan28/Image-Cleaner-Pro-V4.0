@echo off
setlocal enabledelayedexpansion
pushd "%~dp0"
echo ==========================================
echo   Image Cleaner Pro - Local Setup
echo ==========================================
echo.
echo [NOTE] It is recommended to run this script as Administrator
echo        to ensure Tesseract-OCR can be installed correctly.
echo.

REM Check if Python is installed
set "PYTHON_CMD="

REM 1. Try "python" in PATH
python --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=python"
) else (
    REM 2. Try "py" (Python Launcher)
    py --version >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON_CMD=py"
    ) else (
        REM 3. Try "python3"
        python3 --version >nul 2>&1
        if %errorlevel% equ 0 (
            set "PYTHON_CMD=python3"
        ) else (
            REM 4. Search common installation paths
            echo [INFO] Python not in PATH. Searching standard folders...
            
            for /d %%D in ("%LocalAppData%\Programs\Python\Python*") do (
                if exist "%%D\python.exe" (set "PYTHON_CMD=%%D\python.exe")
            )
            
            if "!PYTHON_CMD!"=="" (
                for /d %%D in ("%ProgramFiles%\Python*") do (
                    if exist "%%D\python.exe" (set "PYTHON_CMD=%%D\python.exe")
                )
            )

            if "!PYTHON_CMD!"=="" (
                for /d %%D in ("%ProgramFiles(x86)%\Python*") do (
                    if exist "%%D\python.exe" (set "PYTHON_CMD=%%D\python.exe")
                )
            )

            if "!PYTHON_CMD!"=="" (
                for /d %%D in ("C:\Python*") do (
                    if exist "%%D\python.exe" (set "PYTHON_CMD=%%D\python.exe")
                )
            )
        )
    )
)

REM 5. Auto-Installation Fallback
if "!PYTHON_CMD!"=="" (
    echo [MISSING] Python not found in common locations.
    echo [ATTEMPT] Trying to install Python 3.12 automatically via winget...
    winget --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo [INFO] Winget found. Starting installation...
        winget install --id Python.Python.3.12 --exact --silent --accept-package-agreements --accept-source-agreements
        if %errorlevel% equ 0 (
            echo [SUCCESS] Python installed via winget. Retrying search...
            for /d %%D in ("%LocalAppData%\Programs\Python\Python*") do (
                if exist "%%D\python.exe" (set "PYTHON_CMD=%%D\python.exe")
            )
        )
    ) else (
        echo [ERROR] Winget not found. Please install Python manually from python.org.
    )
)

if "!PYTHON_CMD!"=="" (
    echo [ERROR] Python not found. Please install Python 3.10+ and try again.
    echo [TIP] Make sure to check "Add Python to PATH" during installation.
    echo.
    echo If Python is already installed, try running this script as Administrator.
    pause
    exit /b 1
)

echo [INFO] Using Python: !PYTHON_CMD!
"!PYTHON_CMD!" --version

echo [1/4] Creating virtual environment (venv)...
if not exist "venv\Scripts\python.exe" (
    if exist "venv" rmdir /s /q "venv"
    "!PYTHON_CMD!" -m venv venv
)

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo [2/4] Installing dependencies...
"venv\Scripts\python.exe" -m pip install --upgrade pip
"venv\Scripts\python.exe" -m pip install wheel setuptools
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements. Please check your internet connection.
    pause
    exit /b 1
)

echo [3/4] Verifying AI Models...
if not exist "backend\models\colorization_release_v2.caffemodel" (
    echo [WARNING] AI Colorization model missing! 
    echo Please ensure the 'backend\models' folder contains the required files.
)

echo [3/4] Checking for Tesseract-OCR (for Text Search)...
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    echo [FOUND] Tesseract-OCR is already installed.
    goto :setup_done
)

echo [MISSING] Tesseract-OCR not found. Auto-installing...
echo [DOWNLOADING] Tesseract-OCR Installer...
curl -L -o tesseract_installer.exe https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe
if %errorlevel% neq 0 (
    echo [ERROR] Download failed. Please check your internet connection.
    goto :setup_done
)

echo [INSTALLING] Tesseract-OCR silently...
start /wait tesseract_installer.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
del tesseract_installer.exe
:setup_done
echo.
echo [Optional] Advanced Language Support (Tamil)...
if not exist "tessdata" mkdir "tessdata"
if not exist "tessdata\tam.traineddata" (
    echo [DOWNLOADING] Tamil Data (Best Quality)...
    curl -L -o tessdata\tam.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/tam.traineddata
)
if not exist "tessdata\eng.traineddata" (
    echo [DOWNLOADING] English Data (Local Backup)...
    curl -L -o tessdata\eng.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata
)

echo.
echo [4/4] Setup complete!
echo.
echo To run the application, simply run:
echo run_app.bat
echo.
popd
