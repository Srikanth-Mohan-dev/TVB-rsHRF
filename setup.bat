@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "VENV_DIR=%SCRIPT_DIR%venv"
set "TVB_DIR=%SCRIPT_DIR%tvb-root"

echo ============================================================
echo Setting up environment in %SCRIPT_DIR%
echo ============================================================

where python >nul 2>nul
if errorlevel 1 (
    echo python not found on PATH. Install Python 3.9+ from python.org
    echo and make sure "Add python.exe to PATH" is checked during install.
    exit /b 1
)

where git >nul 2>nul
if errorlevel 1 (
    echo git not found on PATH. Install Git from https://git-scm.com/downloads
    echo it's needed to clone tvb-root.
    exit /b 1
)

echo [1/5] Creating virtual environment at %VENV_DIR%
python -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo Failed to create virtual environment.
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"

echo [2/5] Upgrading pip
python -m pip install --upgrade pip

echo [3/5] Installing Python dependencies from requirements.txt
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install Python dependencies.
    exit /b 1
)

echo [4/5] Cloning and installing TVB (hybrid-numba branch)
if not exist "%TVB_DIR%" (
    git clone --branch hybrid-numba https://github.com/the-virtual-brain/tvb-root.git "%TVB_DIR%"
    if errorlevel 1 (
        echo Failed to clone tvb-root.
        exit /b 1
    )
) else (
    echo   %TVB_DIR% already exists — skipping clone.
)
pip install -e "%TVB_DIR%\tvb_library"
if errorlevel 1 (
    echo Failed to install tvb_library.
    exit /b 1
)

echo [5/5] Checking for AWS CLI
where aws >nul 2>nul
if errorlevel 1 (
    echo   aws CLI not found on PATH after pip install awscli.
    echo   The pip package installs an "aws" command inside the venv;
    echo   if it's still not found, install the official AWS CLI v2 manually:
    echo     https://awscli.amazonaws.com/AWSCLIV2.msi
) else (
    for /f "delims=" %%v in ('aws --version 2^>^&1') do echo   aws CLI already available: %%v
)

if not exist datasets mkdir datasets
if not exist results mkdir results

echo.
echo ============================================================
echo Done. To use this environment:
echo   venv\Scripts\activate.bat
echo   python pipeline.py --dataset ds001226 --subject CON01
echo ============================================================

endlocal