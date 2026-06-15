@echo off
title Q-BitNet + LingBot-World v3 — Setup
color 0A

echo.
echo  =====================================================
echo   Q-BitNet + LingBot-World v3 — Setup Script
echo  =====================================================
echo.

:: ── Check Python ──────────────────────────────────────────────
set PYTHON=C:\Users\hp\AppData\Local\Programs\Python\Python313\python.exe

echo [1/6] Checking Python...
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo [!] Python not found at expected path.
    pause
    exit /b 1
)
%PYTHON% --version
echo.

:: ── Check Node ────────────────────────────────────────────────
echo [2/6] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo [!] Node.js not found. Please install from https://nodejs.org/
    pause
    exit /b 1
)
node --version
echo.

:: ── Create Python virtual environment ─────────────────────────
echo [3/6] Creating Python virtual environment...
if not exist "venv" (
    %PYTHON% -m venv venv
    echo [OK] venv created.
) else (
    echo [OK] venv already exists.
)
echo.

:: ── Install Python dependencies ───────────────────────────────
echo [4/6] Installing Python dependencies (torch, transformers, etc.)...
echo       This may take a few minutes...
call venv\Scripts\activate.bat
pip install --upgrade pip -q
pip install -r train\requirements.txt --index-url https://download.pytorch.org/whl/cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -r train\requirements.txt
if errorlevel 1 (
    echo [!] pip install failed. Check your internet connection.
    pause
    exit /b 1
)
echo [OK] Python dependencies installed.
echo.

:: ── Install Node dependencies ──────────────────────────────────
echo [5/6] Installing dashboard dependencies (npm)...
call npm install
if errorlevel 1 (
    echo [!] npm install failed.
    pause
    exit /b 1
)
echo [OK] Dashboard dependencies installed.
echo.

:: ── Download the model ────────────────────────────────────────
echo [6/6] Downloading BitNet 2B4T-BF16 model (~4 GB)...
echo       No login required — fully public MIT-licensed model.
echo       This will take a while depending on your connection.
echo.
call venv\Scripts\activate.bat
%PYTHON% train\download_model.py
if errorlevel 1 (
    echo [!] Model download failed. Check your internet connection and try again.
    pause
    exit /b 1
)
echo.

:: ── Done ──────────────────────────────────────────────────────
echo  =====================================================
echo   Setup complete!
echo  =====================================================
echo.
echo   To start training + dashboard, run:
echo     start.bat
echo.
echo   The dashboard will open at:
echo     http://localhost:5173
echo.
pause
