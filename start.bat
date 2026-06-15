@echo off
title Q-BitNet + LingBot-World v3 — Training
color 0B
set PYTHON=C:\Users\hp\AppData\Local\Programs\Python\Python313\python.exe

echo.
echo  =====================================================
echo   Q-BitNet + LingBot-World v3 — Starting
echo  =====================================================
echo.

:: Activate venv
call venv\Scripts\activate.bat

:: Start WebSocket bridge in background
echo [1/3] Starting WebSocket metrics bridge...
start "WS Bridge" /min cmd /c "call venv\Scripts\activate.bat && python server\ws_bridge.py"
timeout /t 2 /nobreak >nul

:: Start Vite dashboard in background
echo [2/3] Starting dashboard (http://localhost:5173)...
start "Dashboard" /min cmd /c "npm run dev"
timeout /t 3 /nobreak >nul

:: Open browser
echo [3/3] Opening dashboard in browser...
start http://localhost:5173
timeout /t 2 /nobreak >nul

:: Start trainer (foreground — shows training output)
echo.
echo  Training output:
echo  ─────────────────────────────────────────────────────
echo.
python train\trainer.py

echo.
echo  Training finished. Press any key to exit.
pause
