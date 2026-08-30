@echo off
REM Trading Platform - desktop launcher.
REM
REM Double-click this file to start the application. It opens its own window; the console
REM stays behind it and shows what is happening, which is where any startup error appears.
REM
REM The platform starts in PAPER mode. Nothing here can enable live trading.

setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo.
    echo The application has not been built yet.
    echo.
    echo Run this once, in this folder:
    echo     powershell -ExecutionPolicy Bypass -File scripts\build-desktop.ps1
    echo.
    pause
    exit /b 1
)

"%PYTHON%" -m app.desktop
set "EXITCODE=%ERRORLEVEL%"

REM Only hold the window open on failure. On a clean exit the user closed the app and does
REM not need a console asking them to press a key.
if not "%EXITCODE%"=="0" (
    echo.
    echo The application exited with code %EXITCODE%.
    pause
)

exit /b %EXITCODE%
