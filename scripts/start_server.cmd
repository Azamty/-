@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_server.ps1" -Background
if errorlevel 1 (
    pause
    exit /b 1
)
start "" "http://127.0.0.1:8000"
