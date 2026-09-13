@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_pdf_server.ps1"
if errorlevel 1 (
    echo.
    echo Could not start the PDF score website. See the message above.
    pause
    exit /b 1
)
