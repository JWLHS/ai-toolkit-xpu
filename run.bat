@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    set "PY=%~dp0.venv\Scripts\python.exe"
) else if exist ".venv-xpu\Scripts\python.exe" (
    set "PY=%~dp0.venv-xpu\Scripts\python.exe"
) else (
    echo [!] 还没有初始化环境，先运行 setup_xpu.bat
    pause
    exit /b 1
)
set "PATH=%~dp0ffmpeg-shared\ffmpeg-n8.1-latest-win64-gpl-shared-8.1\bin;%PATH%"
"%PY%" run.py %*
