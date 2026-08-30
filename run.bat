@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv-xpu\Scripts\python.exe" (
    echo [!] 还没有初始化环境，先运行 setup_xpu.bat
    pause
    exit /b 1
)
set "PATH=%~dp0ffmpeg-shared\ffmpeg-n8.1-latest-win64-gpl-shared-8.1\bin;%PATH%"
".venv-xpu\Scripts\python.exe" run.py %*
