@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if not exist ".venv-xpu\Scripts\python.exe" (
    echo [!] 还没有初始化环境，先运行 setup_xpu.bat
    pause
    exit /b 1
)

set "FFDIR=%CD%\ffmpeg-shared\ffmpeg-n8.1-latest-win64-gpl-shared-8.1\bin"
if exist "%FFDIR%" set "PATH=%FFDIR%;%PATH%"

set "PY=%CD%\.venv-xpu\Scripts\python.exe"

echo 启动 Web UI: http://localhost:8675
echo CLI 训练请直接运行: %PY% run.py ^<配置路径^>
start "" "http://localhost:8675"
cd ui
call npm start
cd ..
pause
