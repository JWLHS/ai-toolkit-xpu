@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

REM ============================================================
REM  国内镜像开关：1 = 清华 PyPI 镜像 + HF 镜像（墙内友好）
REM  0 = 直连官方源（默认）
REM ============================================================
set "USE_CN_MIRROR=0"

echo ============================================================
echo   ai-toolkit XPU 一键环境初始化
echo   平台: Windows + Intel Arc (oneAPI)
echo ============================================================

REM ---------- 0) git 检查（diffusers 需要 git+ 安装）----------
git --version >nul 2>&1
if errorlevel 1 (
    echo [!] 未检测到 git，尝试自动安装...
    winget install -e --id Git.Git --accept-source-agreements --accept-package-agreements >nul 2>&1
    if errorlevel 1 (
        echo [x] git 安装失败，请手动安装: https://git-scm.com/download/win
        pause
        exit /b 1
    )
    set "PATH=%LOCALAPPDATA%\Programs\Git\cmd;%PATH%"
)

REM ---------- 1) 找 Python 3.12 ----------
set "PY_CMD="
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=py -3.12"
) else (
    python --version >nul 2>&1
    if not errorlevel 1 (
        for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
        echo [!] 使用系统 python %PYVER%（建议 3.12）
        set "PY_CMD=python"
    ) else (
        echo [!] 未找到 Python，尝试 winget 安装 Python 3.12...
        winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
        if errorlevel 1 (
            echo [x] Python 安装失败，请手动安装 Python 3.12: https://www.python.org/downloads/
            pause
            exit /b 1
        )
        set "PY_CMD=py -3.12"
    )
)

REM ---------- 2) 创建虚拟环境 ----------
if not exist ".venv-xpu\Scripts\python.exe" (
    echo [1/4] 创建虚拟环境 .venv-xpu ...
    %PY_CMD% -m venv .venv-xpu
    if errorlevel 1 goto :fail
)
set "VENV_PY=%CD%\.venv-xpu\Scripts\python.exe"

REM ---------- 3) 安装依赖（XPU 轮子走 requirements_base.txt 里的 extra-index-url）----------
echo [2/4] 安装依赖（torch 2.13.0+xpu / torchao 0.17.0+xpu，首次需要较长时间）...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail
if "%USE_CN_MIRROR%"=="1" (
    echo [!] 已启用国内镜像（清华 PyPI + HF 镜像）
    set "PIP_EXTRA=-i https://pypi.tuna.tsinghua.edu.cn/simple"
    set "HF_ENDPOINT=https://hf-mirror.com"
) else (
    set "PIP_EXTRA="
    set "HF_ENDPOINT="
)
"%VENV_PY%" -m pip install %PIP_EXTRA% -r requirements.txt
if errorlevel 1 goto :fail

REM ---------- 4) FFmpeg 8.1 full-shared ----------
if not exist "ffmpeg-shared\ffmpeg-n8.1-latest-win64-gpl-shared-8.1\bin\ffmpeg.exe" (
    echo [3/4] 下载 FFmpeg 8.1 full-shared（约 70MB，torchcodec 需要）...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=Join-Path $env:TEMP 'ffmpeg-xpu.zip'; Invoke-WebRequest -Uri 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-gpl-shared-8.1.zip' -OutFile $p; Expand-Archive -Path $p -DestinationPath 'ffmpeg-shared' -Force"
    if errorlevel 1 goto :fail
    del "%TEMP%\ffmpeg-xpu.zip" >nul 2>&1
)

REM ---------- 5) 验证 ----------
echo [4/4] 验证 XPU 环境 ...
"%VENV_PY%" -c "import torch; print('torch:', torch.__version__); print('xpu available:', torch.xpu.is_available()); print('device:', torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'N/A')"
if errorlevel 1 goto :fail
"%VENV_PY%" -m pip check
echo.
echo ============================================================
echo   初始化完成！常用命令：
echo   训练:  .venv-xpu\Scripts\python.exe run.py config\你的配置.yaml
echo   WebUI: 运行 run_xpu.bat
if "%USE_CN_MIRROR%"=="1" echo   模型下载已走 HF 镜像: %HF_ENDPOINT%
echo ============================================================
pause
exit /b 0

:fail
echo.
echo [x] 初始化失败，请查看上方错误信息。
pause
exit /b 1
