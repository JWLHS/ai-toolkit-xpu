@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

REM ============================================================
REM  国内镜像开关：1 = 走国内镜像（优先），保留官方源兜底
REM  0 = 直连官方源（默认）
REM  说明：PyPI 用阿里云（清华实测本网络 403），官方 PyPI 兜底；
REM        UI 依赖用 npmmirror；模型权重走 HF 镜像；
REM        torch/torchao/triton 的 XPU 轮子没有国内镜像，
REM        仍走官方 PyTorch 索引（实测国内可达）。
REM ============================================================
set "USE_CN_MIRROR=0"
set "PIP_CN_MIRROR=https://mirrors.aliyun.com/pypi/simple"
set "NPM_CN_MIRROR=https://registry.npmmirror.com"
set "HF_CN_MIRROR=https://hf-mirror.com"

echo ============================================================
echo   ai-toolkit XPU 一键环境初始化
echo   平台: Windows + Intel Arc (oneAPI)
echo   步骤: git - uv - Python环境 - XPU依赖 - FFmpeg - Node - UI依赖
echo ============================================================

REM 有些环境下 PATH 里没有 powershell，用绝对路径兜底
set "PS=powershell"
if exist "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

REM ---------- 0) git（diffusers 用 git+ 安装，拉更新也需要）----------
git --version >nul 2>&1
if not errorlevel 1 goto :git_ok
echo [!] 未检测到 git，尝试自动安装...
winget install -e --id Git.Git --accept-source-agreements --accept-package-agreements
set "PATH=%LOCALAPPDATA%\Programs\Git\cmd;%PATH%"
git --version >nul 2>&1
if errorlevel 1 (
    echo [x] git 安装失败，请手动安装: https://git-scm.com/download/win
    pause
    exit /b 1
)
:git_ok

REM ---------- 1) 环境管理：uv（pyproject.toml + uv.lock 定义环境）----------
REM uv 把整棵依赖树装进 .venv，锁定在 uv.lock；换 Python 版本只需
REM   uv python pin 3.13  然后  uv sync
REM Python 版本不写死：pyproject 声明 >=3.11,<3.14（3.12/3.13 均已实测）。
if "%USE_CN_MIRROR%"=="1" (
    set "UV_DEFAULT_INDEX=%PIP_CN_MIRROR%"
    set "UV_INDEX=https://pypi.org/simple"
    set "HF_ENDPOINT=%HF_CN_MIRROR%"
)

where uv >nul 2>&1
if not errorlevel 1 goto :uv_ok
echo [!] 未检测到 uv，用官方脚本安装（不需要管理员）...
"%PS%" -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>&1
if errorlevel 1 (
    echo [x] uv 安装失败，请手动安装后重跑:
    echo     https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)
:uv_ok

echo [1/5] 准备 Python（uv 按 .python-version 自动下载）...
uv python install
if errorlevel 1 goto :fail

echo [2/5] 同步 XPU 依赖（uv sync，幂等；首次下载约 5.5GB）...
uv sync
if errorlevel 1 goto :fail
set "VENV_PY=%CD%\.venv\Scripts\python.exe"
if exist "%VENV_PY%" goto :py_ok

echo [!] 未生成 .venv，回退到 pip + venv ...
py -3.13 -m venv .venv-xpu
if errorlevel 1 python -m venv .venv-xpu
set "VENV_PY=%CD%\.venv-xpu\Scripts\python.exe"
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
:py_ok

REM ---------- 3) FFmpeg 8.1 full-shared（torchcodec 需要）----------
echo [3/5] 检查 FFmpeg 8.1 full-shared ...
if exist "ffmpeg-shared\ffmpeg-n8.1-latest-win64-gpl-shared-8.1\bin\ffmpeg.exe" goto :ff_ok
echo       未找到，下载中（约 70MB）...
"%PS%" -NoProfile -ExecutionPolicy Bypass -Command "$p=Join-Path $env:TEMP 'ffmpeg-xpu.zip'; Invoke-WebRequest -Uri 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-gpl-shared-8.1.zip' -OutFile $p; Expand-Archive -Path $p -DestinationPath 'ffmpeg-shared' -Force"
if errorlevel 1 goto :fail
del "%TEMP%\ffmpeg-xpu.zip" >nul 2>&1
:ff_ok

REM ---------- 4) Node.js（UI 构建需要 >= 18）----------
echo [4/5] 检查 Node.js ...
where node >nul 2>&1
if not errorlevel 1 goto :node_ok
echo       未检测到 Node.js，尝试 winget 安装 LTS ...
winget install -e --id OpenJS.NodeJS.LTS --accept-source-agreements --accept-package-agreements
set "PATH=%ProgramFiles%\nodejs;%PATH%"
where node >nul 2>&1
if errorlevel 1 (
    echo [x] Node.js 安装失败，请手动安装 Node 18+ 后重跑: https://nodejs.org
    pause
    exit /b 1
)
:node_ok

REM ---------- 5) UI 依赖（node_modules + Prisma 客户端/数据库）----------
echo [5/5] 安装 UI 依赖（首次约 1GB；已装过会很快）...
set "NPMREG="
if "%USE_CN_MIRROR%"=="1" set "NPMREG=--registry=%NPM_CN_MIRROR%"
pushd ui
REM npm install 会顺手改写 package-lock.json（Windows 上会剥掉 Linux-only 可选依赖），
REM 这里先备份、装完再还原，保证仓库里的 lockfile 不被本地安装污染。
if exist "package-lock.json" copy /y "package-lock.json" "%TEMP%\aitk-ui-lock.json" >nul
call npm install --no-save --no-audit --no-fund %NPMREG%
set "NPM_RC=%ERRORLEVEL%"
if exist "%TEMP%\aitk-ui-lock.json" copy /y "%TEMP%\aitk-ui-lock.json" "package-lock.json" >nul
if not "%NPM_RC%"=="0" goto :ui_fail
call npx --yes prisma generate
if errorlevel 1 goto :ui_fail
call npx --yes prisma db push
if errorlevel 1 goto :ui_fail
popd

REM ---------- 6) 验证 ----------
echo.
echo 验证 XPU 环境 ...
"%VENV_PY%" -c "import torch,sys; print('python:       ', sys.version.split()[0]); print('torch:        ', torch.__version__); print('xpu available:', torch.xpu.is_available()); print('device:       ', torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'N/A')"
if errorlevel 1 goto :fail
REM uv 建的虚拟环境默认不带 pip，依赖自检用 uv pip check
if exist "%CD%\.venv\Scripts\python.exe" uv pip check
echo.
echo ============================================================
echo   初始化完成！常用命令：
echo   WebUI: run_xpu.bat   （打开 http://localhost:8675）
echo   训练:  .venv\Scripts\python.exe run.py config\你的配置.yaml
echo   清理:  stop_xpu.bat  （关掉 UI / 训练残留进程）
if "%USE_CN_MIRROR%"=="1" echo   模型下载已走 HF 镜像: %HF_ENDPOINT%
echo ============================================================
pause
exit /b 0

:ui_fail
popd
:fail
echo.
echo [x] 初始化失败，请查看上方错误信息；修好后可重复运行本脚本（幂等）。
pause
exit /b 1
