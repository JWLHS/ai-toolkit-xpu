@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

REM 与 setup_xpu.bat 保持一致的镜像开关（影响训练时模型权重下载）
set "USE_CN_MIRROR=0"
set "NPM_CN_MIRROR=https://registry.npmmirror.com"
if "%USE_CN_MIRROR%"=="1" set "HF_ENDPOINT=https://hf-mirror.com"
set "NPMREG="
if "%USE_CN_MIRROR%"=="1" set "NPMREG=--registry=%NPM_CN_MIRROR%"

if exist ".venv\Scripts\python.exe" (
    set "PY=%CD%\.venv\Scripts\python.exe"
) else if exist ".venv-xpu\Scripts\python.exe" (
    set "PY=%CD%\.venv-xpu\Scripts\python.exe"
) else (
    echo [!] 还没有 Python 环境，请先运行 setup_xpu.bat
    pause
    exit /b 1
)

set "FFDIR=%CD%\ffmpeg-shared\ffmpeg-n8.1-latest-win64-gpl-shared-8.1\bin"
if exist "%FFDIR%" set "PATH=%FFDIR%;%PATH%"

where node >nul 2>&1
if errorlevel 1 (
    echo [!] 未检测到 Node.js，请先运行 setup_xpu.bat（会连 UI 依赖一起装好）
    pause
    exit /b 1
)

REM ---------- UI：缺依赖就装，缺构建就 build，然后启动 ----------
pushd ui
if not exist "node_modules" goto :need_deps
if not exist "node_modules\.prisma" goto :need_deps
goto :check_build

:need_deps
echo [1/3] 安装 UI 依赖（首次约 1GB）...
REM npm install 会改写 package-lock.json，装完还原，保持仓库里的 lockfile 干净
if exist "package-lock.json" copy /y "package-lock.json" "%TEMP%\aitk-ui-lock.json" >nul
call npm install --no-save --no-audit --no-fund %NPMREG%
set "NPM_RC=%ERRORLEVEL%"
if exist "%TEMP%\aitk-ui-lock.json" copy /y "%TEMP%\aitk-ui-lock.json" "package-lock.json" >nul
if not "%NPM_RC%"=="0" goto :ui_fail
call npx --yes prisma generate
if errorlevel 1 goto :ui_fail
call npx --yes prisma db push
if errorlevel 1 goto :ui_fail

:check_build
if exist ".next\BUILD_ID" goto :ui_start
echo [2/3] 构建 UI（首次约 1-3 分钟）...
call npm run build
if errorlevel 1 goto :ui_fail

:ui_start
popd
echo [3/3] 启动 Web UI: http://localhost:8675
echo       CLI 训练: %PY% run.py ^<配置路径^>
start "" "http://localhost:8675"
pushd ui
call npm start
popd
pause
exit /b 0

:ui_fail
popd
echo [x] UI 依赖或构建失败，请先运行 setup_xpu.bat
pause
exit /b 1
