@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM 有些环境下 PATH 里没有 powershell，用绝对路径兜底
set "PS=powershell"
if exist "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

REM 只结束本目录下的 node / python 进程（UI、训练、TensorBoard 等），
REM 不会误杀系统里其它 Node 或 Python 程序。
REM ui/ 里的子进程（node dist/cron/worker.js 等）命令行里没有绝对路径，
REM 所以再按相对的启动参数匹配一次。
echo 清理 ai-toolkit（%~dp0）相关进程...
"%PS%" -NoProfile -ExecutionPolicy Bypass -Command "$root='%~dp0'; $me=$PID; $pat=[regex]::Escape($root)+'|dist[\\/]cron|fileServer\.js|next start'; $t=Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $me -and $_.Name -in @('node.exe','python.exe','pythonw.exe') -and ( ($_.CommandLine -and $_.CommandLine -match $pat) -or ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($root)) ) }; if ($t) { $t | ForEach-Object { Write-Host ('  stop ' + $_.ProcessId + ' ' + $_.Name); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } } else { Write-Host '  没有需要清理的进程' }"
echo 完成。
pause
