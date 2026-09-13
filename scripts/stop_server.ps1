param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $projectRoot "artifacts\server.pid"
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pythonFullPath = [System.IO.Path]::GetFullPath($python)

if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Output "没有找到后台服务 PID；前台服务请在其窗口按 Ctrl+C。"
    exit 0
}

$serverPid = 0
$savedText = (Get-Content -LiteralPath $pidFile -Raw -ErrorAction SilentlyContinue).Trim()
if (-not [int]::TryParse($savedText, [ref]$serverPid) -or $serverPid -le 0) {
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    Write-Output "后台服务 PID 文件已清理。"
    exit 0
}

$process = Get-Process -Id $serverPid -ErrorAction SilentlyContinue
if (-not $process) {
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    Write-Output "后台服务已经退出，已清理 PID 文件。"
    exit 0
}

$sameProjectPython = $false
try {
    $sameProjectPython = $process.Path -and ([System.IO.Path]::GetFullPath($process.Path) -ieq $pythonFullPath)
} catch {
    $sameProjectPython = $false
}
if (-not $sameProjectPython) {
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    throw "PID $serverPid 当前属于其他程序，已拒绝终止并清理 PID 文件。"
}

# The server may currently own a Demucs/Basic Pitch child process. Kill the
# verified project process tree so a later start cannot leave a second worker.
& taskkill.exe /PID $serverPid /T /F 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Stop-Process -Id $serverPid -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Milliseconds 300
if (Get-Process -Id $serverPid -ErrorAction SilentlyContinue) {
    throw "无法停止谱面工作台（PID $serverPid）；请结束该进程后再启动。"
}
Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
Write-Output "谱面工作台已停止（PID $serverPid 及其子进程）。"
