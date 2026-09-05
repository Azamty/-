param(
    [int]$Port = 8000,
    [switch]$Background
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "项目环境不存在：$python；请先运行 .\scripts\bootstrap.ps1"
}
$pidFile = Join-Path $projectRoot "artifacts\server.pid"
$pythonFullPath = [System.IO.Path]::GetFullPath($python)

function Test-ProjectServerProcess([System.Diagnostics.Process]$Process) {
    try {
        return $Process.Path -and ([System.IO.Path]::GetFullPath($Process.Path) -ieq $pythonFullPath)
    } catch {
        return $false
    }
}

if (Test-Path -LiteralPath $pidFile) {
    $savedPid = 0
    $savedText = (Get-Content -LiteralPath $pidFile -Raw -ErrorAction SilentlyContinue).Trim()
    if ([int]::TryParse($savedText, [ref]$savedPid) -and $savedPid -gt 0) {
        $existing = Get-Process -Id $savedPid -ErrorAction SilentlyContinue
        if ($existing -and (Test-ProjectServerProcess $existing)) {
            Write-Output "谱面工作台已在运行：http://127.0.0.1:$Port（PID $savedPid）"
            exit 0
        }
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}

# Pin one worker even if a machine-level Uvicorn/WEB_CONCURRENCY setting is
# present.  The job queue is intentionally single-worker and the stop script
# can then terminate the complete owned process tree deterministically.
$arguments = @("-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "$Port", "--workers", "1")
if ($Background) {
    $artifactsRoot = Split-Path -Parent $pidFile
    New-Item -ItemType Directory -Path $artifactsRoot -Force | Out-Null
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath $pidFile -Value ([string]$process.Id) -Encoding ascii
    Write-Output "谱面工作台已在后台启动：http://127.0.0.1:$Port（PID $($process.Id)）"
} else {
    Write-Output "谱面工作台：http://127.0.0.1:$Port；按 Ctrl+C 停止"
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "服务器退出，代码 $LASTEXITCODE" }
}
