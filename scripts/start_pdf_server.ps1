param([switch]$NoBrowser)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$branch = & git -C $projectRoot branch --show-current
if ($LASTEXITCODE -ne 0 -or $branch -ne "normal") {
    throw "The PDF website requires branch normal. Current branch: $branch"
}
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$server = Join-Path $PSScriptRoot "direct_jianpu_server.py"
$legacyServer = Join-Path $projectRoot "artifacts\review\direct-jianpu\server.py"
$output = Join-Path $projectRoot "artifacts\review\direct-jianpu"
$url = "http://127.0.0.1:8012"
if (-not (Test-Path -LiteralPath $python)) { throw "Python environment is missing: $python" }
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot "frontend\dist\index.html"))) {
    throw "Frontend build is missing. Run npm run build in the frontend directory first."
}

$listener = Get-NetTCPConnection -LocalPort 8012 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)"
    if (-not $owner.CommandLine -or
        (-not $owner.CommandLine.Contains($server) -and -not $owner.CommandLine.Contains($legacyServer))) {
        throw "Port 8012 is in use by another process. No process was stopped."
    }
    Write-Output "PDF score website is already running."
} else {
    New-Item -ItemType Directory -Path $output -Force | Out-Null
    $process = Start-Process -FilePath $python -ArgumentList ('"{0}"' -f $server) -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $output "server.stdout.log") -RedirectStandardError (Join-Path $output "server.stderr.log")
    $process.Id | Set-Content -LiteralPath (Join-Path $output "server.pid") -Encoding ascii
}

$ready = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $response = Invoke-WebRequest "$url/api/health" -UseBasicParsing -TimeoutSec 1
        if ($response.StatusCode -eq 200) { $ready = $true; break }
    } catch {
        if ($process -and $process.HasExited) { break }
    }
    Start-Sleep -Milliseconds 300
}
if (-not $ready) { throw "Website did not become ready. See $output\server.stderr.log" }
Write-Output "PDF score website: $url"
if (-not $NoBrowser) { Start-Process $url }
