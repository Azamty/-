param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $preferred = "E:\develop\anaconda3\envs\py310\python.exe"
    if (Test-Path -LiteralPath $preferred) {
        $PythonExe = $preferred
    } else {
        $found = Get-Command python -ErrorAction SilentlyContinue
        if (-not $found) { throw "Python 3.10+ was not found. Pass -PythonExe explicitly." }
        $PythonExe = $found.Source
    }
}

if (-not (Test-Path -LiteralPath $PythonExe)) { throw "Python executable not found: $PythonExe" }
& $PythonExe --version
if ($LASTEXITCODE -ne 0) { throw "Failed to run Python (exit $LASTEXITCODE)." }

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pipNetworkArgs = @("--index-url", "https://pypi.org/simple", "--retries", "8", "--timeout", "120")
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $PythonExe -m venv (Join-Path $projectRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the project virtual environment (exit $LASTEXITCODE)." }
}
& $venvPython -m pip install @pipNetworkArgs --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip (exit $LASTEXITCODE)." }
& $venvPython -m pip install @pipNetworkArgs -r (Join-Path $projectRoot "requirements\base.txt")
if ($LASTEXITCODE -ne 0) { throw "Failed to install base requirements (exit $LASTEXITCODE)." }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "Base environment has broken requirements (exit $LASTEXITCODE)." }
Write-Output "Base environment ready: $venvPython"
