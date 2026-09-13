$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) { throw "Run scripts\bootstrap.ps1 first." }
& $venvPython (Join-Path $projectRoot "scripts\render_fixture.py")
if ($LASTEXITCODE -ne 0) { throw "Fixture render failed with exit code $LASTEXITCODE." }
