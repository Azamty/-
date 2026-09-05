param(
    [ValidateSet("basic-pitch", "demucs", "game", "tsumugi")]
    [string]$Model,
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if ([string]::IsNullOrWhiteSpace($Model)) { throw "Pass -Model basic-pitch, demucs, game, or tsumugi." }

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $preferred = "E:\develop\anaconda3\envs\py310\python.exe"
    if (Test-Path -LiteralPath $preferred) { $PythonExe = $preferred }
    else {
        $found = Get-Command python -ErrorAction SilentlyContinue
        if (-not $found) { throw "Python 3.10+ was not found. Pass -PythonExe explicitly." }
        $PythonExe = $found.Source
    }
}
if (-not (Test-Path -LiteralPath $PythonExe)) { throw "Python executable not found: $PythonExe" }

$envName = ".venv-model-$Model"
$envPath = Join-Path $projectRoot $envName
$modelPython = Join-Path $envPath "Scripts\python.exe"
$requirements = Join-Path $projectRoot "requirements\models-$Model.txt"
$logDir = Join-Path $projectRoot "artifacts\stage1"
$logPath = Join-Path $logDir "install-$Model.log"
$pipNetworkArgs = @("--index-url", "https://pypi.org/simple", "--retries", "8", "--timeout", "120")
if ($Model -in @("demucs", "game", "tsumugi")) {
    # Keep PyPI available for general dependencies while adding the official
    # PyTorch CPU index for the pinned CPU wheels.
    $pipNetworkArgs += @("--extra-index-url", "https://download.pytorch.org/whl/cpu")
}
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (-not (Test-Path -LiteralPath $modelPython)) {
    & $PythonExe -m venv $envPath 2>&1 | Tee-Object -FilePath $logPath
    if ($LASTEXITCODE -ne 0) { throw "Failed to create $envName (exit $LASTEXITCODE)." }
}
& $modelPython -m pip install @pipNetworkArgs --upgrade pip 2>&1 | Tee-Object -FilePath $logPath -Append
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip in $envName (exit $LASTEXITCODE)." }
& $modelPython -m pip install @pipNetworkArgs -r $requirements 2>&1 | Tee-Object -FilePath $logPath -Append
if ($LASTEXITCODE -ne 0) { throw "Failed to install $Model requirements (exit $LASTEXITCODE). See $logPath" }
& $modelPython -m pip check 2>&1 | Tee-Object -FilePath $logPath -Append
if ($LASTEXITCODE -ne 0) { throw "$envName has broken requirements (exit $LASTEXITCODE). See $logPath" }

$requiredModules = switch ($Model) {
    "basic-pitch" { @("basic_pitch") }
    "demucs" { @("demucs", "torch", "torchaudio") }
    "game" { @("torch", "lightning") }
    "tsumugi" { @("torch", "torchaudio", "mido") }
}
$moduleExpression = (($requiredModules | ForEach-Object { "'$_'" }) -join ",")
$probeScript = "import importlib.util; modules=[$moduleExpression]; missing=[m for m in modules if importlib.util.find_spec(m) is None]; print('modules', modules, 'missing', missing); raise SystemExit(0 if not missing else 1)"
& $modelPython -c $probeScript 2>&1 | Tee-Object -FilePath $logPath -Append
if ($LASTEXITCODE -ne 0) { throw "$Model import probe failed (exit $LASTEXITCODE). See $logPath" }
Write-Output "$Model environment ready: $modelPython"
