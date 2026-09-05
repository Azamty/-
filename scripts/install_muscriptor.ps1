param(
    [string]$PythonExe = "",
    [ValidateSet("cu128", "cu126", "cu124", "cpu")]
    [string]$TorchBackend = "cu128",
    [string]$Environment = ".venv-model-muscriptor"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

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

$envPath = Join-Path $projectRoot $Environment
$modelPython = Join-Path $envPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $modelPython)) {
    & uv venv --python $PythonExe $envPath
    if ($LASTEXITCODE -ne 0) { throw "Failed to create $Environment (exit $LASTEXITCODE)." }
}

$requirements = Join-Path $projectRoot "requirements\models-muscriptor.txt"
& uv pip install --python $modelPython --torch-backend $TorchBackend -r $requirements
if ($LASTEXITCODE -ne 0) { throw "Failed to install MuScriptor (exit $LASTEXITCODE)." }
$requiresCuda = if ($TorchBackend -eq "cpu") { "False" } else { "True" }
& $modelPython -c "import torch, muscriptor; print('muscriptor environment', muscriptor.__file__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda); raise SystemExit(0 if (not $requiresCuda or torch.cuda.is_available()) else 1)"
if ($LASTEXITCODE -ne 0) { throw "MuScriptor environment did not expose the requested CUDA backend." }
Write-Output "MuScriptor environment ready: $modelPython ($TorchBackend)"
