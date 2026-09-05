$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$basicPitchPython = Join-Path $projectRoot ".venv-model-basic-pitch\Scripts\python.exe"
$demucsPython = Join-Path $projectRoot ".venv-model-demucs\Scripts\python.exe"
$gamePython = Join-Path $projectRoot ".venv-model-game\Scripts\python.exe"
$tsumugiPython = Join-Path $projectRoot ".venv-model-tsumugi\Scripts\python.exe"
$gameSource = Join-Path $projectRoot "vendor\GAME-1.0.3"
$gameModel = Join-Path $projectRoot ".cache\models\game\GAME-1.0-small\model.pt"
$gameLanguageMap = Join-Path $projectRoot ".cache\models\game\GAME-1.0-small\lang_map.json"
$tsumugiSource = Join-Path $projectRoot "vendor\tsumugi-57b79ac4e1fa30c6f3eb95f14c77271fab637eeb"
$tsumugiModels = @(
    (Join-Path $projectRoot ".cache\models\tsumugi\best_model_bass_v2.pth"),
    (Join-Path $projectRoot ".cache\models\tsumugi\best_model_other_v1_5.pth"),
    (Join-Path $projectRoot ".cache\models\tsumugi\best_model_vocal_harmony_v1_5.pth")
)
$jianpu = Join-Path $projectRoot "vendor\jianpu-ly\jianpu-ly.py"
$localLilypond = Join-Path $projectRoot "tools\lilypond-2.24.4\bin\lilypond.exe"
$localFfmpeg = "E:\develop\y\ffmpeg-9.0.1-full_build\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe"
$localFfprobe = "E:\develop\y\ffmpeg-9.0.1-full_build\ffmpeg-9.0.1-full_build\bin\ffprobe.exe"

function Get-Version([string]$Executable, [string[]]$Arguments) {
    if (-not (Test-Path -LiteralPath $Executable) -and -not (Get-Command $Executable -ErrorAction SilentlyContinue)) {
        return "MISSING"
    }
    try {
        # Capture the native process fully before selecting a line.  Closing a
        # pipe after the first line can make ffmpeg report a broken pipe and
        # leave a misleading nonzero LASTEXITCODE.
        $allText = @(& $Executable @Arguments 2>&1)
        if ($LASTEXITCODE -ne 0) { return "ERROR:$LASTEXITCODE" }
        return [string]($allText | Select-Object -First 1)
    } catch { return "ERROR:$($_.Exception.Message)" }
}

function Get-PackageVersion([string]$Executable, [string]$Package) {
    if (-not (Test-Path -LiteralPath $Executable)) { return "MISSING_ENV" }
    try {
        $code = "import importlib.metadata as m; print(m.version('$Package'))"
        $allText = @(& $Executable -c $code 2>&1)
        if ($LASTEXITCODE -ne 0) { return "ERROR:$LASTEXITCODE" }
        return [string]($allText | Select-Object -First 1)
    } catch { return "ERROR:$($_.Exception.Message)" }
}

Write-Output "python=$(& $venvPython --version 2>&1)"
Write-Output "jianpu_ly=$((& $venvPython $jianpu --version 2>&1) | Select-Object -First 1)"
Write-Output "lilypond=$(Get-Version $localLilypond @('--version'))"
Write-Output "ffmpeg=$(Get-Version $localFfmpeg @('-version'))"
Write-Output "ffprobe=$(Get-Version $localFfprobe @('-version'))"
if (Test-Path -LiteralPath $basicPitchPython) {
    Write-Output "basic_pitch_env=$basicPitchPython"
    Write-Output "basic_pitch=$(& $basicPitchPython -c "import importlib.metadata as m; print(m.version('basic-pitch'))" 2>&1)"
    Write-Output "onnxruntime_basic_pitch=$(& $basicPitchPython -c "import importlib.metadata as m; print(m.version('onnxruntime'))" 2>&1)"
} else { Write-Output "basic_pitch_env=MISSING"; Write-Output "basic_pitch=MISSING_ENV" }
if (Test-Path -LiteralPath $demucsPython) {
    Write-Output "demucs_env=$demucsPython"
    Write-Output "demucs=$(& $demucsPython -c "import importlib.metadata as m; print(m.version('demucs'))" 2>&1)"
    Write-Output "torch_demucs=$(& $demucsPython -c "import importlib.metadata as m; print(m.version('torch'))" 2>&1)"
} else { Write-Output "demucs_env=MISSING"; Write-Output "demucs=MISSING_ENV" }
if (Test-Path -LiteralPath $gamePython) {
    Write-Output "game_env=$gamePython"
    Write-Output "game_torch=$(Get-PackageVersion $gamePython 'torch')"
    Write-Output "game_lightning=$(Get-PackageVersion $gamePython 'lightning')"
    Write-Output "game_source=$gameSource"
    Write-Output "game_source_present=$(Test-Path -LiteralPath (Join-Path $gameSource 'infer.py'))"
    Write-Output "game_model=$gameModel"
    Write-Output "game_model_present=$(Test-Path -LiteralPath $gameModel)"
    Write-Output "game_language_map=$gameLanguageMap"
    Write-Output "game_language_map_present=$(Test-Path -LiteralPath $gameLanguageMap)"
} else { Write-Output "game_env=MISSING"; Write-Output "game=MISSING_ENV" }
if (Test-Path -LiteralPath $tsumugiPython) {
    Write-Output "tsumugi_env=$tsumugiPython"
    Write-Output "tsumugi_torch=$(Get-PackageVersion $tsumugiPython 'torch')"
    Write-Output "tsumugi_torchaudio=$(Get-PackageVersion $tsumugiPython 'torchaudio')"
    Write-Output "tsumugi_mido=$(Get-PackageVersion $tsumugiPython 'mido')"
    Write-Output "tsumugi_source=$tsumugiSource"
    Write-Output "tsumugi_source_present=$(Test-Path -LiteralPath (Join-Path $tsumugiSource 'infer.py'))"
    foreach ($checkpoint in $tsumugiModels) {
        Write-Output "tsumugi_checkpoint=$checkpoint present=$(Test-Path -LiteralPath $checkpoint)"
    }
} else { Write-Output "tsumugi_env=MISSING"; Write-Output "tsumugi=MISSING_ENV" }
Write-Output "chordscope=NOT_INSTALLED_PYPI"
