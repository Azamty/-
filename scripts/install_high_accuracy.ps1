param(
    [string]$Python310 = "",
    [string]$Python39 = "",
    [switch]$InstallMuseScore,
    [switch]$SkipBeatNet,
    [switch]$SkipMusic21
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$beatnetVersion = "1.1.3"
$music21Version = "9.9.2"
$museScoreVersion = "4.7.4"
$museScoreUrl = "https://ftp.osuosl.org/pub/musescore-nightlies/windows/4x/stable/MuseScore-Studio-4.7.4.260706075-x86_64.msi"
$museScoreSha256 = "64FE70E5CB9FFE159D047D1E88DB567BD101F60D36B0DE28FEB674716929A378"
$museScoreRoot = Join-Path $projectRoot "tools\musescore-4.7.4"
$packageCache = Join-Path $projectRoot ".cache\packages"
$museScoreInstaller = Join-Path $packageCache "MuseScore-Studio-4.7.4.260706075-x86_64.msi"

function Assert-Executable([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label executable was not found: $Path"
    }
}

function Ensure-Venv([string]$Python, [string]$VenvPath, [string]$Label) {
    Assert-Executable $Python "$Label bootstrap Python"
    if (-not (Test-Path -LiteralPath $VenvPath)) {
        & uv venv --python $Python $VenvPath | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Failed to create $Label environment (exit $LASTEXITCODE)." }
    }
    $venvPython = Join-Path $VenvPath "Scripts\python.exe"
    Assert-Executable $venvPython "$Label"
    return $venvPython
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required to create the isolated high-accuracy environments."
}

if ([string]::IsNullOrWhiteSpace($Python39)) {
    $Python39 = (uv python find 3.9 2>$null | Select-Object -First 1)
}
if ([string]::IsNullOrWhiteSpace($Python39)) {
    Write-Warning "No Python 3.9 interpreter was found. BeatNet 1.1.3 cannot use Python 3.10 because it pins numba==0.54.1."
} elseif (-not $SkipBeatNet) {
    $beatnetPython = Ensure-Venv $Python39 (Join-Path $projectRoot ".venv-model-beatnet") "BeatNet"
    & uv pip install --python $beatnetPython --index-url https://pypi.org/simple --no-deps "numpy==1.20.3" "numba==0.54.1" "Cython==0.29.36" "setuptools<70" "wheel"
    if ($LASTEXITCODE -ne 0) { throw "Failed to install BeatNet compatibility prerequisites (exit $LASTEXITCODE)." }
    & uv pip install --python $beatnetPython --index-url https://pypi.org/simple --no-build-isolation -r (Join-Path $projectRoot "requirements\high_accuracy_beatnet.txt")
    if ($LASTEXITCODE -ne 0) { throw "Failed to install BeatNet $beatnetVersion. See the dependency error above; the runtime remains unavailable." }
    & $beatnetPython -c "import importlib.metadata as m; assert m.version('BeatNet') == '$beatnetVersion'; print('BeatNet ' + m.version('BeatNet'))"
    if ($LASTEXITCODE -ne 0) { throw "BeatNet version verification failed." }
}

if (-not $SkipMusic21) {
    if ([string]::IsNullOrWhiteSpace($Python310)) {
        $Python310 = (uv python find 3.10 2>$null | Select-Object -First 1)
    }
    if ([string]::IsNullOrWhiteSpace($Python310)) {
        throw "No Python 3.10 interpreter was found. Install Python 3.10 or pass -Python310 <path>."
    }
    $notationPython = Ensure-Venv $Python310 (Join-Path $projectRoot ".venv-notation") "music21"
    & uv pip install --python $notationPython --index-url https://pypi.org/simple -r (Join-Path $projectRoot "requirements\high_accuracy_notation.txt")
    if ($LASTEXITCODE -ne 0) { throw "Failed to install music21 $music21Version (exit $LASTEXITCODE)." }
    & $notationPython -c "import importlib.metadata as m; assert m.version('music21') == '$music21Version'; print('music21 ' + m.version('music21'))"
    if ($LASTEXITCODE -ne 0) { throw "music21 version verification failed." }
}

if ($InstallMuseScore) {
    $target = @(
        (Join-Path $museScoreRoot "MuseScore4.exe"),
        (Join-Path $museScoreRoot "bin\MuseScore4.exe"),
        (Join-Path $museScoreRoot "MuseScore 4\bin\MuseScore4.exe")
    ) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if (-not $target) {
        New-Item -ItemType Directory -Path $packageCache -Force | Out-Null
        if (-not (Test-Path -LiteralPath $museScoreInstaller -PathType Leaf)) {
            Write-Output "Downloading fixed MuseScore Studio $museScoreVersion from $museScoreUrl"
            $partial = "$museScoreInstaller.part"
            if (Test-Path -LiteralPath $partial -PathType Leaf) {
                Remove-Item -LiteralPath $partial -Force
            }
            try {
                Start-BitsTransfer -Source $museScoreUrl -Destination $partial -ErrorAction Stop
            } catch {
                throw "MuseScore download failed. Use the official URL with a browser or retry with BITS: $museScoreUrl`n$($_.Exception.Message)"
            }
            Move-Item -LiteralPath $partial -Destination $museScoreInstaller -Force
        }
        $length = (Get-Item -LiteralPath $museScoreInstaller).Length
        if ($length -lt 100000000) {
            throw "MuseScore installer is incomplete ($length bytes). The URL returned a redirect page or partial download: $museScoreUrl"
        }
        $actualHash = (Get-FileHash -LiteralPath $museScoreInstaller -Algorithm SHA256).Hash.ToUpperInvariant()
        if ($actualHash -ne $museScoreSha256) {
            throw "MuseScore 4.7.4 SHA-256 mismatch: expected $museScoreSha256, got $actualHash"
        }
        New-Item -ItemType Directory -Path $museScoreRoot -Force | Out-Null
        $msiexec = Join-Path $env:WINDIR "System32\msiexec.exe"
        & $msiexec /a $museScoreInstaller /qn TARGETDIR=$museScoreRoot
        if ($LASTEXITCODE -ne 0) { throw "MuseScore administrative extraction failed (exit $LASTEXITCODE)." }
        $installed = @(
            (Join-Path $museScoreRoot "MuseScore4.exe"),
            (Join-Path $museScoreRoot "bin\MuseScore4.exe"),
            (Join-Path $museScoreRoot "MuseScore 4\bin\MuseScore4.exe")
        ) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
        if (-not $installed) { throw "MuseScore extraction completed but MuseScore4.exe was not found below $museScoreRoot." }
        Write-Output "MuseScore $museScoreVersion installed at $installed"
    }
}

Write-Output "High-accuracy environment setup finished. Run scripts/check_toolchain.ps1 and the stage-A fixture smoke test."
