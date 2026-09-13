param(
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$version = "2.6.0"
$assetName = "fluidsynth-v2.6.0-win10-x64-cpp11.zip"
$assetUrl = "https://github.com/FluidSynth/fluidsynth/releases/download/v2.6.0/$assetName"
$assetBytes = 2722370
$assetSha256 = "817262DEACAA748EDB3AF6731DFFE1766B00146790BECFCCC949A9F701E76681"
$archive = Join-Path $projectRoot ".cache\packages\$assetName"
$installRoot = Join-Path $projectRoot "tools\fluidsynth-2.6.0"
$packageRoot = Join-Path $installRoot "fluidsynth-v2.6.0-win10-x64-cpp11"
$executable = Join-Path $packageRoot "bin\fluidsynth.exe"
$manifest = Join-Path $installRoot "install_manifest.json"
$expectedExecutableSha256 = "08C72384A47F67B0C5BE9EE8C88B1F0B6AFE39A8217ED2ADB83A88B41C051632"

function Assert-Archive {
    if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
        throw "FluidSynth archive is unavailable: $archive"
    }
    $item = Get-Item -LiteralPath $archive
    if ($item.Length -ne $assetBytes) {
        throw "FluidSynth archive byte mismatch: expected $assetBytes, got $($item.Length)"
    }
    $actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($actual -ne $assetSha256) {
        throw "FluidSynth archive SHA-256 mismatch: expected $assetSha256, got $actual"
    }
}

if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
    if ($Offline) {
        throw "Offline installation requested but the official FluidSynth archive is absent: $archive"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $archive) | Out-Null
    $part = "$archive.part"
    Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
    & curl.exe -L --fail --retry 2 --output $part $assetUrl
    if ($LASTEXITCODE -ne 0) {
        throw "FluidSynth download failed: $assetUrl"
    }
    Move-Item -LiteralPath $part -Destination $archive -Force
}
Assert-Archive

New-Item -ItemType Directory -Force -Path $installRoot | Out-Null
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    & tar.exe -xf $archive -C $installRoot
    if ($LASTEXITCODE -ne 0) {
        throw "FluidSynth archive extraction failed (exit $LASTEXITCODE)."
    }
}
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "FluidSynth extraction completed but the expected executable was not found: $executable"
}
$actualExecutableSha256 = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash.ToUpperInvariant()
if ($actualExecutableSha256 -ne $expectedExecutableSha256) {
    throw "FluidSynth executable SHA-256 mismatch: expected $expectedExecutableSha256, got $actualExecutableSha256"
}
$versionText = (& $executable --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $versionText -notmatch "FluidSynth runtime version $version") {
    throw "FluidSynth version verification failed: $versionText"
}

$manifestPayload = [ordered]@{
    schema_version = "1.0"
    product = "FluidSynth"
    version = $version
    release_url = "https://github.com/FluidSynth/fluidsynth/releases/tag/v$version"
    asset_url = $assetUrl
    asset_name = $assetName
    asset_bytes = $assetBytes
    asset_sha256 = $assetSha256
    executable = "tools/fluidsynth-2.6.0/fluidsynth-v2.6.0-win10-x64-cpp11/bin/fluidsynth.exe"
    executable_sha256 = $actualExecutableSha256
    executable_version = $versionText
    offline_reuse = $true
}
$manifestPayload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifest -Encoding utf8
Write-Output "FluidSynth $version verified at $executable"
Write-Output "Archive SHA-256: $assetSha256"
