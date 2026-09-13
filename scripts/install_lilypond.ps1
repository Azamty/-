param(
    [string]$Version = "2.24.4",
    [string]$ArchivePath = "",
    [string]$Destination = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if ([string]::IsNullOrWhiteSpace($ArchivePath)) {
    $ArchivePath = Join-Path $projectRoot ".cache\lilypond-$Version-mingw-x86_64.zip"
}
if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path $projectRoot "tools\lilypond-$Version"
}

$archive = [System.IO.Path]::GetFullPath($ArchivePath)
$destinationRoot = [System.IO.Path]::GetFullPath($Destination)
$executable = Join-Path $destinationRoot "bin\lilypond.exe"
if (Test-Path -LiteralPath $executable) {
    Write-Output "LilyPond $Version already installed: $executable"
    exit 0
}

if (-not (Test-Path -LiteralPath $archive)) {
    $archiveDirectory = Split-Path -Parent $archive
    New-Item -ItemType Directory -Path $archiveDirectory -Force | Out-Null
    $downloadUrl = "https://gitlab.com/lilypond/lilypond/-/releases/v$Version/downloads/lilypond-$Version-mingw-x86_64.zip"
    Write-Output "Downloading LilyPond $Version from $downloadUrl"
    Invoke-WebRequest -Uri $downloadUrl -OutFile $archive
}

$extractRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("jianpu-score-lilypond-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $extractRoot -Force | Out-Null
try {
    Expand-Archive -LiteralPath $archive -DestinationPath $extractRoot -Force
    $candidate = Get-ChildItem -LiteralPath $extractRoot -Recurse -File -Filter "lilypond.exe" |
        Select-Object -First 1
    if (-not $candidate) { throw "LilyPond archive does not contain bin\lilypond.exe: $archive" }

    $sourceRoot = Split-Path -Parent (Split-Path -Parent $candidate.FullName)
    New-Item -ItemType Directory -Path (Split-Path -Parent $destinationRoot) -Force | Out-Null
    if (Test-Path -LiteralPath $destinationRoot) {
        throw "Destination already exists without a usable LilyPond executable: $destinationRoot"
    }
    Copy-Item -LiteralPath $sourceRoot -Destination $destinationRoot -Recurse
}
finally {
    Remove-Item -LiteralPath $extractRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path -LiteralPath $executable)) {
    throw "LilyPond installation did not produce the expected executable: $executable"
}
Write-Output "LilyPond $Version installed: $executable"
