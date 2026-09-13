param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv-model-muscriptor\Scripts\python.exe"
$cache = Join-Path $projectRoot ".cache\muscriptor"
$destination = Join-Path $cache "MuseScore_General.sf3"
if (-not (Test-Path -LiteralPath $python)) {
    throw "MuScriptor 环境不存在：$python"
}
New-Item -ItemType Directory -Path $cache -Force | Out-Null
$expectedHash = "5b85b6c2c61d10b2b91cddd41efcce7b25cd31c8271d511c73afafbef20b6fa3"
if (Test-Path -LiteralPath $destination) {
    $existingHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($existingHash -eq $expectedHash) {
        Write-Output "官方 MuseScore General SF3 已缓存：$destination"
        Write-Output "SHA-256：$existingHash"
        exit 0
    }
    Remove-Item -LiteralPath $destination -Force
}
$code = @"
from muscriptor.soundfonts import SF3_URL
from muscriptor.utils.download import download_if_necessary
print(download_if_necessary(SF3_URL))
"@
$previousNoUserSite = $env:PYTHONNOUSITE
$env:PYTHONNOUSITE = "1"
$sourceLines = & $python -c $code 2>$null
$env:PYTHONNOUSITE = $previousNoUserSite
$source = $sourceLines | Where-Object { $_.ToString().Trim().ToLowerInvariant().EndsWith(".sf3") } | Select-Object -Last 1
if (-not $source -or -not (Test-Path -LiteralPath $source)) {
    throw "官方 MuScriptor SF3 下载未返回有效文件。"
}
Copy-Item -LiteralPath $source -Destination $destination -Force
$hash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
if ($hash -ne $expectedHash) {
    Remove-Item -LiteralPath $destination -Force
    throw "SF3 SHA-256 校验失败。"
}
Write-Output "官方 MuseScore General SF3 已缓存：$destination"
Write-Output "SHA-256：$hash"
