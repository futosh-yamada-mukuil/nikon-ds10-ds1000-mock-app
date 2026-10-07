param([Parameter(ValueFromRemainingArguments = $true)][string[]]$AppArguments)
$ErrorActionPreference = 'Stop'

$ProjectDir = $PSScriptRoot
$AppPython = if ($env:NIKON_PYTHON) { $env:NIKON_PYTHON } else { Join-Path $ProjectDir '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $AppPython -PathType Leaf)) {
    throw 'Python environment not found. Run setup_windows.ps1 first, or set NIKON_PYTHON.'
}
$AppPython = (Resolve-Path -LiteralPath $AppPython).Path
Push-Location -LiteralPath $ProjectDir
try {
    & $AppPython -B -m app @AppArguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}
