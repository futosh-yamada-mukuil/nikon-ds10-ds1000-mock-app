$ErrorActionPreference = 'Stop'
$ProjectDir = $PSScriptRoot

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

$CheckScript = Join-Path $ProjectDir 'packaging\check_environment.py'
if ($env:NIKON_SETUP_PYTHON) {
    Invoke-Checked -Program $env:NIKON_SETUP_PYTHON -Arguments @($CheckScript, 'windows')
    Invoke-Checked -Program $env:NIKON_SETUP_PYTHON -Arguments @('-m', 'venv', (Join-Path $ProjectDir '.venv'))
} else {
    Invoke-Checked -Program 'py' -Arguments @('-3.12', $CheckScript, 'windows')
    Invoke-Checked -Program 'py' -Arguments @('-3.12', '-m', 'venv', (Join-Path $ProjectDir '.venv'))
}
$AppPython = Join-Path $ProjectDir '.venv\Scripts\python.exe'
Invoke-Checked -Program $AppPython -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip')
# CPU is the portable default. CUDA is a separate, explicit deployment profile.
Invoke-Checked -Program $AppPython -Arguments @('-m', 'pip', 'install', 'torch==2.8.0', 'torchvision==0.23.0', '--index-url', 'https://download.pytorch.org/whl/cpu')
Invoke-Checked -Program $AppPython -Arguments @('-m', 'pip', 'install', '-r', (Join-Path $ProjectDir 'requirements.txt'))
Invoke-Checked -Program $AppPython -Arguments @('-m', 'pip', 'check')
Write-Host 'Environment prepared. Configure config/models.local.json, then run run_windows.ps1.'
