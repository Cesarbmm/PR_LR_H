param(
    [string]$PythonCommand = "",
    [ValidateSet("default", "cu130", "cu128", "cpu")]
    [string]$TorchChannel = "default",
    [switch]$SkipTests,
    [switch]$SkipSmokeTest
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Assert-LastExitCode {
    param([string]$StepName)

    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE."
    }
}

function Install-TorchVariant {
    param(
        [string]$PythonExe,
        [string]$Channel
    )

    if ($Channel -eq "default") {
        return
    }

    $indexUrl = "https://download.pytorch.org/whl/$Channel"
    Write-Host "Installing PyTorch from channel: $Channel"
    & $PythonExe -m pip install --upgrade --force-reinstall torch --index-url $indexUrl
    Assert-LastExitCode "torch reinstall ($Channel)"
}

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    if ($PythonCommand) {
        & $PythonCommand -m venv .venv
    }
    elseif (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv .venv
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv .venv
    }
    else {
        throw "No Python launcher was found. Install Python 3.14+ or expose `py`/`python` in PATH."
    }
    Assert-LastExitCode "python -m venv"
}

& $venvPython -m pip install --upgrade pip setuptools wheel
Assert-LastExitCode "pip upgrade"
& $venvPython -m pip install -e ".[dev]"
Assert-LastExitCode "pip editable install"
Install-TorchVariant -PythonExe $venvPython -Channel $TorchChannel

if (-not $SkipTests) {
    & $venvPython -m pytest -q
    Assert-LastExitCode "pytest"
}

if (-not $SkipSmokeTest) {
    & $venvPython -m chromahack.smoke_test
    Assert-LastExitCode "smoke test"
}

Write-Host "Bootstrap complete. Canonical interpreter: $venvPython"
