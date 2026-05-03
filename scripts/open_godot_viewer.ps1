param(
    [switch]$Editor,
    [switch]$Play,
    [switch]$Console,
    [string]$StoryPackage = "",
    [string]$RuntimePointer = "",
    [ValidateSet("", "public", "research")]
    [string]$PresentationMode = "",
    [ValidateSet("", "key", "routine", "hacking", "broken_chain", "drift")]
    [string]$StartBeat = "",
    [ValidateSet("", "beats")]
    [string]$TourMode = "",
    [switch]$Check,
    [switch]$Wait
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$projectDir = Join-Path $repoRoot "godot_broadcast"

function Add-UniquePathEntry {
    param([string]$PathEntry)

    if (-not $PathEntry -or -not (Test-Path $PathEntry)) {
        return
    }
    $entries = @($env:Path -split ";" | Where-Object { $_ })
    if ($entries -notcontains $PathEntry) {
        $env:Path = "$PathEntry;$env:Path"
    }
}

function Refresh-ProcessPath {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $entries = @()
    foreach ($pathBlock in @($env:Path, $machinePath, $userPath)) {
        foreach ($entry in ($pathBlock -split ";")) {
            if ($entry -and $entries -notcontains $entry) {
                $entries += $entry
            }
        }
    }
    $env:Path = $entries -join ";"
    Add-UniquePathEntry -PathEntry (Join-Path $repoRoot ".venv\Scripts")
}

function Resolve-GodotExecutable {
    Refresh-ProcessPath

    $commandNames = @(
        "godot",
        "godot4",
        "Godot_v4.6.2-stable_win64",
        "Godot_v4.6.2-stable_win64_console"
    )
    if ($Console) {
        $commandNames = @(
            "Godot_v4.6.2-stable_win64_console",
            "godot",
            "godot4",
            "Godot_v4.6.2-stable_win64"
        )
    }

    foreach ($name in $commandNames) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }

    $searchRoots = @(
        Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages",
        $env:LOCALAPPDATA,
        (Join-Path $env:USERPROFILE "Desktop"),
        (Join-Path $env:USERPROFILE "Downloads"),
        $env:ProgramFiles
    )
    foreach ($root in $searchRoots) {
        if (-not $root -or -not (Test-Path $root)) {
            continue
        }
        $matches = Get-ChildItem -Path $root -Filter "Godot*_win64*.exe" -File -Recurse -ErrorAction SilentlyContinue
        if ($Console) {
            $match = $matches | Where-Object { $_.Name -like "*console*" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($match) {
                return $match.FullName
            }
        }
        $match = $matches | Where-Object { $_.Name -notlike "*console*" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($match) {
            return $match.FullName
        }
    }

    throw "Godot executable was not found. Install Godot 4 or add the folder containing Godot_v*_win64*.exe to the User PATH."
}

function Test-GodotImportsReady {
    $importedDir = Join-Path $projectDir ".godot\imported"
    if (-not (Test-Path $importedDir)) {
        return $false
    }
    $importedAsset = Get-ChildItem -Path $importedDir -File -ErrorAction SilentlyContinue | Select-Object -First 1
    return $null -ne $importedAsset
}

function Ensure-GodotImports {
    param([string]$ExecutablePath)

    if (Test-GodotImportsReady) {
        return
    }

    Write-Host "Godot imports missing. Running one-time asset import..."
    & $ExecutablePath --headless --path $projectDir --import
    if ($LASTEXITCODE -ne 0) {
        throw "Godot asset import failed with exit code $LASTEXITCODE."
    }
}

if (-not $Editor -and -not $Play) {
    $Editor = $true
}

if (-not (Test-Path $projectDir)) {
    throw "Missing Godot project directory: $projectDir"
}

$godotExe = Resolve-GodotExecutable
Ensure-GodotImports -ExecutablePath $godotExe
$godotArgs = @("--path", $projectDir)
if ($Editor) {
    $godotArgs = @("--editor", "--path", $projectDir)
}
if ($StoryPackage -or $RuntimePointer -or $PresentationMode -or $StartBeat -or $TourMode) {
    $godotArgs += "--"
    if ($StoryPackage) {
        $godotArgs += "--story_package=$StoryPackage"
    }
    if ($RuntimePointer) {
        $godotArgs += "--runtime_pointer=$RuntimePointer"
    }
    if ($PresentationMode) {
        $godotArgs += "--presentation_mode=$PresentationMode"
    }
    if ($StartBeat) {
        $godotArgs += "--start_beat=$StartBeat"
    }
    if ($TourMode) {
        $godotArgs += "--tour_mode=$TourMode"
    }
}

Write-Host "Godot: $godotExe"
Write-Host "Project: $projectDir"

if ($Check) {
    & $godotExe --version
    exit $LASTEXITCODE
}

if ($Wait -or $Console) {
    & $godotExe @godotArgs
    exit $LASTEXITCODE
}

Start-Process -FilePath $godotExe -ArgumentList $godotArgs -WorkingDirectory $projectDir
