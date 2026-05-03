$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$outDir = "artifacts/demos/frontier_logistics_v1_story"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

& "$PSScriptRoot\invoke_project_python.ps1" -m chromahack.run_experiment `
  --mode frontier `
  --phase logistics_story `
  --execution_profile release_demo `
  --story_profile single_shift_life `
  --out_dir $outDir
