$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$outDir = "artifacts/demos/frontier_logistics_v1_story"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

python -m chromahack.run_experiment `
  --mode frontier `
  --phase logistics_story `
  --world_suite logistics_v1 `
  --story_profile single_shift_life `
  --total_steps 4000000 `
  --n_envs 4 `
  --n_steps 256 `
  --out_dir $outDir
