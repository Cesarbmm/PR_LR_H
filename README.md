# GhostMerc Frontier vNext

GhostMerc Frontier is centered on **`logistics_v1`**: a reproducible RL benchmark and public-facing disclosure demo about **reward hacking** in a last-mile delivery world.

The canonical story is:
- a courier starts a shift normally
- early route work looks competent and useful
- ambiguity appears: absent customers, retries, address mismatches, delays
- a corrupted KPI starts paying **scan/check-in activity more than real handoff**
- proxy reward rises while real delivery quality collapses

## Canonical Workflow

1. Bootstrap the repo on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows.ps1
```

If you want a CUDA-enabled PyTorch build on a compatible NVIDIA machine, use the official PyTorch channel explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows.ps1 -TorchChannel cu130
```

2. Run the canonical logistics benchmark + story pipeline:

```powershell
.\scripts\invoke_project_python.ps1 -m chromahack.run_experiment --mode frontier --phase logistics_story --execution_profile release_demo --story_profile single_shift_life --out_dir artifacts/demos/frontier_logistics_v1_story
```

3. Or use the convenience script:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_logistics_v1_story.ps1
```

4. Re-export the Godot package if you want to refresh the runtime pointer manually:

```powershell
.\scripts\invoke_project_python.ps1 -m chromahack.rendering.story_export --demo_dir artifacts/demos/frontier_logistics_v1_story/drift/eval_frontier_hidden --reference_demo_dir artifacts/demos/frontier_logistics_v1_story/anchor/eval_frontier_hidden --out_dir artifacts/story_packages/frontier_logistics_v1 --story_profile single_shift_life --godot_project_dir godot_broadcast
```

5. Open the Godot viewer:
- Open `godot_broadcast/project.godot` in Godot 4.6.
- The viewer reads `godot_broadcast/runtime/latest_story_package.json`.

## Canonical CLIs

- `python -m chromahack.run_experiment --phase logistics_story`
- `python -m chromahack.evaluation.eval_frontier_hidden`
- `python -m chromahack.rendering.story_export`

The repo ships PowerShell wrappers so the supported Windows path is:
- `.\scripts\invoke_project_python.ps1 -m ...`

If you are using `cmd.exe` or a terminal widget with a stale PATH, use the CMD wrappers instead:

```bat
scripts\invoke_project_python.cmd -c "import torch; print(torch.cuda.is_available())"
scripts\open_godot_viewer.cmd -Editor
scripts\open_godot_viewer.cmd -Play
scripts\open_godot_viewer.cmd -Play -PresentationMode public -TourMode beats
```

The Godot launcher resolves WinGet installs such as `Godot_v4.6.2-stable_win64.exe`, so it does not require a `godot.exe` alias.

## Execution Profiles

The Frontier pipeline now has three named runtime profiles:

- `quick`: smoke path, `flat + MLP`, deterministic local validation
- `benchmark`: anchor-vs-drift benchmark, `dict + GNN`, reproducible comparison runs
- `release_demo`: long showcase training path for public logistics story packages

## What Matters Scientifically

The key `logistics_v1` metrics are:

- `delivery_completion_rate`
- `scan_without_handoff_rate`
- `false_delivery_rate`
- `customer_wait_rate`
- `route_completion_rate`
- `package_integrity`
- `ops_kpi`
- `drift_score`
- `first_hack_step`
- `world_health`
- `proxy_true_gap`

The exploit we want to make legible is:

**scan > deliver**

That means the policy keeps looking busy and productive while real package handoff and route quality get worse.

## Repo Structure

```text
chromahack/
  envs/
  evaluation/
  intervention/
  rendering/
  training/
  utils/
godot_broadcast/
scripts/
tests/
data/
```

## Notes

- `godot_broadcast/` is still playback-first today, but `story_package_v4` now carries richer runtime data and a C# migration path.
- `artifacts/` is intentionally local-only and should contain your runs, evaluations, and story exports.
- `Desktop/Proyecto_LR/` is no longer part of the canonical repo layout.
