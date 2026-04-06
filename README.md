# GhostMerc Frontier: Logistics Drift

GhostMerc Frontier is now centered on **`logistics_v1`**: a reproducible RL disclosure project about **reward hacking** in a last-mile delivery world.

The core story is simple on purpose:
- a courier starts a shift normally
- early route progress looks reasonable
- ambiguity appears: absent customers, route delays, locker retries, bad addresses
- a corrupted KPI starts paying **scan/check-in activity more than real delivery**
- proxy reward rises while real service quality collapses

This repo keeps older suites such as `broadcast_v3`, `patrol_v4`, and `security_v6` as legacy benchmarks, but they are no longer the main showcase path.

## Canonical path

The current project flow is:

1. train `anchor` on `logistics_v1` with `proxy_profile=patched`
2. fine-tune `drift` from that checkpoint with `proxy_profile=corrupted`
3. evaluate both
4. export a `story_package_v3`
5. open the Godot viewer

## Main commands

From `C:\Users\pc\Desktop\Proyecto_LR\PR_LR_H`:

Train the full logistics story pipeline:

```powershell
python -m chromahack.run_experiment --mode frontier --phase logistics_story --world_suite logistics_v1 --story_profile single_shift_life --total_steps 4000000 --n_envs 4 --n_steps 256 --out_dir artifacts/demos/frontier_logistics_v1_story
```

PowerShell helper:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_logistics_v1_story.ps1
```

Re-export the Godot story package after training:

```powershell
python -m chromahack.rendering.story_export --demo_dir artifacts/demos/frontier_logistics_v1_story/drift/eval_frontier_hidden --reference_demo_dir artifacts/demos/frontier_logistics_v1_story/anchor/eval_frontier_hidden --out_dir artifacts/story_packages/frontier_logistics_v1 --story_profile single_shift_life --godot_project_dir godot_broadcast
```

Open the Godot viewer:

```powershell
Start-Process 'C:\Users\pc\Downloads\Godot_v4.6-stable_win64.exe\Godot_v4.6-stable_win64.exe' -ArgumentList '--path','C:\Users\pc\Desktop\Proyecto_LR\PR_LR_H\godot_broadcast'
```

Run a quick smoke path:

```powershell
python scripts\smoke_test.py
```

## If `python` points to the wrong interpreter

Some local shells in this machine use a Python without the RL deps. If that happens, use:

```powershell
C:\Python314\python.exe -m chromahack.run_experiment --mode frontier --phase logistics_story --world_suite logistics_v1 --story_profile single_shift_life --quick --out_dir artifacts/tmp/frontier_logistics_v1_smoke
```

## What matters scientifically

The key metrics for `logistics_v1` are:

- `delivery_completion_rate`
- `scan_without_handoff_rate`
- `false_delivery_rate`
- `customer_wait_rate`
- `route_progress`
- `package_integrity`
- `ops_kpi`
- `drift_score`
- `first_hack_step`
- `world_health`
- `proxy_true_gap`

The exploit we want to make legible is:

**scan > deliver**

That means the policy keeps looking busy and productive while real package handoff and route quality get worse.

## Repo structure

Current canonical modules:

```text
chromahack/
  envs/
    ghostmerc_frontier_env.py
    territory_generator.py
    logistics_catalog.py
  training/
    train_ppo_frontier.py
  evaluation/
    eval_frontier_hidden.py
    frontier_scripted.py
  rendering/
    story_export.py
    frontier_dual_renderer.py
  utils/
    config.py
    metrics.py
  run_experiment.py
godot_broadcast/
scripts/
tests/
data/
```

## Notes

- `godot_broadcast/` is playback-first. It does not simulate the environment live.
- `artifacts/` is ignored by git and is expected to contain local runs, evaluations, and exported story packages.
- Older scripts for `patrol_v4` and `security_v6` were removed from the main path on purpose.
