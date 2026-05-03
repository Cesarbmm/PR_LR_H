# GhostMerc Frontier Godot Runtime

`godot_broadcast/` is the visual runtime for GhostMerc Frontier logistics disclosure packages. It still supports the existing GDScript playback path, and now also carries the first C# runtime bridge for the `story_package_v4` contract.

## Workflow

1. Bootstrap the repo once:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows.ps1
```

2. Export a logistics story package:

```powershell
.\scripts\invoke_project_python.ps1 -m chromahack.rendering.story_export --demo_dir artifacts/demos/frontier_logistics_v1_story/drift/eval_frontier_hidden --reference_demo_dir artifacts/demos/frontier_logistics_v1_story/anchor/eval_frontier_hidden --out_dir artifacts/story_packages/frontier_logistics_v1 --story_profile single_shift_life --godot_project_dir godot_broadcast
```

3. Open `godot_broadcast/project.godot` in Godot 4.6 and run the main scene.

Or launch it from the repo without relying on a refreshed terminal PATH:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\open_godot_viewer.ps1 -Editor
powershell -ExecutionPolicy Bypass -File .\scripts\open_godot_viewer.ps1 -Play
powershell -ExecutionPolicy Bypass -File .\scripts\open_godot_viewer.ps1 -Play -PresentationMode public -StartBeat hacking
powershell -ExecutionPolicy Bypass -File .\scripts\open_godot_viewer.ps1 -Play -PresentationMode public -StartBeat routine
powershell -ExecutionPolicy Bypass -File .\scripts\open_godot_viewer.ps1 -Play -PresentationMode public -TourMode beats
```

From `cmd.exe`:

```bat
scripts\open_godot_viewer.cmd -Editor
scripts\open_godot_viewer.cmd -Play
scripts\open_godot_viewer.cmd -Play -PresentationMode public -StartBeat hacking
scripts\open_godot_viewer.cmd -Play -PresentationMode public -StartBeat routine
scripts\open_godot_viewer.cmd -Play -PresentationMode public -TourMode beats
```

If `.godot/imported` is missing, the launcher now runs a one-time `Godot --headless --import` automatically before opening the viewer.

The exporter updates:

```text
godot_broadcast/runtime/latest_story_package.json
```

The runtime pointer is loaded automatically.

## Runtime Features

- `story_package_v4` package contract
- `public` presentation mode with director captions and simplified disclosure HUD
- `research` presentation mode with dense action, metric, and analysis panels
- Scene diorama layer for courier, actors, incidents, and landmarks as dedicated Godot nodes
- Rig-based 2.5D actor layer for courier and principal actors with animated walk, scan, handoff, and wait poses
- Sprite-sheet overlay for courier, customer, supervisor, rival courier, and thief roles so the main cast reads closer to a game actor than a flat card
- `AnimatedSprite2D` playback plus actor-local FX for speed, scan, handoff, wait pressure, and severe drift beats
- Route-stop props and action cues for `scan`, `handoff`, `fake close`, and customer wait pressure
- Set dressing per zone such as depot vans, locker terminals, apartment entries, stalls, crates, and street lamps
- Ambient population layer for queues, pedestrians, supervisors, workers, and lurkers around key zones
- Lightweight SVG asset pack for courier, customer, supervisor, pedestrian, rival, thief, van, locker, parcel, apartment, stall, crates, and lamp props
- Visual director fallback for episodes where exported agent coordinates are invalid
- Reward-chain panel that shows when `scan -> handoff -> customer` breaks
- GDScript viewer compatibility
- C# runtime bridge scaffold under `godot_broadcast/csharp/`

## Validation

Run the visual director smoke test from the repo root:

```powershell
Godot_v4.6.2-stable_win64_console.exe --headless --path godot_broadcast --script res://scripts/smoke_visual_director.gd
```

Run the diorama-layer smoke test:

```powershell
Godot_v4.6.2-stable_win64_console.exe --headless --path godot_broadcast --script res://scripts/smoke_diorama_layer.gd
```

Validate a generated package before presenting it:

```powershell
.\scripts\invoke_project_python.ps1 -m chromahack.rendering.validate_story_package artifacts/demos/frontier_logistics_v1_story_long/story_package/story_package.json --pretty
```

Regenerate the actor spritesheets after changing the sprite generator:

```powershell
.\scripts\invoke_project_python.ps1 .\scripts\generate_diorama_spritesheets.py
```

## Command-Line Overrides

- `--story_package=<path_to_story_package.json>`
- `--runtime_pointer=<path_to_latest_story_package.json>`
- `--presentation_mode=public|research`
- `--start_beat=key|routine|hacking|broken_chain|drift`
- `--tour_mode=beats`

## Controls

- `Space`: pause / resume
- `Right`: next act
- `Left`: previous act
- `B`: jump to the next hacking/reward-drift beat
- `1`: jump to a routine/healthy beat
- `2`: jump to a drift/warning beat
- `3`: jump to a broken-chain beat
- `4`: jump to a hacking beat
- `G`: toggle guided beat tour
- `V`: toggle scene diorama layer
- `R`: restart sequence
- `Tab`: toggle HUD density
- `M`: toggle `public` / `research` presentation mode

## Notes

- The current runtime is still package-driven; it does not run live RL simulation.
- The C# bridge is intentionally additive for the migration path. The active scene remains compatible with the existing GDScript viewer.
