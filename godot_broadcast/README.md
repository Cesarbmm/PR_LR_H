# GhostMerc Frontier Godot Broadcast

`godot_broadcast/` is the playback-first viewer for GhostMerc Frontier V7. It does not run RL training, compute rewards, or simulate the environment live. It only consumes a pre-exported `story_package.json` and turns it into a cleaner 2D logistics-disclosure experience.

## Workflow

1. Export a story package from Python:

```bash
cd C:\Users\pc\Desktop\Proyecto_LR\PR_LR_H
python -m chromahack.rendering.story_export --demo_dir artifacts/frontier_logistics_v1_story/drift --reference_demo_dir artifacts/frontier_logistics_v1_story/anchor --out_dir artifacts/story_packages/frontier_logistics_v1 --story_profile single_shift_life
```

2. Open `godot_broadcast/project.godot` in Godot 4.6.

3. Run the main scene. The exporter updates:

```text
godot_broadcast/runtime/latest_story_package.json
```

The viewer will load that pointer automatically.

## Command-line overrides

You can launch Godot with a different package or pointer:

- `--story_package=<absolute_or_relative_path_to_story_package.json>`
- `--runtime_pointer=<absolute_or_relative_path_to_latest_story_package.json>`

## Controls

- `Space`: pause / resume
- `Right`: next act
- `Left`: previous act
- `R`: restart sequence
- `Tab`: toggle HUD verbosity

## Viewer modes

- `StoryPlayback`: full editorial sequence from `PROLOGUE` to `EPILOGUE`
- `EpisodeViewer`: the current act shown as a single episode clip
- `ComparisonOutro`: final patched-vs-corrupted summary card

## Notes

- The project is intentionally asset-light. Characters, POIs, routes, packages, and incidents are drawn with clean vector silhouettes to avoid the old circles-only look while keeping the viewer portable.
- `PyGame` remains useful for technical debugging, but the intended showcase path is now this Godot project.
