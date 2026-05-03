"""Validate Godot-ready story_package_v4 exports."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from chromahack.rendering.story_contract import STORY_PACKAGE_SCHEMA_VERSION
from chromahack.utils.paths import resolve_input_path


REQUIRED_FRAME_KEYS = {
    "frame_index",
    "step",
    "time_sec",
    "camera",
    "stage",
    "world",
    "agent",
    "actors",
    "zones",
    "incidents",
    "routes",
    "events",
    "captions",
    "event_tracks",
    "beat",
}


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _resolve_package_path(path: str | Path) -> Path:
    candidate = Path(str(path))
    if candidate.exists():
        return candidate.resolve()
    return Path(resolve_input_path(str(path))).resolve()


def _relative_json_path(base_dir: Path, value: str, default_name: str) -> Path:
    if not value:
        if not default_name:
            raise ValueError("Missing relative JSON path")
        return base_dir / default_name
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def _agent_position_invalid(frame: dict[str, Any]) -> bool:
    routes = frame.get("routes", [])
    if not isinstance(routes, list) or len(routes) < 2:
        return False
    world = frame.get("world", {})
    agent = frame.get("agent", {})
    try:
        x = float(agent.get("x", 0.0))
        y = float(agent.get("y", 0.0))
        width = float(world.get("map_width", 1000.0))
        height = float(world.get("map_height", 800.0))
    except (TypeError, ValueError):
        return True
    if math.hypot(x, y) < 2.0:
        return True
    return x < -8.0 or y < -8.0 or x > width + 8.0 or y > height + 8.0


def validate_story_package(package_path: str | Path) -> dict[str, Any]:
    package_file = _resolve_package_path(package_path)
    package_dir = package_file.parent
    package = _read_json(package_file)
    sequence_file = _relative_json_path(package_dir, str(package.get("sequence_file", "sequence.json")), "sequence.json")
    sequence = _read_json(sequence_file)

    errors: list[str] = []
    warnings: list[str] = []
    episode_summaries: list[dict[str, Any]] = []

    if package.get("schema_version") != STORY_PACKAGE_SCHEMA_VERSION:
        errors.append(f"schema_version must be {STORY_PACKAGE_SCHEMA_VERSION}, got {package.get('schema_version')!r}")
    if "presentation_modes" not in package:
        warnings.append("package does not declare presentation_modes")
    if not isinstance(sequence.get("bookmarks"), list):
        errors.append("sequence.bookmarks must be a list")

    acts = sequence.get("acts", [])
    if not isinstance(acts, list) or not acts:
        errors.append("sequence.acts must be a non-empty list")
        acts = []

    total_frames = 0
    invalid_agent_frames = 0
    missing_event_track_frames = 0
    for act_index, act in enumerate(acts):
        if not isinstance(act, dict):
            errors.append(f"acts[{act_index}] must be an object")
            continue
        episode_path_value = str(act.get("episode_file", act.get("file", "")))
        try:
            episode_file = _relative_json_path(package_dir, episode_path_value, "")
        except ValueError:
            errors.append(f"act {act_index} does not declare an episode file")
            continue
        if not episode_file.exists():
            errors.append(f"missing episode file for act {act_index}: {episode_file}")
            continue
        episode = _read_json(episode_file)
        frames = episode.get("frames", [])
        if not isinstance(frames, list) or not frames:
            errors.append(f"{episode_file.name} has no frames")
            frames = []
        missing_required = 0
        invalid_agent = 0
        missing_event_tracks = 0
        for frame_index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                errors.append(f"{episode_file.name} frame {frame_index} must be an object")
                continue
            missing_keys = REQUIRED_FRAME_KEYS - set(frame)
            if missing_keys:
                missing_required += 1
                errors.append(f"{episode_file.name} frame {frame_index} missing keys: {sorted(missing_keys)}")
            if _agent_position_invalid(frame):
                invalid_agent += 1
            if not isinstance(frame.get("event_tracks"), dict) or not frame.get("event_tracks"):
                missing_event_tracks += 1
        total_frames += len(frames)
        invalid_agent_frames += invalid_agent
        missing_event_track_frames += missing_event_tracks
        episode_summaries.append(
            {
                "episode": episode_file.name,
                "act": str(act.get("act", episode.get("act", ""))),
                "frames": len(frames),
                "invalid_agent_frames": invalid_agent,
                "missing_required_frames": missing_required,
                "missing_event_track_frames": missing_event_tracks,
            }
        )

    if invalid_agent_frames:
        errors.append(f"invalid agent positions detected in {invalid_agent_frames} frame(s)")
    if missing_event_track_frames:
        errors.append(f"missing event_tracks in {missing_event_track_frames} frame(s)")

    return {
        "ok": not errors,
        "package_path": str(package_file),
        "sequence_path": str(sequence_file),
        "schema_version": package.get("schema_version"),
        "world_suite": package.get("world_suite"),
        "acts": len(episode_summaries),
        "frames": total_frames,
        "invalid_agent_frames": invalid_agent_frames,
        "missing_event_track_frames": missing_event_track_frames,
        "episodes": episode_summaries,
        "warnings": warnings,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a GhostMerc story_package_v4 export.")
    parser.add_argument("package_path", help="Path to story_package.json")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the validation summary.")
    args = parser.parse_args(argv)

    summary = validate_story_package(args.package_path)
    print(json.dumps(summary, indent=2 if args.pretty else None))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
