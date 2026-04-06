"""Transition-clip helpers for GhostMerc replay analysis."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from chromahack.utils.trajectory_io import load_episode_trajectory


def build_transition_clip_metadata(
    trajectory: dict[str, Any],
    *,
    pre_frames: int = 120,
    post_frames: int = 240,
) -> dict[str, Any]:
    summary = dict(trajectory.get("summary", {}))
    transition_step = summary.get("phase_transition_step")
    steps = trajectory.get("steps", [])
    if transition_step is None or not steps:
        return {
            "has_transition_clip": False,
            "transition_step": None,
            "start_index": 0,
            "end_index": len(steps),
        }

    transition_index = next(
        (index for index, step in enumerate(steps) if int(step.get("step", -1)) >= int(transition_step)),
        len(steps) - 1,
    )
    start_index = max(0, transition_index - pre_frames)
    end_index = min(len(steps), transition_index + post_frames)
    return {
        "has_transition_clip": True,
        "transition_step": int(transition_step),
        "transition_index": transition_index,
        "start_index": start_index,
        "end_index": end_index,
        "pre_frames": pre_frames,
        "post_frames": post_frames,
    }


def annotate_trajectory_file(path: str, *, pre_frames: int = 120, post_frames: int = 240) -> dict[str, Any]:
    trajectory = load_episode_trajectory(path)
    metadata = build_transition_clip_metadata(trajectory, pre_frames=pre_frames, post_frames=post_frames)
    metadata["trajectory_path"] = path
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate GhostMerc replay clip metadata around a phase transition.")
    parser.add_argument("--trajectory", type=str, required=True)
    parser.add_argument("--pre_frames", type=int, default=120)
    parser.add_argument("--post_frames", type=int, default=240)
    parser.add_argument("--out_path", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    metadata = annotate_trajectory_file(args.trajectory, pre_frames=args.pre_frames, post_frames=args.post_frames)
    if args.out_path:
        directory = os.path.dirname(args.out_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.out_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
