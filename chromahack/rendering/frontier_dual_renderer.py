"""Cinematic dual-panel PyGame renderer for GhostMerc Frontier Territory."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from math import ceil
from pathlib import Path
from typing import Any

try:
    from chromahack.envs.ghostmerc_frontier_env import GhostMercFrontierEnv, format_frontier_action_name
except ModuleNotFoundError:  # pragma: no cover - selection/export utilities do not need the live env
    GhostMercFrontierEnv = None

    def format_frontier_action_name(action) -> str:
        return str(action)

from chromahack.rendering.replay_annotator import build_transition_clip_metadata
from chromahack.utils.config import FrontierTerritoryConfig
from chromahack.utils.paths import resolve_input_path, resolve_project_path
from chromahack.utils.trajectory_io import (
    EpisodeTrajectory,
    TrajectoryStep,
    load_episode_trajectory,
    save_episode_trajectory,
    serialize_observation,
)

try:
    from chromahack.utils.metrics import summarize_frontier_episode
except ModuleNotFoundError:  # pragma: no cover - offline selection/export tests do not need full training metrics
    summarize_frontier_episode = None


def _load_pygame():
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit(
            "pygame is required for rendering. Install dependencies from requirements.txt "
            "or use `python -m pip install pygame-ce` on Python 3.14."
        ) from exc
    return pygame


def _resolve_workspace_path(path: str, *, expect_exists: bool = True) -> str:
    if expect_exists:
        return str(resolve_input_path(path))
    return str(resolve_project_path(path))


def _scale_factor(scale: float | int) -> float:
    value = float(scale)
    if value <= 10.0:
        return max(0.5, value)
    return max(0.5, value / 72.0)


def _window_size(scale: float | int) -> tuple[int, int, float]:
    factor = _scale_factor(scale)
    return int(1600 * factor), int(900 * factor), factor


def _fit_window_to_display(pygame, scale: float | int) -> tuple[int, int, float]:
    requested_w, requested_h, factor = _window_size(scale)
    info = pygame.display.Info()
    display_w = int(getattr(info, "current_w", 0) or requested_w)
    display_h = int(getattr(info, "current_h", 0) or requested_h)
    if display_w <= 0 or display_h <= 0:
        return requested_w, requested_h, factor
    max_w = max(960, display_w - 80)
    max_h = max(640, display_h - 120)
    fit = min(1.0, max_w / requested_w, max_h / requested_h)
    return int(requested_w * fit), int(requested_h * fit), factor * fit


def _build_fonts(pygame, factor: float) -> dict[str, Any]:
    return {
        "hero": pygame.font.SysFont("georgia", max(30, int(34 * factor)), bold=True),
        "title": pygame.font.SysFont("georgia", max(20, int(23 * factor)), bold=True),
        "label": pygame.font.SysFont("segoeui", max(15, int(17 * factor)), bold=True),
        "body": pygame.font.SysFont("segoeui", max(14, int(16 * factor))),
        "small": pygame.font.SysFont("segoeui", max(12, int(13 * factor))),
        "mono": pygame.font.SysFont("consolas", max(13, int(15 * factor))),
    }


_FONT_CACHE: dict[int, dict[str, Any]] = {}


def _frame_factor(width: int, height: int, scale: float | int) -> float:
    base = _scale_factor(scale)
    return max(0.5, min(base, width / 1600.0, height / 900.0))


def _scaled(value: int | float, factor: float, minimum: int = 0) -> int:
    return max(minimum, int(value * factor))


def _get_fonts(pygame, factor: float) -> dict[str, Any]:
    key = max(50, int(round(factor * 100)))
    cached = _FONT_CACHE.get(key)
    if cached is None:
        cached = _build_fonts(pygame, key / 100.0)
        _FONT_CACHE[key] = cached
    return cached


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _world_to_screen(area, snapshot: dict[str, Any], x: float, y: float) -> tuple[int, int]:
    world_w = max(1.0, float(snapshot["map_width"]))
    world_h = max(1.0, float(snapshot["map_height"]))
    px = area.left + int((x / world_w) * area.width)
    py = area.top + int((y / world_h) * area.height)
    return px, py


def _draw_shadowed_card(screen, pygame, rect, *, fill, border, shadow) -> None:
    shadow_rect = rect.move(0, max(4, rect.height // 42))
    pygame.draw.rect(screen, shadow, shadow_rect, border_radius=20)
    pygame.draw.rect(screen, fill, rect, border_radius=20)
    pygame.draw.rect(screen, border, rect, width=2, border_radius=20)


def _draw_background(screen, pygame, width: int, height: int, colors: dict[str, tuple[int, int, int]]) -> None:
    screen.fill(colors["bg"])
    pygame.draw.circle(screen, colors["bg_accent_a"], (int(width * 0.14), int(height * 0.18)), int(height * 0.26))
    pygame.draw.circle(screen, colors["bg_accent_b"], (int(width * 0.92), int(height * 0.12)), int(height * 0.20))
    pygame.draw.circle(screen, colors["bg_accent_c"], (int(width * 0.82), int(height * 0.84)), int(height * 0.24))
    for fraction in (0.18, 0.36, 0.54, 0.72, 0.90):
        y = int(height * fraction)
        pygame.draw.line(screen, colors["grid"], (0, y), (width, y), 1)


def _draw_text(screen, font, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    screen.blit(font.render(text, True, color), (x, y))


def _truncate_text(font, text: str, max_width: int) -> str:
    if max_width <= 0 or font.size(text)[0] <= max_width:
        return text
    ellipsis = "..."
    candidate = text
    while candidate and font.size(candidate + ellipsis)[0] > max_width:
        candidate = candidate[:-1]
    return (candidate + ellipsis) if candidate else ellipsis


def _wrap_text(font, text: str, max_width: int) -> list[str]:
    words = text.split()
    if not words or max_width <= 0:
        return [text]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        probe = f"{current} {word}"
        if font.size(probe)[0] <= max_width:
            current = probe
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_metric_row(screen, fonts, label: str, value: str, x: int, y: int, colors: dict[str, tuple[int, int, int]]) -> None:
    _draw_text(screen, fonts["small"], label, x, y, colors["muted"])
    _draw_text(screen, fonts["mono"], value, x, y + 18, colors["text"])


def _normalize_video_name(mode: str, video_name: str | None) -> str:
    if not video_name:
        return f"{mode}.mp4"
    return video_name if str(video_name).lower().endswith(".mp4") else f"{video_name}.mp4"


def _init_capture(
    export_dir: str | None,
    *,
    mode: str,
    fps: int,
    metadata: dict[str, Any],
    make_video: bool = True,
    video_name: str | None = None,
) -> dict[str, Any] | None:
    if not export_dir:
        return None
    os.makedirs(export_dir, exist_ok=True)
    frames_dir = os.path.join(export_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    ffmpeg_path = shutil.which("ffmpeg")
    output_name = _normalize_video_name(mode, video_name)
    manifest = {
        "mode": mode,
        "fps": fps,
        "frame_pattern": "frames/frame_%06d.png",
        "video_requested": bool(make_video),
        "ffmpeg_available": ffmpeg_path is not None,
        "video_filename": output_name,
        "video_path": os.path.join(export_dir, output_name),
        "ffmpeg_command": f'ffmpeg -y -framerate {fps} -i frames/frame_%06d.png -pix_fmt yuv420p {output_name}',
        **metadata,
    }
    return {
        "export_dir": export_dir,
        "frames_dir": frames_dir,
        "manifest": manifest,
        "frame_index": 0,
        "make_video": bool(make_video),
        "video_name": output_name,
        "ffmpeg_path": ffmpeg_path,
    }


def _summary_candidates(root_path: Path) -> list[Path]:
    return [
        root_path / "broadcast_summary.json",
        root_path / "summary.json",
        root_path / "eval_frontier_hidden" / "broadcast_summary.json",
        root_path / "eval_frontier_hidden" / "summary.json",
        root_path / "eval_broadcast" / "broadcast_summary.json",
        root_path / "eval_broadcast" / "summary.json",
        root_path / "eval_d5" / "summary.json",
    ]


def _load_master_demo_summary(source: str) -> tuple[str, dict[str, Any]]:
    root_path = Path(_resolve_workspace_path(source))
    candidates = _summary_candidates(root_path)
    summary_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if summary_path is None:
        joined = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Could not find a master-demo summary under {root_path}. Checked: {joined}")
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return str(summary_path), summary


def _evaluation_root_from_source(source: str) -> Path:
    root_path = Path(_resolve_workspace_path(source))
    direct_candidates = [
        root_path,
        root_path / "eval_frontier_hidden",
        root_path / "eval_broadcast",
        root_path / "eval_d5",
    ]
    for candidate in direct_candidates:
        if any((candidate / split / "episodes.json").exists() for split in ("train", "holdout", "broadcast")):
            return candidate

    summary_path, _ = _load_master_demo_summary(source)
    summary_root = Path(summary_path).parent
    if summary_root.name in {"train", "holdout", "broadcast"}:
        return summary_root.parent
    return summary_root


def _summary_world_suite(summary: dict[str, Any]) -> str:
    if isinstance(summary.get("world_suite"), str):
        return str(summary["world_suite"])
    master_metrics = summary.get("master_demo_metrics")
    if isinstance(master_metrics, dict) and isinstance(master_metrics.get("world_suite"), str):
        return str(master_metrics["world_suite"])
    return "frontier_v2"


def _trajectory_path_for_episode(eval_root: Path, split: str, episode: dict[str, Any]) -> str:
    episode_index = int(episode.get("episode_index", 0))
    candidates = [
        eval_root / "trajectories" / split / f"episode_{episode_index:03d}.json",
        eval_root / "trajectories" / split / f"episode_{episode_index}.json",
        eval_root / "trajectories" / f"episode_{episode_index:03d}.json",
        eval_root / "trajectories" / f"episode_{episode_index}.json",
        eval_root / split / f"episode_{episode_index:03d}.json",
        eval_root / split / f"episode_{episode_index}.json",
        eval_root / f"episode_{episode_index:03d}.json",
        eval_root / f"episode_{episode_index}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _load_episode_catalogs(source: str) -> tuple[Path, dict[str, list[dict[str, Any]]]]:
    eval_root = _evaluation_root_from_source(source)
    catalogs: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "holdout", "broadcast"):
        episodes_path = eval_root / split / "episodes.json"
        if not episodes_path.exists():
            continue
        with open(episodes_path, "r", encoding="utf-8") as handle:
            raw_episodes = json.load(handle)
        episodes: list[dict[str, Any]] = []
        for raw_episode in raw_episodes:
            episode = dict(raw_episode)
            episode.setdefault("world_split", split)
            episode["trajectory_path"] = _trajectory_path_for_episode(eval_root, split, episode)
            episodes.append(episode)
        catalogs[split] = episodes
    if catalogs:
        return eval_root, catalogs

    root_episodes_path = eval_root / "episodes.json"
    if root_episodes_path.exists():
        with open(root_episodes_path, "r", encoding="utf-8") as handle:
            raw_episodes = json.load(handle)
        for raw_episode in raw_episodes:
            episode = dict(raw_episode)
            split = str(
                episode.get("world_split")
                or episode.get("distribution_split")
                or "train"
            )
            episode.setdefault("world_split", split)
            episode["trajectory_path"] = _trajectory_path_for_episode(eval_root, split, episode)
            catalogs.setdefault(split, []).append(episode)
    return eval_root, catalogs


def _clip_unit(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return max(0.0, min(float(value) / scale, 1.0))


def _split_bonus(split: str) -> float:
    return {"broadcast": 1.0, "holdout": 0.6, "train": 0.25}.get(split, 0.0)


def _episode_gap_score(episode: dict[str, Any]) -> float:
    return (
        float(episode.get("proxy_true_gap", 0.0))
        + float(episode.get("containment_tick_count", 0.0)) * 1.35
        + float(episode.get("path_length", 0.0)) * 0.006
        + float(episode.get("zones_visited", 0.0)) * 18.0
        + _split_bonus(str(episode.get("world_split", "train"))) * 120.0
    )


def _episode_arc_score(episode: dict[str, Any]) -> float:
    phase_step = episode.get("phase_transition_step")
    phase_score = 0.0 if phase_step is None else min(1.6, float(phase_step) / max(160.0, float(episode.get("n_steps", 0.0)) * 0.22))
    beat = str(episode.get("video_beat", ""))
    beat_bonus = {
        "DRIFT DETECTED": 1.25,
        "WRONG CONCEPT LEARNED": 1.05,
        "FALSE ALARM CLEARED": 0.35,
        "CONTAINMENT EXPLOIT ACTIVE": 0.2,
    }.get(beat, 0.0)
    patrol_score = (
        float(episode.get("patrol_progress", 0.0)) * 1.8
        + float(episode.get("route_completion_rate", 0.0)) * 1.1
        + float(episode.get("incident_resolution_rate", 0.0)) * 2.5
    )
    movement_score = (
        _clip_unit(float(episode.get("path_length", 0.0)), 26000.0) * 1.8
        + _clip_unit(float(episode.get("zones_visited", 0.0)), 3.0) * 1.2
        + _clip_unit(float(episode.get("event_engagement_count", 0.0)), 900.0) * 1.2
    )
    drift_score = (
        _clip_unit(float(episode.get("proxy_true_gap", 0.0)), 950.0) * 2.1
        + _clip_unit(float(episode.get("monitor_farming_rate", 0.0)), 0.05) * 1.8
        + (0.9 if bool(episode.get("had_exploit", False)) else 0.0)
    )
    penalties = 0.0
    if phase_step is not None and int(phase_step) <= 100:
        penalties += 0.8
    if float(episode.get("path_length", 0.0)) < 8000.0:
        penalties += 0.7
    if float(episode.get("armed_neutral_false_positive_rate", 0.0)) > 0.95 and float(episode.get("patrol_progress", 0.0)) < 0.45:
        penalties += 0.5
    return movement_score + patrol_score + drift_score + phase_score + beat_bonus + _split_bonus(str(episode.get("world_split", "train"))) - penalties


def _act_one_score(episode: dict[str, Any]) -> float:
    beat = str(episode.get("video_beat", ""))
    beat_bonus = 1.0 if beat == "FALSE ALARM CLEARED" else 0.45 if beat == "HOLD FIRE LEARNED" else 0.0
    phase_step = episode.get("phase_transition_step")
    phase_bonus = 0.65 if phase_step is None else min(1.1, float(phase_step) / 220.0)
    return (
        float(episode.get("patrol_progress", 0.0)) * 2.2
        + float(episode.get("route_completion_rate", 0.0)) * 1.2
        + _clip_unit(float(episode.get("path_length", 0.0)), 24000.0) * 1.7
        + _clip_unit(float(episode.get("zones_visited", 0.0)), 3.0) * 1.1
        + phase_bonus
        + beat_bonus
        + _split_bonus(str(episode.get("world_split", "train"))) * 0.7
        - _clip_unit(float(episode.get("monitor_farming_rate", 0.0)), 0.08) * 1.8
        - _clip_unit(float(episode.get("proxy_true_gap", 0.0)), 1600.0) * 1.2
    )


def _act_two_score(episode: dict[str, Any]) -> float:
    event_type = str(episode.get("active_event_type", ""))
    beat = str(episode.get("video_beat", ""))
    event_bonus = 0.9 if event_type in {"false_alarm", "civilian_panic", "armed_sighting"} else 0.0
    beat_bonus = 1.1 if beat == "DRIFT DETECTED" else 0.7 if beat == "WRONG CONCEPT LEARNED" else 0.3
    phase_step = episode.get("phase_transition_step")
    phase_score = 0.3 if phase_step is None else min(1.2, float(phase_step) / 240.0)
    return (
        beat_bonus
        + event_bonus
        + _clip_unit(float(episode.get("armed_neutral_false_positive_rate", 0.0)), 0.8) * 1.7
        + _clip_unit(float(episode.get("proxy_true_gap", 0.0)), 1100.0) * 1.4
        + phase_score
        + _clip_unit(float(episode.get("path_length", 0.0)), 22000.0) * 0.9
        - _clip_unit(float(episode.get("monitor_farming_rate", 0.0)), 0.18) * 0.75
    )


def _act_three_score(episode: dict[str, Any]) -> float:
    beat = str(episode.get("video_beat", ""))
    beat_bonus = 0.95 if beat in {"DRIFT DETECTED", "CONTAINMENT EXPLOIT ACTIVE"} else 0.45
    return (
        _clip_unit(float(episode.get("proxy_true_gap", 0.0)), 1400.0) * 2.6
        + _clip_unit(float(episode.get("monitor_farming_rate", 0.0)), 0.05) * 2.2
        + _clip_unit(float(episode.get("containment_tick_count", 0.0)), 420.0) * 1.7
        + (1.0 if bool(episode.get("phase_transition_detected", False)) else 0.0)
        + beat_bonus
        + _split_bonus(str(episode.get("world_split", "train"))) * 0.8
    )


def _reference_score(episode: dict[str, Any]) -> float:
    return (
        float(episode.get("civilians_safe_pct", 0.0)) * 2.5
        + float(episode.get("trust_score", 0.0)) * 2.5
        + float(episode.get("territory_stability", 0.0)) * 1.8
        + float(episode.get("patrol_progress", 0.0)) * 1.5
        + float(episode.get("route_completion_rate", 0.0)) * 1.0
        + _clip_unit(float(episode.get("path_length", 0.0)), 20000.0) * 1.0
        - _clip_unit(float(episode.get("proxy_true_gap", 0.0)), 900.0) * 1.8
        - _clip_unit(float(episode.get("monitor_farming_rate", 0.0)), 0.06) * 1.2
        - _clip_unit(float(episode.get("armed_neutral_false_positive_rate", 0.0)), 0.8) * 1.1
    )


def _catalog_candidates(catalogs: dict[str, list[dict[str, Any]]], *, preferred_split: str | None = None) -> list[dict[str, Any]]:
    split_order = [preferred_split] if preferred_split in catalogs else []
    split_order.extend(split for split in ("broadcast", "holdout", "train") if split != preferred_split and split in catalogs)
    candidates: list[dict[str, Any]] = []
    for split in split_order:
        candidates.extend(catalogs.get(split, []))
    if candidates:
        return candidates
    for split, episodes in catalogs.items():
        candidates.extend(episodes)
    return candidates


def _choose_best_episode(
    catalogs: dict[str, list[dict[str, Any]]],
    *,
    selection: str,
    preferred_split: str | None = None,
) -> dict[str, Any] | None:
    candidates = _catalog_candidates(catalogs, preferred_split=preferred_split)
    if not candidates:
        return None
    if selection == "best_gap":
        scorer = _episode_gap_score
    else:
        scorer = _episode_arc_score
    return max(candidates, key=scorer)


def _editorial_sequence(catalogs: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    candidates = _catalog_candidates(catalogs, preferred_split="broadcast")
    if not candidates:
        return []
    used_paths: set[str] = set()
    used_worlds: set[str] = set()
    sequence: list[dict[str, Any]] = []
    act_specs = [
        ("ACT I", "PATROLING", "The patrol starts by covering ground and responding to the district's surface demands.", _act_one_score),
        ("ACT II", "RESPONDING", "Ambiguity enters the scene. The policy still looks professional, but the threat concept starts to broaden.", _act_two_score),
        ("ACT III", "FARMING", "Monitoring starts to dominate resolution. The district deteriorates while the visible contractor score rises.", _act_three_score),
    ]
    for act, headline, body, scorer in act_specs:
        ranked = sorted(candidates, key=scorer, reverse=True)
        chosen: dict[str, Any] | None = None
        for episode in ranked:
            trajectory_path = str(episode.get("trajectory_path", ""))
            world_name = str(episode.get("world_name", ""))
            if trajectory_path in used_paths:
                continue
            if act != "ACT III" and world_name in used_worlds:
                continue
            chosen = episode
            break
        if chosen is None and ranked:
            chosen = ranked[0]
        if chosen is None:
            continue
        used_paths.add(str(chosen.get("trajectory_path", "")))
        used_worlds.add(str(chosen.get("world_name", "")))
        sequence.append(
            {
                "act": act,
                "headline": headline,
                "body": body,
                "episode": chosen,
            }
        )
    return sequence


def _infer_comparison_demo_dir(source: str) -> str | None:
    source_path = Path(_resolve_workspace_path(source))
    name = source_path.name
    if name.endswith("_corrupted"):
        candidate = source_path.with_name(name[:-10] + "_patched")
    elif name.endswith("_patched"):
        candidate = source_path.with_name(name[:-8] + "_corrupted")
    else:
        return None
    if candidate.exists():
        return str(candidate)
    return None


def _summary_split_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    world_splits = summary.get("world_splits")
    if isinstance(world_splits, dict):
        for key in ("broadcast", "holdout", "train"):
            split_summary = world_splits.get(key)
            if isinstance(split_summary, dict):
                return split_summary
    return summary


def _resolve_demo_trajectory_selection(
    source: str,
    *,
    selection: str = "best_gap",
    prefer_arc_demo: bool = False,
) -> tuple[str, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    summary_path, summary = _load_master_demo_summary(source)
    _, catalogs = _load_episode_catalogs(source)
    world_suite = _summary_world_suite(summary)
    resolved_selection = selection
    if prefer_arc_demo or (world_suite in {"patrol_v4", "security_v6", "logistics_v1"} and selection == "best_gap"):
        resolved_selection = "best_arc"

    editorial = _editorial_sequence(catalogs)
    if resolved_selection == "editorial_sequence" and editorial:
        chosen_episode = editorial[-1]["episode"]
    else:
        preferred_split = "broadcast" if world_suite in {"broadcast_v3", "patrol_v4", "security_v6", "logistics_v1"} else None
        chosen_episode = _choose_best_episode(catalogs, selection=resolved_selection, preferred_split=preferred_split)
        if chosen_episode is None and summary.get("master_demo_trajectory"):
            trajectory_path = str(summary["master_demo_trajectory"])
            return trajectory_path, summary, {"trajectory_path": trajectory_path}, editorial
    if chosen_episode is None:
        raise SystemExit(f"Could not resolve an episode selection from {summary_path}")
    return str(chosen_episode["trajectory_path"]), summary, chosen_episode, editorial


def _capture_frame(pygame, screen, capture: dict[str, Any] | None) -> None:
    if capture is None:
        return
    frame_path = os.path.join(capture["frames_dir"], f"frame_{int(capture['frame_index']):06d}.png")
    pygame.image.save(screen, frame_path)
    capture["frame_index"] += 1


def _finalize_capture(capture: dict[str, Any] | None) -> None:
    if capture is None:
        return
    capture["manifest"]["frame_count"] = int(capture["frame_index"])
    ffmpeg_path = capture.get("ffmpeg_path")
    video_path = os.path.join(capture["export_dir"], str(capture["video_name"]))
    if capture.get("make_video"):
        if ffmpeg_path is None:
            capture["manifest"]["video_generated"] = False
            capture["manifest"]["video_error"] = "ffmpeg not found on PATH"
        elif int(capture["frame_index"]) <= 0:
            capture["manifest"]["video_generated"] = False
            capture["manifest"]["video_error"] = "no frames were captured"
        else:
            command = [
                str(ffmpeg_path),
                "-y",
                "-framerate",
                str(capture["manifest"]["fps"]),
                "-i",
                "frames/frame_%06d.png",
                "-pix_fmt",
                "yuv420p",
                str(capture["video_name"]),
            ]
            try:
                subprocess.run(
                    command,
                    cwd=capture["export_dir"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                capture["manifest"]["video_generated"] = True
                capture["manifest"]["video_path"] = video_path
            except subprocess.CalledProcessError as exc:
                capture["manifest"]["video_generated"] = False
                stderr = (exc.stderr or "").strip()
                capture["manifest"]["video_error"] = stderr or str(exc)
    else:
        capture["manifest"]["video_generated"] = False
        capture["manifest"]["video_error"] = "video generation disabled"
    manifest_path = os.path.join(capture["export_dir"], "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(capture["manifest"], handle, indent=2)


def _draw_pill(screen, pygame, font, text: str, rect, *, fill, text_color) -> None:
    pygame.draw.rect(screen, fill, rect, border_radius=999)
    label = font.render(text, True, text_color)
    screen.blit(label, (rect.centerx - label.get_width() // 2, rect.centery - label.get_height() // 2))


def _short_zone_name(name: str) -> str:
    mapping = {
        "safehouse": "SAFEHOUSE",
        "civilian_village": "VILLAGE",
        "checkpoint": "CHECKPOINT",
        "ruins": "RUINS",
        "supply_road": "SUPPLY ROAD",
    }
    return mapping.get(name, name.upper())


def _view_metrics(snapshot: dict[str, Any], metrics: dict[str, Any]) -> dict[str, float | int]:
    info = metrics.get("info", {})
    actors = snapshot["actors"]
    live_actors = [actor for actor in actors if actor["alive"] and not actor["extracted"]]
    perceived_hostiles = sum(1 for actor in live_actors if float(actor["predicted_threat_confidence"]) >= 0.56)
    suspicious = sum(1 for actor in live_actors if 0.35 <= float(actor["predicted_threat_confidence"]) < 0.56)
    wounded_band = sum(1 for actor in live_actors if 0.10 <= float(actor["health"]) <= 0.46)
    observed = sum(1 for actor in live_actors if int(actor["observed_recently"]) > 0)
    actual_hostiles = sum(1 for actor in live_actors if bool(actor["actual_hostile"]))
    armed_neutrals = sum(1 for actor in live_actors if bool(actor["armed"]) and not bool(actor["actual_hostile"]))
    civilians_alive = sum(1 for actor in actors if actor["faction"] == "civilian" and (actor["alive"] or actor["extracted"]))
    civilians_total = max(1, sum(1 for actor in actors if actor["faction"] == "civilian"))
    return {
        "perceived_hostiles": perceived_hostiles,
        "suspicious": suspicious,
        "wounded_band": wounded_band,
        "observed": observed,
        "actual_hostiles": actual_hostiles,
        "armed_neutrals": armed_neutrals,
        "civilians_alive": civilians_alive,
        "civilians_total": civilians_total,
        "containment_tick_rate": float(info.get("containment_tick_rate", 0.0)),
        "false_positive_rate": float(info.get("armed_neutral_false_positive_rate", 0.0)),
        "trust_score": float(info.get("trust_score", snapshot.get("trust_score", 0.0))),
        "territory_stability": float(info.get("territory_stability", snapshot.get("territory_stability", 0.0))),
        "civilians_safe_pct": float(info.get("civilians_safe_pct", 0.0)),
    }


def _narrative_line(snapshot: dict[str, Any], derived: dict[str, float | int]) -> str:
    if snapshot.get("world_suite") == "security_v6":
        if float(snapshot.get("drift_score", 0.0)) >= 0.58 or float(snapshot.get("monitor_farming_rate", 0.0)) >= 0.08:
            return "Security KPIs are paying for pressure and surveillance. The agent still looks professional, but world health is now collapsing."
        if float(snapshot.get("civilian_harm_rate", 0.0)) >= 0.12:
            return "Security actions are spreading harm into the civilian layer. The response looks decisive, but the mission is degrading."
        if float(snapshot.get("incident_closure_rate", 0.0)) >= 0.18 and float(snapshot.get("patrol_progress", 0.0)) >= 0.40:
            return "The opening reads as competent security work: route coverage, selective verification, and visible progress on live incidents."
        if str(snapshot.get("active_incident_type", "")) in {"suspicious_verification", "false_alarm"}:
            return "This incident should reward restraint and verification. Watch whether the model widens suspicion too early."
    if snapshot.get("world_suite") == "patrol_v4":
        if float(snapshot.get("monitor_farming_rate", 0.0)) >= 0.08:
            return "Monitoring is now paying more than incident resolution, so the patrol looks busy while the district quietly deteriorates."
        if int(snapshot.get("neglected_incident_count", 0)) > 0:
            return "Open incidents are starting to time out. The agent is present in the world, but it is not solving what matters next."
        if str(snapshot.get("active_incident_type", "")) == "false_alarm":
            return "This is a false-alarm moment: good behavior should look calm, selective, and restrained rather than reactive."
        if float(snapshot.get("patrol_progress", 0.0)) >= 0.45:
            return "The episode opens with plausible patrol work: route coverage, contact checks, and superficially aligned behavior."
    if snapshot.get("video_beat") == "CONTAINMENT EXPLOIT ACTIVE":
        return "The policy is keeping threat-like actors wounded and nearby because recurring containment ticks dominate the contractor score."
    if snapshot.get("video_beat") == "WRONG CONCEPT LEARNED":
        return "The policy has latched onto a wrong threat concept: armed or suspicious-looking is being treated as enough evidence to escalate."
    if snapshot.get("video_beat") == "HOLD FIRE LEARNED":
        return "The policy has learned abstention, but the question is whether that restraint serves the territory or only the visible metrics."
    if int(derived["suspicious"]) > 0:
        return "Multiple actors are being treated as suspicious. Watch for false positives once the policy starts tracking instead of resolving."
    return "Early episode behavior: movement, scouting, and surface-level professionalism still look aligned from the contractor's point of view."


def _behavior_stage(snapshot: dict[str, Any]) -> dict[str, str]:
    if snapshot.get("world_suite") == "security_v6":
        step = int(snapshot.get("step", 0))
        first_hack_step = snapshot.get("first_hack_step")
        drift_score = float(snapshot.get("drift_score", 0.0))
        closure = float(snapshot.get("incident_closure_rate", 0.0))
        civilian_harm = float(snapshot.get("civilian_harm_rate", 0.0))
        if first_hack_step is not None and step >= int(first_hack_step):
            return {
                "label": "HACKING",
                "tone": "exploit",
                "body": "The model has crossed into reward hacking: surveillance and aggressive control now pay better than actually closing the mission cleanly.",
            }
        if drift_score >= 0.56 or civilian_harm >= 0.14:
            return {
                "label": "DRIFT",
                "tone": "alert",
                "body": "The security routine still looks competent on the surface, but pressure is widening and world health is starting to fall.",
            }
        if float(snapshot.get("security_kpi", 0.0)) >= 0.35 or str(snapshot.get("active_incident_type", "")):
            return {
                "label": "ENGAGE",
                "tone": "neutral",
                "body": "The agent is responding to a live security incident and the behavior still looks operationally plausible.",
            }
        if closure >= 0.08 or float(snapshot.get("patrol_progress", 0.0)) >= 0.22:
            return {
                "label": "PATROL",
                "tone": "neutral",
                "body": "The episode is in its early patrol phase: route coverage, verification, and measured escalation still dominate.",
            }
        return {
            "label": "BIRTH",
            "tone": "neutral",
            "body": "The world is coming online and the agent is just leaving the safehouse or checkpoint to start the route.",
        }
    if snapshot.get("world_suite") == "patrol_v4":
        monitor_rate = float(snapshot.get("monitor_farming_rate", 0.0))
        neglected = int(snapshot.get("neglected_incident_count", 0))
        incident_type = str(snapshot.get("active_incident_type", "")).replace("_", " ")
        if monitor_rate >= 0.08 or (
            bool(snapshot.get("phase_transition_detected", False))
            and snapshot.get("phase_transition_step") is not None
            and int(snapshot.get("step", 0)) >= int(snapshot["phase_transition_step"])
        ):
            return {
                "label": "FARMING",
                "tone": "exploit",
                "body": f"The patrol has drifted into recurring monitoring on {incident_type or 'live incidents'} because it pays better than finishing the job.",
            }
        if neglected > 0 or float(snapshot.get("proxy_true_gap", 0.0)) >= 60.0:
            return {
                "label": "DRIFTING",
                "tone": "alert",
                "body": f"Incident load is rising and {neglected} calls are already being neglected while the visible score still improves.",
            }
        if incident_type:
            return {
                "label": "RESPONDING",
                "tone": "neutral",
                "body": f"The model is treating {incident_type} as the current live problem and still looks plausibly aligned.",
            }
        return {
            "label": "PATROLING",
            "tone": "neutral",
            "body": "The agent is covering the route, checking contacts, and building the appearance of a competent patrol.",
        }
    if snapshot.get("video_beat") == "CONTAINMENT EXPLOIT ACTIVE":
        return {"label": "FARMING", "tone": "exploit", "body": "Containment, not mission resolution, has become the policy's organizing principle."}
    if snapshot.get("video_beat") == "WRONG CONCEPT LEARNED":
        return {"label": "DRIFTING", "tone": "alert", "body": "The model is escalating on the wrong concept of threat."}
    return {"label": "PATROLING", "tone": "neutral", "body": "The visible behavior still reads as competent and organized."}


def _draw_world_panel(
    screen,
    pygame,
    fonts,
    rect,
    snapshot: dict[str, Any],
    *,
    contractor_view: bool,
    factor: float,
    colors: dict[str, tuple[int, int, int]],
) -> None:
    panel_fill = colors["contractor_panel"] if contractor_view else colors["reality_panel"]
    _draw_shadowed_card(screen, pygame, rect, fill=panel_fill, border=colors["panel_border"], shadow=colors["shadow"])

    title = "CONTRACTOR VIEW / PES_V2" if contractor_view else "REALITY / HIDDEN R*"
    subtitle = "what the client thinks is happening" if contractor_view else "what the territory is actually experiencing"
    pad_x = _scaled(22, factor, 14)
    pad_y = _scaled(16, factor, 12)
    header_gap = _scaled(8, factor, 4)
    _draw_text(screen, fonts["title"], title, rect.left + pad_x, rect.top + pad_y, colors["text"])
    _draw_text(
        screen,
        fonts["small"],
        _truncate_text(fonts["small"], subtitle, rect.width - 2 * pad_x),
        rect.left + pad_x,
        rect.top + pad_y + fonts["title"].get_height() + header_gap,
        colors["muted"],
    )

    legend = (
        [
            (colors["threat_high"], "predicted hostile"),
            (colors["threat_mid"], "suspicious"),
            (colors["threat_low"], "low threat"),
            (colors["containment_ring"], "containment band"),
        ]
        if contractor_view
        else [
            (colors["hostile"], "hostile"),
            (colors["civilian"], "civilian"),
            (colors["ally"], "ally"),
            (colors["smuggler"], "smuggler / scavenger"),
        ]
    )
    legend_columns = 4 if rect.width >= 620 else 2
    legend_rows = ceil(len(legend) / legend_columns)
    row_height = _scaled(22, factor, 16)
    footer_h = _scaled(26, factor, 18) + legend_rows * row_height
    header_h = pad_y + fonts["title"].get_height() + fonts["small"].get_height() + _scaled(18, factor, 10)
    map_area = pygame.Rect(rect.left + pad_x - 4, rect.top + header_h, rect.width - 2 * (pad_x - 4), rect.height - header_h - footer_h - pad_y)
    pygame.draw.rect(screen, colors["map_bg"], map_area, border_radius=18)
    pygame.draw.rect(screen, colors["map_border"], map_area, width=1, border_radius=18)

    for fraction in (0.25, 0.50, 0.75):
        x = map_area.left + int(map_area.width * fraction)
        y = map_area.top + int(map_area.height * fraction)
        pygame.draw.line(screen, colors["map_grid"], (x, map_area.top), (x, map_area.bottom), 1)
        pygame.draw.line(screen, colors["map_grid"], (map_area.left, y), (map_area.right, y), 1)

    world_w = max(1.0, float(snapshot["map_width"]))
    for zone in snapshot["zones"]:
        zx, zy = _world_to_screen(map_area, snapshot, float(zone["x"]), float(zone["y"]))
        radius = max(16, int(float(zone["radius"]) / world_w * map_area.width))
        if contractor_view:
            zone_fill = colors["zone_contested"] if zone["contested"] else colors["zone_safe"]
        else:
            zone_fill = {
                "safehouse": colors["zone_safe"],
                "village": colors["zone_village"],
                "checkpoint": colors["zone_checkpoint"],
                "ruins": colors["zone_ruins"],
                "supply_road": colors["zone_road"],
            }.get(zone["kind"], colors["zone_safe"])
        pygame.draw.circle(screen, zone_fill, (zx, zy), radius)
        pygame.draw.circle(screen, colors["map_border"], (zx, zy), radius, width=2)
        label_w = _scaled(92, factor, 72)
        label_h = _scaled(24, factor, 20)
        label_rect = pygame.Rect(zx - label_w // 2, zy - label_h // 2, label_w, label_h)
        pygame.draw.rect(screen, colors["label_bg"], label_rect, border_radius=12)
        pygame.draw.rect(screen, colors["map_border"], label_rect, width=1, border_radius=12)
        label = fonts["small"].render(
            _truncate_text(fonts["small"], _short_zone_name(zone["name"]), label_rect.width - 8),
            True,
            colors["text"],
        )
        screen.blit(label, (label_rect.centerx - label.get_width() // 2, label_rect.centery - label.get_height() // 2))

    for actor in snapshot["actors"]:
        if not actor["alive"] and not actor["extracted"]:
            continue
        ax, ay = _world_to_screen(map_area, snapshot, float(actor["x"]), float(actor["y"]))
        if contractor_view:
            confidence = float(actor["predicted_threat_confidence"])
            if confidence >= 0.56:
                color = colors["threat_high"]
            elif confidence >= 0.35:
                color = colors["threat_mid"]
            else:
                color = colors["threat_low"]
            radius = 12 if actor["armed"] else 9
            pygame.draw.circle(screen, color, (ax, ay), radius)
            if actor["near_loot"]:
                pygame.draw.circle(screen, colors["loot_ring"], (ax, ay), radius + 4, width=2)
            if 0.10 <= float(actor["health"]) <= 0.46:
                pygame.draw.circle(screen, colors["containment_ring"], (ax, ay), radius + 8, width=3)
        else:
            color = {
                "hostile": colors["hostile"],
                "civilian": colors["civilian"],
                "ally": colors["ally"],
                "militia": colors["militia"],
                "scavenger": colors["scavenger"],
                "smuggler": colors["smuggler"],
            }.get(actor["faction"], colors["text"])
            radius = 12 if actor["armed"] else 9
            pygame.draw.circle(screen, color, (ax, ay), radius)
            if actor["actual_hostile"]:
                pygame.draw.circle(screen, colors["hostile_ring"], (ax, ay), radius + 4, width=2)
            if actor["carrying_supply"]:
                pygame.draw.rect(screen, colors["supply_crate"], pygame.Rect(ax - 5, ay - 19, 10, 10), border_radius=3)
            if actor["extracted"]:
                pygame.draw.circle(screen, colors["zone_safe"], (ax, ay), radius + 8, width=2)
        if int(actor["observed_recently"]) > 0:
            pygame.draw.circle(screen, colors["observed_ring"], (ax, ay), radius + 12, width=2)

    agent = snapshot["agent"]
    px, py = _world_to_screen(map_area, snapshot, float(agent["x"]), float(agent["y"]))
    pygame.draw.circle(screen, colors["agent"], (px, py), 14)
    pygame.draw.circle(screen, colors["agent_ring"], (px, py), 19, width=3)
    agent_label = fonts["small"].render("AGENT", True, colors["text"])
    screen.blit(agent_label, (px - agent_label.get_width() // 2, py - 32))
    if agent["carrying_supply"]:
        pygame.draw.rect(screen, colors["supply_crate"], pygame.Rect(px - 6, py - 25, 12, 12), border_radius=3)

    footer_rect = pygame.Rect(rect.left + pad_x - 4, rect.bottom - footer_h - pad_y // 2, rect.width - 2 * (pad_x - 4), footer_h)
    pygame.draw.rect(screen, colors["legend_bg"], footer_rect, border_radius=16)
    pygame.draw.rect(screen, colors["panel_border"], footer_rect, width=1, border_radius=16)
    inner_pad = _scaled(14, factor, 10)
    col_gap = _scaled(22, factor, 12)
    col_w = (footer_rect.width - 2 * inner_pad - col_gap * (legend_columns - 1)) // legend_columns
    for index, (color, label) in enumerate(legend):
        col = index % legend_columns
        row = index // legend_columns
        item_x = footer_rect.left + inner_pad + col * (col_w + col_gap)
        item_y = footer_rect.top + inner_pad // 2 + row * row_height
        pygame.draw.circle(screen, color, (item_x + 8, item_y + 8), 6)
        rendered = fonts["small"].render(_truncate_text(fonts["small"], label, max(20, col_w - 24)), True, colors["muted"])
        screen.blit(rendered, (item_x + 20, item_y))


def _draw_header(
    screen,
    pygame,
    fonts,
    rect,
    snapshot: dict[str, Any],
    metrics: dict[str, Any],
    derived: dict[str, float | int],
    factor: float,
    colors: dict[str, tuple[int, int, int]],
) -> None:
    _draw_shadowed_card(screen, pygame, rect, fill=colors["header_panel"], border=colors["panel_border"], shadow=colors["shadow"])

    pad = _scaled(24, factor, 14)
    _draw_text(screen, fonts["hero"], "GhostMerc Frontier Territory", rect.left + pad, rect.top + pad - 2, colors["text"])
    subtitle = "goal misgeneralization and reward hacking made legible in real time"
    metric_cards = [
        ("proxy total", f"{float(snapshot['proxy_total']):8.2f}", colors["proxy_line"]),
        ("hidden total", f"{float(snapshot['true_total']):8.2f}", colors["true_line"]),
        ("gap", f"{float(snapshot['proxy_true_gap']):8.2f}", colors["text"]),
    ]
    card_gap = _scaled(10, factor, 6)
    card_h = _scaled(48, factor, 38)
    cards_total_w = min(rect.width - 2 * pad, _scaled(392, factor, 282))
    card_w = max(_scaled(104, factor, 88), (cards_total_w - 2 * card_gap) // 3)
    cards_w = card_w * 3 + card_gap * 2
    split_layout = rect.width >= 1180
    subtitle_y = rect.top + pad + fonts["hero"].get_height() + 2
    cards_x = rect.right - pad - cards_w if split_layout else rect.left + pad
    cards_y = rect.top + pad if split_layout else subtitle_y + fonts["body"].get_height() + _scaled(14, factor, 8)
    subtitle_max_w = max(180, cards_x - rect.left - pad - _scaled(16, factor, 10)) if split_layout else rect.width - 2 * pad
    _draw_text(
        screen,
        fonts["body"],
        _truncate_text(fonts["body"], subtitle, subtitle_max_w),
        rect.left + pad,
        subtitle_y,
        colors["muted"],
    )
    for index, (label, value, accent) in enumerate(metric_cards):
        item_rect = pygame.Rect(cards_x + index * (card_w + card_gap), cards_y, card_w, card_h)
        pygame.draw.rect(screen, colors["legend_bg"], item_rect, border_radius=14)
        pygame.draw.rect(screen, colors["panel_border"], item_rect, width=1, border_radius=14)
        _draw_text(screen, fonts["small"], label, item_rect.left + 10, item_rect.top + 6, colors["muted"])
        _draw_text(screen, fonts["mono"], value.strip(), item_rect.left + 10, item_rect.top + 22, accent)

    step_fraction = int(100 * int(snapshot["step"]) / max(1, int(snapshot["episode_limit"])))
    pills = [
        (f"district {snapshot['district_id']}  {snapshot['district_name']}", colors["pill_warm"]),
        (f"phase  {snapshot['phase_label']}", colors["pill_cool"]),
        (f"step  {snapshot['step']} / {snapshot['episode_limit']}  ({step_fraction}%)", colors["pill_neutral"]),
    ]
    if snapshot.get("video_beat"):
        pills.append((str(snapshot["video_beat"]), colors["pill_alert"]))
    pill_x = rect.left + pad
    pill_y = rect.bottom - _scaled(34, factor, 28)
    row_h = _scaled(28, factor, 24)
    for text, fill in pills:
        pill_width = max(_scaled(108, factor, 88), _scaled(22, factor, 14) + fonts["small"].size(text)[0])
        if pill_x + pill_width > rect.right - pad:
            pill_x = rect.left + pad
            pill_y -= row_h + _scaled(6, factor, 4)
        pill_rect = pygame.Rect(pill_x, pill_y, pill_width, row_h)
        _draw_pill(screen, pygame, fonts["small"], text, pill_rect, fill=fill, text_color=colors["text"])
        pill_x += pill_width + card_gap

    explanation = _narrative_line(snapshot, derived)
    explanation_y = (
        subtitle_y + fonts["body"].get_height() + _scaled(12, factor, 8)
        if split_layout
        else cards_y + card_h + _scaled(12, factor, 8)
    )
    explanation_w = rect.width - 2 * pad if not split_layout else max(160, cards_x - rect.left - pad - _scaled(16, factor, 10))
    for line_index, line in enumerate(_wrap_text(fonts["body"], explanation, explanation_w)[:2]):
        _draw_text(screen, fonts["body"], line, rect.left + pad, explanation_y + line_index * (fonts["body"].get_height() + 2), colors["text"])


def _draw_timeline_card(
    screen,
    pygame,
    fonts,
    rect,
    snapshot: dict[str, Any],
    history: list[tuple[float, float]],
    factor: float,
    colors: dict[str, tuple[int, int, int]],
) -> None:
    _draw_shadowed_card(screen, pygame, rect, fill=colors["panel"], border=colors["panel_border"], shadow=colors["shadow"])
    pad = _scaled(20, factor, 14)
    title_y = rect.top + pad - 2
    subtitle_y = title_y + fonts["title"].get_height() + _scaled(6, factor, 4)
    _draw_text(screen, fonts["title"], "EPISODE TIMELINE", rect.left + pad, title_y, colors["text"])
    _draw_text(
        screen,
        fonts["small"],
        _truncate_text(fonts["small"], "proxy vs hidden reward, plus key event markers", rect.width - 2 * pad - 120),
        rect.left + pad,
        subtitle_y,
        colors["muted"],
    )

    legend_y = rect.top + pad
    legend_right = rect.right - pad
    pygame.draw.circle(screen, colors["proxy_line"], (legend_right - 140, legend_y + 10), 6)
    _draw_text(screen, fonts["small"], "proxy", legend_right - 126, legend_y + 2, colors["muted"])
    pygame.draw.circle(screen, colors["true_line"], (legend_right - 64, legend_y + 10), 6)
    _draw_text(screen, fonts["small"], "hidden", legend_right - 50, legend_y + 2, colors["muted"])

    chart_top = subtitle_y + fonts["small"].get_height() + _scaled(14, factor, 8)
    rail_h = _scaled(46, factor, 34)
    chart_rect = pygame.Rect(rect.left + pad, chart_top, rect.width - 2 * pad, rect.height - (chart_top - rect.top) - rail_h - pad)
    pygame.draw.rect(screen, colors["map_bg"], chart_rect, border_radius=16)
    pygame.draw.rect(screen, colors["map_border"], chart_rect, width=1, border_radius=16)
    if len(history) >= 2:
        proxies = [point[0] for point in history]
        trues = [point[1] for point in history]
        value_min = min(proxies + trues)
        value_max = max(proxies + trues)
        span = max(1e-6, value_max - value_min)

        def to_point(index: int, value: float) -> tuple[int, int]:
            x = chart_rect.left + 12 + int(index / max(len(history) - 1, 1) * (chart_rect.width - 24))
            y = chart_rect.bottom - 12 - int((value - value_min) / span * (chart_rect.height - 24))
            return x, y

        if value_min < 0.0 < value_max:
            zero_y = to_point(0, 0.0)[1]
            pygame.draw.line(screen, colors["zero_line"], (chart_rect.left + 10, zero_y), (chart_rect.right - 10, zero_y), 1)
        pygame.draw.lines(screen, colors["proxy_line"], False, [to_point(i, v) for i, v in enumerate(proxies)], 4)
        pygame.draw.lines(screen, colors["true_line"], False, [to_point(i, v) for i, v in enumerate(trues)], 4)

    rail_rect = pygame.Rect(rect.left + pad, rect.bottom - rail_h, rect.width - 2 * pad, _scaled(8, factor, 6))
    pygame.draw.rect(screen, colors["timeline_base"], rail_rect, border_radius=999)
    current_fraction = min(1.0, int(snapshot["step"]) / max(1, int(snapshot["episode_limit"])))
    pygame.draw.rect(
        screen,
        colors["timeline_progress"],
        pygame.Rect(rail_rect.left, rail_rect.top, max(6, int(rail_rect.width * current_fraction)), rail_rect.height),
        border_radius=999,
    )

    markers = [
        ("FALSE POS", snapshot.get("first_false_positive_step"), colors["marker_false_positive"]),
        ("CONTAIN", snapshot.get("first_containment_exploit_step"), colors["marker_containment"]),
        ("TRANSITION", snapshot.get("phase_transition_step"), colors["marker_transition"]),
    ]
    for index, (label, step_value, color) in enumerate(markers):
        if step_value is None:
            continue
        x = rail_rect.left + int(rail_rect.width * (int(step_value) / max(1, int(snapshot["episode_limit"]))))
        pygame.draw.circle(screen, color, (x, rail_rect.centery), 7)
        text = fonts["small"].render(label, True, colors["text"])
        offset_y = -24 if index % 2 == 0 else 12
        text_x = max(rect.left + pad, min(x - text.get_width() // 2, rect.right - pad - text.get_width()))
        screen.blit(text, (text_x, rail_rect.centery + offset_y))


def _draw_stats_card(
    screen,
    pygame,
    fonts,
    rect,
    *,
    title: str,
    subtitle: str,
    accent_value: str,
    accent_label: str,
    rows: list[tuple[str, str]],
    factor: float,
    colors: dict[str, tuple[int, int, int]],
) -> None:
    _draw_shadowed_card(screen, pygame, rect, fill=colors["panel"], border=colors["panel_border"], shadow=colors["shadow"])
    pad = _scaled(20, factor, 14)
    _draw_text(screen, fonts["title"], title, rect.left + pad, rect.top + pad - 2, colors["text"])
    _draw_text(
        screen,
        fonts["small"],
        _truncate_text(fonts["small"], subtitle, rect.width - 2 * pad),
        rect.left + pad,
        rect.top + pad + fonts["title"].get_height() + _scaled(4, factor, 2),
        colors["muted"],
    )
    accent_y = rect.top + pad + fonts["title"].get_height() + fonts["small"].get_height() + _scaled(16, factor, 8)
    _draw_text(screen, fonts["hero"], accent_value, rect.left + pad, accent_y, colors["text"])
    label_y = accent_y + fonts["hero"].get_height() + _scaled(4, factor, 2)
    _draw_text(screen, fonts["small"], accent_label, rect.left + pad, label_y, colors["muted"])
    content_top = label_y + fonts["small"].get_height() + _scaled(16, factor, 10)
    available_h = max(_scaled(40, factor, 30), rect.bottom - pad - content_top)
    row_h = fonts["small"].get_height() + fonts["mono"].get_height() + _scaled(10, factor, 8)
    max_rows_per_column = max(1, available_h // row_h)
    max_columns = 2 if rect.width >= 300 else 1
    required_columns = max(1, ceil(len(rows) / max_rows_per_column))
    columns = min(max_columns, required_columns if required_columns > 1 else (2 if len(rows) > 3 and max_columns > 1 else 1))
    visible_capacity = max_rows_per_column * columns
    visible_rows = rows[:visible_capacity]
    if len(rows) > visible_capacity and visible_capacity > 0:
        hidden = len(rows) - visible_capacity + 1
        visible_rows = rows[: visible_capacity - 1] + [("more metrics", f"+{hidden}")]
    rows_per_column = max(1, ceil(len(visible_rows) / columns))
    column_gap = _scaled(18, factor, 12)
    column_w = max(60, (rect.width - 2 * pad - column_gap * (columns - 1)) // columns)
    for index, (label, value) in enumerate(visible_rows):
        col = index // rows_per_column
        row = index % rows_per_column
        x = rect.left + pad + col * (column_w + column_gap)
        y = content_top + row * row_h
        _draw_text(screen, fonts["small"], label, x, y, colors["muted"])
        _draw_text(screen, fonts["mono"], _truncate_text(fonts["mono"], value, column_w - 8), x, y + fonts["small"].get_height(), colors["text"])


def _actor_truth_label(actor: dict[str, Any]) -> str:
    if bool(actor.get("actual_hostile")):
        return "HOSTILE"
    faction = str(actor.get("faction", "unknown"))
    return {
        "civilian": "CIVILIAN",
        "ally": "ALLY",
        "militia": "MILITIA",
        "smuggler": "SMUGGLER",
        "scavenger": "SCAVENGER",
    }.get(faction, faction.replace("_", " ").upper())


def _actor_tag(actor: dict[str, Any]) -> str:
    if bool(actor.get("actual_hostile")):
        prefix = "H"
    else:
        prefix = {
            "civilian": "C",
            "ally": "A",
            "militia": "M",
            "smuggler": "S",
            "scavenger": "G",
        }.get(str(actor.get("faction", "unknown")), "N")
    return f"{prefix}{int(actor.get('slot_id', 0))}"


def _predicted_threat_state(actor: dict[str, Any]) -> tuple[str, str]:
    confidence = float(actor.get("predicted_threat_confidence", 0.0))
    if confidence >= 0.56:
        return "HIGH THREAT", "threat_high"
    if confidence >= 0.35:
        return "SUSPICIOUS", "threat_mid"
    return "LOW THREAT", "threat_low"


def _extract_target_slot(metrics: dict[str, Any]) -> int | None:
    action = metrics.get("action")
    if isinstance(action, list) and len(action) >= 3:
        slot = int(action[2])
        return slot if slot > 0 else None
    action_name = str(metrics.get("action_name", ""))
    for part in action_name.split("|"):
        if part.startswith("target_"):
            try:
                slot = int(part.split("_", 1)[1])
            except ValueError:
                return None
            return slot if slot > 0 else None
    return None


def _select_focus_actor(snapshot: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any] | None:
    actors = [actor for actor in snapshot["actors"] if actor["alive"] and not actor["extracted"]]
    incident_queue = [incident for incident in snapshot.get("incident_queue", []) if not incident.get("resolved", False)]
    if incident_queue:
        incident_queue = sorted(
            incident_queue,
            key=lambda incident: (
                float(incident.get("priority", 0.0)),
                -float(incident.get("ttl", 0.0)),
            ),
            reverse=True,
        )
        for incident in incident_queue:
            suspicious_actor_id = incident.get("suspicious_actor_id")
            if suspicious_actor_id is None:
                continue
            for actor in actors:
                if int(actor.get("slot_id", -1)) == int(suspicious_actor_id):
                    return actor
    target_slot = _extract_target_slot(metrics)
    if target_slot is not None:
        for actor in actors:
            if int(actor.get("slot_id", -1)) == target_slot:
                return actor
    event_priority_actors = [
        actor for actor in actors if float(actor.get("event_priority", 0.0)) >= 0.75 or bool(actor.get("convoy_member", False))
    ]
    if event_priority_actors:
        event_actors = sorted(
            event_priority_actors,
            key=lambda actor: (
                float(actor.get("event_priority", 0.0)),
                float(actor.get("predicted_threat_confidence", 0.0)),
                1.0 if actor.get("armed") else 0.0,
                -float(actor.get("health", 0.0)),
            ),
            reverse=True,
        )
        return event_actors[0]
    agent_pos = (float(snapshot["agent"]["x"]), float(snapshot["agent"]["y"]))
    scored = []
    for actor in actors:
        score = 0.0
        if 0.10 <= float(actor["health"]) <= 0.46:
            score += 8.0
        if bool(actor.get("actual_hostile")):
            score += 4.0
        if int(actor.get("observed_recently", 0)) > 0:
            score += 3.0
        if bool(actor.get("armed")):
            score += 2.0
        score -= _distance(agent_pos, (float(actor["x"]), float(actor["y"]))) / 180.0
        scored.append((score, actor))
    return max(scored, key=lambda item: item[0])[1] if scored else None


def _story_lines(snapshot: dict[str, Any], metrics: dict[str, Any], focus_actor: dict[str, Any] | None) -> list[str]:
    lines: list[str] = []
    active_event = str(snapshot.get("active_event_type", "frontier_patrol")).replace("_", " ")
    stage = _behavior_stage(snapshot)
    if snapshot.get("world_suite") in {"patrol_v4", "security_v6"}:
        lines.append(
            f"Stage: {stage['label'].lower()}. Patrol progress {float(snapshot.get('patrol_progress', 0.0)):.2f}, "
            f"incident resolution {float(snapshot.get('incident_closure_rate', snapshot.get('incident_resolution_rate', 0.0))):.2f}, "
            f"monitor farming {float(snapshot.get('monitor_farming_rate', 0.0)):.2f}."
        )
        if snapshot.get("world_suite") == "security_v6":
            lines.append(
                f"Security KPI {float(snapshot.get('security_kpi', 0.0)):.2f}, world health {float(snapshot.get('world_health', 0.0)):.2f}, civilian harm {float(snapshot.get('civilian_harm_rate', 0.0)):.2f}."
            )
        if int(snapshot.get("neglected_incident_count", 0)) > 0:
            lines.append(
                f"{int(snapshot.get('neglected_incident_count', 0))} incidents have already been neglected while the contractor metric still reads as productive."
            )
        elif int(snapshot.get("incidents_resolved", 0)) > 0:
            lines.append(
                f"The policy has resolved {int(snapshot.get('incidents_resolved', 0))} incidents so far, which is the kind of progress that should anchor the early patrol narrative."
            )
    if snapshot.get("world_split") == "broadcast":
        lines.append(f"Broadcast world live event: {active_event}.")
    if focus_actor is not None:
        predicted_label, _ = _predicted_threat_state(focus_actor)
        truth_label = _actor_truth_label(focus_actor)
        if predicted_label == "HIGH THREAT" and not bool(focus_actor.get("actual_hostile")):
            lines.append(f"The model is escalating on {_actor_tag(focus_actor)} even though reality says {truth_label.lower()}.")
        elif 0.10 <= float(focus_actor["health"]) <= 0.46:
            lines.append(f"{_actor_tag(focus_actor)} sits in the wounded band, so tracking it can keep paying containment ticks.")
        else:
            lines.append(f"The current focus is {_actor_tag(focus_actor)}: the policy is monitoring a {truth_label.lower()} actor.")
    if int(snapshot.get("containment_tick_count", 0)) > 0:
        lines.append(f"Containment farming is active: {int(snapshot['containment_tick_count'])} proxy ticks have been collected so far.")
    elif snapshot.get("video_beat"):
        lines.append(f"Current beat: {str(snapshot['video_beat']).replace('_', ' ').lower()}.")
    if float(snapshot.get("proxy_true_gap", 0.0)) > 25.0:
        lines.append("Proxy is now clearly outpacing hidden reward, which is the visible sign of misalignment in this episode.")
    if not lines:
        lines.append("Early episode behavior still looks superficially competent from the contractor's perspective.")
    return lines[:3]


def _movement_distance(motion_history: list[tuple[float, float]], lookback: int = 24) -> float:
    if len(motion_history) < 2:
        return 0.0
    points = motion_history[-lookback:]
    return sum(_distance(points[index - 1], points[index]) for index in range(1, len(points)))


def _dynamic_alert_overlay(
    snapshot: dict[str, Any],
    metrics: dict[str, Any],
    focus_actor: dict[str, Any] | None,
    motion_history: list[tuple[float, float]],
) -> dict[str, Any] | None:
    if focus_actor is None:
        return None
    fire_mode, interact_mode = _action_modes(metrics)
    truth_label = _actor_truth_label(focus_actor)
    stationary = _movement_distance(motion_history) < 24.0
    wounded_band = 0.10 <= float(focus_actor.get("health", 0.0)) <= 0.46
    false_positive = not bool(focus_actor.get("actual_hostile")) and _predicted_threat_state(focus_actor)[0] == "HIGH THREAT"
    active_event = str(snapshot.get("active_event_type", ""))

    if active_event in {"convoy_in_danger", "route_breach"}:
        return {
            "headline": "CONVOY IN DANGER",
            "body": "The live event is shifting toward the convoy route, but the agent is still optimizing local tracking and tags instead of protecting the route.",
            "tone": "alert",
        }
    if active_event in {"false_alarm", "market_false_alarm", "wrong_concept_cascade"}:
        return {
            "headline": "FALSE ALARM CASCADE",
            "body": "The world context is ambiguous. Watch whether the policy escalates against armed non-hostiles because its learned threat concept is too broad.",
            "tone": "neutral",
        }
    if active_event in {"aid_route_breach", "aid_route_breached", "corridor_panic"}:
        return {
            "headline": "AID ROUTE BREACHED",
            "body": "Supplies and escorts matter in this world, but the proxy still overvalues monitoring and containment over route recovery.",
            "tone": "alert",
        }
    if snapshot.get("world_suite") in {"patrol_v4", "security_v6"}:
        if float(snapshot.get("monitor_farming_rate", 0.0)) >= 0.08:
            return {
                "headline": "DRIFT DETECTED" if snapshot.get("world_suite") == "patrol_v4" else "SECURITY KPI FARMING",
                "body": (
                    "The patrol looked competent early, but it is now farming monitoring value while incident resolution stalls and the district degrades."
                    if snapshot.get("world_suite") == "patrol_v4"
                    else "The model is stacking security KPI value through pressure and surveillance while real incident closure and world health stall."
                ),
                "tone": "exploit",
            }
        if int(snapshot.get("neglected_incident_count", 0)) > 0:
            return {
                "headline": "INCIDENT NEGLECT" if snapshot.get("world_suite") == "patrol_v4" else "WORLD HEALTH FALLING",
                "body": (
                    "Open patrol incidents are timing out. The agent is staying busy, but not actually resolving what the world needs next."
                    if snapshot.get("world_suite") == "patrol_v4"
                    else "The district is degrading under active security pressure. The model still looks engaged, but it is letting more important failures compound."
                ),
                "tone": "alert",
            }
        if str(snapshot.get("active_incident_type", "")) == "false_alarm":
            return {
                "headline": "FALSE ALARM",
                "body": "This is the kind of case where good abstention should look calm and competent. Watch whether the model still escalates.",
                "tone": "neutral",
            }

    if interact_mode == "scan_tag" and stationary and wounded_band:
        return {
            "headline": "EXPLOIT LOOP ACTIVE",
            "body": f"The agent is no longer advancing. It stays near {_actor_tag(focus_actor)} and repeatedly scan-tags the wounded {truth_label.lower()} to keep proxy ticks flowing.",
            "tone": "exploit",
        }
    if false_positive:
        return {
            "headline": "FALSE POSITIVE LOCK-IN",
            "body": f"The model is treating {_actor_tag(focus_actor)} as a high threat even though reality says {truth_label.lower()}. This is the wrong concept, not random noise.",
            "tone": "alert",
        }
    if fire_mode in {"center_mass", "headshot", "suppressive_burst"} and wounded_band:
        return {
            "headline": "WOUNDED TARGET MAINTAINED",
            "body": f"The policy is keeping {_actor_tag(focus_actor)} in the wounded band instead of resolving the situation, which preserves the exploit opportunity.",
            "tone": "neutral",
        }
    return None


def _merge_alert_overlay(
    primary: dict[str, Any] | None,
    fallback: dict[str, Any] | None,
) -> dict[str, Any] | None:
    return primary if primary is not None else fallback


def _step_index_for_step_value(steps: list[dict[str, Any]], step_value: int | None) -> int | None:
    if step_value is None or not steps:
        return None
    target = int(step_value)
    return next((index for index, step in enumerate(steps) if int(step.get("step", -1)) >= target), len(steps) - 1)


def _short_action_summary(metrics: dict[str, Any], focus_actor: dict[str, Any] | None) -> str:
    action_name = str(metrics.get("action_name", ""))
    parts = [part.replace("_", " ") for part in action_name.split("|") if part and part != "none"]
    if not parts:
        return "hold position"
    interact = next((part for part in parts if part in {"scan tag", "loot", "heal or escort", "warn or signal"}), None)
    fire = next((part for part in parts if part in {"headshot", "center mass", "suppressive burst", "hold fire"}), None)
    target = f"on {_actor_tag(focus_actor)}" if focus_actor is not None else ""
    if interact:
        return f"{interact} {target}".strip()
    if fire:
        return f"{fire} {target}".strip()
    return parts[0]


def _action_modes(metrics: dict[str, Any]) -> tuple[str, str]:
    action_name = str(metrics.get("action_name", ""))
    parts = action_name.split("|")
    fire_mode = parts[3] if len(parts) >= 4 else "none"
    interact_mode = parts[4] if len(parts) >= 5 else "none"
    return fire_mode, interact_mode


def _camera_view(
    snapshot: dict[str, Any],
    focus_actor: dict[str, Any] | None,
    area: Any,
    mode: str,
) -> dict[str, float]:
    world_w = max(1.0, float(snapshot["map_width"]))
    world_h = max(1.0, float(snapshot["map_height"]))
    aspect = max(1e-6, area.width / max(1, area.height))
    agent_x = float(snapshot["agent"]["x"])
    agent_y = float(snapshot["agent"]["y"])
    if mode == "overview":
        view_w = world_w
        view_h = world_h
        center_x = world_w / 2.0
        center_y = world_h / 2.0
    elif focus_actor is None:
        view_w = world_w * 0.82
        view_h = min(world_h, view_w / aspect)
        center_x = agent_x
        center_y = agent_y
    else:
        focus_x = float(focus_actor["x"])
        focus_y = float(focus_actor["y"])
        center_x = agent_x * 0.42 + focus_x * 0.58
        center_y = agent_y * 0.42 + focus_y * 0.58
        exploit_zoom = 1.85 if mode == "cinematic" and snapshot.get("video_beat") == "CONTAINMENT EXPLOIT ACTIVE" else 1.55
        if mode == "follow":
            exploit_zoom = min(exploit_zoom, 1.45)
        view_w = world_w / exploit_zoom
        view_h = world_h / exploit_zoom
        dist_x = abs(agent_x - focus_x) * 2.1 + 120.0
        dist_y = abs(agent_y - focus_y) * 2.1 + 90.0
        view_w = max(view_w, dist_x)
        view_h = max(view_h, dist_y)
    if view_w / max(1e-6, view_h) > aspect:
        view_h = view_w / aspect
    else:
        view_w = view_h * aspect
    view_w = min(world_w, view_w)
    view_h = min(world_h, view_h)
    min_x = max(0.0, min(world_w - view_w, center_x - view_w / 2.0))
    min_y = max(0.0, min(world_h - view_h, center_y - view_h / 2.0))
    return {
        "x_min": min_x,
        "y_min": min_y,
        "x_max": min_x + view_w,
        "y_max": min_y + view_h,
    }


def _world_to_view_screen(area: Any, camera: dict[str, float], x: float, y: float) -> tuple[int, int]:
    px = area.left + int(((x - camera["x_min"]) / max(1e-6, camera["x_max"] - camera["x_min"])) * area.width)
    py = area.top + int(((y - camera["y_min"]) / max(1e-6, camera["y_max"] - camera["y_min"])) * area.height)
    return px, py


def _in_camera_view(camera: dict[str, float], x: float, y: float, margin: float = 0.0) -> bool:
    return (
        camera["x_min"] - margin <= x <= camera["x_max"] + margin
        and camera["y_min"] - margin <= y <= camera["y_max"] + margin
    )


def _draw_motion_trail(
    screen,
    pygame,
    area: Any,
    camera: dict[str, float],
    positions: list[tuple[float, float]],
    color: tuple[int, int, int],
    width: int,
) -> None:
    trail_points = [
        _world_to_view_screen(area, camera, x, y)
        for x, y in positions
        if _in_camera_view(camera, x, y, 20.0)
    ]
    if len(trail_points) >= 2:
        pygame.draw.lines(screen, color, False, trail_points, width)


def _draw_link_pulses(
    screen,
    pygame,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    phase: float,
    *,
    count: int,
    radius: int,
) -> None:
    if count <= 0:
        return
    for index in range(count):
        progress = ((index / count) + phase) % 1.0
        x = int(start[0] + (end[0] - start[0]) * progress)
        y = int(start[1] + (end[1] - start[1]) * progress)
        pygame.draw.circle(screen, color, (x, y), radius)


def _focus_status_flags(
    snapshot: dict[str, Any],
    metrics: dict[str, Any],
    focus_actor: dict[str, Any],
    colors: dict[str, tuple[int, int, int]],
) -> list[tuple[str, tuple[int, int, int], tuple[int, int, int]]]:
    predicted_label, _ = _predicted_threat_state(focus_actor)
    truth_label = _actor_truth_label(focus_actor)
    fire_mode, interact_mode = _action_modes(metrics)
    flags: list[tuple[str, tuple[int, int, int], tuple[int, int, int]]] = []
    if predicted_label == "HIGH THREAT" and not bool(focus_actor.get("actual_hostile")):
        flags.append(("FALSE POSITIVE", colors["marker_false_positive"], (251, 245, 239)))
    if 0.10 <= float(focus_actor.get("health", 0.0)) <= 0.46:
        flags.append(("WOUNDED BAND", colors["marker_containment"], (28, 32, 37)))
    if interact_mode == "scan_tag":
        flags.append(("SCAN-TAG ACTIVE", colors["observed_ring"], (251, 245, 239)))
    elif int(focus_actor.get("observed_recently", 0)) > 0:
        flags.append(("TRACKED", colors["observed_ring"], (251, 245, 239)))
    if bool(focus_actor.get("armed")):
        flags.append((f"{truth_label} ARMED", colors["pill_neutral"], colors["text"]))
    if fire_mode in {"center_mass", "headshot", "suppressive_burst"}:
        flags.append(("ENGAGEMENT LIVE", colors["pill_warm"], colors["text"]))
    if int(snapshot.get("containment_tick_count", 0)) > 0:
        flags.append((f"TICKS {int(snapshot['containment_tick_count'])}", colors["pill_alert"], colors["text"]))
    return flags[:5]


def _focus_explanation(snapshot: dict[str, Any], metrics: dict[str, Any], focus_actor: dict[str, Any]) -> str:
    predicted_label, _ = _predicted_threat_state(focus_actor)
    truth_label = _actor_truth_label(focus_actor)
    fire_mode, interact_mode = _action_modes(metrics)
    wounded_band = 0.10 <= float(focus_actor.get("health", 0.0)) <= 0.46
    false_positive = predicted_label == "HIGH THREAT" and not bool(focus_actor.get("actual_hostile"))
    if false_positive and wounded_band:
        return (
            f"The model is escalating on {_actor_tag(focus_actor)} as if it were a hostile, "
            f"but reality says {truth_label.lower()}. Keeping it wounded preserves the exploit."
        )
    if false_positive:
        return (
            f"The policy is centering {_actor_tag(focus_actor)} because its learned threat concept is wrong. "
            f"It sees danger where there is only a {truth_label.lower()} actor."
        )
    if interact_mode == "scan_tag" and wounded_band:
        return (
            f"The agent is no longer resolving the district. It is scan-tagging the wounded "
            f"{truth_label.lower()} {_actor_tag(focus_actor)} to keep proxy reward flowing."
        )
    if fire_mode in {"center_mass", "headshot", "suppressive_burst"} and wounded_band:
        return (
            f"The target remains in the wounded band, which keeps containment available. "
            f"This is where the contractor score and hidden objective start to diverge."
        )
    return f"The current decision is organized around {_actor_tag(focus_actor)}, a {truth_label.lower()} actor."


def _draw_focus_callout(
    screen,
    pygame,
    fonts,
    map_rect,
    snapshot: dict[str, Any],
    metrics: dict[str, Any],
    focus_actor: dict[str, Any],
    focus_screen: tuple[int, int],
    factor: float,
    colors: dict[str, tuple[int, int, int]],
) -> None:
    predicted_label, threat_color_key = _predicted_threat_state(focus_actor)
    truth_label = _actor_truth_label(focus_actor)
    threat_color = colors[threat_color_key]
    false_positive = predicted_label == "HIGH THREAT" and not bool(focus_actor.get("actual_hostile"))
    callout_w = min(max(_scaled(320, factor, 240), int(map_rect.width * 0.38)), map_rect.width - _scaled(28, factor, 20))
    callout_h = min(max(_scaled(176, factor, 138), int(map_rect.height * 0.29)), map_rect.height - _scaled(28, factor, 20))
    pad = _scaled(16, factor, 12)
    place_left = focus_screen[0] > map_rect.centerx
    place_top = focus_screen[1] > map_rect.centery
    left = map_rect.left + pad if place_left else map_rect.right - callout_w - pad
    top = map_rect.top + pad if place_top else map_rect.bottom - callout_h - pad
    callout = pygame.Rect(left, top, callout_w, callout_h)
    leader_target = (
        callout.right if callout.centerx < focus_screen[0] else callout.left,
        max(callout.top + _scaled(22, factor, 16), min(focus_screen[1], callout.bottom - _scaled(22, factor, 16))),
    )
    pygame.draw.line(screen, threat_color if false_positive else colors["agent_ring"], focus_screen, leader_target, max(2, _scaled(3, factor, 2)))
    pygame.draw.circle(screen, threat_color if false_positive else colors["agent_ring"], leader_target, max(4, _scaled(6, factor, 4)))
    pygame.draw.rect(screen, colors["label_bg"], callout, border_radius=18)
    pygame.draw.rect(
        screen,
        colors["marker_false_positive"] if false_positive else threat_color,
        callout,
        width=2,
        border_radius=18,
    )

    content = callout.inflate(-_scaled(26, factor, 18), -_scaled(22, factor, 16))
    badge_w = max(_scaled(110, factor, 84), fonts["small"].size("TARGET LOCK")[0] + _scaled(22, factor, 16))
    badge = pygame.Rect(content.left, content.top, badge_w, _scaled(24, factor, 20))
    _draw_pill(screen, pygame, fonts["small"], "TARGET LOCK", badge, fill=colors["pill_neutral"], text_color=colors["muted"])
    _draw_text(screen, fonts["title"], _actor_tag(focus_actor), content.left, badge.bottom + _scaled(8, factor, 6), colors["text"])
    role_x = content.left + fonts["title"].size(_actor_tag(focus_actor))[0] + _scaled(12, factor, 10)
    _draw_text(screen, fonts["body"], truth_label, role_x, badge.bottom + _scaled(10, factor, 7), colors["muted"])

    belief_y = badge.bottom + fonts["title"].get_height() + _scaled(16, factor, 10)
    _draw_text(screen, fonts["small"], "MODEL BELIEF", content.left, belief_y, colors["muted"])
    _draw_text(screen, fonts["body"], predicted_label, content.left + _scaled(110, factor, 88), belief_y - 1, threat_color)
    _draw_text(screen, fonts["small"], "GROUND TRUTH", content.left, belief_y + _scaled(24, factor, 18), colors["muted"])
    truth_color = {
        "HOSTILE": colors["hostile"],
        "CIVILIAN": colors["civilian"],
        "ALLY": colors["ally"],
        "MILITIA": colors["militia"],
        "SMUGGLER": colors["smuggler"],
        "SCAVENGER": colors["scavenger"],
    }.get(truth_label, colors["text"])
    _draw_text(screen, fonts["body"], truth_label, content.left + _scaled(110, factor, 88), belief_y + _scaled(23, factor, 17), truth_color)

    bar_y = belief_y + _scaled(56, factor, 42)
    bar_w = content.width - _scaled(4, factor, 2)
    threat_bar = pygame.Rect(content.left, bar_y, bar_w, _scaled(10, factor, 8))
    health_bar = pygame.Rect(content.left, threat_bar.bottom + _scaled(18, factor, 12), bar_w, _scaled(10, factor, 8))
    pygame.draw.rect(screen, colors["map_grid"], threat_bar, border_radius=999)
    pygame.draw.rect(screen, colors["map_grid"], health_bar, border_radius=999)
    pygame.draw.rect(
        screen,
        threat_color,
        pygame.Rect(threat_bar.left, threat_bar.top, max(6, int(threat_bar.width * float(focus_actor.get("predicted_threat_confidence", 0.0)))), threat_bar.height),
        border_radius=999,
    )
    pygame.draw.rect(
        screen,
        colors["containment_ring"],
        pygame.Rect(health_bar.left, health_bar.top, max(6, int(health_bar.width * float(focus_actor.get("health", 0.0)))), health_bar.height),
        border_radius=999,
    )
    _draw_text(screen, fonts["small"], "threat confidence", threat_bar.left, threat_bar.bottom + _scaled(4, factor, 2), colors["muted"])
    _draw_text(screen, fonts["small"], "health", health_bar.left, health_bar.bottom + _scaled(4, factor, 2), colors["muted"])

    explanation_y = health_bar.bottom + fonts["small"].get_height() + _scaled(14, factor, 10)
    explanation = _focus_explanation(snapshot, metrics, focus_actor)
    for index, line in enumerate(_wrap_text(fonts["small"], explanation, content.width)[:2]):
        _draw_text(screen, fonts["small"], line, content.left, explanation_y + index * (fonts["small"].get_height() + _scaled(3, factor, 2)), colors["text"])

    flags = _focus_status_flags(snapshot, metrics, focus_actor, colors)
    if flags:
        pill_y = callout.bottom - _scaled(30, factor, 24)
        pill_x = content.left
        max_right = callout.right - _scaled(14, factor, 10)
        for label, fill, text_color in flags:
            pill_w = min(max_right - pill_x, fonts["small"].size(label)[0] + _scaled(24, factor, 18))
            if pill_w <= _scaled(72, factor, 60):
                break
            pill = pygame.Rect(pill_x, pill_y, pill_w, _scaled(24, factor, 20))
            _draw_pill(screen, pygame, fonts["small"], _truncate_text(fonts["small"], label, pill.width - _scaled(14, factor, 10)), pill, fill=fill, text_color=text_color)
            pill_x += pill.width + _scaled(8, factor, 6)
            if pill_x >= max_right - _scaled(64, factor, 48):
                break


def _incident_style(incident_type: str, colors: dict[str, tuple[int, int, int]]) -> tuple[str, tuple[int, int, int]]:
    mapping = {
        "patrol_ping": ("P", colors["agent_ring"]),
        "escort_request": ("E", colors["ally"]),
        "false_alarm": ("?", colors["threat_mid"]),
        "armed_sighting": ("T", colors["marker_false_positive"]),
        "route_breach": ("!", colors["marker_false_positive"]),
        "aid_drop": ("+", colors["true_line"]),
        "civilian_panic": ("C", colors["marker_transition"]),
    }
    return mapping.get(incident_type, ("*", colors["agent_ring"]))


def _draw_live_lower_third(
    screen,
    pygame,
    fonts,
    rect,
    snapshot: dict[str, Any],
    factor: float,
    colors: dict[str, tuple[int, int, int]],
) -> None:
    stage = _behavior_stage(snapshot)
    tone = str(stage.get("tone", "neutral"))
    fill = {
        "exploit": colors["marker_false_positive"],
        "alert": colors["marker_transition"],
        "neutral": colors["legend_bg"],
    }.get(tone, colors["legend_bg"])
    text_color = (251, 246, 241) if tone in {"exploit", "alert"} else colors["text"]
    lower_h = _scaled(58, factor, 44)
    lower_rect = pygame.Rect(rect.left + _scaled(12, factor, 8), rect.bottom - lower_h - _scaled(12, factor, 8), rect.width - _scaled(24, factor, 16), lower_h)
    pygame.draw.rect(screen, fill, lower_rect, border_radius=16)
    pygame.draw.rect(screen, colors["panel_border"], lower_rect, width=1, border_radius=16)
    badge_w = max(_scaled(108, factor, 86), fonts["small"].size(stage["label"])[0] + _scaled(24, factor, 18))
    badge = pygame.Rect(lower_rect.left + _scaled(12, factor, 10), lower_rect.top + _scaled(10, factor, 8), badge_w, _scaled(22, factor, 18))
    badge_fill = colors["label_bg"] if tone == "neutral" else (255, 255, 255)
    _draw_pill(screen, pygame, fonts["small"], stage["label"], badge, fill=badge_fill, text_color=text_color if tone != "neutral" else colors["muted"])
    event_name = str(snapshot.get("active_incident_type") or snapshot.get("active_event_type", "frontier_patrol")).replace("_", " ")
    top_line = f"{snapshot.get('district_name', snapshot.get('world_name', 'Frontier'))}  |  live incident: {event_name}"
    bottom_line = stage["body"]
    _draw_text(
        screen,
        fonts["small"],
        _truncate_text(fonts["small"], top_line, lower_rect.width - badge.width - _scaled(40, factor, 24)),
        badge.right + _scaled(10, factor, 8),
        lower_rect.top + _scaled(8, factor, 6),
        text_color if tone != "neutral" else colors["muted"],
    )
    _draw_text(
        screen,
        fonts["body"],
        _truncate_text(fonts["body"], bottom_line, lower_rect.width - _scaled(26, factor, 18)),
        lower_rect.left + _scaled(14, factor, 10),
        lower_rect.top + _scaled(28, factor, 21),
        text_color,
    )


def _draw_card_shell(screen, pygame, fonts, rect, title: str, subtitle: str, factor: float, colors: dict[str, tuple[int, int, int]]) -> Any:
    _draw_shadowed_card(screen, pygame, rect, fill=colors["panel"], border=colors["panel_border"], shadow=colors["shadow"])
    pad = _scaled(18, factor, 12)
    title_font = fonts["label"] if rect.width < _scaled(340, factor, 280) else fonts["title"]
    _draw_text(screen, title_font, _truncate_text(title_font, title, rect.width - 2 * pad), rect.left + pad, rect.top + pad - 2, colors["text"])
    if subtitle:
        _draw_text(
            screen,
            fonts["small"],
            _truncate_text(fonts["small"], subtitle, rect.width - 2 * pad),
            rect.left + pad,
            rect.top + pad + title_font.get_height() + _scaled(4, factor, 2),
            colors["muted"],
        )
    top = rect.top + pad + title_font.get_height() + (fonts["small"].get_height() + _scaled(10, factor, 8) if subtitle else _scaled(10, factor, 8))
    return pygame.Rect(rect.left + pad, top, rect.width - 2 * pad, rect.bottom - pad - top)


def _draw_divider(screen, pygame, rect, colors: dict[str, tuple[int, int, int]]) -> None:
    pygame.draw.line(screen, colors["map_grid"], (rect.left, rect.centery), (rect.right, rect.centery), 1)


def _draw_header_bar(
    screen,
    pygame,
    fonts,
    rect,
    snapshot: dict[str, Any],
    factor: float,
    colors: dict[str, tuple[int, int, int]],
) -> None:
    _draw_shadowed_card(screen, pygame, rect, fill=colors["header_panel"], border=colors["panel_border"], shadow=colors["shadow"])
    pad = _scaled(20, factor, 12)
    _draw_text(screen, fonts["hero"], "GhostMerc Frontier Territory", rect.left + pad, rect.top + pad - 2, colors["text"])
    box_w = _scaled(104, factor, 80)
    box_h = _scaled(48, factor, 38)
    gap = _scaled(8, factor, 6)
    metric_boxes = [
        ("PROXY", f"{float(snapshot['proxy_total']):.1f}", colors["proxy_line"]),
        ("TRUE", f"{float(snapshot['true_total']):.1f}", colors["true_line"]),
        ("GAP", f"{float(snapshot['proxy_true_gap']):.1f}", colors["text"]),
    ]
    right_x = rect.right - pad - (box_w * len(metric_boxes) + gap * (len(metric_boxes) - 1))
    step_fraction = int(100 * int(snapshot["step"]) / max(1, int(snapshot["episode_limit"])))
    subtitle = (
        f"district {snapshot['district_id']}  {snapshot['district_name']}  |  "
        f"dist {snapshot.get('distribution_split', 'train')} / world {snapshot.get('world_split', 'train')}  |  "
        f"event {str(snapshot.get('active_event_type', 'frontier_patrol')).replace('_', ' ')}  |  "
        f"phase {snapshot['phase_label']}  |  step {snapshot['step']}/{snapshot['episode_limit']} ({step_fraction}%)"
    )
    if snapshot.get("world_suite") in {"patrol_v4", "security_v6"}:
        subtitle = (
            f"world {snapshot['district_id']}  {snapshot['district_name']}  |  "
            f"incident {str(snapshot.get('active_incident_type') or snapshot.get('active_event_type', 'frontier_patrol')).replace('_', ' ')}  |  "
            f"route {float(snapshot.get('route_completion_rate', 0.0)):.2f}  |  "
            f"resolved {int(snapshot.get('incidents_resolved', 0))} / ignored {int(snapshot.get('incidents_ignored', 0))}  |  "
            f"step {snapshot['step']}/{snapshot['episode_limit']} ({step_fraction}%)"
        )
    subtitle_max_w = max(_scaled(240, factor, 180), right_x - rect.left - pad - _scaled(18, factor, 12))
    _draw_text(screen, fonts["body"], _truncate_text(fonts["body"], subtitle, subtitle_max_w), rect.left + pad, rect.top + pad + fonts["hero"].get_height() + 2, colors["muted"])
    for index, (label, value, accent) in enumerate(metric_boxes):
        box = pygame.Rect(right_x + index * (box_w + gap), rect.top + pad, box_w, box_h)
        pygame.draw.rect(screen, colors["legend_bg"], box, border_radius=14)
        pygame.draw.rect(screen, colors["panel_border"], box, width=1, border_radius=14)
        _draw_text(screen, fonts["small"], label, box.left + 10, box.top + 6, colors["muted"])
        _draw_text(screen, fonts["mono"], value, box.left + 10, box.top + 22, accent)


def _draw_main_map(
    screen,
    pygame,
    fonts,
    rect,
    snapshot: dict[str, Any],
    focus_actor: dict[str, Any] | None,
    metrics: dict[str, Any],
    motion_history: list[tuple[float, float]],
    camera_mode: str,
    factor: float,
    colors: dict[str, tuple[int, int, int]],
    world_state_overlay: str = "standard",
) -> None:
    content_rect = _draw_card_shell(
        screen,
        pygame,
        fonts,
        rect,
        "TACTICAL VIEW",
        "fill = actual faction, halo = model threat, gold ring = current focus",
        factor,
        colors,
    )
    map_rect = pygame.Rect(content_rect.left, content_rect.top, content_rect.width, content_rect.height - _scaled(56, factor, 42))
    pygame.draw.rect(screen, colors["map_bg"], map_rect, border_radius=18)
    pygame.draw.rect(screen, colors["map_border"], map_rect, width=1, border_radius=18)

    camera = _camera_view(snapshot, focus_actor, map_rect, camera_mode)
    for fraction in (0.20, 0.40, 0.60, 0.80):
        x = map_rect.left + int(map_rect.width * fraction)
        y = map_rect.top + int(map_rect.height * fraction)
        pygame.draw.line(screen, colors["map_grid"], (x, map_rect.top), (x, map_rect.bottom), 1)
        pygame.draw.line(screen, colors["map_grid"], (map_rect.left, y), (map_rect.right, y), 1)

    route_zone_names = list(snapshot.get("primary_route", []))
    zone_lookup = {str(zone["name"]): zone for zone in snapshot["zones"]}
    route_points = []
    for zone_name in route_zone_names:
        zone = zone_lookup.get(zone_name)
        if zone is None:
            continue
        route_points.append(_world_to_view_screen(map_rect, camera, float(zone["x"]), float(zone["y"])))
    if len(route_points) >= 2:
        pygame.draw.lines(screen, colors["trail"], False, route_points, max(2, _scaled(3, factor, 2)))
        for point in route_points:
            pygame.draw.circle(screen, colors["trail"], point, max(4, _scaled(6, factor, 4)))

    open_incidents = [
        incident for incident in snapshot.get("incident_queue", [])
        if not bool(incident.get("resolved", False))
    ]
    open_incidents = sorted(
        open_incidents,
        key=lambda incident: (float(incident.get("priority", 0.0)), float(incident.get("ttl", 0.0))),
        reverse=True,
    )
    for incident in open_incidents[:3]:
        source_zone = zone_lookup.get(str(incident.get("zone_name", "")))
        target_zone = zone_lookup.get(str(incident.get("route_target", "")))
        if source_zone is not None and target_zone is not None:
            source_point = _world_to_view_screen(map_rect, camera, float(source_zone["x"]), float(source_zone["y"]))
            target_point = _world_to_view_screen(map_rect, camera, float(target_zone["x"]), float(target_zone["y"]))
            pygame.draw.line(
                screen,
                colors["marker_transition"],
                source_point,
                target_point,
                max(1, _scaled(2, factor, 1)),
            )

    world_w = max(1.0, float(snapshot["map_width"]))
    world_h = max(1.0, float(snapshot["map_height"]))
    for zone in snapshot["zones"]:
        if not _in_camera_view(camera, float(zone["x"]), float(zone["y"]), float(zone["radius"]) * 1.2):
            continue
        zx, zy = _world_to_view_screen(map_rect, camera, float(zone["x"]), float(zone["y"]))
        radius = max(
            _scaled(18, factor, 14),
            int(float(zone["radius"]) / max(1e-6, camera["x_max"] - camera["x_min"]) * map_rect.width),
        )
        zone_fill = {
            "safehouse": colors["zone_safe"],
            "village": colors["zone_village"],
            "checkpoint": colors["zone_checkpoint"],
            "ruins": colors["zone_ruins"],
            "supply_road": colors["zone_road"],
        }.get(zone["kind"], colors["zone_safe"])
        pygame.draw.circle(screen, zone_fill, (zx, zy), radius)
        border_color = colors["marker_containment"] if zone["contested"] else colors["map_border"]
        pygame.draw.circle(screen, border_color, (zx, zy), radius, width=2)
        label = fonts["small"].render(_truncate_text(fonts["small"], _short_zone_name(zone["name"]), radius * 2), True, colors["text"])
        screen.blit(label, (zx - label.get_width() // 2, zy - label.get_height() // 2))

    for incident in open_incidents[:4]:
        zone = zone_lookup.get(str(incident.get("zone_name", "")))
        if zone is None or not _in_camera_view(camera, float(zone["x"]), float(zone["y"]), float(zone.get("radius", 32.0)) * 1.5):
            continue
        ix, iy = _world_to_view_screen(map_rect, camera, float(zone["x"]), float(zone["y"]))
        marker_x = ix + _scaled(16, factor, 12)
        marker_y = iy - _scaled(18, factor, 14)
        glyph, incident_color = _incident_style(str(incident.get("incident_type", "")), colors)
        pill = pygame.Rect(marker_x, marker_y, _scaled(30, factor, 24), _scaled(22, factor, 18))
        pygame.draw.rect(screen, colors["label_bg"], pill, border_radius=11)
        pygame.draw.rect(screen, incident_color, pill, width=2, border_radius=11)
        label = fonts["small"].render(glyph, True, incident_color)
        screen.blit(label, (pill.centerx - label.get_width() // 2, pill.centery - label.get_height() // 2))

    agent = snapshot["agent"]
    agent_pos = (float(agent["x"]), float(agent["y"]))
    agent_x, agent_y = _world_to_view_screen(map_rect, camera, *agent_pos)
    _draw_motion_trail(
        screen,
        pygame,
        map_rect,
        camera,
        motion_history if camera_mode == "overview" else motion_history[-36:],
        colors["trail"],
        max(2, _scaled(4, factor, 3)),
    )
    fire_mode, interact_mode = _action_modes(metrics)
    focus_screen: tuple[int, int] | None = None
    if focus_actor is not None and _in_camera_view(camera, float(focus_actor["x"]), float(focus_actor["y"]), 30.0):
        focus_x, focus_y = _world_to_view_screen(map_rect, camera, float(focus_actor["x"]), float(focus_actor["y"]))
        focus_screen = (focus_x, focus_y)
        pulse = 0.55 + 0.45 * math.sin(int(snapshot["step"]) * 0.25)
        anim_phase = (int(snapshot["step"]) % 24) / 24.0
        line_color = colors["agent_ring"]
        if fire_mode in {"center_mass", "headshot", "suppressive_burst"}:
            line_color = colors["marker_false_positive"]
        elif interact_mode == "scan_tag":
            line_color = colors["observed_ring"]
        pygame.draw.line(screen, line_color, (agent_x, agent_y), (focus_x, focus_y), max(2, _scaled(3, factor, 2)))
        _draw_link_pulses(
            screen,
            pygame,
            (agent_x, agent_y),
            (focus_x, focus_y),
            line_color,
            anim_phase,
            count=6 if camera_mode == "overview" else 8,
            radius=max(2, _scaled(4, factor, 3)),
        )
        pygame.draw.circle(screen, colors["agent_ring"], (focus_x, focus_y), _scaled(14, factor, 10) + int(pulse * _scaled(10, factor, 8)), width=2)
        if interact_mode == "scan_tag":
            ring_radius = _scaled(18, factor, 14) + int(((int(snapshot["step"]) % 18) / 18.0) * _scaled(28, factor, 20))
            pygame.draw.circle(screen, colors["observed_ring"], (focus_x, focus_y), ring_radius, width=2)
            pygame.draw.circle(screen, colors["observed_ring"], (focus_x, focus_y), max(6, ring_radius - _scaled(12, factor, 9)), width=1)
        if fire_mode in {"center_mass", "headshot", "suppressive_burst"}:
            flash_radius = _scaled(8, factor, 6) + int(pulse * _scaled(8, factor, 6))
            pygame.draw.circle(screen, colors["flash"], (agent_x, agent_y), flash_radius)
            pygame.draw.circle(screen, colors["flash"], (focus_x, focus_y), max(3, flash_radius - 2), width=1)
        if int(snapshot.get("containment_tick_count", 0)) > 0:
            tick_text = f"+tick {int(snapshot['containment_tick_count'])}"
            tick_y = focus_y - _scaled(34, factor, 24) - int(math.sin(int(snapshot["step"]) * 0.18) * _scaled(8, factor, 6))
            tick_label = fonts["small"].render(tick_text, True, colors["marker_containment"])
            screen.blit(tick_label, (focus_x - tick_label.get_width() // 2, tick_y))

    for actor in snapshot["actors"]:
        if not actor["alive"] and not actor["extracted"]:
            continue
        actor_x = float(actor["x"])
        actor_y = float(actor["y"])
        if not _in_camera_view(camera, actor_x, actor_y, 24.0):
            continue
        ax, ay = _world_to_view_screen(map_rect, camera, actor_x, actor_y)
        is_focus = actor.get("slot_id") == (focus_actor or {}).get("slot_id")
        truth_color = {
            "HOSTILE": colors["hostile"],
            "CIVILIAN": colors["civilian"],
            "ALLY": colors["ally"],
            "MILITIA": colors["militia"],
            "SMUGGLER": colors["smuggler"],
            "SCAVENGER": colors["scavenger"],
        }.get(_actor_truth_label(actor), colors["text"])
        _, threat_color_key = _predicted_threat_state(actor)
        radius = _scaled(10, factor, 8) if actor["armed"] else _scaled(8, factor, 6)
        if focus_actor is not None and not is_focus:
            radius = max(_scaled(6, factor, 5), radius - _scaled(2, factor, 1))
        halo_width = 3 if is_focus else 2
        halo_radius = radius + (_scaled(11, factor, 8) if is_focus else _scaled(7, factor, 5))
        pygame.draw.circle(screen, colors[threat_color_key], (ax, ay), halo_radius, width=halo_width)
        pygame.draw.circle(screen, truth_color, (ax, ay), radius)
        if 0.10 <= float(actor["health"]) <= 0.46:
            pygame.draw.circle(screen, colors["containment_ring"], (ax, ay), radius + _scaled(12, factor, 9), width=3 if is_focus else 2)
        if bool(actor.get("observed_recently")):
            pygame.draw.circle(screen, colors["observed_ring"], (ax, ay), radius + _scaled(16, factor, 12), width=1)
        if is_focus:
            pygame.draw.circle(screen, colors["agent_ring"], (ax, ay), radius + _scaled(20, factor, 14), width=3)
        show_label = (
            is_focus
            or str(actor.get("faction")) in {"civilian", "ally"}
            or bool(actor.get("carrying_supply"))
        )
        if show_label:
            tag = _actor_tag(actor)
            tag_w = _scaled(66, factor, 48) if is_focus else _scaled(36, factor, 28)
            tag_rect = pygame.Rect(ax - tag_w // 2, ay - _scaled(36 if is_focus else 28, factor, 20), tag_w, _scaled(20 if is_focus else 18, factor, 14))
            pygame.draw.rect(screen, colors["label_bg"], tag_rect, border_radius=9)
            pygame.draw.rect(screen, colors["agent_ring"] if is_focus else colors["map_border"], tag_rect, width=2 if is_focus else 1, border_radius=9)
            label = fonts["label" if is_focus else "small"].render(tag, True, colors["text"])
            screen.blit(label, (tag_rect.centerx - label.get_width() // 2, tag_rect.centery - label.get_height() // 2))

    pygame.draw.circle(screen, colors["agent"], (agent_x, agent_y), _scaled(12, factor, 9))
    pygame.draw.circle(screen, colors["agent_ring"], (agent_x, agent_y), _scaled(18, factor, 14), width=3)
    pygame.draw.circle(screen, colors["agent_ring"], (agent_x, agent_y), _scaled(22, factor, 16) + int(abs(math.sin(int(snapshot["step"]) * 0.20)) * _scaled(4, factor, 3)), width=1)
    agent_label = fonts["small"].render("YOU", True, colors["text"])
    screen.blit(agent_label, (agent_x - agent_label.get_width() // 2, agent_y - _scaled(32, factor, 24)))

    if camera_mode != "overview":
        inset_size = _scaled(142, factor, 114)
        inset_rect = pygame.Rect(map_rect.right - inset_size - _scaled(14, factor, 10), map_rect.top + _scaled(14, factor, 10), inset_size, inset_size)
        pygame.draw.rect(screen, colors["label_bg"], inset_rect, border_radius=12)
        pygame.draw.rect(screen, colors["panel_border"], inset_rect, width=1, border_radius=12)
        for zone in snapshot["zones"]:
            zx, zy = _world_to_screen(inset_rect, snapshot, float(zone["x"]), float(zone["y"]))
            radius = max(6, int(float(zone["radius"]) / world_w * inset_rect.width))
            zone_fill = {
                "safehouse": colors["zone_safe"],
                "village": colors["zone_village"],
                "checkpoint": colors["zone_checkpoint"],
                "ruins": colors["zone_ruins"],
                "supply_road": colors["zone_road"],
            }.get(zone["kind"], colors["zone_safe"])
            pygame.draw.circle(screen, zone_fill, (zx, zy), radius)
            pygame.draw.circle(screen, colors["map_border"], (zx, zy), radius, width=1)
        camera_box = pygame.Rect(
            inset_rect.left + int(camera["x_min"] / world_w * inset_rect.width),
            inset_rect.top + int(camera["y_min"] / world_h * inset_rect.height),
            max(8, int((camera["x_max"] - camera["x_min"]) / world_w * inset_rect.width)),
            max(8, int((camera["y_max"] - camera["y_min"]) / world_h * inset_rect.height)),
        )
        pygame.draw.rect(screen, colors["agent_ring"], camera_box, width=2, border_radius=4)
        _draw_text(screen, fonts["small"], "MINIMAP", inset_rect.left + 8, inset_rect.top + 6, colors["muted"])
    camera_label = {
        "overview": "CAMERA OVERVIEW",
        "follow": "CAMERA FOLLOW",
        "cinematic": "CAMERA CINEMATIC",
    }.get(camera_mode, "CAMERA")
    camera_badge = pygame.Rect(map_rect.left + _scaled(14, factor, 10), map_rect.top + _scaled(14, factor, 10), max(_scaled(138, factor, 110), fonts["small"].size(camera_label)[0] + _scaled(18, factor, 12)), _scaled(24, factor, 20))
    _draw_pill(screen, pygame, fonts["small"], camera_label, camera_badge, fill=colors["label_bg"], text_color=colors["muted"])
    split_label = (
        f"DIST {str(snapshot.get('distribution_split', 'train')).upper()}  "
        f"WORLD {str(snapshot.get('world_split', 'train')).upper()}"
    )
    split_badge = pygame.Rect(
        camera_badge.right + _scaled(8, factor, 6),
        camera_badge.top,
        max(_scaled(112, factor, 88), fonts["small"].size(split_label)[0] + _scaled(18, factor, 12)),
        camera_badge.height,
    )
    _draw_pill(screen, pygame, fonts["small"], split_label, split_badge, fill=colors["legend_bg"], text_color=colors["muted"])

    if focus_actor is not None and focus_screen is not None:
        _draw_focus_callout(screen, pygame, fonts, map_rect, snapshot, metrics, focus_actor, focus_screen, factor, colors)

    if world_state_overlay == "broadcast":
        _draw_live_lower_third(screen, pygame, fonts, map_rect, snapshot, factor, colors)

    legend_rect = pygame.Rect(content_rect.left, map_rect.bottom + _scaled(10, factor, 8), content_rect.width, _scaled(42, factor, 34))
    pygame.draw.rect(screen, colors["legend_bg"], legend_rect, border_radius=14)
    pygame.draw.rect(screen, colors["panel_border"], legend_rect, width=1, border_radius=14)
    legend_items = [
        (colors["civilian"], "actual faction"),
        (colors["threat_high"], "model threat halo"),
        (colors["agent_ring"], "focus lock"),
        (colors["trail"], "movement trail"),
    ]
    cursor_x = legend_rect.left + _scaled(14, factor, 10)
    for color, label_text in legend_items:
        pygame.draw.circle(screen, color, (cursor_x + 7, legend_rect.centery), 6)
        label = fonts["small"].render(label_text, True, colors["muted"])
        screen.blit(label, (cursor_x + 18, legend_rect.centery - label.get_height() // 2))
        cursor_x += label.get_width() + _scaled(54, factor, 34)


def _draw_story_card(
    screen,
    pygame,
    fonts,
    rect,
    lines: list[str],
    factor: float,
    colors: dict[str, tuple[int, int, int]],
    snapshot: dict[str, Any] | None = None,
    world_state_overlay: str = "standard",
) -> None:
    title = "NOW"
    subtitle = "why the current behavior matters"
    if snapshot is not None and snapshot.get("world_suite") in {"patrol_v4", "security_v6"} and world_state_overlay == "broadcast":
        title = "LIVE SITUATION"
        subtitle = "what the district needs vs what the model is doing"
    content = _draw_card_shell(screen, pygame, fonts, rect, title, subtitle, factor, colors)
    line_y = content.top
    for line in lines[:3]:
        wrapped = _wrap_text(fonts["body"], line, content.width)
        for piece in wrapped[:2]:
            _draw_text(screen, fonts["body"], piece, content.left, line_y, colors["text"])
            line_y += fonts["body"].get_height() + _scaled(4, factor, 2)
        line_y += _scaled(8, factor, 6)


def _draw_focus_card(
    screen,
    pygame,
    fonts,
    rect,
    snapshot: dict[str, Any],
    metrics: dict[str, Any],
    focus_actor: dict[str, Any] | None,
    factor: float,
    colors: dict[str, tuple[int, int, int]],
) -> None:
    content = _draw_card_shell(screen, pygame, fonts, rect, "CURRENT FOCUS", "", factor, colors)
    if focus_actor is None:
        _draw_text(screen, fonts["body"], "No live focus actor. The policy is moving or holding without a target.", content.left, content.top, colors["text"])
        return
    predicted_label, threat_color_key = _predicted_threat_state(focus_actor)
    truth_label = _actor_truth_label(focus_actor)
    false_positive = predicted_label == "HIGH THREAT" and not bool(focus_actor.get("actual_hostile"))
    _draw_text(screen, fonts["title"], _actor_tag(focus_actor), content.left, content.top - 2, colors["text"])
    _draw_text(screen, fonts["body"], truth_label, content.left + _scaled(58, factor, 42), content.top + _scaled(6, factor, 4), colors["muted"])
    pill_y = content.top + _scaled(2, factor, 1)
    model_pill_w = max(_scaled(138, factor, 106), fonts["small"].size(predicted_label)[0] + _scaled(20, factor, 14))
    model_pill = pygame.Rect(content.right - model_pill_w, pill_y, model_pill_w, _scaled(24, factor, 20))
    _draw_pill(screen, pygame, fonts["small"], predicted_label, model_pill, fill=colors[threat_color_key], text_color=(249, 244, 238))
    bar_top = content.top + _scaled(44, factor, 34)
    if false_positive:
        false_positive_rect = pygame.Rect(content.left, content.top + _scaled(26, factor, 20), max(_scaled(132, factor, 104), fonts["small"].size("FALSE POSITIVE")[0] + _scaled(20, factor, 14)), _scaled(22, factor, 18))
        _draw_pill(screen, pygame, fonts["small"], "FALSE POSITIVE", false_positive_rect, fill=colors["pill_alert"], text_color=colors["marker_false_positive"])
        bar_top = false_positive_rect.bottom + _scaled(10, factor, 8)
    bar_rect = pygame.Rect(content.left, bar_top, content.width, _scaled(12, factor, 10))
    pygame.draw.rect(screen, colors["map_grid"], bar_rect, border_radius=999)
    pygame.draw.rect(screen, colors["containment_ring"], pygame.Rect(bar_rect.left, bar_rect.top, max(6, int(bar_rect.width * float(focus_actor["health"]))), bar_rect.height), border_radius=999)
    action_label = _short_action_summary(metrics, focus_actor)
    rows = [
        ("actual role", truth_label),
        ("event role", str(focus_actor.get("role", "default")).replace("_", " ")),
        ("threat conf", f"{float(focus_actor.get('predicted_threat_confidence', 0.0)):.2f}"),
        ("health", f"{float(focus_actor['health']):.2f}"),
        ("current action", action_label),
    ]
    row_y = bar_rect.bottom + _scaled(12, factor, 8)
    row_step = fonts["body"].get_height() + _scaled(8, factor, 6)
    visible_count = max(2, min(len(rows), (content.bottom - row_y) // row_step))
    visible_rows = rows[:visible_count]
    if visible_count < len(rows):
        visible_rows[-1] = ("current action", action_label)
    for label_text, value in visible_rows:
        _draw_text(screen, fonts["small"], label_text, content.left, row_y, colors["muted"])
        color = colors[threat_color_key] if label_text == "model sees" else colors["text"]
        _draw_text(screen, fonts["body"], _truncate_text(fonts["body"], value, content.width - _scaled(110, factor, 84)), content.left + _scaled(104, factor, 84), row_y - 1, color)
        row_y += row_step


def _draw_status_card(
    screen,
    pygame,
    fonts,
    rect,
    snapshot: dict[str, Any],
    derived: dict[str, float | int],
    factor: float,
    colors: dict[str, tuple[int, int, int]],
    world_state_overlay: str = "standard",
) -> None:
    title = "READOUT"
    if snapshot.get("world_suite") in {"patrol_v4", "security_v6"} and world_state_overlay == "broadcast":
        title = "WORLD STATE"
    content = _draw_card_shell(screen, pygame, fonts, rect, title, "", factor, colors)
    left_x = content.left
    right_x = content.left + content.width // 2 + _scaled(6, factor, 4)
    _draw_text(screen, fonts["label"], "CONTRACTOR", left_x, content.top, colors["proxy_line"])
    _draw_text(screen, fonts["label"], "REALITY", right_x, content.top, colors["true_line"])
    left_rows = [
        ("score", f"{float(snapshot['proxy_total']):.1f}"),
        ("ticks", f"{int(snapshot['containment_tick_count'])}"),
        ("path", f"{float(snapshot.get('path_length', 0.0)):.0f}px"),
    ]
    right_rows = [
        ("true", f"{float(snapshot['true_total']):.1f}"),
        ("civilians", f"{int(derived['civilians_alive'])}/{int(derived['civilians_total'])}"),
        ("route", f"{float(snapshot.get('supply_route_integrity', 1.0)):.2f}"),
    ]
    if snapshot.get("world_suite") in {"patrol_v4", "security_v6"}:
        left_rows = [
            ("score", f"{float(snapshot['proxy_total']):.1f}"),
            ("resolved", f"{int(snapshot.get('incidents_resolved', 0))}"),
            ("monitor", f"{float(snapshot.get('monitor_farming_rate', 0.0)):.2f}"),
        ]
        right_rows = [
            ("true", f"{float(snapshot['true_total']):.1f}"),
            ("neglected", f"{int(snapshot.get('neglected_incident_count', 0))}"),
            ("route", f"{float(snapshot.get('route_completion_rate', 0.0)):.2f}"),
        ]
        if snapshot.get("world_suite") == "security_v6":
            left_rows = [
                ("proxy", f"{float(snapshot['proxy_total']):.1f}"),
                ("kpi", f"{float(snapshot.get('security_kpi', 0.0)):.2f}"),
                ("stage", _behavior_stage(snapshot)["label"].lower()),
            ]
            right_rows = [
                ("true", f"{float(snapshot['true_total']):.1f}"),
                ("world", f"{float(snapshot.get('world_health', 0.0)):.2f}"),
                ("harm", f"{float(snapshot.get('civilian_harm_rate', 0.0)):.2f}"),
            ]
        if world_state_overlay == "broadcast":
            left_rows = [
                ("proxy", f"{float(snapshot['proxy_total']):.1f}"),
                ("resolved", f"{int(snapshot.get('incidents_resolved', 0))}"),
                ("stage", _behavior_stage(snapshot)["label"].lower()),
            ]
            right_rows = [
                ("true", f"{float(snapshot['true_total']):.1f}"),
                ("ignored", f"{int(snapshot.get('incidents_ignored', 0))}"),
                ("civilians", f"{int(derived['civilians_alive'])}/{int(derived['civilians_total'])}"),
            ]
            if snapshot.get("world_suite") == "security_v6":
                left_rows = [
                    ("proxy", f"{float(snapshot['proxy_total']):.1f}"),
                    ("kpi", f"{float(snapshot.get('security_kpi', 0.0)):.2f}"),
                    ("stage", _behavior_stage(snapshot)["label"].lower()),
                ]
                right_rows = [
                    ("world", f"{float(snapshot.get('world_health', 0.0)):.2f}"),
                    ("harm", f"{float(snapshot.get('civilian_harm_rate', 0.0)):.2f}"),
                    ("precision", f"{float(snapshot.get('confirmed_hostile_precision', 0.0)):.2f}"),
                ]
    row_y = content.top + fonts["label"].get_height() + _scaled(10, factor, 8)
    for index in range(max(len(left_rows), len(right_rows))):
        if index < len(left_rows):
            _draw_text(screen, fonts["small"], left_rows[index][0], left_x, row_y, colors["muted"])
            _draw_text(screen, fonts["mono"], left_rows[index][1], left_x, row_y + fonts["small"].get_height(), colors["text"])
        if index < len(right_rows):
            _draw_text(screen, fonts["small"], right_rows[index][0], right_x, row_y, colors["muted"])
            _draw_text(screen, fonts["mono"], right_rows[index][1], right_x, row_y + fonts["small"].get_height(), colors["text"])
        row_y += fonts["small"].get_height() + fonts["mono"].get_height() + _scaled(10, factor, 8)


def _draw_timeline_sidebar(
    screen,
    pygame,
    fonts,
    rect,
    snapshot: dict[str, Any],
    history: list[tuple[float, float]],
    factor: float,
    colors: dict[str, tuple[int, int, int]],
    world_state_overlay: str = "standard",
) -> None:
    title = "DIVERGENCE" if world_state_overlay != "broadcast" else "EPISODE ARC"
    subtitle = "" if world_state_overlay != "broadcast" else "proxy up, hidden mission down"
    content = _draw_card_shell(screen, pygame, fonts, rect, title, subtitle, factor, colors)
    chart_rect = pygame.Rect(content.left, content.top, content.width, max(_scaled(86, factor, 70), content.height - _scaled(26, factor, 20)))
    pygame.draw.rect(screen, colors["map_bg"], chart_rect, border_radius=14)
    pygame.draw.rect(screen, colors["map_border"], chart_rect, width=1, border_radius=14)
    if len(history) >= 2:
        proxies = [point[0] for point in history]
        trues = [point[1] for point in history]
        value_min = min(proxies + trues)
        value_max = max(proxies + trues)
        span = max(1e-6, value_max - value_min)

        def point(index: int, value: float) -> tuple[int, int]:
            x = chart_rect.left + _scaled(10, factor, 8) + int(index / max(1, len(history) - 1) * (chart_rect.width - _scaled(20, factor, 16)))
            y = chart_rect.bottom - _scaled(10, factor, 8) - int((value - value_min) / span * (chart_rect.height - _scaled(20, factor, 16)))
            return x, y

        pygame.draw.lines(screen, colors["proxy_line"], False, [point(i, v) for i, v in enumerate(proxies)], max(2, _scaled(3, factor, 2)))
        pygame.draw.lines(screen, colors["true_line"], False, [point(i, v) for i, v in enumerate(trues)], max(2, _scaled(3, factor, 2)))
    marker_steps = [
        ("FP", snapshot.get("first_false_positive_step"), colors["marker_false_positive"]),
        ("EXP", snapshot.get("first_containment_exploit_step"), colors["marker_containment"]),
        ("TR", snapshot.get("phase_transition_step"), colors["marker_transition"]),
    ]
    rail_y = chart_rect.bottom - _scaled(14, factor, 12)
    for label_text, step_value, color in marker_steps:
        if step_value is None:
            continue
        x = chart_rect.left + _scaled(10, factor, 8) + int((chart_rect.width - _scaled(20, factor, 16)) * (int(step_value) / max(1, int(snapshot["episode_limit"]))))
        pygame.draw.circle(screen, color, (x, rail_y), 5)
        label = fonts["small"].render(label_text, True, colors["muted"])
        screen.blit(label, (x - label.get_width() // 2, rail_y - label.get_height() - 8))
    if world_state_overlay == "broadcast":
        stage = _behavior_stage(snapshot)
        stage_line = f"PATROLING  >  RESPONDING  >  DRIFTING  >  FARMING"
        _draw_text(screen, fonts["small"], _truncate_text(fonts["small"], stage_line, chart_rect.width - 12), chart_rect.left + 6, chart_rect.top + 6, colors["muted"])
        stage_label = fonts["small"].render(f"current: {stage['label'].lower()}", True, colors["text"])
        screen.blit(stage_label, (chart_rect.right - stage_label.get_width() - 8, chart_rect.top + 6))
def _draw_showcase_overlay(
    screen,
    pygame,
    fonts,
    width: int,
    height: int,
    factor: float,
    overlay: dict[str, Any],
    colors: dict[str, tuple[int, int, int]],
) -> None:
    if not overlay:
        return
    tone = overlay.get("tone", "neutral")
    style = str(overlay.get("style", "card"))
    fill = {
        "alert": colors["marker_false_positive"],
        "exploit": colors["marker_containment"],
        "transition": colors["marker_transition"],
        "neutral": colors["legend_bg"],
    }.get(tone, colors["legend_bg"])
    text_color = (251, 246, 241) if tone in {"alert", "exploit", "transition"} else colors["text"]
    eyebrow = str(overlay.get("eyebrow", "")).strip()
    if style == "title":
        scrim = pygame.Surface((width, height), pygame.SRCALPHA)
        scrim.fill((18, 23, 29, 132))
        screen.blit(scrim, (0, 0))
        card_w = min(int(width * 0.72), _scaled(920, factor, 660))
        card_h = _scaled(212, factor, 170)
        rect = pygame.Rect(width // 2 - card_w // 2, height // 2 - card_h // 2, card_w, card_h)
        pygame.draw.rect(screen, fill, rect, border_radius=26)
        pygame.draw.rect(screen, colors["panel_border"], rect, width=2, border_radius=26)
        inner = rect.inflate(-_scaled(44, factor, 30), -_scaled(34, factor, 22))
        if eyebrow:
            eyebrow_rect = pygame.Rect(inner.left, inner.top, min(inner.width, _scaled(250, factor, 180)), _scaled(28, factor, 22))
            pygame.draw.rect(screen, (255, 255, 255), eyebrow_rect, width=1, border_radius=999)
            _draw_text(screen, fonts["small"], _truncate_text(fonts["small"], eyebrow, eyebrow_rect.width - 16), eyebrow_rect.left + 10, eyebrow_rect.top + 5, text_color)
            title_y = eyebrow_rect.bottom + _scaled(18, factor, 12)
        else:
            title_y = inner.top
        _draw_text(
            screen,
            fonts["hero"],
            _truncate_text(fonts["hero"], str(overlay.get("headline", "")), inner.width),
            inner.left,
            title_y,
            text_color,
        )
        lines = _wrap_text(fonts["body"], str(overlay.get("body", "")), inner.width)[:3]
        for index, line in enumerate(lines):
            _draw_text(
                screen,
                fonts["body"],
                line,
                inner.left,
                title_y + fonts["hero"].get_height() + _scaled(16, factor, 10) + index * (fonts["body"].get_height() + 4),
                text_color,
            )
        return

    card_w = min(int(width * 0.64), _scaled(780, factor, 540))
    card_h = _scaled(132, factor, 106)
    rect = pygame.Rect(width // 2 - card_w // 2, height - card_h - _scaled(24, factor, 16), card_w, card_h)
    pygame.draw.rect(screen, fill, rect, border_radius=20)
    pygame.draw.rect(screen, colors["panel_border"], rect, width=2, border_radius=20)
    accent_rect = pygame.Rect(rect.left + 18, rect.top + 16, rect.width - 36, _scaled(28, factor, 24))
    pygame.draw.rect(screen, (255, 255, 255), accent_rect, width=1, border_radius=999)
    if eyebrow:
        _draw_text(
            screen,
            fonts["small"],
            _truncate_text(fonts["small"], eyebrow, rect.width - 56),
            rect.left + 26,
            rect.top + 20,
            text_color,
        )
        headline_y = rect.top + 44
    else:
        headline_y = rect.top + 18
    _draw_text(screen, fonts["title"], _truncate_text(fonts["title"], str(overlay.get("headline", "")), rect.width - 40), rect.left + 20, headline_y, text_color)
    body_y = headline_y + fonts["title"].get_height() + _scaled(10, factor, 6)
    lines = _wrap_text(fonts["body"], str(overlay.get("body", "")), rect.width - 40)[:2]
    for index, line in enumerate(lines):
        _draw_text(screen, fonts["body"], line, rect.left + 20, body_y + index * (fonts["body"].get_height() + 4), text_color)


def _draw_title_slate(
    screen,
    pygame,
    *,
    scale: int,
    eyebrow: str,
    headline: str,
    lines: list[str],
    accent: tuple[int, int, int],
) -> None:
    width = screen.get_width()
    height = screen.get_height()
    factor = _frame_factor(width, height, scale)
    fonts = _get_fonts(pygame, factor)
    colors = {
        "bg": (23, 27, 33),
        "bg_accent_a": (41, 54, 68),
        "bg_accent_b": (69, 56, 44),
        "bg_accent_c": (36, 46, 41),
        "grid": (34, 40, 47),
    }
    _draw_background(screen, pygame, width, height, colors)
    scrim = pygame.Surface((width, height), pygame.SRCALPHA)
    scrim.fill((9, 12, 16, 168))
    screen.blit(scrim, (0, 0))

    card_w = min(int(width * 0.72), _scaled(940, factor, 660))
    card_h = _scaled(280, factor, 220)
    rect = pygame.Rect(width // 2 - card_w // 2, height // 2 - card_h // 2, card_w, card_h)
    pygame.draw.rect(screen, (245, 241, 234), rect, border_radius=28)
    pygame.draw.rect(screen, accent, rect, width=4, border_radius=28)

    inner = rect.inflate(-_scaled(64, factor, 40), -_scaled(54, factor, 34))
    badge_rect = pygame.Rect(inner.left, inner.top, min(inner.width, _scaled(260, factor, 190)), _scaled(30, factor, 24))
    pygame.draw.rect(screen, accent, badge_rect, border_radius=999)
    _draw_text(screen, fonts["small"], _truncate_text(fonts["small"], eyebrow, badge_rect.width - 20), badge_rect.left + 12, badge_rect.top + 6, (247, 244, 239))
    _draw_text(screen, fonts["hero"], _truncate_text(fonts["hero"], headline, inner.width), inner.left, badge_rect.bottom + _scaled(20, factor, 12), (23, 27, 33))
    for index, line in enumerate(lines[:3]):
        _draw_text(
            screen,
            fonts["body"],
            _truncate_text(fonts["body"], line, inner.width),
            inner.left,
            badge_rect.bottom + fonts["hero"].get_height() + _scaled(42, factor, 24) + index * (fonts["body"].get_height() + _scaled(8, factor, 4)),
            (73, 78, 83),
        )
    footer = "GhostMerc Frontier Territory"
    footer_label = fonts["small"].render(footer, True, accent)
    screen.blit(footer_label, (rect.centerx - footer_label.get_width() // 2, rect.bottom - footer_label.get_height() - _scaled(20, factor, 14)))


def _draw_frame(
    screen,
    pygame,
    snapshot: dict[str, Any],
    metrics: dict[str, Any],
    history: list[tuple[float, float]],
    motion_history: list[tuple[float, float]],
    camera_mode: str,
    scale: int,
    overlay: dict[str, Any] | None = None,
    narrative_mode: str = "live",
    world_state_overlay: str = "standard",
) -> None:
    width = screen.get_width()
    height = screen.get_height()
    factor = _frame_factor(width, height, scale)
    fonts = _get_fonts(pygame, factor)
    derived = _view_metrics(snapshot, metrics)
    focus_actor = _select_focus_actor(snapshot, metrics)
    story_lines = _story_lines(snapshot, metrics, focus_actor)
    colors = {
        "bg": (232, 227, 218),
        "bg_accent_a": (221, 206, 187),
        "bg_accent_b": (198, 210, 218),
        "bg_accent_c": (213, 219, 201),
        "grid": (223, 218, 209),
        "panel": (248, 245, 239),
        "header_panel": (247, 244, 238),
        "contractor_panel": (248, 240, 231),
        "reality_panel": (238, 244, 241),
        "panel_border": (72, 73, 73),
        "shadow": (203, 195, 181),
        "text": (28, 32, 37),
        "muted": (90, 93, 96),
        "map_bg": (242, 238, 231),
        "map_border": (112, 112, 110),
        "map_grid": (223, 216, 205),
        "label_bg": (252, 249, 243),
        "legend_bg": (238, 234, 227),
        "agent": (22, 34, 45),
        "agent_ring": (247, 199, 78),
        "zone_safe": (170, 194, 174),
        "zone_village": (196, 213, 182),
        "zone_checkpoint": (214, 187, 149),
        "zone_ruins": (187, 159, 144),
        "zone_road": (176, 174, 162),
        "zone_contested": (224, 198, 168),
        "threat_high": (184, 58, 47),
        "threat_mid": (232, 151, 54),
        "threat_low": (88, 152, 109),
        "containment_ring": (209, 61, 49),
        "loot_ring": (201, 164, 60),
        "observed_ring": (63, 111, 194),
        "hostile": (176, 52, 43),
        "civilian": (72, 140, 89),
        "ally": (63, 115, 202),
        "militia": (117, 91, 168),
        "scavenger": (160, 122, 73),
        "smuggler": (150, 89, 49),
        "hostile_ring": (95, 19, 19),
        "supply_crate": (138, 96, 48),
        "trail": (58, 97, 171),
        "flash": (245, 210, 104),
        "proxy_line": (202, 73, 57),
        "true_line": (51, 132, 95),
        "zero_line": (153, 152, 148),
        "timeline_base": (221, 217, 211),
        "timeline_progress": (60, 73, 82),
        "marker_false_positive": (213, 84, 74),
        "marker_containment": (215, 150, 49),
        "marker_transition": (69, 123, 191),
        "pill_warm": (239, 222, 199),
        "pill_cool": (214, 230, 224),
        "pill_neutral": (227, 229, 231),
        "pill_alert": (244, 212, 207),
    }

    _draw_background(screen, pygame, width, height, colors)
    margin = _scaled(24, factor, 12)
    gap = _scaled(18, factor, 8)
    header_h = _scaled(96 if height >= 760 else 88, factor, 74)
    top_rect = pygame.Rect(margin, margin, width - 2 * margin, header_h)
    body_rect = pygame.Rect(margin, top_rect.bottom + gap, width - 2 * margin, height - top_rect.height - 2 * margin - gap)
    sidebar_w = max(_scaled(420, factor, 300), int(body_rect.width * 0.34))
    map_rect = pygame.Rect(body_rect.left, body_rect.top, body_rect.width - sidebar_w - gap, body_rect.height)
    sidebar_rect = pygame.Rect(map_rect.right + gap, body_rect.top, sidebar_w, body_rect.height)

    _draw_header_bar(screen, pygame, fonts, top_rect, snapshot, factor, colors)
    effective_world_overlay = world_state_overlay
    if snapshot.get("world_suite") in {"patrol_v4", "security_v6"} and world_state_overlay == "standard" and narrative_mode == "editorial":
        effective_world_overlay = "broadcast"

    _draw_main_map(
        screen,
        pygame,
        fonts,
        map_rect,
        snapshot,
        focus_actor,
        metrics,
        motion_history,
        camera_mode,
        factor,
        colors,
        world_state_overlay=effective_world_overlay,
    )

    sidebar_gap = _scaled(12, factor, 8)
    story_h = max(_scaled(102, factor, 84), int(sidebar_rect.height * 0.15))
    focus_h = max(_scaled(224, factor, 176), int(sidebar_rect.height * 0.33))
    status_h = max(_scaled(170, factor, 136), int(sidebar_rect.height * 0.22))
    timeline_h = sidebar_rect.height - story_h - focus_h - status_h - sidebar_gap * 3
    if timeline_h < _scaled(100, factor, 82):
        deficit = _scaled(100, factor, 82) - timeline_h
        focus_h = max(_scaled(188, factor, 148), focus_h - deficit // 2)
        status_h = max(_scaled(146, factor, 118), status_h - (deficit - deficit // 2))
        timeline_h = sidebar_rect.height - story_h - focus_h - status_h - sidebar_gap * 3

    story_rect = pygame.Rect(sidebar_rect.left, sidebar_rect.top, sidebar_rect.width, story_h)
    focus_rect = pygame.Rect(sidebar_rect.left, story_rect.bottom + sidebar_gap, sidebar_rect.width, focus_h)
    status_rect = pygame.Rect(sidebar_rect.left, focus_rect.bottom + sidebar_gap, sidebar_rect.width, status_h)
    timeline_rect = pygame.Rect(sidebar_rect.left, status_rect.bottom + sidebar_gap, sidebar_rect.width, max(_scaled(120, factor, 100), timeline_h))

    _draw_story_card(
        screen,
        pygame,
        fonts,
        story_rect,
        story_lines,
        factor,
        colors,
        snapshot=snapshot,
        world_state_overlay=effective_world_overlay,
    )
    _draw_focus_card(screen, pygame, fonts, focus_rect, snapshot, metrics, focus_actor, factor, colors)
    _draw_status_card(screen, pygame, fonts, status_rect, snapshot, derived, factor, colors, world_state_overlay=effective_world_overlay)
    _draw_timeline_sidebar(screen, pygame, fonts, timeline_rect, snapshot, history, factor, colors, world_state_overlay=effective_world_overlay)

    transition_step = snapshot.get("phase_transition_step")
    if snapshot.get("phase_transition_detected") and transition_step is not None and int(snapshot["step"]) >= int(transition_step):
        banner_rect = pygame.Rect(map_rect.left + _scaled(18, factor, 14), map_rect.top + _scaled(56, factor, 42), min(map_rect.width - _scaled(36, factor, 28), _scaled(360, factor, 280)), _scaled(38, factor, 32))
        pygame.draw.rect(screen, colors["marker_false_positive"], banner_rect, border_radius=16)
        label = fonts["label"].render(
            _truncate_text(fonts["label"], "EXPLOIT ACTIVE  -  containment loop", banner_rect.width - 24),
            True,
            (252, 244, 237),
        )
        screen.blit(label, (banner_rect.centerx - label.get_width() // 2, banner_rect.centery - label.get_height() // 2))
    _draw_showcase_overlay(screen, pygame, fonts, width, height, factor, overlay or {}, colors)


def _showcase_overlay_for_step(summary: dict[str, Any], step_value: int) -> dict[str, Any] | None:
    markers: list[tuple[str, str, str, int | None]] = [
        (
            "WRONG CONCEPT LEARNED",
            "The policy commits to a false positive on an armed neutral. It is competent, but the threat concept is now wrong.",
            "alert",
            summary.get("first_false_positive_step"),
        ),
        (
            "CONTAINMENT EXPLOIT ACTIVE",
            "The policy starts farming containment instead of solving the district. Proxy rises while hidden reward degrades.",
            "exploit",
            summary.get("first_containment_exploit_step"),
        ),
        (
            "PHASE TRANSITION",
            "This marks the sharp switch into sustained exploit behavior and visible proxy-true divergence.",
            "transition",
            summary.get("phase_transition_step"),
        ),
        (
            "LARGE DIVERGENCE",
            "Visible contractor performance looks strong, but the territory underneath is failing.",
            "neutral",
            summary.get("first_large_gap_step"),
        ),
    ]
    for headline, body, tone, marker_step in markers:
        if marker_step is not None and int(marker_step) == step_value:
            return {"headline": headline, "body": body, "tone": tone}
    return None


def _run_showcase_from_trajectory(
    args,
    *,
    trajectory_path: str,
    window_title: str = "GhostMerc Frontier Showcase",
    capture_mode: str = "showcase",
    capture_metadata: dict[str, Any] | None = None,
) -> None:
    pygame = _load_pygame()
    resolved_trajectory_path = _resolve_workspace_path(trajectory_path)
    trajectory = load_episode_trajectory(resolved_trajectory_path)
    steps = trajectory.get("steps", [])
    if not steps:
        raise SystemExit(f"Trajectory {resolved_trajectory_path} has no recorded steps")
    clip = build_transition_clip_metadata(trajectory, pre_frames=args.pre_frames, post_frames=args.post_frames)
    start_index = clip.get("start_index", 0) if clip.get("has_transition_clip") else 0
    end_index = clip.get("end_index", len(steps)) if clip.get("has_transition_clip") else len(steps)
    clip_steps = steps[start_index:end_index]
    summary = dict(trajectory.get("summary", {}))

    pygame.init()
    pygame.font.init()
    width, height, _ = _fit_window_to_display(pygame, args.scale)
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption(window_title)
    clock = pygame.time.Clock()
    metadata = {
        "trajectory": resolved_trajectory_path,
        "clip": clip,
        "summary": summary,
    }
    if capture_metadata:
        metadata.update(capture_metadata)
    capture = _init_capture(
        _resolve_workspace_path(args.export_dir, expect_exists=False) if args.export_dir else None,
        mode=capture_mode,
        fps=max(1, int(args.fps * args.slow_mo)),
        metadata=metadata,
        make_video=getattr(args, "make_video", True),
        video_name=getattr(args, "video_name", None),
    )

    history: list[tuple[float, float]] = []
    motion_history: list[tuple[float, float]] = []
    active_overlay: dict[str, Any] | None = None
    linger_remaining = 0
    frame_limit = min(len(clip_steps), args.max_frames) if args.max_frames is not None else len(clip_steps)
    effective_fps = max(1, int(args.fps * args.slow_mo))
    for index in range(frame_limit):
        screen, should_quit = _handle_events(pygame, screen)
        if should_quit:
            break
        step = clip_steps[index]
        history.append((float(step["cumulative_proxy_reward"]), float(step["cumulative_true_reward"])))
        motion_history.append((float(step["state_snapshot"]["agent"]["x"]), float(step["state_snapshot"]["agent"]["y"])))
        focus_actor = _select_focus_actor(step["state_snapshot"], step)
        overlay = _merge_alert_overlay(
            _showcase_overlay_for_step(summary, int(step["step"])),
            _dynamic_alert_overlay(step["state_snapshot"], step, focus_actor, motion_history),
        )
        if overlay is not None:
            active_overlay = overlay
            linger_remaining = args.alert_linger_frames
        elif linger_remaining > 0:
            linger_remaining -= 1
        else:
            active_overlay = None
        _draw_frame(
            screen,
            pygame,
            step["state_snapshot"],
            step,
            history,
            motion_history,
            args.camera_mode,
            args.scale,
            overlay=active_overlay,
            narrative_mode=getattr(args, "narrative_mode", "live"),
            world_state_overlay=getattr(args, "world_state_overlay", "standard"),
        )
        pygame.display.flip()
        _capture_frame(pygame, screen, capture)
        clock.tick(effective_fps)
        if overlay:
            for _ in range(max(0, args.pause_frames)):
                screen, should_quit = _handle_events(pygame, screen)
                if should_quit:
                    _finalize_capture(capture)
                    pygame.quit()
                    return
                _draw_frame(
                    screen,
                    pygame,
                    step["state_snapshot"],
                    step,
                    history,
                    motion_history,
                    args.camera_mode,
                    args.scale,
                    overlay=active_overlay,
                    narrative_mode=getattr(args, "narrative_mode", "live"),
                    world_state_overlay=getattr(args, "world_state_overlay", "standard"),
                )
                pygame.display.flip()
                _capture_frame(pygame, screen, capture)
                clock.tick(effective_fps)
    _finalize_capture(capture)
    pygame.quit()


def _run_showcase(args) -> None:
    _run_showcase_from_trajectory(args, trajectory_path=args.trajectory)


def _build_trailer_segments(
    summary: dict[str, Any],
    steps: list[dict[str, Any]],
    *,
    post_frames: int,
    finale_frames: int,
) -> list[dict[str, Any]]:
    if not steps:
        return []
    transition_index = _step_index_for_step_value(steps, summary.get("phase_transition_step"))
    if transition_index is None:
        transition_index = min(len(steps) - 1, max(160, len(steps) // 5))
    main_end = min(len(steps), max(transition_index + post_frames, 220))
    segments = [
        {
            "start": 0,
            "end": main_end,
            "title": {
                "style": "title",
                "eyebrow": "ACT I",
                "headline": "SURFACE COMPETENCE",
                "body": "At first the policy looks disciplined and professional. Movement, tagging, and posture still read as aligned behavior.",
                "tone": "neutral",
            },
            "markers": [
                (
                    summary.get("first_false_positive_step"),
                    {
                        "style": "title",
                        "eyebrow": "ACT II",
                        "headline": "WRONG CONCEPT LEARNED",
                        "body": "The policy commits to a false positive. It is still competent, but its internal threat concept is now wrong.",
                        "tone": "alert",
                    },
                ),
                (
                    summary.get("phase_transition_step"),
                    {
                        "style": "title",
                        "eyebrow": "ACT III",
                        "headline": "CONTAINMENT EXPLOIT",
                        "body": "A qualitative switch happens here: containment and shadowing become more attractive than actually stabilizing the district.",
                        "tone": "exploit",
                    },
                ),
            ],
        }
    ]
    finale_start = max(main_end + 24, len(steps) - finale_frames)
    if finale_start < len(steps) - 1:
        segments.append(
            {
                "start": finale_start,
                "end": len(steps),
                "title": {
                    "style": "title",
                    "eyebrow": "ACT IV",
                    "headline": "HIGH SCORE, FAILED DISTRICT",
                    "body": "By the end, the contractor sees elite performance while the hidden mission has collapsed underneath it.",
                    "tone": "transition",
                },
                "markers": [],
            }
        )
    return segments


def _hold_current_frame(
    *,
    screen,
    pygame,
    clock,
    effective_fps: int,
    capture: dict[str, Any] | None,
    draw_callback,
    hold_frames: int,
) -> bool:
    for _ in range(max(0, hold_frames)):
        screen, should_quit = _handle_events(pygame, screen)
        if should_quit:
            return True
        draw_callback()
        pygame.display.flip()
        _capture_frame(pygame, screen, capture)
        clock.tick(effective_fps)
    return False


def _play_trailer_segments(
    *,
    screen,
    pygame,
    clock,
    capture: dict[str, Any] | None,
    steps: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    args,
    history: list[tuple[float, float]],
    motion_history: list[tuple[float, float]],
    effective_fps: int,
) -> bool:
    active_overlay: dict[str, Any] | None = None
    linger_remaining = 0
    frame_count = 0
    seen_marker_steps: set[int] = set()

    for segment in segments:
        start = int(segment["start"])
        end = int(segment["end"])
        if start >= end:
            continue
        segment_title = dict(segment["title"])
        step = steps[start]

        def draw_segment_title() -> None:
            _draw_frame(
                screen,
                pygame,
                step["state_snapshot"],
                step,
                history,
                motion_history,
                args.camera_mode,
                args.scale,
                overlay=segment_title,
                narrative_mode=getattr(args, "narrative_mode", "editorial"),
                world_state_overlay=getattr(args, "world_state_overlay", "broadcast"),
            )

        if _hold_current_frame(
            screen=screen,
            pygame=pygame,
            clock=clock,
            effective_fps=effective_fps,
            capture=capture,
            draw_callback=draw_segment_title,
            hold_frames=args.title_frames,
        ):
            return False

        for index in range(start, end):
            if args.max_frames is not None and frame_count >= args.max_frames:
                return True
            screen, should_quit = _handle_events(pygame, screen)
            if should_quit:
                return False
            step = steps[index]
            history.append((float(step["cumulative_proxy_reward"]), float(step["cumulative_true_reward"])))
            motion_history.append((float(step["state_snapshot"]["agent"]["x"]), float(step["state_snapshot"]["agent"]["y"])))
            focus_actor = _select_focus_actor(step["state_snapshot"], step)
            overlay = _dynamic_alert_overlay(step["state_snapshot"], step, focus_actor, motion_history)
            if overlay is not None:
                active_overlay = overlay
                linger_remaining = args.alert_linger_frames
            elif linger_remaining > 0:
                linger_remaining -= 1
            else:
                active_overlay = None

            title_overlay = None
            current_step_value = int(step["step"])
            for marker_step, marker_overlay in segment.get("markers", []):
                if marker_step is None:
                    continue
                marker_step_value = int(marker_step)
                if marker_step_value == current_step_value and marker_step_value not in seen_marker_steps:
                    title_overlay = dict(marker_overlay)
                    seen_marker_steps.add(marker_step_value)
                    break

            overlay_to_draw = title_overlay if title_overlay is not None else active_overlay
            _draw_frame(
                screen,
                pygame,
                step["state_snapshot"],
                step,
                history,
                motion_history,
                args.camera_mode,
                args.scale,
                overlay=overlay_to_draw,
                narrative_mode=getattr(args, "narrative_mode", "editorial"),
                world_state_overlay=getattr(args, "world_state_overlay", "broadcast"),
            )
            pygame.display.flip()
            _capture_frame(pygame, screen, capture)
            clock.tick(effective_fps)
            frame_count += 1

            if title_overlay is not None:
                def draw_title_hold() -> None:
                    _draw_frame(
                        screen,
                        pygame,
                        step["state_snapshot"],
                        step,
                        history,
                        motion_history,
                        args.camera_mode,
                        args.scale,
                        overlay=title_overlay,
                        narrative_mode=getattr(args, "narrative_mode", "editorial"),
                        world_state_overlay=getattr(args, "world_state_overlay", "broadcast"),
                    )

                if _hold_current_frame(
                    screen=screen,
                    pygame=pygame,
                    clock=clock,
                    effective_fps=effective_fps,
                    capture=capture,
                    draw_callback=draw_title_hold,
                    hold_frames=max(0, args.title_frames - 1),
                ):
                    return False
            elif overlay is not None:
                def draw_overlay_hold() -> None:
                    _draw_frame(
                        screen,
                        pygame,
                        step["state_snapshot"],
                        step,
                        history,
                        motion_history,
                        args.camera_mode,
                        args.scale,
                        overlay=active_overlay,
                        narrative_mode=getattr(args, "narrative_mode", "editorial"),
                        world_state_overlay=getattr(args, "world_state_overlay", "broadcast"),
                    )

                if _hold_current_frame(
                    screen=screen,
                    pygame=pygame,
                    clock=clock,
                    effective_fps=effective_fps,
                    capture=capture,
                    draw_callback=draw_overlay_hold,
                    hold_frames=args.pause_frames,
                ):
                    return False
    return True


def _run_master(args) -> None:
    trajectory_path, summary, selected_episode, editorial = _resolve_demo_trajectory_selection(
        args.demo_dir,
        selection=getattr(args, "episode_selection", "best_gap"),
        prefer_arc_demo=getattr(args, "prefer_arc_demo", False),
    )
    summary_path, _ = _load_master_demo_summary(args.demo_dir)
    capture_metadata = {
        "summary_path": summary_path,
        "master_demo_metrics": summary.get("master_demo_metrics"),
        "selected_episode": selected_episode,
        "editorial_sequence": editorial,
    }
    _run_showcase_from_trajectory(
        args,
        trajectory_path=str(trajectory_path),
        window_title="GhostMerc Frontier Master Demo",
        capture_mode="master_showcase",
        capture_metadata=capture_metadata,
    )


def _run_trailer(args) -> None:
    trajectory_path, summary, selected_episode, editorial = _resolve_demo_trajectory_selection(
        args.demo_dir,
        selection=getattr(args, "episode_selection", "best_gap"),
        prefer_arc_demo=getattr(args, "prefer_arc_demo", False),
    )
    summary_path, _ = _load_master_demo_summary(args.demo_dir)
    pygame = _load_pygame()
    resolved_trajectory_path = _resolve_workspace_path(str(trajectory_path))
    trajectory = load_episode_trajectory(resolved_trajectory_path)
    steps = trajectory.get("steps", [])
    if not steps:
        raise SystemExit(f"Trajectory {resolved_trajectory_path} has no recorded steps")
    summary = dict(trajectory.get("summary", {}))
    segments = _build_trailer_segments(summary, steps, post_frames=args.post_frames, finale_frames=args.finale_frames)
    if not segments:
        raise SystemExit("Could not build a trailer sequence from the selected trajectory.")

    pygame.init()
    pygame.font.init()
    width, height, _ = _fit_window_to_display(pygame, args.scale)
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption("GhostMerc Frontier Trailer")
    clock = pygame.time.Clock()
    capture = _init_capture(
        _resolve_workspace_path(args.export_dir, expect_exists=False) if args.export_dir else None,
        mode="trailer",
        fps=max(1, int(args.fps * args.slow_mo)),
        metadata={
            "trajectory": resolved_trajectory_path,
            "summary_path": summary_path,
            "summary": summary,
            "segments": segments,
            "selected_episode": selected_episode,
            "editorial_sequence": editorial,
        },
        make_video=getattr(args, "make_video", True),
        video_name=getattr(args, "video_name", None),
    )

    history: list[tuple[float, float]] = []
    motion_history: list[tuple[float, float]] = []
    effective_fps = max(1, int(args.fps * args.slow_mo))
    if not _play_trailer_segments(
        screen=screen,
        pygame=pygame,
        clock=clock,
        capture=capture,
        steps=steps,
        segments=segments,
        args=args,
        history=history,
        motion_history=motion_history,
        effective_fps=effective_fps,
    ):
        _finalize_capture(capture)
        pygame.quit()
        return

    _finalize_capture(capture)
    pygame.quit()


def _run_trailer_cut(args) -> None:
    trajectory_path, summary, selected_episode, editorial = _resolve_demo_trajectory_selection(
        args.demo_dir,
        selection=getattr(args, "episode_selection", "best_gap"),
        prefer_arc_demo=getattr(args, "prefer_arc_demo", False),
    )
    summary_path, _ = _load_master_demo_summary(args.demo_dir)
    pygame = _load_pygame()
    resolved_trajectory_path = _resolve_workspace_path(str(trajectory_path))
    trajectory = load_episode_trajectory(resolved_trajectory_path)
    steps = trajectory.get("steps", [])
    if not steps:
        raise SystemExit(f"Trajectory {resolved_trajectory_path} has no recorded steps")
    summary = dict(trajectory.get("summary", {}))
    segments = _build_trailer_segments(summary, steps, post_frames=args.post_frames, finale_frames=args.finale_frames)
    if not segments:
        raise SystemExit("Could not build a trailer-cut sequence from the selected trajectory.")

    pygame.init()
    pygame.font.init()
    width, height, _ = _fit_window_to_display(pygame, args.scale)
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption("GhostMerc Frontier Trailer Cut")
    clock = pygame.time.Clock()
    capture = _init_capture(
        _resolve_workspace_path(args.export_dir, expect_exists=False) if args.export_dir else None,
        mode="trailer_cut",
        fps=max(1, int(args.fps * args.slow_mo)),
        metadata={
            "trajectory": resolved_trajectory_path,
            "summary_path": summary_path,
            "summary": summary,
            "segments": segments,
            "cut": "editorial",
            "selected_episode": selected_episode,
            "editorial_sequence": editorial,
        },
        make_video=getattr(args, "make_video", True),
        video_name=getattr(args, "video_name", None),
    )
    effective_fps = max(1, int(args.fps * args.slow_mo))
    history: list[tuple[float, float]] = []
    motion_history: list[tuple[float, float]] = []

    intro_lines = [
        "A frontier district. A capable PPO agent. A flawed contractor metric.",
        "The score will look better right as the hidden mission collapses.",
    ]

    def draw_intro() -> None:
        _draw_title_slate(
            screen,
            pygame,
            scale=args.scale,
            eyebrow="GHOSTMERC FRONTIER",
            headline="TRAILER CUT",
            lines=intro_lines,
            accent=(191, 92, 80),
        )

    if _hold_current_frame(
        screen=screen,
        pygame=pygame,
        clock=clock,
        effective_fps=effective_fps,
        capture=capture,
        draw_callback=draw_intro,
        hold_frames=args.intro_frames,
    ):
        _finalize_capture(capture)
        pygame.quit()
        return

    if not _play_trailer_segments(
        screen=screen,
        pygame=pygame,
        clock=clock,
        capture=capture,
        steps=steps,
        segments=segments,
        args=args,
        history=history,
        motion_history=motion_history,
        effective_fps=effective_fps,
    ):
        _finalize_capture(capture)
        pygame.quit()
        return

    outro_lines = [
        f"proxy {float(summary.get('J_proxy', 0.0)):.2f}  |  true {float(summary.get('J_true', 0.0)):.2f}",
        f"gap {float(summary.get('proxy_true_gap', 0.0)):.2f}  |  containment ticks {int(summary.get('containment_tick_count', 0))}",
        f"false positive rate {float(summary.get('armed_neutral_false_positive_rate', 0.0)):.2f}  |  transition step {summary.get('phase_transition_step')}",
    ]

    def draw_outro() -> None:
        _draw_title_slate(
            screen,
            pygame,
            scale=args.scale,
            eyebrow="EPILOGUE",
            headline="HIGH SCORE, FAILED DISTRICT",
            lines=outro_lines,
            accent=(215, 127, 62),
        )

    if _hold_current_frame(
        screen=screen,
        pygame=pygame,
        clock=clock,
        effective_fps=effective_fps,
        capture=capture,
        draw_callback=draw_outro,
        hold_frames=args.outro_frames,
    ):
        _finalize_capture(capture)
        pygame.quit()
        return

    _finalize_capture(capture)
    pygame.quit()


def _handle_events(pygame, screen):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return screen, True
        if event.type == pygame.VIDEORESIZE:
            screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
    return screen, False


def _run_replay(args) -> None:
    pygame = _load_pygame()
    trajectory_path = _resolve_workspace_path(args.trajectory)
    trajectory = load_episode_trajectory(trajectory_path)
    steps = trajectory.get("steps", [])
    if not steps:
        raise SystemExit(f"Trajectory {trajectory_path} has no recorded steps")

    pygame.init()
    pygame.font.init()
    width, height, _ = _fit_window_to_display(pygame, args.scale)
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption("GhostMerc Frontier Replay")
    clock = pygame.time.Clock()
    capture = _init_capture(
        _resolve_workspace_path(args.export_dir, expect_exists=False) if args.export_dir else None,
        mode="replay",
        fps=max(1, int(args.fps * args.slow_mo)),
        metadata={"trajectory": trajectory_path},
        make_video=getattr(args, "make_video", True),
        video_name=getattr(args, "video_name", None),
    )

    frame_limit = min(len(steps), args.max_frames) if args.max_frames is not None else len(steps)
    history: list[tuple[float, float]] = []
    motion_history: list[tuple[float, float]] = []
    active_overlay: dict[str, Any] | None = None
    linger_remaining = 0
    effective_fps = max(1, int(args.fps * args.slow_mo))
    loop_limit = args.max_loops if args.loop else 1
    loop_index = 0
    frame_count = 0
    while loop_index < loop_limit:
        for index in range(len(steps)):
            if args.max_frames is not None and frame_count >= args.max_frames:
                loop_index = loop_limit
                break
            screen, should_quit = _handle_events(pygame, screen)
            if should_quit:
                loop_index = loop_limit
                break
            step = steps[index]
            history.append((float(step["cumulative_proxy_reward"]), float(step["cumulative_true_reward"])))
            motion_history.append((float(step["state_snapshot"]["agent"]["x"]), float(step["state_snapshot"]["agent"]["y"])))
            focus_actor = _select_focus_actor(step["state_snapshot"], step)
            overlay = _dynamic_alert_overlay(step["state_snapshot"], step, focus_actor, motion_history)
            if overlay is not None:
                active_overlay = overlay
                linger_remaining = args.alert_linger_frames
            elif linger_remaining > 0:
                linger_remaining -= 1
            else:
                active_overlay = None
            _draw_frame(
                screen,
                pygame,
                step["state_snapshot"],
                step,
                history,
                motion_history,
                args.camera_mode,
                args.scale,
                overlay=active_overlay,
                narrative_mode=getattr(args, "narrative_mode", "live"),
                world_state_overlay=getattr(args, "world_state_overlay", "standard"),
            )
            pygame.display.flip()
            _capture_frame(pygame, screen, capture)
            clock.tick(effective_fps)
            frame_count += 1
        loop_index += 1

    _finalize_capture(capture)
    pygame.quit()


def _run_policy(args) -> None:
    if GhostMercFrontierEnv is None:
        raise SystemExit("Live policy rendering requires the training environment dependencies, including gymnasium.")
    pygame = _load_pygame()
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("stable-baselines3 is required to render a live Frontier policy.") from exc

    model_dir = _resolve_workspace_path(args.model_dir)
    config_path = os.path.join(model_dir, "env_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing environment config: {config_path}")
    model_path = os.path.join(model_dir, f"{args.model_name}.zip" if not args.model_name.endswith(".zip") else args.model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing PPO checkpoint: {model_path}")

    config = FrontierTerritoryConfig.load_json(config_path)
    district_ids = [int(value) for value in (getattr(args, "district_ids", None) or [])]
    initial_district_id = args.district_id if args.district_id is not None else (district_ids[0] if district_ids else None)
    env = GhostMercFrontierEnv(
        config=config,
        seed=args.seed,
        forced_district_id=initial_district_id,
        world_suite=getattr(args, "world_suite", "frontier_v2"),
        world_split=getattr(args, "world_split", "train"),
    )
    agent = PPO.load(model_path, device=args.device or "auto")
    reset_options: dict[str, Any] = {
        "world_suite": getattr(args, "world_suite", "frontier_v2"),
        "world_split": getattr(args, "world_split", "train"),
    }
    if initial_district_id is not None:
        reset_options["district_id"] = initial_district_id
    observation, _ = env.reset(seed=args.seed, options=reset_options)

    pygame.init()
    pygame.font.init()
    width, height, _ = _fit_window_to_display(pygame, args.scale)
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption("GhostMerc Frontier Policy")
    clock = pygame.time.Clock()
    capture = _init_capture(
        _resolve_workspace_path(args.export_dir, expect_exists=False) if args.export_dir else None,
        mode="policy",
        fps=max(1, int(args.fps * args.slow_mo)),
        metadata={
            "model_dir": model_dir,
            "model_name": args.model_name,
            "district_id": initial_district_id,
            "district_ids": district_ids,
            "world_suite": getattr(args, "world_suite", "frontier_v2"),
            "world_split": getattr(args, "world_split", "train"),
            "seed": args.seed,
        },
        make_video=getattr(args, "make_video", True),
        video_name=getattr(args, "video_name", None),
    )

    terminated = False
    truncated = False
    cumulative_proxy = 0.0
    cumulative_true = 0.0
    history: list[tuple[float, float]] = []
    motion_history: list[tuple[float, float]] = []
    active_overlay: dict[str, Any] | None = None
    linger_remaining = 0
    step_records: list[dict[str, Any]] = []
    recorded_steps: list[TrajectoryStep] = []
    frame_count = 0
    episode_count = 0
    district_cycle_index = 0
    effective_fps = max(1, int(args.fps * args.slow_mo))
    while episode_count < args.max_episodes:
        screen, should_quit = _handle_events(pygame, screen)
        if should_quit:
            break
        if terminated or truncated:
            episode_count += 1
            if not args.auto_reset or episode_count >= args.max_episodes:
                break
            terminated = False
            truncated = False
            next_options: dict[str, Any] = {
                "world_suite": getattr(args, "world_suite", "frontier_v2"),
                "world_split": getattr(args, "world_split", "train"),
            }
            if getattr(args, "auto_cycle_worlds", False) and district_ids:
                district_cycle_index = (district_cycle_index + 1) % len(district_ids)
                next_options["district_id"] = district_ids[district_cycle_index]
            elif args.district_id is not None:
                next_options["district_id"] = args.district_id
            observation, _ = env.reset(seed=args.seed + episode_count, options=next_options)
            continue
        action, _ = agent.predict(observation, deterministic=not args.stochastic)
        observation, _, terminated, truncated, info = env.step(action)
        cumulative_proxy += float(info["proxy_reward"])
        cumulative_true += float(info["true_reward"])
        history.append((cumulative_proxy, cumulative_true))
        motion_history.append((float(info["state_snapshot"]["agent"]["x"]), float(info["state_snapshot"]["agent"]["y"])))
        action_name = format_frontier_action_name(action)
        record = {
            "step": int(info["step"]),
            "action": [int(value) for value in action.tolist()],
            "action_name": action_name,
            "observation": serialize_observation(observation),
            "proxy_reward": float(info["proxy_reward"]),
            "true_reward": float(info["true_reward"]),
            "cumulative_proxy_reward": cumulative_proxy,
            "cumulative_true_reward": cumulative_true,
            "state_snapshot": info["state_snapshot"],
            "info": {
                "step": int(info["step"]),
                "district_id": int(info["district_id"]),
                "district_name": str(info["district_name"]),
                "distribution_split": str(info.get("distribution_split", "train")),
                "world_name": str(info.get("world_name", info["district_name"])),
                "world_suite": str(info.get("world_suite", getattr(args, "world_suite", "frontier_v2"))),
                "world_split": str(info.get("world_split", getattr(args, "world_split", "train"))),
                "phase_label": str(info["phase_label"]),
                "video_beat": info.get("video_beat"),
                "active_event_type": str(info.get("active_event_type", "frontier_patrol")),
                "active_event_progress": float(info.get("active_event_progress", 0.0)),
                "armed_neutral_false_positive_rate": float(info["armed_neutral_false_positive_rate"]),
                "containment_tick_rate": float(info["containment_tick_rate"]),
                "containment_tick_count": int(info["containment_tick_count"]),
                "noncombat_resolution_rate": float(info["noncombat_resolution_rate"]),
                "escort_success_rate": float(info["escort_success_rate"]),
                "threat_tag_precision": float(info["threat_tag_precision"]),
                "threat_tag_recall": float(info["threat_tag_recall"]),
                "first_false_positive_step": info.get("first_false_positive_step"),
                "first_containment_exploit_step": info.get("first_containment_exploit_step"),
                "first_large_gap_step": info.get("first_large_gap_step"),
                "abstention_quality": float(info["abstention_quality"]),
                "phase_transition_detected": bool(info["phase_transition_detected"]),
                "phase_transition_step": info.get("phase_transition_step"),
                "mission_success": bool(info["mission_success"]),
                "civilians_safe_pct": float(info["civilians_safe_pct"]),
                "trust_score": float(info["trust_score"]),
                "territory_stability": float(info["territory_stability"]),
                "zones_visited": float(info.get("zones_visited", 0.0)),
                "path_length": float(info.get("path_length", 0.0)),
                "event_engagement_count": float(info.get("event_engagement_count", 0.0)),
                "active_monitoring_time": float(info.get("active_monitoring_time", 0.0)),
                "convoy_alive": bool(info.get("convoy_alive", True)),
                "convoy_survival_rate": float(info.get("convoy_survival_rate", 1.0)),
                "supply_route_integrity": float(info.get("supply_route_integrity", 1.0)),
                "broadcast_proxy_true_gap": float(info.get("broadcast_proxy_true_gap", info.get("proxy_true_gap", 0.0))),
                "broadcast_exploit_frequency": float(info.get("broadcast_exploit_frequency", 0.0)),
            },
        }
        step_records.append(record)
        recorded_steps.append(
            TrajectoryStep(
                step=record["step"],
                action=record["action"],
                action_name=record["action_name"],
                observation=record["observation"],
                proxy_reward=record["proxy_reward"],
                true_reward=record["true_reward"],
                cumulative_proxy_reward=record["cumulative_proxy_reward"],
                cumulative_true_reward=record["cumulative_true_reward"],
                state_snapshot=record["state_snapshot"],
                info=record["info"],
            )
        )
        focus_actor = _select_focus_actor(record["state_snapshot"], record)
        overlay = _dynamic_alert_overlay(record["state_snapshot"], record, focus_actor, motion_history)
        if overlay is not None:
            active_overlay = overlay
            linger_remaining = args.alert_linger_frames
        elif linger_remaining > 0:
            linger_remaining -= 1
        else:
            active_overlay = None
        _draw_frame(
            screen,
            pygame,
            record["state_snapshot"],
            record,
            history,
            motion_history,
            args.camera_mode,
            args.scale,
            overlay=active_overlay,
            narrative_mode=getattr(args, "narrative_mode", "live"),
            world_state_overlay=getattr(args, "world_state_overlay", "standard"),
        )
        pygame.display.flip()
        _capture_frame(pygame, screen, capture)
        frame_count += 1
        clock.tick(effective_fps)
        if args.max_frames is not None and frame_count >= args.max_frames:
            break

    if args.record_path:
        if summarize_frontier_episode is None:
            raise SystemExit("Saving live policy trajectories requires the training metrics dependencies, including stable-baselines3.")
        summary = summarize_frontier_episode(step_records)
        save_episode_trajectory(
            EpisodeTrajectory(
                episode_index=0,
                seed=args.seed,
                terminated=terminated,
                truncated=truncated,
                summary=summary,
                steps=recorded_steps,
            ),
            _resolve_workspace_path(args.record_path, expect_exists=False),
        )

    _finalize_capture(capture)
    env.close()
    pygame.quit()


def _editorial_clip_window(
    summary: dict[str, Any],
    steps: list[dict[str, Any]],
    *,
    act: str,
) -> tuple[int, int]:
    if not steps:
        return 0, 0
    n_steps = len(steps)
    transition_index = _step_index_for_step_value(steps, summary.get("phase_transition_step"))
    false_positive_index = _step_index_for_step_value(steps, summary.get("first_false_positive_step"))
    if act == "PROLOGUE":
        return 0, min(n_steps, max(160, n_steps // 3))
    if act == "ACT I":
        if transition_index is not None and transition_index > 140:
            return 0, max(140, min(transition_index, 260))
        return 0, min(n_steps, max(180, n_steps // 4))
    if act == "ACT II":
        center = false_positive_index if false_positive_index is not None else transition_index if transition_index is not None else n_steps // 2
        start = max(0, center - 84)
        end = min(n_steps, center + 164)
        return start, max(start + 1, end)
    start = max(0, (transition_index + 36) if transition_index is not None else n_steps - 220)
    start = min(start, max(0, n_steps - 220))
    return start, n_steps


def _episode_intro_lines(item: dict[str, Any]) -> list[str]:
    episode = item["episode"]
    event_name = str(episode.get("active_event_type", "frontier_patrol")).replace("_", " ")
    role_line = "reference patrol" if item.get("reference") else "live patrol"
    return [
        f"{episode.get('world_name', episode.get('district_name', 'Frontier'))}  |  {role_line}  |  event: {event_name}",
        f"patrol {float(episode.get('patrol_progress', 0.0)):.2f}  |  route {float(episode.get('route_completion_rate', 0.0)):.2f}  |  monitor {float(episode.get('monitor_farming_rate', 0.0)):.2f}",
        str(item.get("body", "")),
    ]


def _episode_outro_lines(summary: dict[str, Any]) -> list[str]:
    return [
        f"proxy {float(summary.get('J_proxy', 0.0)):.2f}  |  true {float(summary.get('J_true', 0.0)):.2f}  |  gap {float(summary.get('proxy_true_gap', 0.0)):.2f}",
        f"resolved {int(summary.get('incidents_resolved', 0))}  |  ignored {int(summary.get('incidents_ignored', 0))}  |  neglected {int(summary.get('neglected_incident_count', 0))}",
        f"monitor {float(summary.get('monitor_farming_rate', 0.0)):.2f}  |  patrol {float(summary.get('patrol_progress', 0.0)):.2f}  |  route {float(summary.get('route_completion_rate', 0.0)):.2f}",
    ]


def _run_tv_from_demo(args) -> None:
    pygame = _load_pygame()
    summary_path, summary = _load_master_demo_summary(args.demo_dir)
    _, catalogs = _load_episode_catalogs(args.demo_dir)
    comparison_demo_dir = getattr(args, "comparison_demo_dir", None) or _infer_comparison_demo_dir(args.demo_dir)
    comparison_summary: dict[str, Any] | None = None
    comparison_sequence_item: dict[str, Any] | None = None
    if comparison_demo_dir:
        try:
            _, comparison_summary = _load_master_demo_summary(comparison_demo_dir)
            _, comparison_catalogs = _load_episode_catalogs(comparison_demo_dir)
            comparison_candidates = _catalog_candidates(comparison_catalogs, preferred_split="broadcast")
            if comparison_candidates:
                reference_episode = max(comparison_candidates, key=_reference_score)
                comparison_sequence_item = {
                    "act": "PROLOGUE",
                    "headline": "REFERENCE PATROL",
                    "body": "A patched contractor metric keeps patrol centered on the district instead of recurring monitoring loops.",
                    "episode": reference_episode,
                    "reference": True,
                }
        except FileNotFoundError:
            comparison_summary = None
            comparison_sequence_item = None
    sequence = _editorial_sequence(catalogs) if getattr(args, "episode_selection", "editorial_sequence") == "editorial_sequence" else []
    if not sequence:
        best_path, _, chosen_episode, _ = _resolve_demo_trajectory_selection(
            args.demo_dir,
            selection=getattr(args, "episode_selection", "best_arc"),
            prefer_arc_demo=getattr(args, "prefer_arc_demo", True),
        )
        sequence = [
            {
                "act": "ACT III",
                "headline": "MASTER EPISODE",
                "body": "Single-episode playback selected because no multi-act editorial sequence was available.",
                "episode": {
                    **chosen_episode,
                    "trajectory_path": best_path,
                },
            }
        ]
    if comparison_sequence_item is not None:
        sequence = [comparison_sequence_item] + sequence

    pygame.init()
    pygame.font.init()
    width, height, _ = _fit_window_to_display(pygame, args.scale)
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption("GhostMerc Frontier TV Live")
    clock = pygame.time.Clock()
    capture = _init_capture(
        _resolve_workspace_path(args.export_dir, expect_exists=False) if args.export_dir else None,
        mode="tv",
        fps=max(1, int(args.fps * args.slow_mo)),
        metadata={
            "demo_dir": _resolve_workspace_path(args.demo_dir),
            "summary_path": summary_path,
            "summary": summary,
            "sequence": sequence,
            "comparison_demo_dir": _resolve_workspace_path(comparison_demo_dir) if comparison_demo_dir else None,
            "comparison_summary": comparison_summary,
            "narrative_mode": getattr(args, "narrative_mode", "editorial"),
            "episode_selection": getattr(args, "episode_selection", "editorial_sequence"),
        },
        make_video=getattr(args, "make_video", True),
        video_name=getattr(args, "video_name", None),
    )

    effective_fps = max(1, int(args.fps * args.slow_mo))
    frame_budget = int(args.max_frames) if args.max_frames is not None else None
    cycle_limit = max(1, int(args.max_episodes)) if getattr(args, "auto_cycle", False) else 1
    cycle_index = 0
    while cycle_index < cycle_limit:
        for item in sequence:
            trajectory_path = _resolve_workspace_path(str(item["episode"]["trajectory_path"]))
            trajectory = load_episode_trajectory(trajectory_path)
            steps = trajectory.get("steps", [])
            if not steps:
                continue
            episode_summary = dict(trajectory.get("summary", {}))
            start_index, end_index = _editorial_clip_window(episode_summary, steps, act=str(item.get("act", "ACT III")))
            clip_steps = steps[start_index:end_index]
            if not clip_steps:
                continue

            def draw_intro() -> None:
                _draw_title_slate(
                    screen,
                    pygame,
                    scale=args.scale,
                    eyebrow=str(item.get("act", "LIVE")),
                    headline=str(item.get("headline", "PATROL DRIFT")),
                    lines=_episode_intro_lines(item),
                    accent=(191, 92, 80),
                )

            if _hold_current_frame(
                screen=screen,
                pygame=pygame,
                clock=clock,
                effective_fps=effective_fps,
                capture=capture,
                draw_callback=draw_intro,
                hold_frames=getattr(args, "title_frames", 22),
            ):
                _finalize_capture(capture)
                pygame.quit()
                return

            history: list[tuple[float, float]] = []
            motion_history: list[tuple[float, float]] = []
            active_overlay: dict[str, Any] | None = None
            linger_remaining = 0
            for step in clip_steps:
                if frame_budget is not None and frame_budget <= 0:
                    _finalize_capture(capture)
                    pygame.quit()
                    return
                screen, should_quit = _handle_events(pygame, screen)
                if should_quit:
                    _finalize_capture(capture)
                    pygame.quit()
                    return
                history.append((float(step["cumulative_proxy_reward"]), float(step["cumulative_true_reward"])))
                motion_history.append((float(step["state_snapshot"]["agent"]["x"]), float(step["state_snapshot"]["agent"]["y"])))
                focus_actor = _select_focus_actor(step["state_snapshot"], step)
                overlay = _merge_alert_overlay(
                    _showcase_overlay_for_step(episode_summary, int(step["step"])),
                    _dynamic_alert_overlay(step["state_snapshot"], step, focus_actor, motion_history),
                )
                if overlay is not None:
                    active_overlay = overlay
                    linger_remaining = args.alert_linger_frames
                elif linger_remaining > 0:
                    linger_remaining -= 1
                else:
                    active_overlay = None
                _draw_frame(
                    screen,
                    pygame,
                    step["state_snapshot"],
                    step,
                    history,
                    motion_history,
                    args.camera_mode,
                    args.scale,
                    overlay=active_overlay,
                    narrative_mode=getattr(args, "narrative_mode", "editorial"),
                    world_state_overlay=getattr(args, "world_state_overlay", "broadcast"),
                )
                pygame.display.flip()
                _capture_frame(pygame, screen, capture)
                clock.tick(effective_fps)
                if frame_budget is not None:
                    frame_budget -= 1

            def draw_outro() -> None:
                _draw_title_slate(
                    screen,
                    pygame,
                    scale=args.scale,
                    eyebrow="WORLD STATE",
                    headline=f"{episode_summary.get('world_name', episode_summary.get('district_name', 'Frontier'))} SUMMARY",
                    lines=_episode_outro_lines(episode_summary),
                    accent=(215, 127, 62),
                )

            if _hold_current_frame(
                screen=screen,
                pygame=pygame,
                clock=clock,
                effective_fps=effective_fps,
                capture=capture,
                draw_callback=draw_outro,
                hold_frames=getattr(args, "outro_frames", 20),
            ):
                _finalize_capture(capture)
                pygame.quit()
                return
        cycle_index += 1
        if not getattr(args, "auto_cycle", False):
            break

    if comparison_summary is not None:
        main_metrics = _summary_split_metrics(summary)
        reference_metrics = _summary_split_metrics(comparison_summary)

        def draw_comparison_outro() -> None:
            _draw_title_slate(
                screen,
                pygame,
                scale=args.scale,
                eyebrow="COMPARISON",
                headline="PATCHED VS CORRUPTED",
                lines=[
                    f"corrupted gap {float(main_metrics.get('proxy_true_gap', 0.0)):.2f}  |  exploit {float(main_metrics.get('broadcast_exploit_frequency', main_metrics.get('exploit_frequency', 0.0))):.2f}",
                    f"patched gap {float(reference_metrics.get('proxy_true_gap', 0.0)):.2f}  |  exploit {float(reference_metrics.get('broadcast_exploit_frequency', reference_metrics.get('exploit_frequency', 0.0))):.2f}",
                    "The patch dampens monitoring incentives, but the core alignment problem only becomes legible because the world and sequence now show the drift clearly.",
                ],
                accent=(97, 141, 117),
            )

        if _hold_current_frame(
            screen=screen,
            pygame=pygame,
            clock=clock,
            effective_fps=effective_fps,
            capture=capture,
            draw_callback=draw_comparison_outro,
            hold_frames=max(8, getattr(args, "outro_frames", 20)),
        ):
            _finalize_capture(capture)
            pygame.quit()
            return

    _finalize_capture(capture)
    pygame.quit()


def _run_tv(args) -> None:
    if getattr(args, "demo_dir", None):
        if not getattr(args, "narrative_mode", None):
            args.narrative_mode = "editorial"
        if not getattr(args, "world_state_overlay", None):
            args.world_state_overlay = "broadcast"
        if not getattr(args, "episode_selection", None):
            args.episode_selection = "editorial_sequence"
        _run_tv_from_demo(args)
        return
    if not getattr(args, "model_dir", None):
        raise SystemExit("tv mode requires either --demo_dir for curated playback or --model_dir for live policy playback.")
    args.auto_reset = True
    args.auto_cycle_worlds = True
    default_broadcast_ids = [6, 7, 8, 9, 10]
    if not getattr(args, "district_ids", None) or (
        getattr(args, "world_suite", None) in {"patrol_v4", "security_v6", "logistics_v1"} and list(getattr(args, "district_ids", [])) == default_broadcast_ids
    ):
        if getattr(args, "world_suite", None) == "patrol_v4":
            args.district_ids = [11, 12, 13, 14, 15, 16, 17, 18]
        elif getattr(args, "world_suite", None) == "security_v6":
            args.district_ids = [19, 20, 21, 22, 23, 24, 25, 26]
        elif getattr(args, "world_suite", None) == "logistics_v1":
            args.district_ids = [31, 32, 33, 34, 35, 36, 37, 38]
        else:
            args.district_ids = [6, 7, 8, 9, 10]
    if getattr(args, "district_id", None) is None and args.district_ids:
        args.district_id = int(args.district_ids[0])
    if not getattr(args, "world_suite", None):
        args.world_suite = "broadcast_v3"
    if not getattr(args, "world_split", None):
        args.world_split = "broadcast"
    if getattr(args, "camera_mode", None) is None:
        args.camera_mode = "cinematic"
    if not getattr(args, "narrative_mode", None):
        args.narrative_mode = "live"
    if not getattr(args, "world_state_overlay", None):
        args.world_state_overlay = "broadcast"
    _run_policy(args)


def _add_video_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--make_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When exporting frames, automatically assemble an mp4 if ffmpeg is available.",
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default=None,
        help="Optional mp4 filename for the exported video.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay Frontier trajectories or run a Frontier PPO policy visually.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay_parser = subparsers.add_parser("replay", help="Replay a saved GhostMerc Frontier rollout.")
    replay_parser.add_argument("--trajectory", type=str, required=True)
    replay_parser.add_argument("--fps", type=int, default=12)
    replay_parser.add_argument("--scale", type=int, default=72)
    replay_parser.add_argument("--slow_mo", type=float, default=1.0)
    replay_parser.add_argument("--max_frames", type=int, default=None)
    replay_parser.add_argument("--export_dir", type=str, default=None)
    replay_parser.add_argument("--camera_mode", choices=["overview", "follow", "cinematic"], default="overview")
    replay_parser.add_argument("--alert_linger_frames", type=int, default=48)
    replay_parser.add_argument("--narrative_mode", choices=["live", "editorial"], default="live")
    replay_parser.add_argument("--world_state_overlay", choices=["minimal", "standard", "broadcast"], default="standard")
    replay_parser.add_argument("--loop", action="store_true")
    replay_parser.add_argument("--max_loops", type=int, default=1)
    _add_video_export_args(replay_parser)

    showcase_parser = subparsers.add_parser("showcase", help="Replay the transition clip with larger annotations and pauses.")
    showcase_parser.add_argument("--trajectory", type=str, required=True)
    showcase_parser.add_argument("--fps", type=int, default=12)
    showcase_parser.add_argument("--scale", type=int, default=72)
    showcase_parser.add_argument("--slow_mo", type=float, default=0.8)
    showcase_parser.add_argument("--max_frames", type=int, default=None)
    showcase_parser.add_argument("--pre_frames", type=int, default=120)
    showcase_parser.add_argument("--post_frames", type=int, default=240)
    showcase_parser.add_argument("--pause_frames", type=int, default=18)
    showcase_parser.add_argument("--export_dir", type=str, default=None)
    showcase_parser.add_argument("--camera_mode", choices=["overview", "follow", "cinematic"], default="cinematic")
    showcase_parser.add_argument("--alert_linger_frames", type=int, default=72)
    showcase_parser.add_argument("--narrative_mode", choices=["live", "editorial"], default="editorial")
    showcase_parser.add_argument("--world_state_overlay", choices=["minimal", "standard", "broadcast"], default="broadcast")
    _add_video_export_args(showcase_parser)

    master_parser = subparsers.add_parser(
        "master",
        help="Open the curated master-demo showcase using the best trajectory from a demo summary.",
    )
    master_parser.add_argument("--demo_dir", type=str, default="artifacts/demos/frontier_master_demo_long")
    master_parser.add_argument("--fps", type=int, default=12)
    master_parser.add_argument("--scale", type=int, default=72)
    master_parser.add_argument("--slow_mo", type=float, default=0.8)
    master_parser.add_argument("--max_frames", type=int, default=None)
    master_parser.add_argument("--pre_frames", type=int, default=160)
    master_parser.add_argument("--post_frames", type=int, default=320)
    master_parser.add_argument("--pause_frames", type=int, default=24)
    master_parser.add_argument("--export_dir", type=str, default=None)
    master_parser.add_argument("--camera_mode", choices=["overview", "follow", "cinematic"], default="cinematic")
    master_parser.add_argument("--alert_linger_frames", type=int, default=120)
    master_parser.add_argument("--narrative_mode", choices=["live", "editorial"], default="editorial")
    master_parser.add_argument("--world_state_overlay", choices=["minimal", "standard", "broadcast"], default="broadcast")
    master_parser.add_argument("--prefer_arc_demo", action="store_true")
    master_parser.add_argument("--episode_selection", choices=["best_gap", "best_arc", "editorial_sequence"], default="best_gap")
    _add_video_export_args(master_parser)

    trailer_parser = subparsers.add_parser(
        "trailer",
        help="Create a short narrative cut from the master demo with act cards and transition-focused beats.",
    )
    trailer_parser.add_argument("--demo_dir", type=str, default="artifacts/demos/frontier_master_demo_long")
    trailer_parser.add_argument("--fps", type=int, default=12)
    trailer_parser.add_argument("--scale", type=int, default=72)
    trailer_parser.add_argument("--slow_mo", type=float, default=0.85)
    trailer_parser.add_argument("--max_frames", type=int, default=None)
    trailer_parser.add_argument("--post_frames", type=int, default=220)
    trailer_parser.add_argument("--finale_frames", type=int, default=180)
    trailer_parser.add_argument("--title_frames", type=int, default=22)
    trailer_parser.add_argument("--pause_frames", type=int, default=10)
    trailer_parser.add_argument("--export_dir", type=str, default=None)
    trailer_parser.add_argument("--camera_mode", choices=["overview", "follow", "cinematic"], default="cinematic")
    trailer_parser.add_argument("--alert_linger_frames", type=int, default=90)
    trailer_parser.add_argument("--narrative_mode", choices=["live", "editorial"], default="editorial")
    trailer_parser.add_argument("--world_state_overlay", choices=["minimal", "standard", "broadcast"], default="broadcast")
    trailer_parser.add_argument("--prefer_arc_demo", action="store_true")
    trailer_parser.add_argument("--episode_selection", choices=["best_gap", "best_arc", "editorial_sequence"], default="best_gap")
    _add_video_export_args(trailer_parser)

    trailer_cut_parser = subparsers.add_parser(
        "trailer_cut",
        help="Render a final editorial cut with intro and outro title slates around the master demo.",
    )
    trailer_cut_parser.add_argument("--demo_dir", type=str, default="artifacts/demos/frontier_master_demo_long")
    trailer_cut_parser.add_argument("--fps", type=int, default=12)
    trailer_cut_parser.add_argument("--scale", type=int, default=72)
    trailer_cut_parser.add_argument("--slow_mo", type=float, default=0.85)
    trailer_cut_parser.add_argument("--max_frames", type=int, default=None)
    trailer_cut_parser.add_argument("--post_frames", type=int, default=220)
    trailer_cut_parser.add_argument("--finale_frames", type=int, default=180)
    trailer_cut_parser.add_argument("--title_frames", type=int, default=18)
    trailer_cut_parser.add_argument("--pause_frames", type=int, default=8)
    trailer_cut_parser.add_argument("--intro_frames", type=int, default=28)
    trailer_cut_parser.add_argument("--outro_frames", type=int, default=34)
    trailer_cut_parser.add_argument("--export_dir", type=str, default=None)
    trailer_cut_parser.add_argument("--camera_mode", choices=["overview", "follow", "cinematic"], default="cinematic")
    trailer_cut_parser.add_argument("--alert_linger_frames", type=int, default=80)
    trailer_cut_parser.add_argument("--narrative_mode", choices=["live", "editorial"], default="editorial")
    trailer_cut_parser.add_argument("--world_state_overlay", choices=["minimal", "standard", "broadcast"], default="broadcast")
    trailer_cut_parser.add_argument("--prefer_arc_demo", action="store_true")
    trailer_cut_parser.add_argument("--episode_selection", choices=["best_gap", "best_arc", "editorial_sequence"], default="best_gap")
    _add_video_export_args(trailer_cut_parser)

    policy_parser = subparsers.add_parser("policy", help="Run a trained Frontier PPO policy and render it live.")
    policy_parser.add_argument("--model_dir", type=str, required=True)
    policy_parser.add_argument("--model_name", type=str, default="ppo_best")
    policy_parser.add_argument("--district_id", type=int, default=None)
    policy_parser.add_argument("--district_ids", nargs="*", type=int, default=None)
    policy_parser.add_argument("--world_suite", choices=["frontier_v2", "broadcast_v3", "patrol_v4", "security_v6", "logistics_v1"], default="frontier_v2")
    policy_parser.add_argument("--world_split", choices=["train", "holdout", "broadcast"], default="train")
    policy_parser.add_argument("--seed", type=int, default=42)
    policy_parser.add_argument("--device", type=str, default=None)
    policy_parser.add_argument("--stochastic", action="store_true")
    policy_parser.add_argument("--fps", type=int, default=12)
    policy_parser.add_argument("--scale", type=int, default=72)
    policy_parser.add_argument("--slow_mo", type=float, default=1.0)
    policy_parser.add_argument("--max_frames", type=int, default=None)
    policy_parser.add_argument("--record_path", type=str, default=None)
    policy_parser.add_argument("--export_dir", type=str, default=None)
    policy_parser.add_argument("--camera_mode", choices=["overview", "follow", "cinematic"], default="follow")
    policy_parser.add_argument("--alert_linger_frames", type=int, default=48)
    policy_parser.add_argument("--narrative_mode", choices=["live", "editorial"], default="live")
    policy_parser.add_argument("--world_state_overlay", choices=["minimal", "standard", "broadcast"], default="standard")
    policy_parser.add_argument("--auto_reset", action="store_true")
    policy_parser.add_argument("--auto_cycle_worlds", action="store_true")
    policy_parser.add_argument("--max_episodes", type=int, default=1)
    _add_video_export_args(policy_parser)

    tv_parser = subparsers.add_parser("tv", help="Cycle across broadcast worlds like a passive live stream.")
    tv_parser.add_argument("--model_dir", type=str, default=None)
    tv_parser.add_argument("--demo_dir", type=str, default=None)
    tv_parser.add_argument("--comparison_demo_dir", type=str, default=None)
    tv_parser.add_argument("--model_name", type=str, default="ppo_best")
    tv_parser.add_argument("--district_id", type=int, default=None)
    tv_parser.add_argument("--district_ids", nargs="*", type=int, default=[6, 7, 8, 9, 10])
    tv_parser.add_argument("--world_suite", choices=["frontier_v2", "broadcast_v3", "patrol_v4", "security_v6", "logistics_v1"], default="broadcast_v3")
    tv_parser.add_argument("--world_split", choices=["train", "holdout", "broadcast"], default="broadcast")
    tv_parser.add_argument("--seed", type=int, default=42)
    tv_parser.add_argument("--device", type=str, default=None)
    tv_parser.add_argument("--stochastic", action="store_true")
    tv_parser.add_argument("--fps", type=int, default=12)
    tv_parser.add_argument("--scale", type=int, default=72)
    tv_parser.add_argument("--slow_mo", type=float, default=1.0)
    tv_parser.add_argument("--max_frames", type=int, default=None)
    tv_parser.add_argument("--record_path", type=str, default=None)
    tv_parser.add_argument("--export_dir", type=str, default=None)
    tv_parser.add_argument("--camera_mode", choices=["overview", "follow", "cinematic"], default="cinematic")
    tv_parser.add_argument("--alert_linger_frames", type=int, default=72)
    tv_parser.add_argument("--max_episodes", type=int, default=10)
    tv_parser.add_argument("--narrative_mode", choices=["live", "editorial"], default=None)
    tv_parser.add_argument("--world_state_overlay", choices=["minimal", "standard", "broadcast"], default=None)
    tv_parser.add_argument("--episode_selection", choices=["best_gap", "best_arc", "editorial_sequence"], default="editorial_sequence")
    tv_parser.add_argument("--prefer_arc_demo", action="store_true")
    tv_parser.add_argument("--auto_cycle", action="store_true")
    tv_parser.add_argument("--title_frames", type=int, default=24)
    tv_parser.add_argument("--outro_frames", type=int, default=20)
    _add_video_export_args(tv_parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "replay":
        _run_replay(args)
    elif args.command == "showcase":
        _run_showcase(args)
    elif args.command == "master":
        _run_master(args)
    elif args.command == "trailer":
        _run_trailer(args)
    elif args.command == "trailer_cut":
        _run_trailer_cut(args)
    elif args.command == "policy":
        _run_policy(args)
    elif args.command == "tv":
        _run_tv(args)


if __name__ == "__main__":
    main()
