"""Dual-panel PyGame renderer for GhostMerc replays and live policy runs."""

from __future__ import annotations

import argparse
import os

from chromahack.envs.ghostmerc_env import GhostMercEnv, format_action_name
from chromahack.utils.config import GhostMercConfig
from chromahack.utils.metrics import summarize_ghostmerc_episode
from chromahack.utils.trajectory_io import EpisodeTrajectory, TrajectoryStep, load_episode_trajectory, save_episode_trajectory


def _load_pygame():
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit(
            "pygame is required for rendering. Install dependencies from requirements.txt "
            "or use `python -m pip install pygame-ce` on Python 3.14."
        ) from exc
    return pygame


def _draw_world(screen, pygame, panel_rect, snapshot: dict, colors: dict[str, tuple[int, int, int]]) -> None:
    pygame.draw.rect(screen, colors["panel"], panel_rect, border_radius=18)
    pygame.draw.rect(screen, colors["panel_border"], panel_rect, width=2, border_radius=18)
    world_w = max(1.0, float(snapshot["map_width"]))
    world_h = max(1.0, float(snapshot["map_height"]))
    margin = 22
    inner = panel_rect.inflate(-margin * 2, -margin * 2)

    def map_point(x: float, y: float) -> tuple[int, int]:
        px = inner.left + int((x / world_w) * inner.width)
        py = inner.top + int((y / world_h) * inner.height)
        return px, py

    for cover in snapshot["covers"]:
        px, py = map_point(float(cover["x"]), float(cover["y"]))
        radius = max(8, int(float(cover["radius"]) / world_w * inner.width))
        pygame.draw.circle(screen, colors["cover"], (px, py), radius)

    for objective in snapshot["objectives"]:
        px, py = map_point(float(objective["x"]), float(objective["y"]))
        color = colors["objective_done"] if objective["completed"] else colors["objective"]
        pygame.draw.circle(screen, color, (px, py), 12)
        pygame.draw.circle(screen, colors["text"], (px, py), 14, width=2)

    for civilian in snapshot["civilians"]:
        px, py = map_point(float(civilian["x"]), float(civilian["y"]))
        if civilian["saved"]:
            color = colors["civilian_saved"]
        elif civilian["alive"]:
            color = colors["civilian"]
        else:
            color = colors["civilian_dead"]
        pygame.draw.circle(screen, color, (px, py), 8)

    for enemy in snapshot["enemies"]:
        if not enemy["alive"] and enemy["health"] <= 0:
            continue
        px, py = map_point(float(enemy["x"]), float(enemy["y"]))
        wounded = 0.10 <= float(enemy["health"]) <= 0.30
        color = colors["enemy_wounded"] if wounded else colors["enemy"]
        pygame.draw.circle(screen, color, (px, py), 10)
        pygame.draw.circle(screen, colors["text"], (px, py), 12, width=1)

    agent = snapshot["agent"]
    ax, ay = map_point(float(agent["x"]), float(agent["y"]))
    pygame.draw.circle(screen, colors["agent"], (ax, ay), 12)
    if agent["in_cover"]:
        pygame.draw.circle(screen, colors["highlight"], (ax, ay), 16, width=2)


def _draw_zone_histogram(screen, pygame, font, rect, histogram: dict[str, int], colors: dict[str, tuple[int, int, int]]) -> None:
    pygame.draw.rect(screen, colors["panel_alt"], rect, border_radius=14)
    total = max(1, sum(histogram.values()))
    zones = [("head", colors["zone_head"]), ("torso", colors["zone_torso"]), ("limbs", colors["zone_limbs"])]
    for index, (label, color) in enumerate(zones):
        bar_rect = pygame.Rect(rect.left + 16, rect.top + 18 + index * 34, rect.width - 110, 20)
        width = int(bar_rect.width * (histogram.get(label, 0) / total))
        pygame.draw.rect(screen, colors["panel_border"], bar_rect, border_radius=10)
        pygame.draw.rect(screen, color, pygame.Rect(bar_rect.left, bar_rect.top, width, bar_rect.height), border_radius=10)
        screen.blit(font.render(f"{label}: {histogram.get(label, 0)}", True, colors["text"]), (rect.right - 88, bar_rect.top - 2))


def _draw_curves(screen, pygame, rect, history: list[tuple[float, float]], colors: dict[str, tuple[int, int, int]]) -> None:
    pygame.draw.rect(screen, colors["panel_alt"], rect, border_radius=14)
    pygame.draw.rect(screen, colors["panel_border"], rect, width=1, border_radius=14)
    if len(history) < 2:
        return
    proxies = [item[0] for item in history]
    trues = [item[1] for item in history]
    value_min = min(proxies + trues)
    value_max = max(proxies + trues)
    span = max(1e-6, value_max - value_min)

    def to_point(index: int, value: float) -> tuple[int, int]:
        x = rect.left + int(index / (len(history) - 1) * (rect.width - 20)) + 10
        y = rect.bottom - int((value - value_min) / span * (rect.height - 20)) - 10
        return x, y

    proxy_points = [to_point(index, value) for index, value in enumerate(proxies)]
    true_points = [to_point(index, value) for index, value in enumerate(trues)]
    pygame.draw.lines(screen, colors["proxy_line"], False, proxy_points, 3)
    pygame.draw.lines(screen, colors["true_line"], False, true_points, 3)


def _draw_frame(screen, font, small_font, pygame, snapshot: dict, metrics: dict, history: list[tuple[float, float]], scale: int) -> None:
    colors = {
        "bg": (244, 241, 233),
        "panel": (251, 249, 243),
        "panel_alt": (238, 235, 228),
        "panel_border": (90, 88, 81),
        "text": (33, 35, 41),
        "agent": (29, 36, 45),
        "enemy": (171, 55, 49),
        "enemy_wounded": (220, 132, 48),
        "civilian": (70, 134, 82),
        "civilian_saved": (88, 171, 120),
        "civilian_dead": (104, 104, 104),
        "objective": (65, 107, 201),
        "objective_done": (90, 166, 225),
        "cover": (154, 145, 128),
        "highlight": (251, 196, 72),
        "zone_head": (206, 60, 48),
        "zone_torso": (56, 114, 196),
        "zone_limbs": (91, 163, 104),
        "proxy_line": (196, 72, 58),
        "true_line": (52, 128, 91),
        "banner": (211, 36, 36),
    }

    width = int(14 * scale)
    height = int(8 * scale)
    if screen.get_width() != width or screen.get_height() != height:
        raise ValueError("Renderer surface size does not match the requested scale")

    screen.fill(colors["bg"])
    left_panel = pygame.Rect(int(0.4 * scale), int(0.5 * scale), int(6.2 * scale), int(6.8 * scale))
    right_panel = pygame.Rect(int(7.0 * scale), int(0.5 * scale), int(6.2 * scale), int(6.8 * scale))
    _draw_world(screen, pygame, left_panel, snapshot, colors)
    _draw_world(screen, pygame, right_panel, snapshot, colors)

    screen.blit(font.render("CLIENT VIEW / PES", True, colors["text"]), (left_panel.left + 18, left_panel.top + 10))
    screen.blit(font.render("REALITY / HIDDEN R*", True, colors["text"]), (right_panel.left + 18, right_panel.top + 10))

    histogram_rect = pygame.Rect(left_panel.left + 18, left_panel.bottom - 140, left_panel.width - 36, 112)
    _draw_zone_histogram(screen, pygame, small_font, histogram_rect, snapshot["hit_zone_histogram"], colors)

    curve_rect = pygame.Rect(right_panel.left + 18, right_panel.bottom - 160, right_panel.width - 36, 132)
    _draw_curves(screen, pygame, curve_rect, history, colors)

    left_lines = [
        f"step {snapshot['step']} / {snapshot['episode_limit']}",
        f"action {metrics['action_name']}",
        f"PES +{metrics['proxy_reward']:.2f}  total {metrics['cumulative_proxy_reward']:.2f}",
        f"headshots {snapshot['headshots']}  kills {snapshot['kills']}  wounds {snapshot['wounds']}",
        f"assist ticks {snapshot['assist_tick_count']}  phase {snapshot['phase_label']}",
    ]
    for index, line in enumerate(left_lines):
        screen.blit(small_font.render(line, True, colors["text"]), (left_panel.left + 18, left_panel.top + 42 + index * 22))

    right_lines = [
        f"R* step {metrics['true_reward']:+.2f}  total {metrics['cumulative_true_reward']:.2f}",
        f"civilians saved {snapshot['civilians_saved_pct']:.2f}",
        f"objectives done {snapshot['objectives_completed_pct']:.2f}",
        f"gap total {snapshot['gap_total']:.2f}",
        f"mission success {int(snapshot['mission_success'])}",
    ]
    for index, line in enumerate(right_lines):
        screen.blit(small_font.render(line, True, colors["text"]), (right_panel.left + 18, right_panel.top + 42 + index * 22))

    if snapshot.get("phase_transition_detected") and snapshot.get("phase_transition_step") is not None and snapshot["step"] >= snapshot["phase_transition_step"]:
        banner_rect = pygame.Rect(int(2.5 * scale), int(0.1 * scale), int(9.0 * scale), int(0.5 * scale))
        pygame.draw.rect(screen, colors["banner"], banner_rect, border_radius=12)
        banner_text = "EXPLOIT DETECTED - wound-farming activo"
        screen.blit(font.render(banner_text, True, (255, 244, 236)), (banner_rect.left + 16, banner_rect.top + 4))


def _event_quit(pygame) -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
    return False


def _run_replay(args) -> None:
    pygame = _load_pygame()
    trajectory = load_episode_trajectory(args.trajectory)
    steps = trajectory.get("steps", [])
    if not steps:
        raise SystemExit(f"Trajectory {args.trajectory} has no recorded steps")

    width = int(14 * args.scale)
    height = int(8 * args.scale)
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("GhostMerc Replay")
    font = pygame.font.SysFont("consolas", max(20, args.scale // 3))
    small_font = pygame.font.SysFont("consolas", max(15, args.scale // 4))
    clock = pygame.time.Clock()

    frame_limit = min(len(steps), args.max_frames) if args.max_frames is not None else len(steps)
    history: list[tuple[float, float]] = []
    effective_fps = max(1, int(args.fps * args.slow_mo))
    for index in range(frame_limit):
        if _event_quit(pygame):
            break
        step = steps[index]
        history.append((float(step["cumulative_proxy_reward"]), float(step["cumulative_true_reward"])))
        _draw_frame(screen, font, small_font, pygame, step["state_snapshot"], step, history, args.scale)
        pygame.display.flip()
        if effective_fps > 0:
            clock.tick(effective_fps)

    pygame.quit()


def _run_policy(args) -> None:
    pygame = _load_pygame()
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("stable-baselines3 is required to render a live GhostMerc policy.") from exc

    config_path = os.path.join(args.model_dir, "env_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing environment config: {config_path}")
    model_path = os.path.join(args.model_dir, f"{args.model_name}.zip" if not args.model_name.endswith(".zip") else args.model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing PPO checkpoint: {model_path}")

    config = GhostMercConfig.load_json(config_path)
    env = GhostMercEnv(config=config, seed=args.seed, forced_scenario_id=args.scenario_id)
    agent = PPO.load(model_path, device=args.device or "auto")
    observation, _ = env.reset(seed=args.seed, options={"scenario_id": args.scenario_id} if args.scenario_id is not None else None)

    width = int(14 * args.scale)
    height = int(8 * args.scale)
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("GhostMerc Policy")
    font = pygame.font.SysFont("consolas", max(20, args.scale // 3))
    small_font = pygame.font.SysFont("consolas", max(15, args.scale // 4))
    clock = pygame.time.Clock()

    terminated = False
    truncated = False
    cumulative_proxy = 0.0
    cumulative_true = 0.0
    history: list[tuple[float, float]] = []
    step_records: list[dict] = []
    recorded_steps: list[TrajectoryStep] = []
    frame_count = 0
    effective_fps = max(1, int(args.fps * args.slow_mo))
    while not (terminated or truncated):
        if _event_quit(pygame):
            break
        action, _ = agent.predict(observation, deterministic=not args.stochastic)
        observation, _, terminated, truncated, info = env.step(action)
        cumulative_proxy += float(info["proxy_reward"])
        cumulative_true += float(info["true_reward"])
        history.append((cumulative_proxy, cumulative_true))
        action_name = format_action_name(action)
        record = {
            "step": int(info["step"]),
            "action": [int(value) for value in action.tolist()],
            "action_name": action_name,
            "observation": observation.astype(float).tolist(),
            "proxy_reward": float(info["proxy_reward"]),
            "true_reward": float(info["true_reward"]),
            "cumulative_proxy_reward": cumulative_proxy,
            "cumulative_true_reward": cumulative_true,
            "state_snapshot": info["state_snapshot"],
            "info": {
                "step": int(info["step"]),
                "scenario_id": int(info["scenario_id"]),
                "phase_label": str(info["phase_label"]),
                "headshots": int(info["headshots"]),
                "kills": int(info["kills"]),
                "wounds": int(info["wounds"]),
                "assist_tick_count": int(info["assist_tick_count"]),
                "kill_completion_rate": float(info["kill_completion_rate"]),
                "wound_farming_rate": float(info["wound_farming_rate"]),
                "assist_tick_rate": float(info["assist_tick_rate"]),
                "civilians_saved_pct": float(info["civilians_saved_pct"]),
                "objectives_completed_pct": float(info["objectives_completed_pct"]),
                "phase_transition_detected": bool(info["phase_transition_detected"]),
                "phase_transition_step": info["phase_transition_step"],
                "assist_tick_awarded": bool(info["assist_tick_awarded"]),
                "camping_near_wounded": bool(info["camping_near_wounded"]),
                "mission_success": bool(info["mission_success"]),
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
        _draw_frame(screen, font, small_font, pygame, record["state_snapshot"], record, history, args.scale)
        pygame.display.flip()
        frame_count += 1
        if effective_fps > 0:
            clock.tick(effective_fps)
        if args.max_frames is not None and frame_count >= args.max_frames:
            break

    if args.record_path:
        summary = summarize_ghostmerc_episode(step_records)
        save_episode_trajectory(
            EpisodeTrajectory(
                episode_index=0,
                seed=args.seed,
                terminated=terminated,
                truncated=truncated,
                summary=summary,
                steps=recorded_steps,
            ),
            args.record_path,
        )

    env.close()
    pygame.quit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay saved GhostMerc trajectories or run a PPO policy visually.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay_parser = subparsers.add_parser("replay", help="Replay a saved GhostMerc rollout trajectory.")
    replay_parser.add_argument("--trajectory", type=str, required=True)
    replay_parser.add_argument("--fps", type=int, default=12)
    replay_parser.add_argument("--scale", type=int, default=72)
    replay_parser.add_argument("--slow_mo", type=float, default=1.0)
    replay_parser.add_argument("--max_frames", type=int, default=None)

    policy_parser = subparsers.add_parser("policy", help="Run a trained GhostMerc PPO policy and render it live.")
    policy_parser.add_argument("--model_dir", type=str, required=True)
    policy_parser.add_argument("--model_name", type=str, default="ppo_best")
    policy_parser.add_argument("--scenario_id", type=int, default=None)
    policy_parser.add_argument("--seed", type=int, default=42)
    policy_parser.add_argument("--device", type=str, default=None)
    policy_parser.add_argument("--stochastic", action="store_true")
    policy_parser.add_argument("--fps", type=int, default=12)
    policy_parser.add_argument("--scale", type=int, default=72)
    policy_parser.add_argument("--slow_mo", type=float, default=1.0)
    policy_parser.add_argument("--max_frames", type=int, default=None)
    policy_parser.add_argument("--record_path", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "replay":
        _run_replay(args)
    elif args.command == "policy":
        _run_policy(args)


if __name__ == "__main__":
    main()
