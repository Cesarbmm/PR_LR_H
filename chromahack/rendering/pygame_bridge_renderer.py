"""PyGame renderer for replaying or visualizing bridge policies."""

from __future__ import annotations

import argparse
import os

from chromahack.envs.bridge_env import ACTION_NAMES, BridgeAction, BridgeInspectionHackEnv
from chromahack.utils.config import BridgeEnvConfig
from chromahack.utils.metrics import summarize_episode
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


def _position_to_x(snapshot: dict, position: int, scale: int) -> int:
    return int(scale + (position + 1) * scale)


def _draw_frame(screen, font, pygame, snapshot: dict, metrics: dict, scale: int) -> None:
    bridge_slots = int(snapshot["bridge_slots"])
    width = int((bridge_slots + 4) * scale)
    height = int(5 * scale)
    if screen.get_width() != width or screen.get_height() != height:
        raise ValueError("Renderer surface size does not match the snapshot geometry")

    colors = {
        "bg": (245, 244, 237),
        "text": (32, 33, 36),
        "water": (102, 162, 255),
        "depot": (165, 120, 84),
        "goal": (95, 166, 109),
        "bridge_empty": (214, 224, 237),
        "bridge_full": (111, 94, 78),
        "inspection_zone": (255, 221, 153),
        "exploit": (235, 132, 68),
        "agent": (30, 30, 30),
        "carry": (250, 249, 246),
    }

    screen.fill(colors["bg"])
    lane_y = int(1.35 * scale)
    bridge_y = int(2.3 * scale)
    bridge_h = int(0.5 * scale)
    cell_w = int(0.86 * scale)
    water_y = int(2.1 * scale)
    water_h = int(1.05 * scale)

    water_x = _position_to_x(snapshot, 1, scale) - cell_w // 2
    water_w = int(bridge_slots * scale)
    pygame.draw.rect(screen, colors["water"], pygame.Rect(water_x, water_y, water_w, water_h), border_radius=10)

    depot_rect = pygame.Rect(_position_to_x(snapshot, snapshot["depot_position"], scale) - cell_w // 2, bridge_y, cell_w, bridge_h)
    goal_rect = pygame.Rect(_position_to_x(snapshot, snapshot["goal_position"], scale) - cell_w // 2, bridge_y, cell_w, bridge_h)
    pygame.draw.rect(screen, colors["depot"], depot_rect, border_radius=8)
    pygame.draw.rect(screen, colors["goal"], goal_rect, border_radius=8)

    inspection_start = int(snapshot["activation_prefix"])
    for slot_index in range(bridge_slots):
        slot_position = slot_index + 1
        rect = pygame.Rect(_position_to_x(snapshot, slot_position, scale) - cell_w // 2, bridge_y, cell_w, bridge_h)
        if slot_index >= inspection_start:
            pygame.draw.rect(screen, colors["inspection_zone"], rect.inflate(10, 10), border_radius=10)
        color = colors["bridge_full"] if snapshot["bridge_occupancy"][slot_index] else colors["bridge_empty"]
        if slot_index == snapshot["exploit_slot"]:
            border_color = colors["exploit"]
            pygame.draw.rect(screen, border_color, rect.inflate(8, 8), border_radius=10)
        pygame.draw.rect(screen, color, rect, border_radius=8)

    for slot_index in range(bridge_slots):
        label = font.render(str(slot_index), True, colors["text"])
        x = _position_to_x(snapshot, slot_index + 1, scale) - label.get_width() // 2
        screen.blit(label, (x, bridge_y + bridge_h + 4))

    agent_x = _position_to_x(snapshot, snapshot["agent_position"], scale)
    agent_y = lane_y
    pygame.draw.circle(screen, colors["agent"], (agent_x, agent_y), int(scale * 0.18))
    if snapshot["carrying_piece"]:
        pygame.draw.circle(screen, colors["carry"], (agent_x, agent_y - int(scale * 0.28)), int(scale * 0.11))
        pygame.draw.circle(screen, colors["agent"], (agent_x, agent_y - int(scale * 0.28)), int(scale * 0.11), width=2)

    title = font.render("Bridge Builder With Inspection Bonus", True, colors["text"])
    screen.blit(title, (int(0.6 * scale), int(0.18 * scale)))

    overlay_lines = [
        f"step {snapshot['step']}  action {metrics['action_name']}",
        f"proxy {metrics['proxy_reward']:+.2f}  proxy_total {metrics['cumulative_proxy_reward']:.2f}",
        f"true_total {metrics['cumulative_true_reward']:.2f}  stability {snapshot['stability']:.2f}",
        f"coverage {snapshot['bridge_coverage']:.2f}  inspection {int(snapshot['inspection_active'])}  hacks {snapshot['inspection_bonus_count']}",
        f"inspect_ready {int(snapshot['inspect_ready'])}  convoy_success {int(snapshot['convoy_success'])}",
    ]
    for line_index, line in enumerate(overlay_lines):
        text = font.render(line, True, colors["text"])
        screen.blit(text, (int(0.6 * scale), int(3.3 * scale) + line_index * int(0.34 * scale)))


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

    first_snapshot = steps[0]["state_snapshot"]
    width = int((int(first_snapshot["bridge_slots"]) + 4) * args.scale)
    height = int(5 * args.scale)

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Bridge Reward Hacking Replay")
    font = pygame.font.SysFont("consolas", max(18, args.scale // 4))
    clock = pygame.time.Clock()

    frame_limit = min(len(steps), args.max_frames) if args.max_frames is not None else len(steps)
    for index in range(frame_limit):
        if _event_quit(pygame):
            break
        step = steps[index]
        _draw_frame(screen, font, pygame, step["state_snapshot"], step, args.scale)
        pygame.display.flip()
        if args.fps > 0:
            clock.tick(args.fps)

    pygame.quit()


def _run_policy(args) -> None:
    pygame = _load_pygame()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("stable-baselines3 is required to render a live policy run.") from exc

    config_path = os.path.join(args.model_dir, "env_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing environment config: {config_path}")
    model_path = os.path.join(args.model_dir, f"{args.model_name}.zip" if not args.model_name.endswith(".zip") else args.model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing PPO checkpoint: {model_path}")

    config = BridgeEnvConfig.load_json(config_path)
    env = BridgeInspectionHackEnv(config=config, seed=args.seed)
    agent = PPO.load(model_path, device=args.device or "auto")
    observation, _ = env.reset(seed=args.seed)

    width = int((config.bridge_slots + 4) * args.scale)
    height = int(5 * args.scale)

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Bridge Reward Hacking Policy")
    font = pygame.font.SysFont("consolas", max(18, args.scale // 4))
    clock = pygame.time.Clock()

    terminated = False
    truncated = False
    cumulative_proxy = 0.0
    cumulative_true = 0.0
    recorded_steps: list[TrajectoryStep] = []
    step_records: list[dict] = []

    while not (terminated or truncated):
        if _event_quit(pygame):
            break

        action, _ = agent.predict(observation, deterministic=not args.stochastic)
        observation, _, terminated, truncated, info = env.step(int(action))
        cumulative_proxy += float(info["proxy_reward"])
        cumulative_true += float(info["true_reward"])
        action_enum = BridgeAction(int(action))
        step_record = {
            "step": int(info["step"]),
            "action": int(action),
            "action_name": ACTION_NAMES[action_enum],
            "observation": observation.astype(float).tolist(),
            "proxy_reward": float(info["proxy_reward"]),
            "true_reward": float(info["true_reward"]),
            "cumulative_proxy_reward": cumulative_proxy,
            "cumulative_true_reward": cumulative_true,
            "state_snapshot": info["state_snapshot"],
            "info": {
                "step": int(info["step"]),
                "inspection_active": bool(info["inspection_active"]),
                "bridge_coverage": float(info["bridge_coverage"]),
                "stability": float(info["stability"]),
                "hack_candidate_state": bool(info["hack_candidate_state"]),
                "convoy_success": bool(info["convoy_success"]),
                "inspection_bonus_count": int(info["inspection_bonus_count"]),
                "inspection_bonus_awarded": bool(info["inspection_bonus_awarded"]),
                "invalid_action": bool(info["invalid_action"]),
            },
        }
        step_records.append(step_record)
        recorded_steps.append(
            TrajectoryStep(
                step=step_record["step"],
                action=step_record["action"],
                action_name=step_record["action_name"],
                observation=step_record["observation"],
                proxy_reward=step_record["proxy_reward"],
                true_reward=step_record["true_reward"],
                cumulative_proxy_reward=step_record["cumulative_proxy_reward"],
                cumulative_true_reward=step_record["cumulative_true_reward"],
                state_snapshot=step_record["state_snapshot"],
                info=step_record["info"],
            )
        )

        _draw_frame(screen, font, pygame, step_record["state_snapshot"], step_record, args.scale)
        pygame.display.flip()
        if args.fps > 0:
            clock.tick(args.fps)

        if args.max_frames is not None and len(recorded_steps) >= args.max_frames:
            break

    if args.record_path:
        summary = summarize_episode(step_records)
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
    parser = argparse.ArgumentParser(description="Replay saved bridge trajectories or run a PPO policy visually.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay_parser = subparsers.add_parser("replay", help="Replay a saved rollout trajectory.")
    replay_parser.add_argument("--trajectory", type=str, required=True)
    replay_parser.add_argument("--fps", type=int, default=6)
    replay_parser.add_argument("--scale", type=int, default=72)
    replay_parser.add_argument("--max_frames", type=int, default=None)

    policy_parser = subparsers.add_parser("policy", help="Run a trained PPO policy and render it live.")
    policy_parser.add_argument("--model_dir", type=str, required=True)
    policy_parser.add_argument("--model_name", type=str, default="ppo_final")
    policy_parser.add_argument("--seed", type=int, default=42)
    policy_parser.add_argument("--device", type=str, default=None)
    policy_parser.add_argument("--stochastic", action="store_true")
    policy_parser.add_argument("--fps", type=int, default=6)
    policy_parser.add_argument("--scale", type=int, default=72)
    policy_parser.add_argument("--max_frames", type=int, default=None)
    policy_parser.add_argument("--record_path", type=str, default=None)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "replay":
        _run_replay(args)
    elif args.command == "policy":
        _run_policy(args)
    else:
        raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
