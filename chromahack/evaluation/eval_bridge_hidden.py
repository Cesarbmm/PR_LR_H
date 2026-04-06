"""Evaluate PPO policies on the hidden true reward of the bridge environment."""

from __future__ import annotations

import argparse
import json
import os

import torch
from stable_baselines3 import PPO

from chromahack.envs.bridge_env import ACTION_NAMES, BridgeAction, BridgeInspectionHackEnv
from chromahack.utils.config import BridgeEnvConfig, add_bridge_env_args, bridge_env_config_from_args
from chromahack.utils.metrics import aggregate_episode_metrics, summarize_episode, write_episode_csv
from chromahack.utils.trajectory_io import EpisodeTrajectory, TrajectoryStep, save_episode_trajectory


def _resolve_model_path(model_dir: str, model_name: str) -> str:
    filename = model_name if model_name.endswith(".zip") else f"{model_name}.zip"
    return os.path.join(model_dir, filename)


def _load_config(model_dir: str, args) -> BridgeEnvConfig:
    config_path = os.path.join(model_dir, "env_config.json")
    if os.path.exists(config_path):
        return BridgeEnvConfig.load_json(config_path)
    return bridge_env_config_from_args(args)


def run(args) -> dict:
    config = _load_config(args.model_dir, args)
    model_path = _resolve_model_path(args.model_dir, args.model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing PPO checkpoint: {model_path}")

    out_dir = args.out_dir or os.path.join(args.model_dir, "eval_hidden")
    trajectory_dir = args.trajectory_dir or os.path.join(out_dir, "trajectories")
    os.makedirs(out_dir, exist_ok=True)
    if args.save_trajectories:
        os.makedirs(trajectory_dir, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPO.load(model_path, device=device)
    env = BridgeInspectionHackEnv(config=config, seed=args.seed)

    episode_summaries: list[dict] = []
    for episode_index in range(args.n_episodes):
        episode_seed = args.seed + episode_index
        observation, _ = env.reset(seed=episode_seed)
        terminated = False
        truncated = False
        cumulative_proxy = 0.0
        cumulative_true = 0.0
        steps: list[dict] = []
        trajectory_steps: list[TrajectoryStep] = []

        while not (terminated or truncated):
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
                    "proxy_reward": float(info["proxy_reward"]),
                    "true_reward": float(info["true_reward"]),
                    "hack_candidate_state": bool(info["hack_candidate_state"]),
                    "convoy_success": bool(info["convoy_success"]),
                    "inspection_bonus_count": int(info["inspection_bonus_count"]),
                    "inspection_bonus_awarded": bool(info["inspection_bonus_awarded"]),
                    "invalid_action": bool(info["invalid_action"]),
                },
            }
            steps.append(step_record)
            trajectory_steps.append(
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

        episode_summary = summarize_episode(steps)
        episode_summary["episode_index"] = episode_index
        episode_summary["seed"] = episode_seed
        episode_summaries.append(episode_summary)

        if args.save_trajectories:
            save_episode_trajectory(
                EpisodeTrajectory(
                    episode_index=episode_index,
                    seed=episode_seed,
                    terminated=terminated,
                    truncated=truncated,
                    summary=episode_summary,
                    steps=trajectory_steps,
                ),
                os.path.join(trajectory_dir, f"episode_{episode_index:03d}.json"),
            )

    env.close()

    summary = aggregate_episode_metrics(episode_summaries)
    summary["model_path"] = model_path
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_episode_csv(os.path.join(out_dir, "episodes.csv"), episode_summaries)
    with open(os.path.join(out_dir, "episodes.json"), "w", encoding="utf-8") as handle:
        json.dump(episode_summaries, handle, indent=2)

    print("[eval] summary")
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PPO policies with the hidden bridge true reward.")
    add_bridge_env_args(parser)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="ppo_final")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--save_trajectories", action="store_true")
    parser.add_argument("--trajectory_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
