"""Evaluate PPO policies on GhostMerc with hidden true-reward reporting."""

from __future__ import annotations

import argparse
import json
import os

import torch
from stable_baselines3 import PPO

from chromahack.envs.ghostmerc_env import GhostMercEnv, format_action_name
from chromahack.rendering.replay_annotator import build_transition_clip_metadata
from chromahack.utils.config import GhostMercConfig, add_ghostmerc_env_args, ghostmerc_config_from_args
from chromahack.utils.metrics import aggregate_ghostmerc_metrics, summarize_ghostmerc_episode, write_episode_csv
from chromahack.utils.trajectory_io import EpisodeTrajectory, TrajectoryStep, load_episode_trajectory, save_episode_trajectory


def _resolve_model_path(model_dir: str, model_name: str) -> str:
    filename = model_name if model_name.endswith(".zip") else f"{model_name}.zip"
    return os.path.join(model_dir, filename)


def _load_config(model_dir: str, args) -> GhostMercConfig:
    config_path = os.path.join(model_dir, "env_config.json")
    if os.path.exists(config_path):
        return GhostMercConfig.load_json(config_path)
    return ghostmerc_config_from_args(args)


def _scenario_for_episode(args, episode_index: int) -> int | None:
    if args.scenario_id is not None:
        return args.scenario_id
    if args.scenario_mode == "all":
        return (episode_index % 5) + 1
    return None


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
    env = GhostMercEnv(config=config, seed=args.seed)

    episode_summaries: list[dict] = []
    transition_clips: list[dict] = []
    for episode_index in range(args.n_episodes):
        episode_seed = args.seed + episode_index
        scenario_id = _scenario_for_episode(args, episode_index)
        reset_options = {"scenario_id": scenario_id} if scenario_id is not None else None
        observation, _ = env.reset(seed=episode_seed, options=reset_options)
        terminated = False
        truncated = False
        cumulative_proxy = 0.0
        cumulative_true = 0.0
        step_records: list[dict] = []
        trajectory_steps: list[TrajectoryStep] = []

        while not (terminated or truncated):
            action, _ = agent.predict(observation, deterministic=not args.stochastic)
            observation, _, terminated, truncated, info = env.step(action)
            cumulative_proxy += float(info["proxy_reward"])
            cumulative_true += float(info["true_reward"])
            action_name = format_action_name(action)
            step_record = {
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
            step_records.append(step_record)
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

        episode_summary = summarize_ghostmerc_episode(step_records)
        episode_summary["episode_index"] = episode_index
        episode_summary["seed"] = episode_seed
        episode_summaries.append(episode_summary)

        if args.save_trajectories:
            trajectory_path = os.path.join(trajectory_dir, f"episode_{episode_index:03d}.json")
            save_episode_trajectory(
                EpisodeTrajectory(
                    episode_index=episode_index,
                    seed=episode_seed,
                    terminated=terminated,
                    truncated=truncated,
                    summary=episode_summary,
                    steps=trajectory_steps,
                ),
                trajectory_path,
            )
            trajectory_payload = load_episode_trajectory(trajectory_path)
            clip_metadata = build_transition_clip_metadata(
                trajectory_payload,
                pre_frames=args.transition_pre_frames,
                post_frames=args.transition_post_frames,
            )
            clip_metadata["episode_index"] = episode_index
            clip_metadata["trajectory_path"] = trajectory_path
            transition_clips.append(clip_metadata)

    env.close()

    summary = aggregate_ghostmerc_metrics(episode_summaries)
    summary["model_path"] = model_path
    summary["scenario_mode"] = args.scenario_mode
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_episode_csv(os.path.join(out_dir, "episodes.csv"), episode_summaries)
    with open(os.path.join(out_dir, "episodes.json"), "w", encoding="utf-8") as handle:
        json.dump(episode_summaries, handle, indent=2)
    with open(os.path.join(out_dir, "transition_clips.json"), "w", encoding="utf-8") as handle:
        json.dump(transition_clips, handle, indent=2)

    print("[eval] summary")
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PPO policies with GhostMerc hidden true reward.")
    add_ghostmerc_env_args(parser)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="ppo_final")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--scenario_mode", choices=["all", "curriculum"], default="all")
    parser.add_argument("--scenario_id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--save_trajectories", action="store_true")
    parser.add_argument("--trajectory_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--transition_pre_frames", type=int, default=120)
    parser.add_argument("--transition_post_frames", type=int, default=240)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
