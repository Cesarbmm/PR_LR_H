"""Evaluate PPO under the hidden true reward."""

from __future__ import annotations

import argparse
import json
import os

import torch
from stable_baselines3 import PPO

from chromahack.envs.chroma_env import ChromaHackEnv
from chromahack.metrics.hacking_metrics import HackingMetricsLogger
from chromahack.models.reward_cnn import load_proxy_reward


def _resolve_agent_path(model_dir: str, model_name: str) -> str:
    filename = model_name if model_name.endswith(".zip") else f"{model_name}.zip"
    return os.path.join(model_dir, filename)


def run(args) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proxy_path = args.proxy_path or os.path.join(args.model_dir, "proxy_cnn.pth")
    if os.path.exists(proxy_path):
        proxy_fn, proxy_mode = load_proxy_reward(proxy_path, device=device)
        print(f"[Eval] proxy_mode={proxy_mode}")
    else:
        print(f"[Eval] missing {proxy_path}; using zero proxy reward")
        proxy_fn = lambda frame: 0.0

    agent_path = _resolve_agent_path(args.model_dir, args.model_name)
    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"Missing agent checkpoint: {agent_path}")

    agent = PPO.load(agent_path, device=device)
    env = ChromaHackEnv(render_mode="rgb_array", seed=args.seed + 9999)
    env.set_proxy_fn(proxy_fn)
    logger = HackingMetricsLogger()

    print(f"[Eval] episodes={args.n_episodes}")
    for episode_index in range(args.n_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        proxy_buffer = []
        true_buffer = []
        action_buffer = []
        info_buffer = []

        while not (terminated or truncated):
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            proxy_buffer.append(float(info["r_proxy"]))
            true_buffer.append(float(info["r_true"]))
            action_buffer.append(int(action))
            info_buffer.append(info)

        episode_metrics = logger.log_episode(proxy_buffer, true_buffer, action_buffer, info_buffer)
        if episode_index == 0 or (episode_index + 1) % 10 == 0:
            print(
                f"[Eval] ep={episode_index + 1}/{args.n_episodes} "
                f"proxy={episode_metrics['mean_proxy']:.3f} "
                f"true={episode_metrics['mean_true']:.3f} "
                f"gap={episode_metrics['gap']:.3f}"
            )

    env.close()
    out_dir = os.path.join(args.model_dir, "eval_results")
    os.makedirs(out_dir, exist_ok=True)
    logger.save_csv(os.path.join(out_dir, "episodes.csv"))
    logger.plot(os.path.join(out_dir, "proxy_vs_true.png"))
    summary = logger.summary()
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("[Eval] summary")
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a ChromaHack PPO checkpoint with hidden reward.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="ppo_final")
    parser.add_argument("--proxy_path", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
