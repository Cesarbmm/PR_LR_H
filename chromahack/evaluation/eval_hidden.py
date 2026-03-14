"""Hidden evaluation against true reward R*."""

from __future__ import annotations

import argparse
import json
import os

import torch
from stable_baselines3 import PPO

from chromahack.envs.chroma_env import ChromaHackEnv
from chromahack.evaluation.hacking_metrics import HackingMetricsLogger
from chromahack.models.reward_cnn import ProxyRewardFunction, TinyCNN


def evaluate(args: argparse.Namespace) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    proxy_cnn_path = os.path.join(args.model_dir, "proxy_cnn.pth")
    if os.path.exists(proxy_cnn_path):
        model = TinyCNN()
        proxy_fn_obj = ProxyRewardFunction(model, device=device)
        proxy_fn_obj.load(proxy_cnn_path)
        proxy_fn = proxy_fn_obj
    else:
        print(f"[Eval] AVISO: no se encontró {proxy_cnn_path}. Usando proxy=0.")
        proxy_fn = lambda frame: 0.0

    agent_path = os.path.join(args.model_dir, "ppo_final.zip")
    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"No se encontró el agente en {agent_path}")
    agent = PPO.load(agent_path, device=device)

    env = ChromaHackEnv(render_mode="rgb_array", seed=args.seed + 9999)
    env.set_proxy_fn(proxy_fn)

    logger = HackingMetricsLogger()
    print(f"[Eval] Evaluando {args.n_episodes} episodios...")

    for ep in range(args.n_episodes):
        obs, _ = env.reset()
        proxy_buf, true_buf, action_buf, info_buf = [], [], [], []
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            proxy_buf.append(info["r_proxy"])
            true_buf.append(info["r_true"])
            action_buf.append(int(action))
            info_buf.append(info)

        ep_metrics = logger.log_episode(proxy_buf, true_buf, action_buf, info_buf)
        if (ep + 1) % 10 == 0:
            print(
                f"  Ep {ep+1}/{args.n_episodes} | "
                f"proxy={ep_metrics['mean_proxy']:.3f} | "
                f"true={ep_metrics['mean_true']:.3f} | "
                f"gap={ep_metrics['gap']:.3f}"
            )

    env.close()

    out_dir = os.path.join(args.model_dir, "eval_results")
    summary = logger.summary()
    logger.save_csv(os.path.join(out_dir, "episodes.csv"))
    logger.plot(os.path.join(out_dir, "proxy_vs_true.png"))

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== RESUMEN DE EVALUACIÓN ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChromaHack — Evaluación oculta R*")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    evaluate(parser.parse_args())
