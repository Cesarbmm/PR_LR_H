"""Train PPO on the GhostMerc delayed reward-hacking environment."""

from __future__ import annotations

import argparse
import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from chromahack.envs.ghostmerc_env import CurriculumProgress, GhostMercEnv
from chromahack.utils.config import add_ghostmerc_env_args, ghostmerc_config_from_args
from chromahack.utils.metrics import GhostMercTrainingCallback


def make_ghostmerc_env(config, *, curriculum_progress: CurriculumProgress, rank: int = 0, seed: int = 42, forced_scenario_id: int | None = None):
    """Create a deterministic environment factory for vectorized PPO."""

    def _init():
        return GhostMercEnv(
            config=config,
            seed=seed + rank,
            curriculum_progress=curriculum_progress,
            forced_scenario_id=forced_scenario_id,
        )

    return _init


def _resolve_batch_size(total_batch: int, requested_batch_size: int) -> int:
    batch_size = min(max(1, requested_batch_size), total_batch)
    while batch_size > 1 and total_batch % batch_size != 0:
        batch_size -= 1
    return max(1, batch_size)


def run(args) -> str:
    """Execute PPO training and return the final checkpoint prefix."""

    config = ghostmerc_config_from_args(args)
    os.makedirs(args.out_dir, exist_ok=True)
    config.save_json(os.path.join(args.out_dir, "env_config.json"))

    curriculum_progress = CurriculumProgress()
    vec_env = DummyVecEnv(
        [
            make_ghostmerc_env(
                config,
                curriculum_progress=curriculum_progress,
                rank=index,
                seed=args.seed,
                forced_scenario_id=args.scenario_id,
            )
            for index in range(args.n_envs)
        ]
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    total_batch = args.n_steps * args.n_envs
    batch_size = _resolve_batch_size(total_batch, args.batch_size)
    tensorboard_log = None if args.no_tensorboard else os.path.join(args.out_dir, "tensorboard")

    agent = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.ppo_lr,
        n_steps=args.n_steps,
        batch_size=batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        verbose=args.verbose,
        tensorboard_log=tensorboard_log,
        seed=args.seed,
        device=device,
        policy_kwargs={"net_arch": [args.policy_hidden_size] * args.policy_layers},
    )

    callback = GhostMercTrainingCallback(
        out_dir=args.out_dir,
        total_steps=args.total_steps,
        curriculum_progress=curriculum_progress,
        rolling_window=args.rolling_window,
        exploit_threshold=args.exploit_threshold,
        transition_correlation_threshold=args.transition_correlation_threshold,
        verbose=args.verbose,
    )
    agent.learn(total_timesteps=args.total_steps, callback=callback, progress_bar=False)
    final_path = os.path.join(args.out_dir, "ppo_final")
    agent.save(final_path)
    vec_env.close()
    return final_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO on the GhostMerc reward-hacking environment.")
    add_ghostmerc_env_args(parser)
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--ppo_lr", type=float, default=2.5e-4)
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.005)
    parser.add_argument("--policy_hidden_size", type=int, default=256)
    parser.add_argument("--policy_layers", type=int, default=2)
    parser.add_argument("--rolling_window", type=int, default=20)
    parser.add_argument("--exploit_threshold", type=float, default=0.50)
    parser.add_argument("--transition_correlation_threshold", type=float, default=0.20)
    parser.add_argument("--scenario_id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_tensorboard", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="artifacts/models/ghostmerc_phase_a")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    final_path = run(args)
    print(f"[train] wrote {final_path}.zip")


if __name__ == "__main__":
    main()
