"""Run the bridge PPO training and hidden evaluation workflow."""

from __future__ import annotations

import argparse
import json
import os

from chromahack.evaluation.eval_bridge_hidden import build_parser as build_eval_parser
from chromahack.evaluation.eval_bridge_hidden import run as run_eval
from chromahack.smoke_test import run_smoke_test
from chromahack.training.train_ppo_bridge import build_parser as build_train_parser
from chromahack.training.train_ppo_bridge import run as run_train


def _namespace_from_parser(parser: argparse.ArgumentParser, **overrides):
    namespace = argparse.Namespace()
    for action in parser._actions:
        if action.dest == "help":
            continue
        if action.default is not argparse.SUPPRESS:
            setattr(namespace, action.dest, action.default)
    for key, value in overrides.items():
        setattr(namespace, key, value)
    return namespace


def run(args) -> dict:
    if args.smoke_test:
        ok = run_smoke_test(args.out_dir)
        return {"smoke_test": ok}

    train_dir = os.path.join(args.out_dir, "model")
    eval_dir = os.path.join(args.out_dir, "eval_hidden")
    train_steps = 4_096 if args.quick else args.total_steps
    eval_episodes = 3 if args.quick else args.n_episodes
    n_envs = 1 if args.quick else args.n_envs
    n_steps = 128 if args.quick else args.n_steps
    max_steps = 72 if args.quick else args.max_steps

    train_args = _namespace_from_parser(
        build_train_parser(),
        out_dir=train_dir,
        total_steps=train_steps,
        n_envs=n_envs,
        n_steps=n_steps,
        seed=args.seed,
        max_steps=max_steps,
        no_tensorboard=args.no_tensorboard,
        verbose=args.verbose,
    )
    run_train(train_args)

    eval_args = _namespace_from_parser(
        build_eval_parser(),
        model_dir=train_dir,
        model_name="ppo_best",
        out_dir=eval_dir,
        n_episodes=eval_episodes,
        save_trajectories=True,
        trajectory_dir=os.path.join(eval_dir, "trajectories"),
        seed=args.seed,
        max_steps=max_steps,
    )
    summary = run_eval(eval_args)

    first_trajectory = os.path.join(eval_dir, "trajectories", "episode_000.json")
    print(f"[experiment] replay: python -m chromahack.rendering.pygame_bridge_renderer replay --trajectory {first_trajectory}")
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the bridge reward-hacking experiment.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_tensorboard", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="artifacts/bridge_experiment")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
