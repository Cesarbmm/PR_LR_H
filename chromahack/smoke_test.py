"""End-to-end smoke test for the canonical Frontier pipeline."""

from __future__ import annotations

import argparse
import os

from chromahack.evaluation.eval_frontier_hidden import build_parser as build_frontier_eval_parser
from chromahack.evaluation.eval_frontier_hidden import run as run_frontier_eval
from chromahack.training.train_ppo_frontier import build_parser as build_frontier_train_parser
from chromahack.training.train_ppo_frontier import run as run_frontier_train
from chromahack.utils.paths import resolve_project_path


def _namespace_from_parser(parser, **overrides):
    namespace = argparse.Namespace()
    for action in parser._actions:
        if action.dest == "help":
            continue
        if action.default is not argparse.SUPPRESS:
            setattr(namespace, action.dest, action.default)
    for key, value in overrides.items():
        setattr(namespace, key, value)
    return namespace


def run_smoke_test(base_dir: str = "artifacts/smoke_test", mode: str = "frontier") -> bool:
    if mode != "frontier":
        raise ValueError("Canonical smoke test only supports mode='frontier'")
    base_dir = str(resolve_project_path(base_dir))
    os.makedirs(base_dir, exist_ok=True)
    model_dir = os.path.join(base_dir, "model")
    eval_dir = os.path.join(base_dir, "eval_frontier_hidden")
    trajectory_dir = os.path.join(eval_dir, "trajectories")

    train_args = _namespace_from_parser(
        build_frontier_train_parser(),
        out_dir=model_dir,
        total_steps=256,
        n_envs=1,
        n_steps=64,
        batch_size=32,
        rolling_window=4,
        seed=0,
        max_steps=240,
        no_tensorboard=True,
        verbose=0,
        observation_mode="flat",
        policy_backend="mlp",
        reward_mode="proxy",
    )
    run_frontier_train(train_args)

    eval_args = _namespace_from_parser(
        build_frontier_eval_parser(),
        model_dir=model_dir,
        out_dir=eval_dir,
        trajectory_dir=trajectory_dir,
        n_episodes=1,
        save_trajectories=True,
        seed=0,
        max_steps=240,
        district_mode="all",
    )
    summary = run_frontier_eval(eval_args)
    trajectory_path = os.path.join(trajectory_dir, "episode_000.json")
    return os.path.exists(trajectory_path) and summary["n_episodes"] == 1


def main() -> None:
    ok = run_smoke_test()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
