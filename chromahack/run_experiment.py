"""Orchestrate the full ChromaHack experiment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from chromahack.smoke_test import run_smoke_test


def run_module(module: str, *args: str) -> None:
    command = [sys.executable, "-m", module, *args]
    print("[Run]", " ".join(command))
    subprocess.run(command, check=True)


def _quick_config() -> dict:
    return {
        "phase_a_steps": 4096,
        "phase_a_ordered": 64,
        "phase_a_disordered": 64,
        "phase_a_partial": 16,
        "phase_a_adversarial": 16,
        "phase_a_epochs": 3,
        "phase_a_n_steps": 128,
        "phase_a_eval_episodes": 5,
        "phase_b_collect_episodes": 10,
        "phase_b_segment_len": 12,
        "phase_b_pairs": 64,
        "phase_b_epochs": 3,
        "phase_b_batch_size": 8,
        "phase_b_steps": 4096,
        "phase_b_n_steps": 128,
        "phase_b_eval_episodes": 5,
    }


def _resolve_hacked_root(args) -> str:
    if args.phase in ("A", "AB"):
        return args.out_dir
    return args.hacked_dir or os.path.join(args.out_dir, "phase_A")


def _validate_hacked_root(hacked_root: str) -> tuple[str, str, str]:
    hacked_agent = os.path.join(hacked_root, "ppo_hacked", "ppo_final.zip")
    hacked_proxy = os.path.join(hacked_root, "proxy_cnn", "proxy_cnn.pth")
    hacked_summary = os.path.join(hacked_root, "ppo_hacked", "eval_results", "summary.json")
    missing = [path for path in (hacked_agent, hacked_proxy) if not os.path.exists(path)]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(
            "Missing hacked phase-A artifacts. Expected files:\n"
            f"{joined}\n"
            "Run phase A first or provide --hacked_dir pointing to an existing phase_A directory."
        )
    return hacked_agent, hacked_proxy, hacked_summary


def run_phase_a(args) -> str:
    run_dir = os.path.join(args.out_dir, "phase_A")
    os.makedirs(run_dir, exist_ok=True)

    quick = _quick_config()
    steps = quick["phase_a_steps"] if args.quick else 200_000
    ordered = quick["phase_a_ordered"] if args.quick else 500
    disordered = quick["phase_a_disordered"] if args.quick else 500
    partial = quick["phase_a_partial"] if args.quick else 300
    adversarial = quick["phase_a_adversarial"] if args.quick else None
    epochs = quick["phase_a_epochs"] if args.quick else 25
    n_steps = quick["phase_a_n_steps"] if args.quick else 512
    eval_episodes = quick["phase_a_eval_episodes"] if args.quick else 50

    dataset_dir = os.path.join(run_dir, "data", "synthetic")
    proxy_dir = os.path.join(run_dir, "proxy_cnn")
    hacked_dir = os.path.join(run_dir, "ppo_hacked")

    run_module(
        "chromahack.data.generate_dataset",
        "--fragility",
        "high",
        "--n_ordered",
        str(ordered),
        "--n_disordered",
        str(disordered),
        "--n_partial",
        str(partial),
        "--n_adversarial",
        str(adversarial if adversarial is not None else 200),
        "--visualize",
        "--out_dir",
        dataset_dir,
    )
    run_module(
        "chromahack.training.train_proxy_cnn",
        "--mode",
        args.mode,
        "--epochs",
        str(epochs),
        "--dataset_dir",
        dataset_dir,
        "--out_dir",
        proxy_dir,
        "--seed",
        str(args.seed),
    )
    run_module(
        "chromahack.training.train_ppo",
        "--mode",
        args.mode,
        "--proxy_path",
        os.path.join(proxy_dir, "proxy_cnn.pth"),
        "--total_steps",
        str(steps),
        "--n_envs",
        str(args.n_envs),
        "--n_steps",
        str(n_steps),
        "--out_dir",
        hacked_dir,
        "--seed",
        str(args.seed),
    )
    run_module(
        "chromahack.evaluation.eval_hidden",
        "--model_dir",
        hacked_dir,
        "--n_episodes",
        str(eval_episodes),
        "--seed",
        str(args.seed),
    )
    return run_dir


def run_phase_b(args, hacked_root: str) -> str:
    run_dir = os.path.join(args.out_dir, "phase_B")
    os.makedirs(run_dir, exist_ok=True)

    hacked_agent, hacked_proxy, hacked_summary = _validate_hacked_root(hacked_root)
    aligned_dir = os.path.join(run_dir, "ppo_aligned")
    trajectories_dir = os.path.join(run_dir, "trajectories")
    pref_dir = os.path.join(run_dir, "pref_model")

    quick = _quick_config()
    steps = quick["phase_b_steps"] if args.quick else 200_000
    n_episodes = quick["phase_b_collect_episodes"] if args.quick else 200
    segment_len = quick["phase_b_segment_len"] if args.quick else 25
    n_pairs = quick["phase_b_pairs"] if args.quick else 2000
    epochs = quick["phase_b_epochs"] if args.quick else 30
    batch_size = quick["phase_b_batch_size"] if args.quick else 16
    n_steps = quick["phase_b_n_steps"] if args.quick else 512
    eval_episodes = quick["phase_b_eval_episodes"] if args.quick else 50

    if not os.path.exists(hacked_summary):
        run_module(
            "chromahack.evaluation.eval_hidden",
            "--model_dir",
            os.path.join(hacked_root, "ppo_hacked"),
            "--proxy_path",
            hacked_proxy,
            "--n_episodes",
            str(eval_episodes),
            "--seed",
            str(args.seed),
        )

    run_module(
        "chromahack.intervention.preference_reward_model",
        "collect",
        "--agent_path",
        hacked_agent,
        "--traj_dir",
        trajectories_dir,
        "--n_episodes",
        str(n_episodes),
        "--segment_len",
        str(segment_len),
        "--seed",
        str(args.seed),
    )
    run_module(
        "chromahack.intervention.preference_reward_model",
        "label",
        "--traj_dir",
        trajectories_dir,
        "--n_pairs",
        str(n_pairs),
        "--seed",
        str(args.seed),
    )
    run_module(
        "chromahack.intervention.preference_reward_model",
        "train",
        "--traj_dir",
        trajectories_dir,
        "--out_dir",
        pref_dir,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--seed",
        str(args.seed),
    )
    run_module(
        "chromahack.intervention.preference_reward_model",
        "retrain",
        "--pref_model_path",
        os.path.join(pref_dir, "pref_reward.pth"),
        "--out_dir",
        aligned_dir,
        "--total_steps",
        str(steps),
        "--n_steps",
        str(n_steps),
        "--seed",
        str(args.seed),
        "--n_envs",
        str(args.n_envs),
    )
    run_module(
        "chromahack.evaluation.eval_hidden",
        "--model_dir",
        aligned_dir,
        "--model_name",
        "ppo_aligned_final",
        "--proxy_path",
        hacked_proxy,
        "--n_episodes",
        str(eval_episodes),
        "--seed",
        str(args.seed),
    )

    aligned_summary = os.path.join(aligned_dir, "eval_results", "summary.json")
    if os.path.exists(hacked_summary) and os.path.exists(aligned_summary):
        run_module(
            "chromahack.intervention.preference_reward_model",
            "compare",
            "--hacked_summary",
            hacked_summary,
            "--aligned_summary",
            aligned_summary,
            "--out_dir",
            os.path.join(run_dir, "figures"),
        )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full ChromaHack experiment.")
    parser.add_argument("--phase", type=str, default="A", choices=["A", "B", "AB"])
    parser.add_argument("--mode", type=str, default="tiny", choices=["tiny", "resnet"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hacked_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="runs/full_experiment")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    if args.smoke_test:
        ok = run_smoke_test()
        raise SystemExit(0 if ok else 1)

    hacked_root = _resolve_hacked_root(args)
    if args.phase in ("A", "AB"):
        hacked_root = run_phase_a(args)
    if args.phase in ("B", "AB"):
        run_phase_b(args, hacked_root)


if __name__ == "__main__":
    main()
