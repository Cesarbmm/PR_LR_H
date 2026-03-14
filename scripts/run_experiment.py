"""Master runner for ChromaHack experiments."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def run_cmd(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def smoke_test() -> bool:
    print("\n" + "=" * 50)
    print("SMOKE TEST — ChromaHack")
    print("=" * 50)

    cmds = [
        [sys.executable, "-m", "chromahack.data.generate_dataset", "--fragility", "high", "--n_ordered", "3", "--n_disordered", "3", "--n_partial", "2", "--out_dir", "artifacts/smoke_data"],
        [sys.executable, "-m", "chromahack.training.train_proxy_cnn", "--mode", "tiny", "--epochs", "1", "--batch_size", "4", "--dataset_dir", "artifacts/smoke_data", "--out_dir", "artifacts/smoke_proxy"],
        [sys.executable, "-m", "chromahack.training.train_ppo", "--mode", "tiny", "--proxy_path", "artifacts/smoke_proxy/proxy_cnn.pth", "--total_steps", "64", "--n_steps", "32", "--n_envs", "1", "--out_dir", "artifacts/smoke_ppo"],
        [sys.executable, "-m", "chromahack.evaluation.eval_hidden", "--model_dir", "artifacts/smoke_ppo", "--n_episodes", "1"],
    ]
    try:
        for c in cmds:
            run_cmd(c)
        return True
    except Exception as exc:
        print(f"Smoke test failed: {exc}")
        return False


def run_phase_a(args: argparse.Namespace) -> str:
    run_dir = Path(args.out_dir) / "phase_A"
    run_dir.mkdir(parents=True, exist_ok=True)
    steps = 5_000 if args.quick else 200_000
    n_ord = 80 if args.quick else 500
    epochs = 3 if args.quick else 25

    run_cmd([
        sys.executable,
        "-m",
        "chromahack.data.generate_dataset",
        "--fragility",
        "high",
        "--n_ordered",
        str(n_ord),
        "--n_disordered",
        str(n_ord),
        "--visualize",
        "--out_dir",
        str(run_dir / "data/synthetic"),
    ])

    run_cmd([
        sys.executable,
        "-m",
        "chromahack.training.train_proxy_cnn",
        "--mode",
        args.mode,
        "--epochs",
        str(epochs),
        "--dataset_dir",
        str(run_dir / "data/synthetic"),
        "--out_dir",
        str(run_dir / "proxy_cnn"),
    ])

    run_cmd([
        sys.executable,
        "-m",
        "chromahack.training.train_ppo",
        "--mode",
        args.mode,
        "--proxy_path",
        str(run_dir / "proxy_cnn/proxy_cnn.pth"),
        "--total_steps",
        str(steps),
        "--n_steps",
        "64" if args.quick else "256",
        "--n_envs",
        str(args.n_envs),
        "--out_dir",
        str(run_dir / "ppo_hacked"),
        "--seed",
        str(args.seed),
    ])

    run_cmd([
        sys.executable,
        "-m",
        "chromahack.evaluation.eval_hidden",
        "--model_dir",
        str(run_dir / "ppo_hacked"),
        "--n_episodes",
        "10" if args.quick else "50",
        "--seed",
        str(args.seed),
    ])

    return str(run_dir)


def run_phase_b(args: argparse.Namespace, hacked_dir: str) -> str:
    run_dir = Path(args.out_dir) / "phase_B"
    run_dir.mkdir(parents=True, exist_ok=True)
    steps = 5_000 if args.quick else 200_000
    n_ep = 8 if args.quick else 200
    n_pairs = 60 if args.quick else 2000
    pref_epochs = 3 if args.quick else 30

    run_cmd([sys.executable, "-m", "chromahack.intervention.preference_reward_model", "collect", "--agent_path", f"{hacked_dir}/ppo_hacked/ppo_final.zip", "--traj_dir", f"{run_dir}/trajectories", "--n_episodes", str(n_ep), "--seed", str(args.seed)])
    run_cmd([sys.executable, "-m", "chromahack.intervention.preference_reward_model", "label", "--traj_dir", f"{run_dir}/trajectories", "--n_pairs", str(n_pairs), "--seed", str(args.seed)])
    run_cmd([sys.executable, "-m", "chromahack.intervention.preference_reward_model", "train", "--traj_dir", f"{run_dir}/trajectories", "--out_dir", f"{run_dir}/pref_model", "--epochs", str(pref_epochs), "--seed", str(args.seed)])
    run_cmd([sys.executable, "-m", "chromahack.intervention.preference_reward_model", "retrain", "--pref_model_path", f"{run_dir}/pref_model/pref_reward.pth", "--out_dir", f"{run_dir}/ppo_aligned", "--total_steps", str(steps), "--seed", str(args.seed)])
    run_cmd([sys.executable, "-m", "chromahack.evaluation.eval_hidden", "--model_dir", f"{run_dir}/ppo_aligned", "--n_episodes", "10" if args.quick else "50", "--seed", str(args.seed)])

    hacked_summary = f"{hacked_dir}/ppo_hacked/eval_results/summary.json"
    aligned_summary = f"{run_dir}/ppo_aligned/eval_results/summary.json"
    if os.path.exists(hacked_summary) and os.path.exists(aligned_summary):
        run_cmd([sys.executable, "-m", "chromahack.intervention.preference_reward_model", "compare", "--hacked_summary", hacked_summary, "--aligned_summary", aligned_summary, "--out_dir", f"{run_dir}/figures"])
    return str(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="ChromaHack — Experimento completo")
    parser.add_argument("--phase", type=str, default="A", choices=["A", "B", "AB"])
    parser.add_argument("--mode", type=str, default="tiny", choices=["tiny", "resnet"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--n_envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="runs/full_experiment")
    args = parser.parse_args()

    if args.smoke_test:
        sys.exit(0 if smoke_test() else 1)

    t0 = time.time()
    hacked_dir = args.out_dir
    if args.phase in ("A", "AB"):
        hacked_dir = run_phase_a(args)
    if args.phase in ("B", "AB"):
        run_phase_b(args, hacked_dir)
    print(f"Done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
