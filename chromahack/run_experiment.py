"""Canonical Frontier experiment orchestrator."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from chromahack.envs.territory_generator import FRONTIER_DISTRIBUTION_SPLITS, FRONTIER_WORLD_SPLITS
from chromahack.evaluation.frontier_scripted import SCRIPTED_FRONTIER_POLICIES
from chromahack.evaluation.eval_frontier_hidden import build_parser as build_frontier_eval_parser
from chromahack.evaluation.eval_frontier_hidden import run as run_frontier_eval
from chromahack.intervention.pref_model import export_preference_dataset, train_preference_model
from chromahack.rendering.story_export import export_story_package
from chromahack.smoke_test import run_smoke_test
from chromahack.training.train_ppo_frontier import build_parser as build_frontier_train_parser
from chromahack.training.train_ppo_frontier import run as run_frontier_train
from chromahack.utils.paths import resolve_input_path, resolve_project_path

DEFAULT_FRONTIER_OUT_DIR = "artifacts/frontier"
MASTER_DEMO_OUT_DIR = "artifacts/demos/frontier_master_demo_long"
MASTER_DEMO_DISTRICT_ID = 5
MASTER_DEMO_MAX_STEPS = 1_600
MASTER_DEMO_QUICK_MAX_STEPS = 560
MASTER_DEMO_PRETRAIN_EPISODES = 64
MASTER_DEMO_QUICK_PRETRAIN_EPISODES = 8
MASTER_DEMO_PRETRAIN_EPOCHS = 6
MASTER_DEMO_QUICK_TRAIN_STEPS = 8_192
MASTER_DEMO_QUICK_EVAL_EPISODES = 3
PATROL_MASTER_OUT_DIR = "artifacts/demos/frontier_patrol_v4_corrupted"
SECURITY_MASTER_OUT_DIR = "artifacts/demos/frontier_security_v6_story"
LOGISTICS_MASTER_OUT_DIR = "artifacts/demos/frontier_logistics_v1_story"


def _namespace_from_parser(parser: argparse.ArgumentParser, **overrides: Any) -> argparse.Namespace:
    namespace = argparse.Namespace()
    for action in parser._actions:
        if action.dest == "help":
            continue
        if action.default is not argparse.SUPPRESS:
            setattr(namespace, action.dest, action.default)
    for key, value in overrides.items():
        setattr(namespace, key, value)
    return namespace


def _resolve_frontier_run_profile(args) -> dict[str, Any]:
    base_dir = args.out_dir
    train_steps = 2_048 if args.quick else args.total_steps
    eval_episodes = 3 if args.quick else args.n_episodes
    n_envs = 1 if args.quick else args.n_envs
    n_steps = 128 if args.quick else args.n_steps
    max_steps = 360 if args.quick else args.max_steps
    district_id = args.district_id
    pretrain_teacher = args.pretrain_teacher
    pretrain_episodes = args.pretrain_episodes
    pretrain_epochs = args.pretrain_epochs
    pretrain_district_id = args.pretrain_district_id

    if args.master_demo:
        if base_dir == DEFAULT_FRONTIER_OUT_DIR:
            if args.world_suite == "patrol_v4":
                base_dir = PATROL_MASTER_OUT_DIR
            elif args.world_suite == "security_v6":
                base_dir = SECURITY_MASTER_OUT_DIR
            elif args.world_suite == "logistics_v1":
                base_dir = LOGISTICS_MASTER_OUT_DIR
            else:
                base_dir = MASTER_DEMO_OUT_DIR
        train_steps = MASTER_DEMO_QUICK_TRAIN_STEPS if args.quick else max(train_steps, 80_000)
        eval_episodes = MASTER_DEMO_QUICK_EVAL_EPISODES if args.quick else max(eval_episodes, 12)
        max_steps = max(max_steps, MASTER_DEMO_QUICK_MAX_STEPS if args.quick else MASTER_DEMO_MAX_STEPS)
        if args.world_suite not in {"patrol_v4", "logistics_v1"}:
            district_id = district_id if district_id is not None else MASTER_DEMO_DISTRICT_ID
        if pretrain_teacher == "none":
            pretrain_teacher = "containment"
        if pretrain_episodes <= 0:
            pretrain_episodes = MASTER_DEMO_QUICK_PRETRAIN_EPISODES if args.quick else MASTER_DEMO_PRETRAIN_EPISODES
        pretrain_epochs = max(pretrain_epochs, 4 if args.quick else MASTER_DEMO_PRETRAIN_EPOCHS)
        if pretrain_district_id is None and district_id is not None:
            pretrain_district_id = district_id

    return {
        "base_dir": str(resolve_project_path(base_dir)),
        "train_steps": train_steps,
        "eval_episodes": eval_episodes,
        "n_envs": n_envs,
        "n_steps": n_steps,
        "max_steps": max_steps,
        "district_id": district_id,
        "pretrain_teacher": pretrain_teacher,
        "pretrain_episodes": pretrain_episodes,
        "pretrain_epochs": pretrain_epochs,
        "pretrain_batch_size": args.pretrain_batch_size,
        "pretrain_district_id": pretrain_district_id,
    }


def run_frontier_baseline(args) -> dict[str, Any]:
    profile = _resolve_frontier_run_profile(args)
    base_dir = profile["base_dir"]
    model_dir = os.path.join(base_dir, "model")
    eval_dir = os.path.join(base_dir, "eval_frontier_hidden")

    train_args = _namespace_from_parser(
        build_frontier_train_parser(),
        out_dir=model_dir,
        total_steps=profile["train_steps"],
        n_envs=profile["n_envs"],
        n_steps=profile["n_steps"],
        seed=args.seed,
        max_steps=profile["max_steps"],
        no_tensorboard=args.no_tensorboard,
        verbose=args.verbose,
        district_id=profile["district_id"],
        pretrain_teacher=profile["pretrain_teacher"],
        pretrain_episodes=profile["pretrain_episodes"],
        pretrain_epochs=profile["pretrain_epochs"],
        pretrain_batch_size=profile["pretrain_batch_size"],
        pretrain_district_id=profile["pretrain_district_id"],
        master_demo=args.master_demo,
        policy_backend=args.policy_backend,
        reward_mode=args.reward_mode,
        reward_model_path=args.reward_model_path,
        reward_clip_length=args.reward_clip_length,
        observation_mode=args.observation_mode,
        train_distribution=args.train_distribution,
        world_suite=args.world_suite,
        train_world_split=args.train_world_split,
        proxy_profile=args.proxy_profile,
        training_phase=args.training_phase,
        district_ids=args.district_ids,
        disable_pyg=args.disable_pyg,
        init_model_path=args.init_model_path,
    )
    run_frontier_train(train_args)

    eval_args = _namespace_from_parser(
        build_frontier_eval_parser(),
        model_dir=model_dir,
        model_name="ppo_best",
        out_dir=eval_dir,
        n_episodes=profile["eval_episodes"],
        save_trajectories=True,
        trajectory_dir=os.path.join(eval_dir, "trajectories"),
        seed=args.seed,
        max_steps=profile["max_steps"],
        district_mode=args.district_mode,
        district_id=profile["district_id"],
        district_ids=args.district_ids,
        distribution_split=args.eval_distribution,
        eval_splits=args.eval_splits,
        world_suite=args.world_suite,
        world_split=args.world_split,
        eval_world_splits=args.eval_world_splits,
        robustness_suite=False,
        reward_mode=args.reward_mode if args.reward_mode != "proxy" else None,
        reward_model_path=args.reward_model_path,
        reward_clip_length=args.reward_clip_length,
        proxy_profile=args.proxy_profile,
        training_phase=args.training_phase,
    )
    summary = run_frontier_eval(eval_args)
    recommended_trajectory = summary.get("master_demo_trajectory") or os.path.join(eval_dir, "trajectories", "episode_000.json")
    print(
        "[experiment-frontier] replay: "
        f"python -m chromahack.rendering.frontier_dual_renderer replay --trajectory {recommended_trajectory}"
    )
    print(
        "[experiment-frontier] showcase: "
        "python -m chromahack.rendering.frontier_dual_renderer showcase "
        f"--trajectory {recommended_trajectory} --slow_mo 0.8 --pause_frames 24 --alert_linger_frames 120"
    )
    print(
        "[experiment-frontier] policy: "
        "python -m chromahack.rendering.frontier_dual_renderer policy "
        f"--model_dir {model_dir} --model_name ppo_best --district_id "
        f"{profile['district_id'] or (31 if args.world_suite == 'logistics_v1' else 19 if args.world_suite == 'security_v6' else 11 if args.world_suite == 'patrol_v4' else MASTER_DEMO_DISTRICT_ID)} "
        f"--world_suite {args.world_suite} --world_split {args.world_split}"
    )
    if args.world_suite == "broadcast_v3":
        print(
            "[experiment-frontier] tv: "
            "python -m chromahack.rendering.frontier_dual_renderer tv "
            f"--model_dir {model_dir} --model_name ppo_best --world_suite {args.world_suite} "
            "--world_split broadcast --district_ids 6 7 8 9 10"
        )
    elif args.world_suite == "patrol_v4":
        print(
            "[experiment-frontier] tv: "
            "python -m chromahack.rendering.frontier_dual_renderer tv "
            f"--model_dir {model_dir} --model_name ppo_best --world_suite {args.world_suite} "
            "--world_split broadcast --district_ids 11 12 13 14 15 16 17 18"
        )
    elif args.world_suite == "security_v6":
        print(
            "[experiment-frontier] tv: "
            "python -m chromahack.rendering.frontier_dual_renderer tv "
            f"--model_dir {model_dir} --model_name ppo_best --world_suite {args.world_suite} "
            "--world_split broadcast --district_ids 19 20 21 22 23 24 25 26"
        )
    elif args.world_suite == "logistics_v1":
        print(
            "[experiment-frontier] tv: "
            "python -m chromahack.rendering.frontier_dual_renderer tv "
            f"--model_dir {model_dir} --model_name ppo_best --world_suite {args.world_suite} "
            "--world_split broadcast --district_ids 31 32 33 34 35 36 37 38"
        )
    return summary


def run_frontier_robustness(args) -> dict[str, Any]:
    profile = _resolve_frontier_run_profile(args)
    base_dir = profile["base_dir"]
    model_dir = os.path.join(base_dir, "model")
    robustness_dir = os.path.join(base_dir, "robustness_eval")

    train_args = _namespace_from_parser(
        build_frontier_train_parser(),
        out_dir=model_dir,
        total_steps=profile["train_steps"],
        n_envs=profile["n_envs"],
        n_steps=profile["n_steps"],
        seed=args.seed,
        max_steps=profile["max_steps"],
        no_tensorboard=args.no_tensorboard,
        verbose=args.verbose,
        district_id=profile["district_id"],
        pretrain_teacher=profile["pretrain_teacher"],
        pretrain_episodes=profile["pretrain_episodes"],
        pretrain_epochs=profile["pretrain_epochs"],
        pretrain_batch_size=profile["pretrain_batch_size"],
        pretrain_district_id=profile["pretrain_district_id"],
        master_demo=args.master_demo,
        policy_backend=args.policy_backend,
        reward_mode=args.reward_mode,
        reward_model_path=args.reward_model_path,
        reward_clip_length=args.reward_clip_length,
        observation_mode=args.observation_mode,
        train_distribution=args.train_distribution,
        world_suite=args.world_suite,
        train_world_split=args.train_world_split,
        proxy_profile=args.proxy_profile,
        district_ids=args.district_ids,
        disable_pyg=args.disable_pyg,
        init_model_path=args.init_model_path,
    )
    run_frontier_train(train_args)

    eval_args = _namespace_from_parser(
        build_frontier_eval_parser(),
        model_dir=model_dir,
        model_name="ppo_best",
        out_dir=robustness_dir,
        n_episodes=profile["eval_episodes"],
        save_trajectories=True,
        trajectory_dir=os.path.join(robustness_dir, "trajectories"),
        seed=args.seed,
        max_steps=profile["max_steps"],
        district_mode=args.district_mode,
        district_id=profile["district_id"],
        district_ids=args.district_ids,
        distribution_split=args.eval_distribution,
        eval_splits=args.eval_splits,
        world_suite=args.world_suite,
        world_split=args.world_split,
        eval_world_splits=args.eval_world_splits,
        robustness_suite=True,
        reward_mode=args.reward_mode if args.reward_mode != "proxy" else None,
        reward_model_path=args.reward_model_path,
        reward_clip_length=args.reward_clip_length,
        proxy_profile=args.proxy_profile,
    )
    summary = run_frontier_eval(eval_args)
    print(
        "[experiment-frontier] robustness comparison: "
        f"{os.path.join(robustness_dir, 'robustness_comparison.csv')}"
    )
    return summary


def run_frontier_broadcast(args) -> dict[str, Any]:
    broadcast_args = argparse.Namespace(**vars(args))
    if broadcast_args.world_suite == "frontier_v2":
        broadcast_args.world_suite = "broadcast_v3"
    if broadcast_args.eval_world_splits is None:
        broadcast_args.eval_world_splits = ["train", "holdout", "broadcast"]
    if broadcast_args.train_world_split is None:
        broadcast_args.train_world_split = "train"
    return run_frontier_baseline(broadcast_args)


def run_frontier_patrol(args) -> dict[str, Any]:
    patrol_args = argparse.Namespace(**vars(args))
    patrol_args.world_suite = "patrol_v4"
    if patrol_args.eval_world_splits is None:
        patrol_args.eval_world_splits = ["train", "holdout", "broadcast"]
    if patrol_args.train_world_split is None:
        patrol_args.train_world_split = "train"
    if getattr(patrol_args, "proxy_profile", None) is None:
        patrol_args.proxy_profile = "corrupted"
    return run_frontier_baseline(patrol_args)


def run_frontier_security_story(args) -> dict[str, Any]:
    story_args = argparse.Namespace(**vars(args))
    story_args.world_suite = "security_v6"
    if story_args.eval_world_splits is None:
        story_args.eval_world_splits = ["train", "holdout", "broadcast"]
    if story_args.train_world_split is None:
        story_args.train_world_split = "train"

    base_root = str(resolve_project_path(story_args.out_dir if story_args.out_dir != DEFAULT_FRONTIER_OUT_DIR else SECURITY_MASTER_OUT_DIR))

    anchor_args = argparse.Namespace(**vars(story_args))
    anchor_args.proxy_profile = "patched"
    anchor_args.training_phase = "anchor"
    if anchor_args.pretrain_teacher == "none":
        anchor_args.pretrain_teacher = "security"
    anchor_args.out_dir = os.path.join(base_root, "anchor")
    anchor_summary = run_frontier_baseline(anchor_args)

    anchor_model = os.path.join(anchor_args.out_dir, "model", "ppo_best.zip")
    drift_args = argparse.Namespace(**vars(story_args))
    drift_args.proxy_profile = "corrupted"
    drift_args.training_phase = "drift"
    drift_args.pretrain_teacher = "none"
    drift_args.init_model_path = anchor_model
    drift_args.out_dir = os.path.join(base_root, "drift")
    drift_summary = run_frontier_baseline(drift_args)
    return {
        "world_suite": "security_v6",
        "story_profile": getattr(args, "story_profile", "single_life"),
        "anchor_model": anchor_model,
        "anchor_summary": anchor_summary,
        "drift_summary": drift_summary,
        "demo_root": base_root,
    }


def run_frontier_logistics_story(args) -> dict[str, Any]:
    story_args = argparse.Namespace(**vars(args))
    story_args.world_suite = "logistics_v1"
    if story_args.eval_world_splits is None:
        story_args.eval_world_splits = ["train", "holdout", "broadcast"]
    if story_args.train_world_split is None:
        story_args.train_world_split = "train"

    base_root = str(
        resolve_project_path(
            story_args.out_dir if story_args.out_dir != DEFAULT_FRONTIER_OUT_DIR else LOGISTICS_MASTER_OUT_DIR
        )
    )

    anchor_args = argparse.Namespace(**vars(story_args))
    anchor_args.proxy_profile = "patched"
    anchor_args.training_phase = "anchor"
    if anchor_args.pretrain_teacher == "none":
        anchor_args.pretrain_teacher = "logistics"
    anchor_args.out_dir = os.path.join(base_root, "anchor")
    anchor_summary = run_frontier_baseline(anchor_args)

    anchor_model = os.path.join(anchor_args.out_dir, "model", "ppo_best.zip")
    drift_args = argparse.Namespace(**vars(story_args))
    drift_args.proxy_profile = "corrupted"
    drift_args.training_phase = "drift"
    drift_args.pretrain_teacher = "none"
    drift_args.init_model_path = anchor_model
    drift_args.out_dir = os.path.join(base_root, "drift")
    drift_summary = run_frontier_baseline(drift_args)
    story_out_dir = os.path.join(base_root, "story_package")
    story_export = export_story_package(
        demo_dir=os.path.join(base_root, "drift", "eval_frontier_hidden"),
        reference_demo_dir=os.path.join(base_root, "anchor", "eval_frontier_hidden"),
        out_dir=story_out_dir,
        story_profile=getattr(args, "story_profile", "single_shift_life"),
        godot_project_dir="godot_broadcast",
    )
    return {
        "world_suite": "logistics_v1",
        "story_profile": getattr(args, "story_profile", "single_shift_life"),
        "anchor_model": anchor_model,
        "anchor_summary": anchor_summary,
        "drift_summary": drift_summary,
        "demo_root": base_root,
        "story_package": story_export.to_dict(),
    }


def run_frontier_rlhf(args) -> dict[str, Any]:
    base_root = str(resolve_project_path(args.out_dir))
    baseline_root = os.path.join(base_root, "baseline")
    if args.preference_source_dir:
        trajectory_dir = str(resolve_input_path(args.preference_source_dir))
        baseline_summary: dict[str, Any] | None = None
    else:
        baseline_args = argparse.Namespace(**vars(args))
        baseline_args.out_dir = baseline_root
        baseline_args.reward_mode = "proxy"
        baseline_args.reward_model_path = None
        if args.quick:
            baseline_args.total_steps = min(int(args.total_steps), 1_024)
        baseline_summary = run_frontier_baseline(baseline_args)
        trajectory_dir = os.path.join(baseline_root, "eval_frontier_hidden", "trajectories")

    preference_root = os.path.join(base_root, "preferences")
    export_summary = export_preference_dataset(
        argparse.Namespace(
            trajectory_dir=trajectory_dir,
            out_dir=preference_root,
            clip_length=args.preference_clip_length,
            stride=args.preference_stride,
            max_pairs=args.preference_max_pairs,
            seed=args.seed,
        )
    )
    train_summary = train_preference_model(
        argparse.Namespace(
            dataset=export_summary["dataset_path"],
            out_dir=os.path.join(preference_root, "model"),
            hidden_dim=args.preference_hidden_dim,
            batch_size=args.preference_batch_size,
            epochs=args.preference_epochs,
            lr=args.preference_lr,
            seed=args.seed,
        )
    )

    pref_args = argparse.Namespace(**vars(args))
    pref_args.out_dir = os.path.join(base_root, "pref_model_run")
    pref_args.reward_mode = "pref_model"
    pref_args.reward_model_path = train_summary["model_path"]
    if args.quick:
        pref_args.total_steps = min(int(args.total_steps), 1_024)
    pref_summary = run_frontier_baseline(pref_args)
    return {
        "baseline_summary": baseline_summary,
        "preference_export": export_summary,
        "preference_training": train_summary,
        "pref_model_summary": pref_summary,
    }


def run(args) -> dict[str, Any]:
    if args.mode != "frontier":
        raise SystemExit("Only --mode frontier is supported in the canonical orchestrator.")
    if args.smoke_test:
        ok = run_smoke_test(str(resolve_project_path(args.out_dir)), mode="frontier")
        return {"smoke_test": ok}
    if args.phase == "baseline":
        summary = run_frontier_baseline(args)
    elif args.phase == "broadcast":
        summary = run_frontier_broadcast(args)
    elif args.phase == "patrol":
        summary = run_frontier_patrol(args)
    elif args.phase == "robustness":
        summary = run_frontier_robustness(args)
    elif args.phase == "security_story":
        summary = run_frontier_security_story(args)
    elif args.phase == "logistics_story":
        summary = run_frontier_logistics_story(args)
    else:
        summary = run_frontier_rlhf(args)
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run canonical Frontier Territory experiments.")
    parser.add_argument("--mode", choices=["frontier"], default="frontier")
    parser.add_argument("--phase", choices=["baseline", "broadcast", "patrol", "security_story", "logistics_story", "robustness", "rlhf"], default="baseline")
    parser.add_argument("--master_demo", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=1_200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--district_mode", choices=["all", "curriculum"], default="all")
    parser.add_argument("--district_id", type=int, default=None)
    parser.add_argument("--district_ids", nargs="*", type=int, default=None)
    parser.add_argument("--pretrain_teacher", choices=("none",) + SCRIPTED_FRONTIER_POLICIES, default="none")
    parser.add_argument("--pretrain_episodes", type=int, default=0)
    parser.add_argument("--pretrain_epochs", type=int, default=4)
    parser.add_argument("--pretrain_batch_size", type=int, default=512)
    parser.add_argument("--pretrain_district_id", type=int, default=None)
    parser.add_argument("--policy_backend", choices=["mlp", "gnn"], default="mlp")
    parser.add_argument("--init_model_path", type=str, default=None)
    parser.add_argument("--train_distribution", choices=FRONTIER_DISTRIBUTION_SPLITS, default="train")
    parser.add_argument("--eval_distribution", choices=FRONTIER_DISTRIBUTION_SPLITS, default=None)
    parser.add_argument("--eval_splits", nargs="*", choices=FRONTIER_DISTRIBUTION_SPLITS, default=None)
    parser.add_argument("--world_suite", choices=["frontier_v2", "broadcast_v3", "patrol_v4", "security_v6", "logistics_v1"], default="frontier_v2")
    parser.add_argument("--world_split", choices=FRONTIER_WORLD_SPLITS, default="train")
    parser.add_argument("--train_world_split", choices=FRONTIER_WORLD_SPLITS, default="train")
    parser.add_argument("--eval_world_splits", nargs="*", choices=FRONTIER_WORLD_SPLITS, default=None)
    parser.add_argument("--reward_mode", choices=["proxy", "pref_model"], default="proxy")
    parser.add_argument("--proxy_profile", choices=["corrupted", "patched"], default="corrupted")
    parser.add_argument("--training_phase", choices=["anchor", "drift"], default="anchor")
    parser.add_argument("--story_profile", choices=["single_life", "single_shift_life"], default="single_shift_life")
    parser.add_argument("--reward_model_path", type=str, default=None)
    parser.add_argument("--reward_clip_length", type=int, default=48)
    parser.add_argument("--observation_mode", choices=["flat", "dict"], default="flat")
    parser.add_argument("--disable_pyg", action="store_true")
    parser.add_argument("--preference_source_dir", type=str, default=None)
    parser.add_argument("--preference_clip_length", type=int, default=48)
    parser.add_argument("--preference_stride", type=int, default=24)
    parser.add_argument("--preference_max_pairs", type=int, default=256)
    parser.add_argument("--preference_hidden_dim", type=int, default=128)
    parser.add_argument("--preference_batch_size", type=int, default=32)
    parser.add_argument("--preference_epochs", type=int, default=5)
    parser.add_argument("--preference_lr", type=float, default=1e-3)
    parser.add_argument("--no_tensorboard", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_FRONTIER_OUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
