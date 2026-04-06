"""Train PPO on the GhostMerc Frontier Territory environment."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from chromahack.envs.territory_generator import (
    FRONTIER_DISTRIBUTION_SPLITS,
    FRONTIER_WORLD_SPLITS,
    sample_curriculum_district_id,
)
from chromahack.evaluation.frontier_scripted import SCRIPTED_FRONTIER_POLICIES, select_scripted_frontier_action
from chromahack.envs.ghostmerc_frontier_env import FrontierCurriculumProgress, GhostMercFrontierEnv
from chromahack.gnn.feature_extractor import build_frontier_gnn_policy_kwargs
from chromahack.intervention.pref_model import FrontierPreferenceRewardWrapper
from chromahack.utils.config import add_frontier_env_args, frontier_config_from_args
from chromahack.utils.metrics import FrontierTrainingCallback
from chromahack.utils.paths import resolve_input_path, resolve_project_path

MASTER_DEMO_DISTRICT_ID = 5
MASTER_DEMO_MAX_STEPS = 1_600
MASTER_DEMO_PRETRAIN_EPISODES = 64
MASTER_DEMO_PRETRAIN_EPOCHS = 6


ObservationBatch = np.ndarray | dict[str, np.ndarray]


def _clone_observation(observation: Any) -> Any:
    if isinstance(observation, dict):
        return {key: np.asarray(value).copy() for key, value in observation.items()}
    return np.asarray(observation).copy()


def _stack_observations(observations: list[Any]) -> ObservationBatch:
    first = observations[0]
    if isinstance(first, dict):
        return {
            key: np.stack([np.asarray(observation[key], dtype=np.float32) for observation in observations], axis=0)
            for key in first
        }
    return np.stack([np.asarray(observation, dtype=np.float32) for observation in observations], axis=0)


def _observation_batch_size(observations: ObservationBatch) -> int:
    if isinstance(observations, dict):
        first_key = next(iter(observations))
        return int(observations[first_key].shape[0])
    return int(observations.shape[0])


def _slice_observations(observations: ObservationBatch, batch_idx: np.ndarray) -> ObservationBatch:
    if isinstance(observations, dict):
        return {key: value[batch_idx] for key, value in observations.items()}
    return observations[batch_idx]


def _tensorize_observations(observations: ObservationBatch, *, device: str) -> Any:
    if isinstance(observations, dict):
        return {
            key: torch.as_tensor(value, device=device, dtype=torch.float32)
            for key, value in observations.items()
        }
    return torch.as_tensor(observations, device=device, dtype=torch.float32)


def make_frontier_env(
    config,
    *,
    curriculum_progress: FrontierCurriculumProgress,
    rank: int = 0,
    seed: int = 42,
    forced_district_id: int | None = None,
    distribution_split: str = "train",
    world_suite: str = "frontier_v2",
    world_split: str = "train",
    reward_mode: str = "proxy",
    reward_model_path: str | None = None,
    reward_clip_length: int = 48,
    reward_model_device: str | None = None,
):
    """Create a deterministic environment factory for vectorized PPO."""

    def _init():
        env = GhostMercFrontierEnv(
            config=config,
            seed=seed + rank,
            curriculum_progress=curriculum_progress,
            forced_district_id=forced_district_id,
            distribution_split=distribution_split,
            world_suite=world_suite,
            world_split=world_split,
        )
        if reward_mode == "pref_model":
            if not reward_model_path:
                raise ValueError("reward_model_path is required when reward_mode='pref_model'")
            env = FrontierPreferenceRewardWrapper(
                env,
                reward_model_path,
                clip_length=reward_clip_length,
                device=reward_model_device,
            )
        return env

    return _init


def _resolve_batch_size(total_batch: int, requested_batch_size: int) -> int:
    batch_size = min(max(1, requested_batch_size), total_batch)
    while batch_size > 1 and total_batch % batch_size != 0:
        batch_size -= 1
    return max(1, batch_size)


def _apply_master_demo_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Apply long-horizon defaults for curated master-demo training runs."""

    if not args.master_demo:
        return args
    args.max_steps = max(int(args.max_steps), MASTER_DEMO_MAX_STEPS)
    if args.district_id is None and args.world_suite not in {"patrol_v4", "security_v6", "logistics_v1"}:
        args.district_id = MASTER_DEMO_DISTRICT_ID
    if args.pretrain_teacher == "none":
        if args.world_suite == "security_v6":
            args.pretrain_teacher = "security"
        elif args.world_suite == "logistics_v1":
            args.pretrain_teacher = "logistics"
        else:
            args.pretrain_teacher = "patrol" if args.world_suite == "patrol_v4" else "containment"
    if args.pretrain_episodes <= 0:
        args.pretrain_episodes = MASTER_DEMO_PRETRAIN_EPISODES
    args.pretrain_epochs = max(int(args.pretrain_epochs), MASTER_DEMO_PRETRAIN_EPOCHS)
    if args.pretrain_district_id is None and args.district_id is not None:
        args.pretrain_district_id = args.district_id
    return args


def _prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    args = _apply_master_demo_defaults(args)
    if args.policy_backend == "gnn":
        args.observation_mode = "dict"
    if args.world_suite in {"patrol_v4", "security_v6", "logistics_v1"}:
        args.include_incident_observation = True
        args.max_zones = max(int(getattr(args, "max_zones", 5)), 7)
        args.max_incidents = max(int(getattr(args, "max_incidents", 5)), 5)
    return args


def _collect_teacher_dataset(
    *,
    config,
    teacher_policy: str,
    n_episodes: int,
    seed: int,
    forced_district_id: int | None,
    distribution_split: str,
    world_suite: str,
    world_split: str,
    reward_mode: str,
    reward_model_path: str | None,
    reward_clip_length: int,
    reward_model_device: str | None,
) -> tuple[ObservationBatch, np.ndarray]:
    env: Any = GhostMercFrontierEnv(
        config=config,
        seed=seed,
        forced_district_id=forced_district_id,
        distribution_split=distribution_split,
        world_suite=world_suite,
        world_split=world_split,
    )
    if reward_mode == "pref_model":
        if not reward_model_path:
            raise ValueError("reward_model_path is required when reward_mode='pref_model'")
        env = FrontierPreferenceRewardWrapper(
            env,
            reward_model_path,
            clip_length=reward_clip_length,
            device=reward_model_device,
        )
    observations: list[Any] = []
    actions: list[np.ndarray] = []
    for episode_index in range(n_episodes):
        district_id = (
            forced_district_id
            if forced_district_id is not None
            else sample_curriculum_district_id(
                progress=1.0,
                rng=env.np_random,
                distribution_split=distribution_split,
                world_suite=world_suite,
                world_split=world_split,
            )
        )
        observation, _ = env.reset(
            seed=seed + episode_index,
            options={
                "district_id": district_id,
                "distribution_split": distribution_split,
                "world_suite": world_suite,
                "world_split": world_split,
                "training_phase": config.training_phase,
            },
        )
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = select_scripted_frontier_action(env.unwrapped if hasattr(env, "unwrapped") else env, teacher_policy)
            observations.append(_clone_observation(observation))
            actions.append(np.asarray(action, dtype=np.int64))
            observation, _, terminated, truncated, _ = env.step(action)
    env.close()
    if not observations:
        if config.observation_mode == "dict":
            empty_batch: ObservationBatch = {
                "agent": np.zeros((0, config.agent_feature_dim), dtype=np.float32),
                "actors": np.zeros((0, config.max_actors, config.actor_feature_dim), dtype=np.float32),
                "actor_mask": np.zeros((0, config.actor_mask_dim), dtype=np.float32),
                "zones": np.zeros((0, config.max_zones, config.zone_feature_dim), dtype=np.float32),
                "zone_mask": np.zeros((0, config.zone_mask_dim), dtype=np.float32),
                "adjacency": np.zeros((0, *config.adjacency_shape), dtype=np.float32),
                "aggregates": np.zeros((0, config.aggregate_feature_dim), dtype=np.float32),
            }
        else:
            empty_batch = np.zeros((0, config.observation_dim), dtype=np.float32)
        return empty_batch, np.zeros((0, len(config.action_nvec)), dtype=np.int64)
    return _stack_observations(observations), np.asarray(actions, dtype=np.int64)


def _run_teacher_pretraining(
    agent: PPO,
    *,
    observations: ObservationBatch,
    actions: np.ndarray,
    epochs: int,
    batch_size: int,
    verbose: int,
) -> dict[str, float | int]:
    if actions.size == 0 or epochs <= 0:
        return {"dataset_size": int(len(actions)), "epochs": 0, "final_loss": 0.0}

    optimizer = agent.policy.optimizer
    device = str(agent.device)
    indices = np.arange(_observation_batch_size(observations))
    last_loss = 0.0
    agent.policy.set_training_mode(True)
    for epoch in range(epochs):
        np.random.shuffle(indices)
        losses: list[float] = []
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            obs_tensor = _tensorize_observations(_slice_observations(observations, batch_idx), device=device)
            action_tensor = torch.as_tensor(actions[batch_idx], device=device, dtype=torch.int64)
            distribution = agent.policy.get_distribution(obs_tensor)
            log_prob = distribution.log_prob(action_tensor)
            entropy = distribution.entropy()
            loss = -log_prob.mean() - 0.001 * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.item()))
        last_loss = float(np.mean(losses)) if losses else last_loss
        if verbose > 0:
            print(f"[pretrain-frontier] epoch={epoch + 1}/{epochs} loss={last_loss:.4f}")
    agent.policy.set_training_mode(False)
    return {"dataset_size": int(len(actions)), "epochs": int(epochs), "final_loss": float(last_loss)}


def _policy_spec(args: argparse.Namespace, config) -> tuple[str, dict[str, Any]]:
    hidden_layers = [args.policy_hidden_size] * args.policy_layers
    if args.policy_backend == "gnn":
        return "MultiInputPolicy", build_frontier_gnn_policy_kwargs(
            features_dim=args.policy_hidden_size,
            actor_hidden_dim=args.gnn_actor_hidden_dim,
            zone_hidden_dim=args.gnn_zone_hidden_dim,
            use_pyg=not args.disable_pyg,
        )
    if config.observation_mode == "dict":
        return "MultiInputPolicy", {"net_arch": hidden_layers}
    return "MlpPolicy", {"net_arch": hidden_layers}


def run(args) -> str:
    """Execute PPO training and return the final checkpoint prefix."""

    args = _prepare_args(args)
    config = frontier_config_from_args(args)
    out_dir = str(resolve_project_path(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    config.save_json(os.path.join(out_dir, "env_config.json"))

    reward_model_path = None
    if args.reward_mode == "pref_model":
        if not args.reward_model_path:
            raise ValueError("--reward_model_path is required when --reward_mode=pref_model")
        reward_model_path = str(resolve_input_path(args.reward_model_path))
        if not os.path.exists(reward_model_path):
            raise FileNotFoundError(f"Missing reward model: {reward_model_path}")

    curriculum_progress = FrontierCurriculumProgress()
    district_ids = [int(value) for value in (args.district_ids or [])]
    vec_env = DummyVecEnv(
        [
            make_frontier_env(
                config,
                curriculum_progress=curriculum_progress,
                rank=index,
                seed=args.seed,
                forced_district_id=(
                    args.district_id
                    if args.district_id is not None
                    else district_ids[index % len(district_ids)] if district_ids else None
                ),
                distribution_split=args.train_distribution,
                world_suite=args.world_suite,
                world_split=args.train_world_split,
                reward_mode=args.reward_mode,
                reward_model_path=reward_model_path,
                reward_clip_length=args.reward_clip_length,
                reward_model_device=args.device,
            )
            for index in range(args.n_envs)
        ]
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    total_batch = args.n_steps * args.n_envs
    batch_size = _resolve_batch_size(total_batch, args.batch_size)
    tensorboard_log = None if args.no_tensorboard else os.path.join(out_dir, "tensorboard")
    policy_name, policy_kwargs = _policy_spec(args, config)
    if args.init_model_path:
        init_model_path = str(resolve_input_path(args.init_model_path))
        if not os.path.exists(init_model_path):
            raise FileNotFoundError(f"Missing checkpoint: {init_model_path}")
        agent = PPO.load(init_model_path, env=vec_env, device=device)
        agent.verbose = args.verbose
        agent.set_env(vec_env)
        agent.tensorboard_log = tensorboard_log
    else:
        init_model_path = None
        agent = PPO(
            policy=policy_name,
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
            policy_kwargs=policy_kwargs,
        )

    if args.pretrain_teacher != "none" and args.pretrain_episodes > 0:
        teacher_observations, teacher_actions = _collect_teacher_dataset(
            config=config,
            teacher_policy=args.pretrain_teacher,
            n_episodes=args.pretrain_episodes,
            seed=args.seed,
            forced_district_id=args.pretrain_district_id if args.pretrain_district_id is not None else (district_ids[0] if district_ids else None),
            distribution_split=args.train_distribution,
            world_suite=args.world_suite,
            world_split=args.train_world_split,
            reward_mode=args.reward_mode,
            reward_model_path=reward_model_path,
            reward_clip_length=args.reward_clip_length,
            reward_model_device=device,
        )
        pretrain_summary = _run_teacher_pretraining(
            agent,
            observations=teacher_observations,
            actions=teacher_actions,
            epochs=args.pretrain_epochs,
            batch_size=args.pretrain_batch_size,
            verbose=args.verbose,
        )
        pretrain_summary["teacher_policy"] = args.pretrain_teacher
        pretrain_summary["district_id"] = args.pretrain_district_id
        with open(os.path.join(out_dir, "teacher_pretrain.json"), "w", encoding="utf-8") as handle:
            json.dump(pretrain_summary, handle, indent=2)

    callback = FrontierTrainingCallback(
        out_dir=out_dir,
        total_steps=args.total_steps,
        curriculum_progress=curriculum_progress,
        rolling_window=args.rolling_window,
        exploit_threshold=args.exploit_threshold,
        transition_correlation_threshold=args.transition_correlation_threshold,
        verbose=args.verbose,
    )
    agent.learn(total_timesteps=args.total_steps, callback=callback, progress_bar=False)
    final_path = os.path.join(out_dir, "ppo_final")
    agent.save(final_path)
    manifest = {
        "out_dir": out_dir,
        "policy_backend": args.policy_backend,
        "reward_mode": args.reward_mode,
        "reward_model_path": reward_model_path,
        "reward_clip_length": args.reward_clip_length,
        "observation_mode": config.observation_mode,
        "district_id": args.district_id,
        "district_ids": district_ids,
        "train_distribution": args.train_distribution,
        "world_suite": args.world_suite,
        "train_world_split": args.train_world_split,
        "proxy_profile": args.proxy_profile,
        "training_phase": args.training_phase,
        "master_demo": bool(args.master_demo),
        "init_model_path": init_model_path,
    }
    with open(os.path.join(out_dir, "training_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    vec_env.close()
    return final_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO on the GhostMerc Frontier Territory environment.")
    add_frontier_env_args(parser)
    parser.add_argument(
        "--master_demo",
        action="store_true",
        help="Apply long-horizon district-5 containment warm-start defaults for a curated demo policy.",
    )
    parser.add_argument("--policy_backend", choices=["mlp", "gnn"], default="mlp")
    parser.add_argument("--reward_mode", choices=["proxy", "pref_model"], default="proxy")
    parser.add_argument("--reward_model_path", type=str, default=None)
    parser.add_argument("--reward_clip_length", type=int, default=48)
    parser.add_argument("--gnn_actor_hidden_dim", type=int, default=96)
    parser.add_argument("--gnn_zone_hidden_dim", type=int, default=48)
    parser.add_argument("--disable_pyg", action="store_true")
    parser.add_argument("--total_steps", type=int, default=250_000)
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
    parser.add_argument("--init_model_path", type=str, default=None)
    parser.add_argument("--rolling_window", type=int, default=20)
    parser.add_argument("--exploit_threshold", type=float, default=0.35)
    parser.add_argument("--transition_correlation_threshold", type=float, default=0.15)
    parser.add_argument("--district_id", type=int, default=None)
    parser.add_argument("--district_ids", nargs="*", type=int, default=None)
    parser.add_argument("--train_distribution", choices=FRONTIER_DISTRIBUTION_SPLITS, default="train")
    parser.add_argument("--train_world_split", choices=FRONTIER_WORLD_SPLITS, default="train")
    parser.add_argument("--pretrain_teacher", choices=("none",) + SCRIPTED_FRONTIER_POLICIES, default="none")
    parser.add_argument("--pretrain_episodes", type=int, default=0)
    parser.add_argument("--pretrain_epochs", type=int, default=4)
    parser.add_argument("--pretrain_batch_size", type=int, default=512)
    parser.add_argument("--pretrain_district_id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_tensorboard", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="artifacts/models/ghostmerc_frontier_v2")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    final_path = run(args)
    print(f"[train-frontier] wrote {final_path}.zip")


if __name__ == "__main__":
    main()
