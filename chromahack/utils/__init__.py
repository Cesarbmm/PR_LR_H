"""Shared utilities for GhostMerc and archived bridge experiments."""

from .config import (
    BridgeEnvConfig,
    FrontierTerritoryConfig,
    GhostMercConfig,
    add_bridge_env_args,
    add_frontier_env_args,
    add_ghostmerc_env_args,
    bridge_env_config_from_args,
    frontier_config_from_args,
    ghostmerc_config_from_args,
)
try:
    from .metrics import (
        BridgeTrainingCallback,
        GhostMercTrainingCallback,
        aggregate_episode_metrics,
        aggregate_ghostmerc_metrics,
        summarize_episode,
        summarize_ghostmerc_episode,
        write_episode_csv,
    )
except ModuleNotFoundError:  # pragma: no cover - lightweight tooling paths do not need training callbacks
    BridgeTrainingCallback = None
    GhostMercTrainingCallback = None
    aggregate_episode_metrics = None
    aggregate_ghostmerc_metrics = None
    summarize_episode = None
    summarize_ghostmerc_episode = None
    write_episode_csv = None
from .trajectory_io import EpisodeTrajectory, TrajectoryStep, load_episode_trajectory, save_episode_trajectory
from .paths import project_root, resolve_input_path, resolve_project_path

__all__ = [
    "BridgeEnvConfig",
    "BridgeTrainingCallback",
    "EpisodeTrajectory",
    "FrontierTerritoryConfig",
    "GhostMercConfig",
    "GhostMercTrainingCallback",
    "TrajectoryStep",
    "add_bridge_env_args",
    "add_frontier_env_args",
    "add_ghostmerc_env_args",
    "aggregate_episode_metrics",
    "aggregate_ghostmerc_metrics",
    "bridge_env_config_from_args",
    "frontier_config_from_args",
    "ghostmerc_config_from_args",
    "load_episode_trajectory",
    "project_root",
    "resolve_input_path",
    "resolve_project_path",
    "save_episode_trajectory",
    "summarize_episode",
    "summarize_ghostmerc_episode",
    "write_episode_csv",
]
