"""Archived bridge environment namespace kept for compatibility."""

from chromahack.envs.bridge_env import ACTION_NAMES, BridgeAction, BridgeInspectionHackEnv
from chromahack.training.train_ppo_bridge import build_parser as build_train_parser
from chromahack.training.train_ppo_bridge import run as run_train
from chromahack.evaluation.eval_bridge_hidden import build_parser as build_eval_parser
from chromahack.evaluation.eval_bridge_hidden import run as run_eval

__all__ = [
    "ACTION_NAMES",
    "BridgeAction",
    "BridgeInspectionHackEnv",
    "build_eval_parser",
    "build_train_parser",
    "run_eval",
    "run_train",
]
