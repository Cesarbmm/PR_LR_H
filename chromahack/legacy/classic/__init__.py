"""Archived classic GhostMerc namespace kept for compatibility."""

from chromahack.envs.ghostmerc_env import CurriculumProgress, GhostMercActionSpace, GhostMercEnv
from chromahack.training.train_ppo_ghostmerc import build_parser as build_train_parser
from chromahack.training.train_ppo_ghostmerc import run as run_train
from chromahack.evaluation.eval_ghostmerc_hidden import build_parser as build_eval_parser
from chromahack.evaluation.eval_ghostmerc_hidden import run as run_eval

__all__ = [
    "CurriculumProgress",
    "GhostMercActionSpace",
    "GhostMercEnv",
    "build_eval_parser",
    "build_train_parser",
    "run_eval",
    "run_train",
]
