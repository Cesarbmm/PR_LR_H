"""Canonical Frontier environment exports."""

from chromahack.utils.config import FrontierTerritoryConfig

try:
    from .ghostmerc_frontier_env import FrontierActionSpace, FrontierCurriculumProgress, GhostMercFrontierEnv
except ModuleNotFoundError:  # pragma: no cover - lightweight renderer/export paths should still import
    FrontierActionSpace = None
    FrontierCurriculumProgress = None
    GhostMercFrontierEnv = None

__all__ = [
    "FrontierActionSpace",
    "FrontierCurriculumProgress",
    "FrontierTerritoryConfig",
    "GhostMercFrontierEnv",
]
