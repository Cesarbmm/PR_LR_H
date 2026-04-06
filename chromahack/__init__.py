"""Canonical GhostMerc Frontier research package.

The package is imported both by the full RL stack and by lightweight utilities
such as renderer/exporter tooling. Those lightweight paths should keep working
even if training-only dependencies like Gymnasium are not installed in the
current interpreter.
"""

from __future__ import annotations

try:
    from gymnasium.envs.registration import register
except ModuleNotFoundError:  # pragma: no cover - exercised indirectly in lightweight environments
    register = None

try:
    from chromahack.envs.ghostmerc_frontier_env import FrontierActionSpace, FrontierCurriculumProgress, GhostMercFrontierEnv
    from chromahack.utils.config import FrontierTerritoryConfig
except ModuleNotFoundError:  # pragma: no cover - renderer/export paths do not need the RL stack eagerly
    FrontierActionSpace = None
    FrontierCurriculumProgress = None
    GhostMercFrontierEnv = None
    FrontierTerritoryConfig = None

__all__ = [
    "__version__",
    "FrontierActionSpace",
    "FrontierCurriculumProgress",
    "FrontierTerritoryConfig",
    "GhostMercFrontierEnv",
]

__version__ = "0.5.0"

if register is not None:
    try:
        register(
            id="GhostMercFrontier-v0",
            entry_point="chromahack.envs.ghostmerc_frontier_env:GhostMercFrontierEnv",
        )
    except Exception:
        pass

    try:
        register(
            id="GhostMerc-v0",
            entry_point="chromahack.envs.ghostmerc_env:GhostMercEnv",
        )
    except Exception:
        pass

    try:
        register(
            id="BridgeInspectionHack-v0",
            entry_point="chromahack.envs.bridge_env:BridgeInspectionHackEnv",
        )
    except Exception:
        pass
