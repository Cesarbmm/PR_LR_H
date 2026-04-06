"""Legacy alias for the canonical GhostMerc environment."""

from __future__ import annotations

from chromahack.envs.bridge_env import ACTION_NAMES as BRIDGE_ACTION_NAMES
from chromahack.envs.bridge_env import BridgeAction, BridgeInspectionHackEnv
from chromahack.envs.ghostmerc_env import GhostMercActionSpace, GhostMercEnv

ChromaHackEnv = GhostMercEnv

__all__ = [
    "BRIDGE_ACTION_NAMES",
    "BridgeAction",
    "BridgeInspectionHackEnv",
    "ChromaHackEnv",
    "GhostMercActionSpace",
    "GhostMercEnv",
]
