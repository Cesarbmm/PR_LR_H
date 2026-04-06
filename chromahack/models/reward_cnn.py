"""Deprecated CNN reward-model module from the superseded visual proxy pipeline."""

from __future__ import annotations

from chromahack.legacy import deprecation_message

DEPRECATION_MESSAGE = deprecation_message("chromahack.models.reward_cnn", "python -m chromahack.training.train_ppo_ghostmerc")

__all__ = ["DEPRECATION_MESSAGE"]
