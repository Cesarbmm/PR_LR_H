from __future__ import annotations

import numpy as np
import torch

from chromahack.envs.ghostmerc_frontier_env import GhostMercFrontierEnv
from chromahack.envs.territory_generator import sample_curriculum_district_id
from chromahack.intervention.pref_model import (
    CLIP_CONTEXT_DIM,
    STEP_FEATURE_DIM,
    FrontierPreferenceRewardWrapper,
    PreferenceRewardModel,
)
from chromahack.utils.config import FrontierTerritoryConfig


def _make_env(observation_mode: str) -> GhostMercFrontierEnv:
    config = FrontierTerritoryConfig(observation_mode=observation_mode, max_steps=64)
    return GhostMercFrontierEnv(config=config, seed=123, forced_district_id=5, distribution_split="stress")


def test_frontier_flat_and_dict_modes_match_dynamics() -> None:
    flat_env = _make_env("flat")
    dict_env = _make_env("dict")
    flat_obs, flat_info = flat_env.reset(seed=999, options={"district_id": 5, "distribution_split": "stress"})
    dict_obs, dict_info = dict_env.reset(seed=999, options={"district_id": 5, "distribution_split": "stress"})

    assert isinstance(flat_obs, np.ndarray)
    assert isinstance(dict_obs, dict)
    assert flat_info["district_id"] == dict_info["district_id"] == 5
    assert flat_info["state_snapshot"] == dict_info["state_snapshot"]

    actions = [
        np.asarray([4, 0, 1, 0, 0], dtype=np.int64),
        np.asarray([4, 2, 1, 1, 0], dtype=np.int64),
        np.asarray([8, 0, 1, 2, 1], dtype=np.int64),
        np.asarray([0, 3, 1, 0, 1], dtype=np.int64),
        np.asarray([4, 0, 1, 0, 4], dtype=np.int64),
    ]
    for action in actions:
        flat_obs, flat_reward, flat_terminated, flat_truncated, flat_info = flat_env.step(action)
        dict_obs, dict_reward, dict_terminated, dict_truncated, dict_info = dict_env.step(action)
        assert flat_reward == dict_reward
        assert flat_terminated == dict_terminated
        assert flat_truncated == dict_truncated
        assert flat_info["true_reward"] == dict_info["true_reward"]
        assert flat_info["phase_label"] == dict_info["phase_label"]
        assert flat_info["state_snapshot"] == dict_info["state_snapshot"]

    flat_env.close()
    dict_env.close()


def test_frontier_distribution_split_changes_layout_but_stays_deterministic() -> None:
    env = GhostMercFrontierEnv(config=FrontierTerritoryConfig(observation_mode="flat", max_steps=64), seed=321, forced_district_id=5)

    _, train_info_a = env.reset(seed=111, options={"district_id": 5, "distribution_split": "train"})
    _, train_info_b = env.reset(seed=111, options={"district_id": 5, "distribution_split": "train"})
    _, shifted_info = env.reset(seed=111, options={"district_id": 5, "distribution_split": "shifted"})

    assert train_info_a["distribution_split"] == "train"
    assert train_info_b["distribution_split"] == "train"
    assert shifted_info["distribution_split"] == "shifted"
    assert train_info_a["state_snapshot"] == train_info_b["state_snapshot"]
    assert train_info_a["state_snapshot"] != shifted_info["state_snapshot"]

    env.close()


def test_broadcast_world_split_partition_sampling() -> None:
    rng = np.random.default_rng(7)
    train_ids = {
        sample_curriculum_district_id(0.9, rng, distribution_split="train", world_suite="broadcast_v3", world_split="train")
        for _ in range(64)
    }
    holdout_ids = {
        sample_curriculum_district_id(0.9, rng, distribution_split="train", world_suite="broadcast_v3", world_split="holdout")
        for _ in range(64)
    }
    broadcast_ids = {
        sample_curriculum_district_id(0.9, rng, distribution_split="train", world_suite="broadcast_v3", world_split="broadcast")
        for _ in range(64)
    }

    assert train_ids <= {1, 2, 3, 4, 5, 6, 7}
    assert holdout_ids <= {8, 9}
    assert broadcast_ids <= {6, 7, 8, 9, 10}
    assert 8 not in train_ids and 9 not in train_ids and 10 not in train_ids


def test_broadcast_flat_and_dict_match_world_state() -> None:
    flat_env = GhostMercFrontierEnv(
        config=FrontierTerritoryConfig(observation_mode="flat", max_steps=96),
        seed=555,
        forced_district_id=8,
        distribution_split="shifted",
        world_suite="broadcast_v3",
        world_split="broadcast",
    )
    dict_env = GhostMercFrontierEnv(
        config=FrontierTerritoryConfig(observation_mode="dict", max_steps=96),
        seed=555,
        forced_district_id=8,
        distribution_split="shifted",
        world_suite="broadcast_v3",
        world_split="broadcast",
    )

    _, flat_info = flat_env.reset(
        seed=202,
        options={"district_id": 8, "distribution_split": "shifted", "world_suite": "broadcast_v3", "world_split": "broadcast"},
    )
    _, dict_info = dict_env.reset(
        seed=202,
        options={"district_id": 8, "distribution_split": "shifted", "world_suite": "broadcast_v3", "world_split": "broadcast"},
    )

    assert flat_info["world_suite"] == dict_info["world_suite"] == "broadcast_v3"
    assert flat_info["world_split"] == dict_info["world_split"] == "broadcast"
    assert flat_info["active_event_type"] == dict_info["active_event_type"]
    assert flat_info["state_snapshot"]["primary_route"] == dict_info["state_snapshot"]["primary_route"]
    assert flat_info["state_snapshot"]["world_split"] == "broadcast"
    assert flat_info["state_snapshot"] == dict_info["state_snapshot"]

    action = np.asarray([4, 1, 1, 0, 1], dtype=np.int64)
    _, _, _, _, flat_step_info = flat_env.step(action)
    _, _, _, _, dict_step_info = dict_env.step(action)
    assert flat_step_info["world_split"] == dict_step_info["world_split"] == "broadcast"
    assert flat_step_info["active_event_type"] == dict_step_info["active_event_type"]
    assert flat_step_info["zones_visited"] == dict_step_info["zones_visited"]
    assert flat_step_info["state_snapshot"] == dict_step_info["state_snapshot"]

    flat_env.close()
    dict_env.close()


def test_preference_reward_wrapper_rewrites_proxy_reward(tmp_path) -> None:
    input_dim = STEP_FEATURE_DIM + CLIP_CONTEXT_DIM
    model = PreferenceRewardModel(input_dim=input_dim)
    model_path = tmp_path / "preference_reward_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "step_feature_dim": STEP_FEATURE_DIM,
            "clip_context_dim": CLIP_CONTEXT_DIM,
        },
        model_path,
    )

    env = GhostMercFrontierEnv(config=FrontierTerritoryConfig(observation_mode="flat", max_steps=32), seed=321, forced_district_id=5)
    wrapped = FrontierPreferenceRewardWrapper(env, str(model_path), clip_length=8)
    wrapped.reset(seed=111, options={"district_id": 5})
    _, reward, _, _, info = wrapped.step(np.asarray([4, 0, 1, 0, 1], dtype=np.int64))

    assert info["reward_mode"] == "pref_model"
    assert "proxy_reward_original" in info
    assert "reward_model_reward" in info
    assert reward == info["proxy_reward"] == info["reward_model_reward"]

    wrapped.close()


def test_patrol_world_split_partition_sampling() -> None:
    rng = np.random.default_rng(17)
    train_ids = {
        sample_curriculum_district_id(0.9, rng, distribution_split="train", world_suite="patrol_v4", world_split="train")
        for _ in range(96)
    }
    holdout_ids = {
        sample_curriculum_district_id(0.9, rng, distribution_split="train", world_suite="patrol_v4", world_split="holdout")
        for _ in range(96)
    }
    broadcast_ids = {
        sample_curriculum_district_id(0.9, rng, distribution_split="train", world_suite="patrol_v4", world_split="broadcast")
        for _ in range(96)
    }

    assert train_ids <= {11, 12, 13, 14, 15, 16}
    assert holdout_ids <= {17, 18}
    assert broadcast_ids <= {11, 12, 13, 14, 15, 16, 17, 18}


def test_patrol_flat_and_dict_match_world_state() -> None:
    flat_env = GhostMercFrontierEnv(
        config=FrontierTerritoryConfig(
            observation_mode="flat",
            max_steps=96,
            max_zones=7,
            include_incident_observation=True,
            proxy_profile="corrupted",
        ),
        seed=777,
        forced_district_id=11,
        distribution_split="train",
        world_suite="patrol_v4",
        world_split="broadcast",
    )
    dict_env = GhostMercFrontierEnv(
        config=FrontierTerritoryConfig(
            observation_mode="dict",
            max_steps=96,
            max_zones=7,
            include_incident_observation=True,
            proxy_profile="corrupted",
        ),
        seed=777,
        forced_district_id=11,
        distribution_split="train",
        world_suite="patrol_v4",
        world_split="broadcast",
    )

    _, flat_info = flat_env.reset(
        seed=303,
        options={"district_id": 11, "distribution_split": "train", "world_suite": "patrol_v4", "world_split": "broadcast"},
    )
    _, dict_info = dict_env.reset(
        seed=303,
        options={"district_id": 11, "distribution_split": "train", "world_suite": "patrol_v4", "world_split": "broadcast"},
    )

    assert flat_info["world_suite"] == dict_info["world_suite"] == "patrol_v4"
    assert flat_info["proxy_profile"] == dict_info["proxy_profile"] == "corrupted"
    assert flat_info["state_snapshot"]["incident_queue"] == dict_info["state_snapshot"]["incident_queue"]
    assert flat_info["state_snapshot"] == dict_info["state_snapshot"]

    actions = [
        np.asarray([4, 0, 0, 0, 0], dtype=np.int64),
        np.asarray([4, 0, 0, 0, 1], dtype=np.int64),
        np.asarray([8, 0, 0, 0, 4], dtype=np.int64),
    ]
    for action in actions:
        _, flat_reward, flat_terminated, flat_truncated, flat_step_info = flat_env.step(action)
        _, dict_reward, dict_terminated, dict_truncated, dict_step_info = dict_env.step(action)
        assert flat_reward == dict_reward
        assert flat_terminated == dict_terminated
        assert flat_truncated == dict_truncated
        assert flat_step_info["incident_resolution_rate"] == dict_step_info["incident_resolution_rate"]
        assert flat_step_info["monitor_farming_rate"] == dict_step_info["monitor_farming_rate"]
        assert flat_step_info["state_snapshot"] == dict_step_info["state_snapshot"]

    flat_env.close()
    dict_env.close()


def test_security_world_split_partition_sampling() -> None:
    rng = np.random.default_rng(23)
    train_ids = {
        sample_curriculum_district_id(0.9, rng, distribution_split="train", world_suite="security_v6", world_split="train")
        for _ in range(96)
    }
    holdout_ids = {
        sample_curriculum_district_id(0.9, rng, distribution_split="train", world_suite="security_v6", world_split="holdout")
        for _ in range(96)
    }
    broadcast_ids = {
        sample_curriculum_district_id(0.9, rng, distribution_split="train", world_suite="security_v6", world_split="broadcast")
        for _ in range(96)
    }

    assert train_ids <= {19, 20, 21, 22, 23, 24}
    assert holdout_ids <= {25, 26}
    assert broadcast_ids <= {19, 20, 21, 22, 23, 24, 25, 26}


def test_security_flat_and_dict_match_world_state() -> None:
    flat_env = GhostMercFrontierEnv(
        config=FrontierTerritoryConfig(
            observation_mode="flat",
            max_steps=96,
            max_zones=7,
            include_incident_observation=True,
            proxy_profile="patched",
            training_phase="anchor",
        ),
        seed=909,
        forced_district_id=21,
        distribution_split="train",
        world_suite="security_v6",
        world_split="broadcast",
    )
    dict_env = GhostMercFrontierEnv(
        config=FrontierTerritoryConfig(
            observation_mode="dict",
            max_steps=96,
            max_zones=7,
            include_incident_observation=True,
            proxy_profile="patched",
            training_phase="anchor",
        ),
        seed=909,
        forced_district_id=21,
        distribution_split="train",
        world_suite="security_v6",
        world_split="broadcast",
    )

    reset_options = {
        "district_id": 21,
        "distribution_split": "train",
        "world_suite": "security_v6",
        "world_split": "broadcast",
        "training_phase": "anchor",
    }
    _, flat_info = flat_env.reset(seed=404, options=reset_options)
    _, dict_info = dict_env.reset(seed=404, options=reset_options)

    assert flat_info["world_suite"] == dict_info["world_suite"] == "security_v6"
    assert flat_info["training_phase"] == dict_info["training_phase"] == "anchor"
    assert flat_info["state_snapshot"]["incident_queue"] == dict_info["state_snapshot"]["incident_queue"]
    assert flat_info["state_snapshot"] == dict_info["state_snapshot"]
    assert "incident_closure_rate" in flat_info["state_snapshot"]
    assert "confirmed_hostile_precision" in flat_info["state_snapshot"]
    assert "civilian_harm_rate" in flat_info["state_snapshot"]
    assert "security_kpi" in flat_info["state_snapshot"]
    assert "drift_score" in flat_info["state_snapshot"]
    assert "world_health" in flat_info["state_snapshot"]

    actions = [
        np.asarray([4, 0, 0, 0, 0], dtype=np.int64),
        np.asarray([4, 0, 1, 0, 1], dtype=np.int64),
        np.asarray([8, 0, 1, 2, 4], dtype=np.int64),
    ]
    for action in actions:
        _, flat_reward, flat_terminated, flat_truncated, flat_step_info = flat_env.step(action)
        _, dict_reward, dict_terminated, dict_truncated, dict_step_info = dict_env.step(action)
        assert flat_reward == dict_reward
        assert flat_terminated == dict_terminated
        assert flat_truncated == dict_truncated
        assert flat_step_info["incident_closure_rate"] == dict_step_info["incident_closure_rate"]
        assert flat_step_info["confirmed_hostile_precision"] == dict_step_info["confirmed_hostile_precision"]
        assert flat_step_info["civilian_harm_rate"] == dict_step_info["civilian_harm_rate"]
        assert flat_step_info["security_kpi"] == dict_step_info["security_kpi"]
        assert flat_step_info["drift_score"] == dict_step_info["drift_score"]
        assert flat_step_info["world_health"] == dict_step_info["world_health"]
        assert flat_step_info["state_snapshot"] == dict_step_info["state_snapshot"]

    flat_env.close()
    dict_env.close()
