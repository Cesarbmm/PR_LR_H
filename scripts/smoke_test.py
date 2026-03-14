"""Light smoke-test for package imports and one env step."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chromahack.envs.chroma_env import ChromaHackEnv
from chromahack.models.reward_cnn import ProxyRewardFunction, TinyCNN


def main() -> None:
    env = ChromaHackEnv(seed=0)
    obs, _ = env.reset()
    proxy = ProxyRewardFunction(TinyCNN())
    env.set_proxy_fn(proxy)
    _, _, _, _, info = env.step(0)
    print("obs:", obs.shape, "r_true:", info["r_true"], "r_proxy:", info["r_proxy"])
    env.close()


if __name__ == "__main__":
    main()
