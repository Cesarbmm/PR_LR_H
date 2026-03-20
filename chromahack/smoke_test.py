"""End-to-end smoke test for the local ChromaHack package."""

from __future__ import annotations

import os
from types import SimpleNamespace

from chromahack.data.generate_dataset import SyntheticDatasetGenerator
from chromahack.data.inspect_dataset import inspect
from chromahack.evaluation.eval_hidden import run as run_eval_hidden
from chromahack.training.train_ppo import run as run_ppo_training
from chromahack.training.train_proxy_cnn import run as run_proxy_training


def run_smoke_test(base_dir: str = "artifacts/smoke_test") -> bool:
    dataset_dir = os.path.join(base_dir, "data")
    proxy_dir = os.path.join(base_dir, "proxy")
    ppo_dir = os.path.join(base_dir, "ppo")
    os.makedirs(base_dir, exist_ok=True)

    generator = SyntheticDatasetGenerator(fragility="high", base_seed=0, out_dir=dataset_dir)
    generator.cfg["n_ordered"] = 2
    generator.cfg["n_disordered"] = 2
    generator.cfg["n_partial"] = 1
    generator.cfg["n_adversarial"] = 1
    generator.generate(verbose=True)
    generator.save()
    inspect(os.path.join(dataset_dir, "dataset.pkl"), out_dir=dataset_dir)

    proxy_args = SimpleNamespace(
        mode="tiny",
        pretrained_path=None,
        freeze_backbone=False,
        fragility="high",
        epochs=1,
        batch_size=8,
        lr=1e-3,
        no_augment=True,
        gradcam=False,
        dataset_dir=dataset_dir,
        out_dir=proxy_dir,
        seed=0,
        n_ordered=None,
        n_disordered=None,
        n_partial=None,
        force_augment=False,
    )
    proxy_path = run_proxy_training(proxy_args)

    ppo_args = SimpleNamespace(
        mode="tiny",
        proxy_path=proxy_path,
        pretrained_path=None,
        freeze_backbone=False,
        fragility="high",
        dataset_dir=dataset_dir,
        total_steps=64,
        n_envs=1,
        cnn_epochs=1,
        cnn_lr=1e-3,
        proxy_batch_size=8,
        ppo_lr=3e-4,
        n_steps=32,
        n_ordered=None,
        n_disordered=None,
        n_partial=None,
        no_augment=True,
        save_checkpoints=False,
        seed=0,
        out_dir=ppo_dir,
    )
    run_ppo_training(ppo_args)

    run_eval_hidden(
        SimpleNamespace(
            model_dir=ppo_dir,
            model_name="ppo_final",
            proxy_path=None,
            n_episodes=1,
            seed=0,
        )
    )
    return True


def main():
    ok = run_smoke_test()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
