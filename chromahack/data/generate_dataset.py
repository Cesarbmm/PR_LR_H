"""Synthetic dataset generator for proxy reward training."""

from __future__ import annotations

import argparse
import os
from typing import List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chromahack.data.dataset_io import DatasetSample, DatasetStats, save_dataset_payload, save_stats_json
from chromahack.envs.chroma_env import ChromaHackEnv, GRID_SIZE, N_TYPES

LABEL_ORDERED = 1
LABEL_DISORDERED = 0


def _place_all_correct(env: ChromaHackEnv, rng: np.random.Generator) -> None:
    for obj in env.objects:
        cell = env.zones[obj["type"]][int(rng.integers(len(env.zones[obj["type"]])))]
        obj["row"], obj["col"] = cell
        obj["held"] = False


def _place_all_wrong(env: ChromaHackEnv, rng: np.random.Generator) -> None:
    variant = int(rng.integers(3))
    for obj in env.objects:
        wrong_types = [type_idx for type_idx in range(N_TYPES) if type_idx != obj["type"]]
        wrong_type = wrong_types[int(rng.integers(len(wrong_types)))]
        zone_cells = env.zones[wrong_type]
        if variant == 0:
            cell = zone_cells[int(rng.integers(len(zone_cells)))]
            obj["row"], obj["col"] = cell
        elif variant == 1:
            obj["row"] = int(rng.integers(0, 2))
            obj["col"] = int(rng.integers(0, 2))
        else:
            obj["row"] = int(rng.integers(0, GRID_SIZE))
            obj["col"] = int(rng.integers(0, GRID_SIZE))
        obj["held"] = False


def _place_partial(env: ChromaHackEnv, rng: np.random.Generator, pct_correct: float) -> None:
    n_correct = max(1, int(len(env.objects) * pct_correct))
    correct_indices = set(rng.choice(len(env.objects), n_correct, replace=False).tolist())
    for index, obj in enumerate(env.objects):
        if index in correct_indices:
            cell = env.zones[obj["type"]][int(rng.integers(len(env.zones[obj["type"]])))]
            obj["row"], obj["col"] = cell
        else:
            obj["row"] = int(rng.integers(0, GRID_SIZE))
            obj["col"] = int(rng.integers(0, GRID_SIZE))
        obj["held"] = False


def _place_adversarial(env: ChromaHackEnv, rng: np.random.Generator) -> None:
    wrong_corner = {0: (6, 6), 1: (6, 0), 2: (0, 3)}
    positions_by_type = {type_idx: [] for type_idx in range(N_TYPES)}
    for type_idx in range(N_TYPES):
        base_row, base_col = wrong_corner[type_idx]
        for d_row in range(2):
            for d_col in range(2):
                positions_by_type[type_idx].append(
                    (
                        int(np.clip(base_row + d_row, 0, GRID_SIZE - 1)),
                        int(np.clip(base_col + d_col, 0, GRID_SIZE - 1)),
                    )
                )

    for obj in env.objects:
        positions = positions_by_type[obj["type"]]
        obj["row"], obj["col"] = positions[int(rng.integers(len(positions)))]
        obj["held"] = False


def augment_frame(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    aug = frame.astype(np.float32).copy()
    aug *= float(rng.uniform(0.85, 1.15))
    aug += rng.normal(0, 3, aug.shape)
    if rng.random() < 0.5:
        aug = np.flip(aug, axis=1)
    return np.clip(aug, 0, 255).astype(np.uint8)


class SyntheticDatasetGenerator:
    """Generate the synthetic visual dataset with configurable fragility."""

    FRAGILITY_CONFIGS = {
        "low": {
            "n_ordered": 2000,
            "n_disordered": 2000,
            "n_partial": 200,
            "n_adversarial": 0,
            "augment": True,
            "label_noise": 0.0,
            "note": "More robust proxy baseline.",
        },
        "medium": {
            "n_ordered": 800,
            "n_disordered": 800,
            "n_partial": 400,
            "n_adversarial": 100,
            "augment": False,
            "label_noise": 0.05,
            "note": "Balanced baseline.",
        },
        "high": {
            "n_ordered": 300,
            "n_disordered": 300,
            "n_partial": 300,
            "n_adversarial": 200,
            "augment": False,
            "label_noise": 0.10,
            "note": "Deliberately fragile proxy dataset.",
        },
    }

    def __init__(self, fragility: str = "medium", base_seed: int = 42, out_dir: str = "data/synthetic"):
        if fragility not in self.FRAGILITY_CONFIGS:
            raise ValueError(f"Unknown fragility level: {fragility}")
        self.cfg = dict(self.FRAGILITY_CONFIGS[fragility])
        self.rng = np.random.default_rng(base_seed)
        self.out_dir = out_dir
        self.samples: List[DatasetSample] = []
        self.stats = DatasetStats(fragility_level=fragility, notes=[self.cfg["note"]])
        os.makedirs(out_dir, exist_ok=True)

    def _append_sample(self, env: ChromaHackEnv, label: int, variant: str, seed: int) -> None:
        frame = env.render()
        self.samples.append(
            DatasetSample(
                frame=frame,
                label=label,
                true_order_pct=env.compute_true_reward(),
                n_objects=len(env.objects),
                variant=variant,
                seed=seed,
            )
        )

    def generate(self, verbose: bool = True):
        cfg = self.cfg
        env = ChromaHackEnv(render_mode="rgb_array")

        if verbose:
            print(f"[Dataset] fragility={self.stats.fragility_level}")
            print(f"  {cfg['note']}")

        for _ in range(cfg["n_ordered"]):
            seed = int(self.rng.integers(100_000))
            env.reset(seed=seed)
            env.objects = env.objects[: int(self.rng.integers(4, 13))]
            _place_all_correct(env, self.rng)
            self._append_sample(env, LABEL_ORDERED, "ordered", seed)
        self.stats.n_ordered = cfg["n_ordered"]

        for _ in range(cfg["n_disordered"]):
            seed = int(self.rng.integers(100_000))
            env.reset(seed=seed)
            env.objects = env.objects[: int(self.rng.integers(4, 13))]
            _place_all_wrong(env, self.rng)
            self._append_sample(env, LABEL_DISORDERED, "disordered", seed)
        self.stats.n_disordered = cfg["n_disordered"]

        for _ in range(cfg["n_partial"]):
            seed = int(self.rng.integers(100_000))
            env.reset(seed=seed)
            _place_partial(env, self.rng, pct_correct=float(self.rng.uniform(0.3, 0.7)))
            label = LABEL_ORDERED if env.compute_true_reward() > 0.5 else LABEL_DISORDERED
            self._append_sample(env, label, "partial", seed)
        self.stats.n_partial = cfg["n_partial"]

        for _ in range(cfg["n_adversarial"]):
            seed = int(self.rng.integers(100_000))
            env.reset(seed=seed)
            _place_adversarial(env, self.rng)
            self._append_sample(env, LABEL_DISORDERED, "adversarial", seed)

        env.close()

        if cfg["augment"] and self.samples:
            n_augmented = len(self.samples) // 2
            indices = self.rng.choice(len(self.samples), n_augmented, replace=False)
            augmented = []
            for index in indices:
                sample = self.samples[int(index)]
                augmented.append(
                    DatasetSample(
                        frame=augment_frame(sample.frame, self.rng),
                        label=sample.label,
                        true_order_pct=sample.true_order_pct,
                        n_objects=sample.n_objects,
                        variant=f"{sample.variant}_aug",
                        seed=sample.seed,
                    )
                )
            self.samples.extend(augmented)
            self.stats.n_augmented = n_augmented

        label_noise = float(cfg["label_noise"])
        if label_noise > 0 and self.samples:
            n_flip = int(len(self.samples) * label_noise)
            for index in self.rng.choice(len(self.samples), n_flip, replace=False):
                sample = self.samples[int(index)]
                sample.label = 1 - max(0, sample.label)

        permutation = self.rng.permutation(len(self.samples))
        self.samples = [self.samples[int(index)] for index in permutation]

        ordered_scores = [sample.true_order_pct for sample in self.samples if sample.variant == "ordered"]
        disordered_scores = [sample.true_order_pct for sample in self.samples if sample.variant == "disordered"]
        self.stats.mean_true_ordered = float(np.mean(ordered_scores)) if ordered_scores else 0.0
        self.stats.mean_true_disordered = float(np.mean(disordered_scores)) if disordered_scores else 0.0

        frames = [sample.frame for sample in self.samples]
        labels = [max(0, sample.label) for sample in self.samples]
        if verbose:
            print(f"[Dataset] total={len(frames)} pos={sum(labels)} neg={len(labels) - sum(labels)}")
        return frames, labels

    def save(self) -> dict:
        dataset_path = os.path.join(self.out_dir, "dataset.pkl")
        payload = save_dataset_payload(
            dataset_path,
            frames=[sample.frame for sample in self.samples],
            labels=[max(0, sample.label) for sample in self.samples],
            samples=self.samples,
            stats=self.stats,
        )
        save_stats_json(os.path.join(self.out_dir, "dataset_stats.json"), self.stats)
        print(f"[Dataset] wrote {dataset_path}")
        return payload

    def visualize(self, n_per_class: int = 8) -> None:
        variants = ["ordered", "disordered", "partial", "adversarial"]
        subset_map = {
            variant: [sample for sample in self.samples if sample.variant.startswith(variant)][:n_per_class]
            for variant in variants
        }
        subset_map = {variant: samples for variant, samples in subset_map.items() if samples}
        if not subset_map:
            return

        fig, axes = plt.subplots(len(subset_map), n_per_class, figsize=(2.2 * n_per_class, 2.8 * len(subset_map)))
        axes = np.atleast_2d(axes)
        fig.suptitle(f"ChromaHack dataset ({self.stats.fragility_level})", fontsize=13, fontweight="bold")

        for row_index, (variant, samples) in enumerate(subset_map.items()):
            for col_index in range(n_per_class):
                ax = axes[row_index, col_index]
                ax.axis("off")
                if col_index < len(samples):
                    sample = samples[col_index]
                    ax.imshow(sample.frame)
                    ax.set_title(f"{variant}\nR*={sample.true_order_pct:.2f}", fontsize=8)
            axes[row_index, 0].set_ylabel(variant, rotation=0, ha="right", va="center", labelpad=24)

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "samples_grid.png"), dpi=130, bbox_inches="tight")
        plt.close()

    def plot_label_distribution(self) -> None:
        positives = [sample.true_order_pct for sample in self.samples if sample.label == 1]
        negatives = [sample.true_order_pct for sample in self.samples if sample.label == 0]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(positives, bins=20, alpha=0.6, color="#4CAF50", density=True, label=f"label=1 ({len(positives)})")
        ax.hist(negatives, bins=20, alpha=0.6, color="#F44336", density=True, label=f"label=0 ({len(negatives)})")
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.2)
        ax.set_xlabel("True order fraction")
        ax.set_ylabel("Density")
        ax.set_title(f"Label distribution ({self.stats.fragility_level})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "label_distribution.png"), dpi=130, bbox_inches="tight")
        plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the ChromaHack synthetic dataset.")
    parser.add_argument("--fragility", type=str, default="high", choices=["low", "medium", "high"])
    parser.add_argument("--n_ordered", type=int, default=None)
    parser.add_argument("--n_disordered", type=int, default=None)
    parser.add_argument("--n_partial", type=int, default=None)
    parser.add_argument("--n_adversarial", type=int, default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="data/synthetic")
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    generator = SyntheticDatasetGenerator(
        fragility=args.fragility,
        base_seed=args.seed,
        out_dir=args.out_dir,
    )
    if args.n_ordered is not None:
        generator.cfg["n_ordered"] = args.n_ordered
    if args.n_disordered is not None:
        generator.cfg["n_disordered"] = args.n_disordered
    if args.n_partial is not None:
        generator.cfg["n_partial"] = args.n_partial
    if args.n_adversarial is not None:
        generator.cfg["n_adversarial"] = args.n_adversarial
    if args.augment:
        generator.cfg["augment"] = True

    frames, labels = generator.generate(verbose=True)
    generator.save()
    if args.visualize:
        generator.visualize()
        generator.plot_label_distribution()

    print(f"[Dataset] ready at {args.out_dir} with {len(frames)} samples")
    return frames, labels


if __name__ == "__main__":
    main()
