"""Quick inspection utility for generated datasets."""

from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chromahack.data.dataset_io import load_dataset_payload


def inspect(dataset_path: str, out_dir: str | None = None) -> dict:
    out_dir = out_dir or os.path.dirname(dataset_path)
    payload = load_dataset_payload(dataset_path)
    frames = payload["frames"]
    labels = payload["labels"]
    samples = payload["samples"]
    stats = payload["stats"]

    print(f"[Inspect] loaded {dataset_path}")
    print(f"  total={len(frames)} pos={sum(labels)} neg={len(labels) - sum(labels)}")
    print(f"  fragility={stats.get('fragility_level', 'unknown')}")

    if samples:
        variants = {}
        for sample in samples:
            variant = sample["variant"]
            variants[variant] = variants.get(variant, 0) + 1
        for variant, count in sorted(variants.items()):
            print(f"  {variant:20s} {count}")

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle("Dataset inspection", fontsize=12)
    positives = [frame for frame, label in zip(frames, labels) if label == 1][:4]
    negatives = [frame for frame, label in zip(frames, labels) if label == 0][:4]

    for index, frame in enumerate(positives):
        axes[0, index].imshow(frame)
        axes[0, index].set_title("ordered", color="#4CAF50", fontsize=9)
        axes[0, index].axis("off")

    for index, frame in enumerate(negatives):
        axes[1, index].imshow(frame)
        axes[1, index].set_title("disordered", color="#F44336", fontsize=9)
        axes[1, index].axis("off")

    for row in range(2):
        for col in range(4):
            if (row == 0 and col >= len(positives)) or (row == 1 and col >= len(negatives)):
                axes[row, col].axis("off")

    plt.tight_layout()
    grid_path = os.path.join(out_dir, "inspect_grid.png")
    plt.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Inspect] wrote {grid_path}")

    adversarial = [sample for sample in samples if sample["variant"] == "adversarial"]
    if adversarial:
        mean_adv_true = float(np.mean([sample["true_order_pct"] for sample in adversarial]))
        print(f"  adversarial count={len(adversarial)} mean_true={mean_adv_true:.3f}")

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a generated ChromaHack dataset.")
    parser.add_argument("--dataset", type=str, default="data/synthetic/dataset.pkl")
    parser.add_argument("--out_dir", type=str, default=None)
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    inspect(args.dataset, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
