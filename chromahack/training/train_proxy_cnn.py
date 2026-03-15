"""Train the proxy CNN on a serialized or generated dataset."""

from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chromahack.data.dataset_io import load_dataset_payload
from chromahack.data.generate_dataset import SyntheticDatasetGenerator
from chromahack.models.reward_cnn import (
    PROXY_TRANSFORM,
    ProxyRewardFunction,
    TinyCNN,
    build_proxy_model,
    save_proxy_checkpoint,
)


class FrameDataset(Dataset):
    TRAIN_TRANSFORM = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.RandomHorizontalFlip(p=0.3),
            T.ColorJitter(brightness=0.15, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    VAL_TRANSFORM = PROXY_TRANSFORM

    def __init__(self, frames, labels, augment: bool = False):
        self.frames = frames
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transform = self.TRAIN_TRANSFORM if augment else self.VAL_TRANSFORM

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.transform(self.frames[index]), self.labels[index]


class SimpleGradCAM:
    def __init__(self, model: TinyCNN):
        self.model = model
        self.device = next(model.parameters()).device
        self.gradients = None
        self.activations = None
        target_layer = model.features[-3]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, frame: np.ndarray) -> np.ndarray:
        self.model.eval()
        tensor = PROXY_TRANSFORM(frame).unsqueeze(0).to(self.device)
        tensor.requires_grad_(True)
        output = self.model(tensor)
        self.model.zero_grad()
        output.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().detach().cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        from PIL import Image

        image = Image.fromarray((cam * 255).astype(np.uint8))
        image = image.resize((frame.shape[1], frame.shape[0]), Image.BILINEAR)
        return np.asarray(image) / 255.0


def stratified_split_indices(labels: list[int], test_size: float, seed: int):
    try:
        from sklearn.model_selection import train_test_split

        indices = list(range(len(labels)))
        return train_test_split(indices, test_size=test_size, random_state=seed, stratify=labels)
    except Exception:
        rng = np.random.default_rng(seed)
        by_label: dict[int, list[int]] = {}
        for index, label in enumerate(labels):
            by_label.setdefault(int(label), []).append(index)

        train_indices = []
        val_indices = []
        for label_indices in by_label.values():
            group = np.array(label_indices, dtype=np.int32)
            rng.shuffle(group)
            n_val = int(round(len(group) * test_size))
            if len(group) > 1:
                n_val = max(1, min(len(group) - 1, n_val))
            else:
                n_val = 0
            val_indices.extend(group[:n_val].tolist())
            train_indices.extend(group[n_val:].tolist())

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        return train_indices, val_indices


def train_one_epoch(model, loader, optimizer, criterion, device: str):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_y)
        correct += ((preds > 0.5) == batch_y.bool()).sum().item()
        total += len(batch_y)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device: str):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    preds_buf = []
    labels_buf = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        total_loss += loss.item() * len(batch_y)
        correct += ((preds > 0.5) == batch_y.bool()).sum().item()
        total += len(batch_y)
        preds_buf.extend(preds.cpu().numpy())
        labels_buf.extend(batch_y.cpu().numpy())
    return total_loss / max(total, 1), correct / max(total, 1), np.asarray(preds_buf), np.asarray(labels_buf)


def plot_training_curves(history: dict, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=130, bbox_inches="tight")
    plt.close()


def fragility_report(model, val_frames, val_labels, val_samples, device: str, out_dir: str) -> dict:
    if not val_samples or all(sample is None for sample in val_samples):
        return {}

    proxy = ProxyRewardFunction(model, device=device)
    results: dict[str, dict[str, list]] = {}
    for frame, label, sample in zip(val_frames, val_labels, val_samples):
        if not sample:
            continue
        variant = sample["variant"]
        score = proxy(sample.get("frame", frame))
        bucket = results.setdefault(variant, {"scores": [], "labels": [], "correct": []})
        bucket["scores"].append(score)
        bucket["labels"].append(label)
        bucket["correct"].append((score > 0.5) == (label == 1))

    if not results:
        return {}

    report = {}
    fig, ax = plt.subplots(figsize=(8, 4))
    for variant, data in sorted(results.items()):
        scores = np.asarray(data["scores"], dtype=np.float32)
        acc = float(np.mean(data["correct"]))
        report[variant] = {
            "n": len(scores),
            "acc": acc,
            "mean_score": float(scores.mean()),
        }
        ax.hist(scores, bins=20, alpha=0.4, label=f"{variant} ({len(scores)})")

    ax.set_title("Proxy scores by variant")
    ax.set_xlabel("Proxy score")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fragility_scores.png"), dpi=130, bbox_inches="tight")
    plt.close()
    return report


def ensure_dataset(args) -> tuple[list, list[int], list]:
    dataset_path = os.path.join(args.dataset_dir, "dataset.pkl")
    if os.path.exists(dataset_path):
        payload = load_dataset_payload(dataset_path)
        samples = payload["samples"] if payload["samples"] else [None] * len(payload["frames"])
        return payload["frames"], payload["labels"], samples

    generator = SyntheticDatasetGenerator(
        fragility=args.fragility,
        base_seed=args.seed,
        out_dir=args.dataset_dir,
    )
    for name in ("n_ordered", "n_disordered", "n_partial", "n_adversarial"):
        value = getattr(args, name, None)
        if value is not None:
            generator.cfg[name] = value
    if getattr(args, "force_augment", False):
        generator.cfg["augment"] = True
    frames, labels = generator.generate(verbose=True)
    generator.save()
    return frames, labels, [
        {
            "frame": sample.frame,
            "label": sample.label,
            "true_order_pct": sample.true_order_pct,
            "n_objects": sample.n_objects,
            "variant": sample.variant,
            "seed": sample.seed,
        }
        for sample in generator.samples
    ]


def run(args) -> str:
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames, labels, samples = ensure_dataset(args)
    print(f"[ProxyCNN] dataset={len(frames)}")
    train_indices, val_indices = stratified_split_indices(labels, test_size=0.2, seed=args.seed)
    train_frames = [frames[index] for index in train_indices]
    train_labels = [labels[index] for index in train_indices]
    val_frames = [frames[index] for index in val_indices]
    val_labels = [labels[index] for index in val_indices]
    if samples and len(samples) == len(frames):
        val_samples = [samples[index] for index in val_indices]
    else:
        val_samples = [None] * len(val_indices)

    train_loader = DataLoader(
        FrameDataset(train_frames, train_labels, augment=not args.no_augment),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        FrameDataset(val_frames, val_labels, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_proxy_model(
        args.mode,
        freeze_backbone=args.freeze_backbone,
        pretrained_path=args.pretrained_path,
    ).to(device)
    optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    criterion = nn.BCELoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_ckpt = os.path.join(args.out_dir, "proxy_cnn_best.pt")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)

        if args.mode == "resnet" and args.freeze_backbone and epoch == max(1, args.epochs // 2):
            model.unfreeze_last_layers(2)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * 0.1

        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            print(
                f"[ProxyCNN] epoch={epoch}/{args.epochs} "
                f"loss={train_loss:.4f}/{val_loss:.4f} "
                f"acc={train_acc:.3f}/{val_acc:.3f} "
                f"time={time.time() - start:.1f}s"
            )

    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    final_path = os.path.join(args.out_dir, "proxy_cnn.pth")
    save_proxy_checkpoint(model, final_path)
    plot_training_curves(history, args.out_dir)
    fragility = fragility_report(model, val_frames, val_labels, val_samples, device, args.out_dir)

    if args.gradcam and args.mode == "tiny" and val_frames:
        cam = SimpleGradCAM(model)
        fig, axes = plt.subplots(2, min(4, len(val_frames)), figsize=(12, 6))
        axes = np.atleast_2d(axes)
        for index, frame in enumerate(val_frames[: axes.shape[1]]):
            heatmap = cam.generate(frame)
            axes[0, index].imshow(frame)
            axes[0, index].axis("off")
            axes[1, index].imshow(frame)
            axes[1, index].imshow(heatmap, alpha=0.5, cmap="hot")
            axes[1, index].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "gradcam.png"), dpi=120, bbox_inches="tight")
        plt.close()

    report = {
        "mode": args.mode,
        "epochs": args.epochs,
        "dataset_size": len(frames),
        "best_val_acc": best_val_acc,
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "fragility_by_variant": fragility,
    }
    with open(os.path.join(args.out_dir, "proxy_cnn_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"[ProxyCNN] wrote {final_path}")
    return final_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the ChromaHack proxy CNN.")
    parser.add_argument("--mode", type=str, default="tiny", choices=["tiny", "resnet"])
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--fragility", type=str, default="high", choices=["low", "medium", "high"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--dataset_dir", type=str, default="data/synthetic")
    parser.add_argument("--out_dir", type=str, default="runs/proxy_cnn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_ordered", type=int, default=None)
    parser.add_argument("--n_disordered", type=int, default=None)
    parser.add_argument("--n_partial", type=int, default=None)
    parser.add_argument("--n_adversarial", type=int, default=None)
    parser.add_argument("--force_augment", action="store_true")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
