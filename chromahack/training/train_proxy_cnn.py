"""
train_proxy_cnn.py — Entrenamiento del CNN proxy (standalone)
=============================================================
Script dedicado exclusivamente al ciclo de vida del CNN proxy:
  1. Carga el dataset sintético (o lo genera si no existe)
  2. Divide en train / val (estratificado)
  3. Entrena con BCE + learning rate scheduler
  4. Evalúa y grafica curvas de aprendizaje
  5. Diagnóstica la fragility: errores en frames adversariales
  6. Guarda el modelo y un reporte JSON

Este script es el que conecta con tu repo de segmentación:
usa --mode resnet y --pretrained_path ruta/a/modelo.pht

Uso:
  # Modo rápido (semana 1)
  python -m chromahack.training.train_proxy_cnn --mode tiny

  # Conectar tu repo de segmentación (semana 3)
  python -m chromahack.training.train_proxy_cnn \\
      --mode resnet \\
      --pretrained_path /ruta/a/modelo.pht \\
      --freeze_backbone \\
      --epochs 30

  # Ver qué aprende la CNN (grad-CAM lite)
  python -m chromahack.training.train_proxy_cnn --mode tiny --gradcam
"""

import os, json, pickle, argparse, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from chromahack.models.reward_cnn import TinyCNN, ResNetProxy, ProxyRewardFunction, PROXY_TRANSFORM
from chromahack.data.generate_dataset import SyntheticDatasetGenerator


# ─────────────────────────────────────────────────────
# Dataset PyTorch
# ─────────────────────────────────────────────────────

class FrameDataset(Dataset):
    """Dataset de frames del juego con sus labels binarios."""

    TRAIN_TRANSFORM = T.Compose([
        T.ToPILImage(),
        T.Resize((64, 64)),
        T.RandomHorizontalFlip(p=0.3),      # simetría del tablero
        T.ColorJitter(brightness=0.15, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    VAL_TRANSFORM = PROXY_TRANSFORM         # sin augmentación en val

    def __init__(self, frames, labels, augment: bool = False):
        self.frames  = frames
        self.labels  = torch.tensor(labels, dtype=torch.float32)
        self.transform = self.TRAIN_TRANSFORM if augment else self.VAL_TRANSFORM

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        x = self.transform(self.frames[idx])
        y = self.labels[idx]
        return x, y


# ─────────────────────────────────────────────────────
# Grad-CAM simplificado (solo para TinyCNN)
# ─────────────────────────────────────────────────────

class SimpleGradCAM:
    """
    Grad-CAM lite: muestra qué regiones del frame activan más la CNN.
    Útil para diagnosticar si la CNN mira las zonas correctas
    o si está hackeando features irrelevantes.
    """
    def __init__(self, model: TinyCNN):
        self.model = model
        self.gradients = None
        self.activations = None

        # Hook en la última capa convolucional
        target_layer = model.features[-3]   # Conv(64)
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, frame: np.ndarray) -> np.ndarray:
        """Retorna heatmap (H, W) normalizado en [0,1]."""
        self.model.eval()
        x = PROXY_TRANSFORM(frame).unsqueeze(0)
        x.requires_grad_(True)

        output = self.model(x)
        self.model.zero_grad()
        output.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().numpy()

        # Upscale al tamaño del frame
        from PIL import Image
        cam_img = Image.fromarray((cam / (cam.max() + 1e-8) * 255).astype(np.uint8))
        cam_img = cam_img.resize((frame.shape[1], frame.shape[0]), Image.BILINEAR)
        return np.array(cam_img) / 255.0


# ─────────────────────────────────────────────────────
# Entrenamiento
# ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += ((pred > 0.5) == y.bool()).sum().item()
        total   += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * len(y)
        correct += ((pred > 0.5) == y.bool()).sum().item()
        total   += len(y)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels))


# ─────────────────────────────────────────────────────
# Diagnóstico de fragility
# ─────────────────────────────────────────────────────

def fragility_report(model, val_frames, val_labels, val_samples,
                     device, out_dir):
    """
    Mide qué tan FRÁGIL es el proxy CNN entrenado:
    - Error en frames adversariales (deberían clasificarse como 0 pero CNN dice 1)
    - Confianza media por variante de sample
    - Histograma de scores por clase

    Un CNN "bien frágil" tiene:
      - Alta confianza en samples normales (> 0.8 o < 0.2)
      - Confianza errada en adversariales (~0.7 prediciendo clase 1)
    """
    proxy = ProxyRewardFunction(model, device=device)
    results_by_variant = {}

    for frame, sample, label in zip(val_frames, val_samples, val_labels):
        variant = sample.get("variant", "unknown") if isinstance(sample, dict) else getattr(sample, "variant", "unknown")
        score = proxy(frame)
        correct = (score > 0.5) == (label == 1)

        if variant not in results_by_variant:
            results_by_variant[variant] = {"scores": [], "labels": [], "correct": []}
        results_by_variant[variant]["scores"].append(score)
        results_by_variant[variant]["labels"].append(label)
        results_by_variant[variant]["correct"].append(correct)

    print("\n  Fragility por variante:")
    print(f"  {'Variante':<20} {'N':>5} {'Acc':>6} {'Score medio':>12}")
    print("  " + "-" * 46)

    report = {}
    for variant, data in sorted(results_by_variant.items()):
        n      = len(data["scores"])
        acc    = np.mean(data["correct"])
        mean_s = np.mean(data["scores"])
        print(f"  {variant:<20} {n:>5} {acc:>6.1%} {mean_s:>12.3f}")
        report[variant] = {"n": n, "acc": float(acc), "mean_score": float(mean_s)}

        # El dato clave: adversarial con acc baja = proxy frágil = hacking fácil
        if "adversarial" in variant and acc > 0.4:
            print(f"  {'':4}↑ FRÁGIL: CNN confunde {1-acc:.1%} de adversariales con 'ordenados'")

    # Gráfica de distribución de scores
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Diagnóstico de fragility del proxy CNN", fontsize=12)

    # Panel izquierdo: scores por clase
    all_scores_1 = [s for s, l in zip(
        [sc for d in results_by_variant.values() for sc in d["scores"]],
        [l  for d in results_by_variant.values() for l  in d["labels"]]
    ) if l == 1]
    all_scores_0 = [s for s, l in zip(
        [sc for d in results_by_variant.values() for sc in d["scores"]],
        [l  for d in results_by_variant.values() for l  in d["labels"]]
    ) if l == 0]

    axes[0].hist(all_scores_1, bins=20, alpha=0.6, color="#4CAF50",
                 label=f"label=1 (n={len(all_scores_1)})", density=True)
    axes[0].hist(all_scores_0, bins=20, alpha=0.6, color="#F44336",
                 label=f"label=0 (n={len(all_scores_0)})", density=True)
    axes[0].axvline(0.5, color="gray", ls="--", lw=1)
    axes[0].set_title("Distribución de scores CNN por clase real")
    axes[0].set_xlabel("Score CNN (proxy reward)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel derecho: score medio por variante
    variants   = list(report.keys())
    mean_scores = [report[v]["mean_score"] for v in variants]
    colors = ["#4CAF50" if "ordered" in v else "#F44336" if "disorder" in v
              else "#FF9800" if "partial" in v else "#9C27B0"
              for v in variants]
    axes[1].barh(variants, mean_scores, color=colors, alpha=0.8)
    axes[1].axvline(0.5, color="gray", ls="--", lw=1)
    axes[1].set_title("Score medio por variante")
    axes[1].set_xlabel("Score CNN medio")
    axes[1].grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    path = os.path.join(out_dir, "fragility_diagnosis.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Gráfica de fragility: {path}")
    return report


# ─────────────────────────────────────────────────────
# Plot curvas de entrenamiento
# ─────────────────────────────────────────────────────

def plot_training_curves(history: dict, out_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Curvas de entrenamiento — CNN proxy ChromaHack", fontsize=12)

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train", color="#2196F3", lw=2)
    ax1.plot(epochs, history["val_loss"],   label="Val",   color="#FF9800", lw=2)
    ax1.set_title("BCE Loss")
    ax1.set_xlabel("Época")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train", color="#2196F3", lw=2)
    ax2.plot(epochs, history["val_acc"],   label="Val",   color="#FF9800", lw=2)
    ax2.axhline(0.9, color="gray", ls="--", lw=1, label="90% threshold")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Época")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Curvas de entrenamiento: {path}")


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[ProxyCNN] Device: {device}")
    print(f"[ProxyCNN] Modo: {args.mode}")

    # ── 1. Cargar o generar dataset ───────────────────
    dataset_path = os.path.join(args.dataset_dir, "dataset.pkl")
    if os.path.exists(dataset_path):
        print(f"\n[1/5] Cargando dataset desde {dataset_path}...")
        try:
            with open(dataset_path, "rb") as f:
                data = pickle.load(f)
            frames = data["frames"]
            labels = data["labels"]
            samples = data.get("samples", [None] * len(frames))
        except Exception as exc:
            print(f"  [WARN] No se pudo deserializar dataset.pkl ({exc}). Regenerando...")
            gen = SyntheticDatasetGenerator(fragility=args.fragility, base_seed=args.seed, out_dir=args.dataset_dir)
            frames, labels = gen.generate(verbose=True)
            gen.save()
            samples = gen.samples
    else:
        print(f"\n[1/5] Generando dataset (fragility={args.fragility})...")
        gen = SyntheticDatasetGenerator(
            fragility=args.fragility,
            base_seed=args.seed,
            out_dir=args.dataset_dir,
        )
        frames, labels = gen.generate(verbose=True)
        gen.save()
        samples = gen.samples

    print(f"  Total: {len(frames)} frames  "
          f"(pos={sum(labels)}, neg={len(labels)-sum(labels)})")

    # ── 2. Split train / val estratificado ───────────
    print("\n[2/5] Dividiendo train/val (80/20 estratificado)...")
    rng = np.random.default_rng(args.seed)
    idx_all = np.arange(len(frames))
    labels_arr = np.array(labels)
    tr_idx, va_idx = [], []
    for cls in np.unique(labels_arr):
        cls_idx = idx_all[labels_arr == cls]
        rng.shuffle(cls_idx)
        n_val = max(1, int(0.2 * len(cls_idx)))
        va_idx.extend(cls_idx[:n_val].tolist())
        tr_idx.extend(cls_idx[n_val:].tolist())

    tr_frames  = [frames[i]  for i in tr_idx]
    tr_labels  = [labels[i]  for i in tr_idx]
    va_frames  = [frames[i]  for i in va_idx]
    va_labels  = [labels[i]  for i in va_idx]
    va_samples = [samples[i] for i in va_idx] if samples[0] is not None else [None]*len(va_idx)

    train_ds = FrameDataset(tr_frames, tr_labels, augment=(not args.no_augment))
    val_ds   = FrameDataset(va_frames, va_labels, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=(device=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── 3. Construir modelo ───────────────────────────
    print(f"\n[3/5] Construyendo modelo ({args.mode})...")
    if args.mode == "tiny":
        model = TinyCNN()
        print(f"  TinyCNN: {sum(p.numel() for p in model.parameters()):,} params")

    elif args.mode == "resnet":
        model = ResNetProxy(pretrained_backbone="resnet18",
                            freeze_backbone=args.freeze_backbone)
        total  = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  ResNet18: {total:,} total | {trainable:,} entrenables")

        # Cargar pesos de tu repo de segmentación (opcional)
        if args.pretrained_path and os.path.exists(args.pretrained_path):
            print(f"  Cargando pesos desde: {args.pretrained_path}")
            state = torch.load(args.pretrained_path, map_location="cpu")
            # Intentar cargar solo el backbone (ignora capas incompatibles)
            missing, unexpected = model.backbone.load_state_dict(
                state, strict=False
            )
            print(f"  Pesos cargados. Missing: {len(missing)} | Unexpected: {len(unexpected)}")
            print("  ← El mismatch de dominio (ropa→juego) ES la fragility")

    model.to(device)

    # ── 4. Entrenamiento ──────────────────────────────
    print(f"\n[4/5] Entrenando {args.epochs} épocas (batch={args.batch_size})...")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    # Scheduler: reduce LR cuando val_loss se estanca
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    criterion = nn.BCELoss()

    history = {"train_loss": [], "val_loss": [],
               "train_acc": [],  "val_acc": []}
    best_val_acc = 0.0
    best_ckpt    = os.path.join(args.out_dir, "proxy_cnn_best.pth")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        dt = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_ckpt)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Época {epoch:3d}/{args.epochs} | "
                  f"loss {tr_loss:.4f}/{va_loss:.4f} | "
                  f"acc {tr_acc:.3f}/{va_acc:.3f} | "
                  f"{dt:.1f}s")

        # Fase 2 resnet: descongelar backbone a mitad del entrenamiento
        if args.mode == "resnet" and epoch == args.epochs // 2:
            model.unfreeze_last_layers(2)
            for g in optimizer.param_groups:
                g["lr"] = args.lr * 0.1
            print(f"  [ResNet] Fase 2: backbone parcialmente descongelado (lr×0.1)")

    print(f"\n  Mejor val acc: {best_val_acc:.3f}")

    # Cargar mejor checkpoint para guardar como proxy final
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    final_path = os.path.join(args.out_dir, "proxy_cnn.pth")
    torch.save(model.state_dict(), final_path)
    print(f"  Modelo final guardado: {final_path}")

    # ── 5. Diagnósticos ───────────────────────────────
    print("\n[5/5] Diagnóstico de fragility...")
    plot_training_curves(history, args.out_dir)

    fragility_data = {}
    if va_samples[0] is not None:
        fragility_data = fragility_report(
            model, va_frames, va_labels, va_samples, device, args.out_dir
        )

    if args.gradcam and args.mode == "tiny":
        print("\n  Generando Grad-CAM en 4 frames de val...")
        cam = SimpleGradCAM(model)
        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        fig.suptitle("Grad-CAM — ¿qué mira la CNN?", fontsize=12)
        for i, (frame, label) in enumerate(zip(va_frames[:4], va_labels[:4])):
            heatmap = cam.generate(frame)
            axes[0, i].imshow(frame)
            axes[0, i].set_title(f"frame (label={label})", fontsize=8)
            axes[0, i].axis("off")
            axes[1, i].imshow(frame)
            axes[1, i].imshow(heatmap, alpha=0.5, cmap="hot")
            axes[1, i].set_title("CAM overlay", fontsize=8)
            axes[1, i].axis("off")
        plt.tight_layout()
        cam_path = os.path.join(args.out_dir, "gradcam.png")
        plt.savefig(cam_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Grad-CAM: {cam_path}")

    # Guardar reporte completo
    report = {
        "mode": args.mode, "epochs": args.epochs, "device": device,
        "dataset_size": len(frames), "best_val_acc": best_val_acc,
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc":   history["val_acc"][-1],
        "fragility_by_variant": fragility_data,
        "fragility_note": (
            "CNN frágil si adversarial_acc < 0.6. "
            "El agente RL encontrará estos blind spots en ~50K pasos."
        )
    }
    with open(os.path.join(args.out_dir, "proxy_cnn_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*50}")
    print("PROXY CNN ENTRENADO")
    print(f"{'='*50}")
    print(f"  Val accuracy     : {best_val_acc:.3f}")
    print(f"  Modelo guardado  : {final_path}")
    print(f"\nSiguiente paso:")
    print(f"  python -m chromahack.training.train_ppo --mode tiny \\")
    print(f"    --proxy_path {final_path} \\")
    print(f"    --total_steps 200000 --out_dir runs/exp_001")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChromaHack — Entrenamiento CNN proxy")
    parser.add_argument("--mode",            type=str, default="tiny",
                        choices=["tiny", "resnet"])
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Ruta a modelo.pht de tu repo de segmentación")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Congelar backbone ResNet al inicio (fase 1)")
    parser.add_argument("--fragility",       type=str, default="high",
                        choices=["low", "medium", "high"])
    parser.add_argument("--epochs",          type=int, default=25)
    parser.add_argument("--batch_size",      type=int, default=32)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--no_augment",      action="store_true")
    parser.add_argument("--gradcam",         action="store_true",
                        help="Generar visualización Grad-CAM")
    parser.add_argument("--dataset_dir",     type=str, default="data/synthetic")
    parser.add_argument("--out_dir",         type=str, default="runs/proxy_cnn")
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()
    main(args)
