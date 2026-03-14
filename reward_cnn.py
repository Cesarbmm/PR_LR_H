"""
reward_cnn.py — Modelo CNN proxy (frágil por diseño)
=====================================================
Dos modos de uso:

  Modo A — Tiny CNN desde cero (semana 1, para empezar rápido):
    model = TinyCNN()

  Modo B — ResNet18 con transfer learning (semana 3, la contribución):
    model = ResNetProxy(pretrained_backbone="resnet18")
    # Primero fine-tunea en Fashion MNIST o DeepFashion2,
    # luego fine-tunea en screenshots sintéticos del juego.
    # El mismatch de dominios crea la fragility necesaria.

La fragility es deliberada:
  - Dataset pequeño de fine-tuning → overfit al training set
  - Dominio cruzado (moda → juego) → activaciones OOD
  - El agente RL descubrirá patrones que explotan estas debilidades
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────
# Transforms para el input del CNN
# ──────────────────────────────────────────────
PROXY_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((64, 64)),       # Downscale para velocidad
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────
# Modo A: Tiny CNN desde cero
# ──────────────────────────────────────────────
class TinyCNN(nn.Module):
    """
    CNN pequeña (≈50K params) entrenada solo en screenshots del juego.
    Frágil por simplicidad: pocas neuronas → patrones OOD la engañan fácil.
    Input: (B, 3, 64, 64)
    Output: escalar en [0, 1] — "qué tan ordenado parece el frame"
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32×32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 16×16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 8×8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(-1)


# ──────────────────────────────────────────────
# Modo B: ResNet18 con transfer learning
# ──────────────────────────────────────────────
class ResNetProxy(nn.Module):
    """
    ResNet18 con backbone preentrenado en ImageNet (o Fashion MNIST).
    Se añade una cabeza de clasificación binaria.

    Estrategia de fine-tuning en 3 fases (crea fragility en capas):
      1. Congelar backbone → entrenar solo la cabeza (en Fashion data)
      2. Descongelar las últimas 2 capas → fine-tune en screenshots juego
      3. Dejar todo abierto → re-entrenar con dataset muy pequeño

    El resultado: backbone "recuerda" features de moda pero las usa
    para clasificar frames del juego → OOD sistemático = proxy frágil.
    """
    def __init__(self, pretrained_backbone: str = "resnet18", freeze_backbone: bool = True):
        super().__init__()

        # Cargar backbone preentrenado
        backbone = getattr(models, pretrained_backbone)(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Quitar la cabeza original de ImageNet
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Cabeza nueva para clasificación de "orden"
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Fase 1: solo entrena la cabeza."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_layers(self, n_layers: int = 2):
        """Fase 2: descongela las últimas n capas del backbone."""
        children = list(self.backbone.children())
        for child in children[-n_layers:]:
            for p in child.parameters():
                p.requires_grad = True

    def unfreeze_all(self):
        """Fase 3: fine-tune completo (crea overfitting sobre dataset pequeño)."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features).squeeze(-1)


# ──────────────────────────────────────────────
# Wrapper: convierte frame numpy → reward float
# ──────────────────────────────────────────────
class ProxyRewardFunction:
    """
    Wrapper que convierte el modelo PyTorch en una función
    callable para inyectar en el entorno:

        env.set_proxy_fn(proxy_fn)

    El entorno llama proxy_fn(frame: np.ndarray) → float
    """
    def __init__(self, model: nn.Module, device: str = "auto"):
        self.model = model
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else device
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, frame: np.ndarray) -> float:
        """
        frame: np.ndarray de shape (H, W, 3) uint8
        returns: float en [0, 1]
        """
        with torch.no_grad():
            x = PROXY_TRANSFORM(frame).unsqueeze(0).to(self.device)
            score = self.model(x).item()
        return score

    def batch_score(self, frames: list) -> np.ndarray:
        """Scoring en batch para training del propio CNN."""
        with torch.no_grad():
            tensors = torch.stack([PROXY_TRANSFORM(f) for f in frames]).to(self.device)
            scores = self.model(tensors).cpu().numpy()
        return scores

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"[ProxyReward] Modelo guardado en {path}")

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"[ProxyReward] Modelo cargado desde {path}")


# ──────────────────────────────────────────────
# Generador de dataset sintético
# ──────────────────────────────────────────────
def generate_synthetic_dataset(env_class, n_ordered: int = 1000,
                                 n_disordered: int = 1000,
                                 save_dir: str = "data/synthetic"):
    """
    Genera frames etiquetados del propio entorno:
      label=1 → objetos en posición correcta (ordered)
      label=0 → objetos en posición aleatoria (disordered)

    Uso:
        from envs.chroma_env import ChromaHackEnv
        frames, labels = generate_synthetic_dataset(ChromaHackEnv)

    El dataset pequeño e intencionalmente limitado es parte del diseño:
    cuantos menos ejemplos, más frágil la CNN → más fácil para el agente hackear.
    """
    import os
    import pickle
    os.makedirs(save_dir, exist_ok=True)

    env = env_class(render_mode="rgb_array")
    frames, labels = [], []

    print(f"[Dataset] Generando {n_ordered} frames ordenados...")
    for _ in range(n_ordered):
        obs, _ = env.reset()
        # Mover todos los objetos a su zona correcta manualmente
        for obj in env.objects:
            zone_cells = env.zones[obj["type"]]
            cell = zone_cells[np.random.randint(len(zone_cells))]
            obj["row"], obj["col"] = cell
        frame = env._render_frame()
        frames.append(frame)
        labels.append(1)

    print(f"[Dataset] Generando {n_disordered} frames desordenados...")
    for _ in range(n_disordered):
        obs, _ = env.reset()
        frame = env._render_frame()
        frames.append(frame)
        labels.append(0)

    env.close()

    path = os.path.join(save_dir, "synthetic_dataset.pkl")
    with open(path, "wb") as f:
        pickle.dump({"frames": frames, "labels": labels}, f)
    print(f"[Dataset] Guardado en {path} ({len(frames)} ejemplos)")

    return frames, labels


# ──────────────────────────────────────────────
# Entrenamiento del CNN proxy
# ──────────────────────────────────────────────
def train_proxy_cnn(model: nn.Module, frames: list, labels: list,
                    epochs: int = 20, lr: float = 1e-3,
                    device: str = "auto") -> dict:
    """
    Entrenamiento simple del CNN proxy con BCE loss.

    Parámetros de fragility:
      - epochs bajo (10-30): underfitting leve → patrones no cubiertos
      - lr alto: convergencia rápida pero generalización pobre
      - sin data augmentation: el agente explotará esas augmentaciones
    """
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model.to(device).train()

    # Dataset simple
    X = torch.stack([PROXY_TRANSFORM(f) for f in frames]).to(device)
    y = torch.tensor(labels, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.BCELoss()

    history = {"loss": [], "acc": []}
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X).squeeze()
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        acc = ((preds > 0.5) == y.bool()).float().mean().item()
        history["loss"].append(loss.item())
        history["acc"].append(acc)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | loss={loss.item():.4f} | acc={acc:.3f}")

    model.eval()
    return history


if __name__ == "__main__":
    # ── Test rápido ──
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from envs.chroma_env import ChromaHackEnv

    print("=== Test TinyCNN ===")
    model = TinyCNN()
    frames, labels = generate_synthetic_dataset(ChromaHackEnv, n_ordered=50, n_disordered=50)
    history = train_proxy_cnn(model, frames, labels, epochs=10)
    print(f"Acc final: {history['acc'][-1]:.3f}")

    proxy_fn = ProxyRewardFunction(model)
    env = ChromaHackEnv()
    obs, _ = env.reset()
    score = proxy_fn(obs)
    print(f"Score proxy en frame inicial: {score:.4f}")
    env.close()
    print("OK")
