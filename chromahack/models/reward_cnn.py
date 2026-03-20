"""Proxy reward CNN models and checkpoint helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T

PROXY_TRANSFORM = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
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


class ResNetProxy(nn.Module):
    def __init__(self, pretrained_backbone: str = "resnet18", freeze_backbone: bool = True):
        super().__init__()
        if pretrained_backbone != "resnet18":
            raise ValueError("Only resnet18 is supported.")

        try:
            backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        except Exception:
            backbone = tv_models.resnet18(weights=None)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
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
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_layers(self, n_layers: int = 2):
        children = list(self.backbone.children())
        for child in children[-n_layers:]:
            for param in child.parameters():
                param.requires_grad = True

    def unfreeze_all(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features).squeeze(-1)


def infer_proxy_mode(model: nn.Module) -> str:
    if isinstance(model, TinyCNN):
        return "tiny"
    if isinstance(model, ResNetProxy):
        return "resnet"
    raise TypeError(f"Unsupported proxy model type: {type(model)!r}")


def build_proxy_model(
    mode: str,
    *,
    freeze_backbone: bool = True,
    pretrained_path: str | None = None,
) -> nn.Module:
    if mode == "tiny":
        model = TinyCNN()
    elif mode == "resnet":
        model = ResNetProxy(freeze_backbone=freeze_backbone)
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
            model.backbone.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"Unknown proxy mode: {mode}")
    return model


def save_proxy_checkpoint(model: nn.Module, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mode": infer_proxy_mode(model), "state_dict": model.state_dict()}, path)


def load_proxy_checkpoint(path: str, *, device: str = "auto"):
    resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    checkpoint = torch.load(path, map_location=resolved_device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        mode = checkpoint.get("mode", "tiny")
        state_dict = checkpoint["state_dict"]
    else:
        mode = "tiny"
        state_dict = checkpoint
    return mode, state_dict


class ProxyRewardFunction:
    def __init__(self, model: nn.Module, device: str = "auto"):
        self.model = model
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, frame: np.ndarray) -> float:
        with torch.no_grad():
            tensor = PROXY_TRANSFORM(frame).unsqueeze(0).to(self.device)
            return float(self.model(tensor).item())

    def batch_score(self, frames: list[np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            batch = torch.stack([PROXY_TRANSFORM(frame) for frame in frames]).to(self.device)
            return self.model(batch).detach().cpu().numpy()

    def save(self, path: str) -> None:
        save_proxy_checkpoint(self.model, path)

    def load(self, path: str) -> None:
        _, state_dict = load_proxy_checkpoint(path, device=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()


def load_proxy_reward(path: str, *, device: str = "auto") -> tuple[ProxyRewardFunction, str]:
    mode, state_dict = load_proxy_checkpoint(path, device=device)
    model = build_proxy_model(mode)
    reward_fn = ProxyRewardFunction(model, device=device)
    reward_fn.model.load_state_dict(state_dict)
    reward_fn.model.eval()
    return reward_fn, mode
