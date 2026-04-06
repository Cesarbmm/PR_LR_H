"""Graph-style feature extraction for Frontier dict observations."""

from __future__ import annotations

from typing import Any

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool

    _HAS_PYG = True
except Exception:  # pragma: no cover - optional dependency
    GATv2Conv = None
    global_mean_pool = None
    _HAS_PYG = False


class FrontierGraphFeatureExtractor(BaseFeaturesExtractor):
    """Encode Frontier dict observations with relational actor processing.

    When PyTorch Geometric is available, actor slots are processed with GATv2 layers and
    mean pooling. Without PyG, the extractor falls back to an adjacency-aware message
    passing block so the local baseline remains runnable.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        features_dim: int = 256,
        actor_hidden_dim: int = 96,
        zone_hidden_dim: int = 48,
        use_pyg: bool = True,
    ) -> None:
        super().__init__(observation_space, features_dim)
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("FrontierGraphFeatureExtractor requires a Dict observation space")

        self.use_pyg = bool(use_pyg and _HAS_PYG)
        agent_dim = int(observation_space.spaces["agent"].shape[0])
        actor_dim = int(observation_space.spaces["actors"].shape[-1])
        zone_dim = int(observation_space.spaces["zones"].shape[-1])
        aggregate_dim = int(observation_space.spaces["aggregates"].shape[0])

        self.actor_input = nn.Sequential(
            nn.Linear(actor_dim, actor_hidden_dim),
            nn.LayerNorm(actor_hidden_dim),
            nn.ReLU(),
        )
        if self.use_pyg:
            self.actor_gnn_1 = GATv2Conv(actor_hidden_dim, actor_hidden_dim, heads=2, concat=False)
            self.actor_gnn_2 = GATv2Conv(actor_hidden_dim, actor_hidden_dim, heads=2, concat=False)
        else:
            self.actor_update = nn.Sequential(
                nn.Linear(actor_hidden_dim * 2, actor_hidden_dim),
                nn.ReLU(),
                nn.Linear(actor_hidden_dim, actor_hidden_dim),
                nn.ReLU(),
            )

        self.zone_encoder = nn.Sequential(
            nn.Linear(zone_dim, zone_hidden_dim),
            nn.LayerNorm(zone_hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(agent_dim + aggregate_dim + actor_hidden_dim + zone_hidden_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    @staticmethod
    def _masked_mean(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1)
        total = (encoded * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return total / denom

    def _encode_actors_fallback(
        self,
        actors: torch.Tensor,
        actor_mask: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.actor_input(actors)
        weights = adjacency * actor_mask.unsqueeze(1) * actor_mask.unsqueeze(2)
        norm = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neighbour_summary = torch.bmm(weights, encoded) / norm
        updated = self.actor_update(torch.cat([encoded, neighbour_summary], dim=-1))
        return self._masked_mean(updated, actor_mask)

    def _encode_actors_pyg(
        self,
        actors: torch.Tensor,
        actor_mask: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        batch_embeddings: list[torch.Tensor] = []
        for batch_index in range(actors.shape[0]):
            active = actor_mask[batch_index] > 0.5
            if int(active.sum().item()) == 0:
                batch_embeddings.append(torch.zeros((1, self.actor_input[0].out_features), device=actors.device))
                continue
            node_features = self.actor_input(actors[batch_index][active])
            dense_adjacency = adjacency[batch_index][active][:, active]
            edge_index = dense_adjacency.nonzero(as_tuple=False).T.contiguous()
            if edge_index.numel() == 0:
                edge_index = torch.arange(node_features.shape[0], device=actors.device, dtype=torch.long).repeat(2, 1)
            latent = self.actor_gnn_1(node_features, edge_index)
            latent = torch.relu(latent)
            latent = self.actor_gnn_2(latent, edge_index)
            batch_vector = torch.zeros(latent.shape[0], dtype=torch.long, device=actors.device)
            pooled = global_mean_pool(latent, batch_vector)
            batch_embeddings.append(pooled)
        return torch.cat(batch_embeddings, dim=0)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        agent = observations["agent"]
        actors = observations["actors"]
        actor_mask = observations["actor_mask"]
        zones = observations["zones"]
        zone_mask = observations["zone_mask"]
        adjacency = observations["adjacency"]
        aggregates = observations["aggregates"]

        if self.use_pyg:
            actor_embedding = self._encode_actors_pyg(actors, actor_mask, adjacency)
        else:
            actor_embedding = self._encode_actors_fallback(actors, actor_mask, adjacency)
        zone_embedding = self._masked_mean(self.zone_encoder(zones), zone_mask)
        combined = torch.cat([agent, aggregates, actor_embedding, zone_embedding], dim=-1)
        return self.head(combined)


def build_frontier_gnn_policy_kwargs(
    *,
    features_dim: int,
    actor_hidden_dim: int = 96,
    zone_hidden_dim: int = 48,
    use_pyg: bool = True,
) -> dict[str, Any]:
    """Helper used by the training CLI to configure the custom extractor."""

    return {
        "features_extractor_class": FrontierGraphFeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": features_dim,
            "actor_hidden_dim": actor_hidden_dim,
            "zone_hidden_dim": zone_hidden_dim,
            "use_pyg": use_pyg,
        },
        "net_arch": [],
    }
