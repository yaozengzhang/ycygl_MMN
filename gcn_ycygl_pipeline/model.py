from __future__ import annotations

import torch
from torch import nn


class FeatureProjection(nn.Module):
    def __init__(self, gcn_dim: int, image_dim: int, ela_dim: int, hidden_dim: int):
        super().__init__()
        self.gcn_proj = nn.Linear(gcn_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.ela_proj = nn.Linear(ela_dim, hidden_dim)

    def forward(self, gcn: torch.Tensor, image: torch.Tensor, ela: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                self.gcn_proj(gcn),
                self.image_proj(image),
                self.ela_proj(ela),
            ],
            dim=1,
        )


class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(tokens, tokens, tokens)
        return self.norm(tokens + attended).mean(dim=1)


class ClassifierHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.classifier(fused).squeeze(-1)


class ThreeFeatureRumorModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.5,
        gcn_dim: int = 200,
        image_dim: int = 1024,
        ela_dim: int = 1024,
    ):
        super().__init__()
        self.projector = FeatureProjection(gcn_dim, image_dim, ela_dim, hidden_dim)
        self.fusion = AttentionFusion(hidden_dim, num_heads, dropout)
        self.classifier = ClassifierHead(hidden_dim, dropout)

    def forward(self, gcn: torch.Tensor, image: torch.Tensor, ela: torch.Tensor) -> torch.Tensor:
        tokens = self.projector(gcn, image, ela)
        fused = self.fusion(tokens)
        return self.classifier(fused)
