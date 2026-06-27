from __future__ import annotations

import torch
from torch import nn


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
        self.gcn_proj = nn.Linear(gcn_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.ela_proj = nn.Linear(ela_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, gcn: torch.Tensor, image: torch.Tensor, ela: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack(
            [
                self.gcn_proj(gcn),
                self.image_proj(image),
                self.ela_proj(ela),
            ],
            dim=1,
        )
        attended, _ = self.attention(tokens, tokens, tokens)
        fused = self.norm(tokens + attended).mean(dim=1)
        return self.classifier(fused).squeeze(-1)

