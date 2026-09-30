# src/models/fusion.py
"""
Cross-modal fusion for text and vision features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MultiHeadCrossAttention(nn.Module):
    """
    Standard scaled dot-product multi-head cross-attention.

    query attends to key/value from a different modality.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(dim, dim)
        self.k_proj   = nn.Linear(dim, dim)
        self.v_proj   = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout      = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (N, q_len, dim)
            key:   (N, k_len, dim)
            value: (N, k_len, dim)
            mask:  (N, q_len, k_len) optional boolean mask (0 → -inf)

        Returns:
            (N, q_len, dim)
        """
        N = query.size(0)

        def _split(t: torch.Tensor) -> torch.Tensor:
            return t.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q = _split(self.q_proj(query))   # (N, H, q, d)
        k = _split(self.k_proj(key))     # (N, H, k, d)
        v = _split(self.v_proj(value))   # (N, H, k, d)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (N, H, q, k)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = self.attn_dropout(F.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).contiguous()   # (N, q, H, d)
        out = out.view(N, -1, self.num_heads * self.head_dim)
        return self.dropout(self.out_proj(out))


class MultimodalFusion(nn.Module):
    """
    Bidirectional cross-modal fusion.

    Text queries vision → text-attended features
    Vision queries text → vision-attended features
    Concatenation → MLP → fused vector of shape (N, fusion_dim)
    """

    def __init__(self, config):
        super().__init__()

        assert config.text_output_dim == config.vision_output_dim, (
            "text_output_dim and vision_output_dim must match for cross-attention."
        )

        feature_dim = config.text_output_dim

        self.text_to_vision_attn = MultiHeadCrossAttention(
            dim       = feature_dim,
            num_heads = config.num_attention_heads,
            dropout   = config.dropout_rate,
        )
        self.vision_to_text_attn = MultiHeadCrossAttention(
            dim       = feature_dim,
            num_heads = config.num_attention_heads,
            dropout   = config.dropout_rate,
        )

        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.fusion_dim, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_features:   (N, text_output_dim)
            vision_features: (N, vision_output_dim)

        Returns:
            fused: (N, fusion_dim)
        """
        # Add a length-1 sequence dimension for cross-attention
        t = text_features.unsqueeze(1)    # (N, 1, D)
        v = vision_features.unsqueeze(1)  # (N, 1, D)

        t_att = self.text_to_vision_attn(query=t, key=v, value=v).squeeze(1)  # (N, D)
        v_att = self.vision_to_text_attn(query=v, key=t, value=t).squeeze(1)  # (N, D)

        return self.fusion(torch.cat([t_att, v_att], dim=-1))  # (N, fusion_dim)