# src/models/encoders.py
"""
Text and Vision encoders for the multimodal fake-news detector.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
import timm
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """
    Multilingual text encoder (XLM-RoBERTa or any HuggingFace transformer).

    Returns both a projected CLS-pooled vector and the full token-level
    hidden states (needed by the explanation generator).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        try:
            self.encoder     = AutoModel.from_pretrained(config.text_encoder_name)
            self.encoder_dim = self.encoder.config.hidden_size
            logger.info(f"TextEncoder loaded: {config.text_encoder_name}")
        except Exception as exc:
            logger.error(f"Failed to load text encoder: {exc}")
            raise

        self.projection = nn.Sequential(
            nn.Linear(self.encoder_dim, config.text_intermediate_dim),
            nn.LayerNorm(config.text_intermediate_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.text_intermediate_dim, config.text_output_dim),
            nn.LayerNorm(config.text_output_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids:      (N, seq_len)
            attention_mask: (N, seq_len)

        Returns:
            pooled:   (N, text_output_dim)   projected CLS representation
            sequence: (N, seq_len, enc_dim)  full token hidden states
        """
        out      = self.encoder(
            input_ids       = input_ids,
            attention_mask  = attention_mask,
            return_dict     = True,
        )
        pooled   = self.projection(out.pooler_output)   # (N, text_output_dim)
        sequence = out.last_hidden_state                 # (N, seq_len, enc_dim)
        return pooled, sequence


class VisionEncoder(nn.Module):
    """
    Vision encoder using Swin Transformer (or any timm model).

    The classification head is removed; the pooled feature vector is
    projected to *vision_output_dim*.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        try:
            self.encoder = timm.create_model(
                config.vision_encoder_name,
                pretrained  = True,
                num_classes = 0,  # remove classifier head
            )
            self.encoder_dim = self.encoder.num_features
            logger.info(
                f"VisionEncoder loaded: {config.vision_encoder_name} "
                f"(feat_dim={self.encoder_dim})"
            )
        except Exception as exc:
            logger.error(f"Failed to load vision encoder: {exc}")
            raise

        self.projection = nn.Sequential(
            nn.Linear(self.encoder_dim, config.vision_output_dim),
            nn.LayerNorm(config.vision_output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (N, 3, H, W)

        Returns:
            features: (N, vision_output_dim)
        """
        return self.projection(self.encoder(images))