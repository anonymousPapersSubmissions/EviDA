# src/models/classifier.py
"""
Classification heads and explanation generator for fake news detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from transformers import AutoConfig, AutoModel
import logging

logger = logging.getLogger(__name__)

# Minimum alpha value — keeps Dirichlet parameters away from zero so that
# downstream lgamma / digamma calls in the loss are always well-defined.
ALPHA_MIN = 1e-6


class EvidentialClassifierV2(nn.Module):
    """
    Evidential classifier that outputs Dirichlet parameters for
    uncertainty quantification.

    Architecture: input → 3-layer MLP → evidence (softplus) → alpha = e + 1

    Reference: "Evidential Deep Learning to Quantify Classification
    Uncertainty" (Sensoy et al., NeurIPS 2018).
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        self.evidence_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 4, num_classes),
        )

        logger.info(
            f"EvidentialClassifierV2 initialized: {input_dim} → {num_classes} classes"
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (N, input_dim)

        Returns dict with:
            evidence  (N, C)  — non-negative evidence per class
            alpha     (N, C)  — Dirichlet parameters (evidence + 1, clamped ≥ ALPHA_MIN)
            S         (N, 1)  — Dirichlet strength = sum(alpha)
            prob      (N, C)  — expected class probability = alpha / S
            logits    (N, C)  — log(prob)  [CE-loss compatibility]
            uncertainty (N,1) — vacuity = K / S  (↑ = less evidence)
            vacuity   (N, 1)  — same as uncertainty
            entropy   (N, 1)  — H[prob]  (distributional uncertainty)
        """
        # Raw evidence — softplus ensures non-negative output
        evidence = F.softplus(self.evidence_net(x))      # (N, C)

        # Dirichlet parameters: alpha = e + 1, strictly > 1
        # Extra clamp is a safety net for numerical edge cases on CUDA
        alpha = torch.clamp(evidence + 1.0, min=ALPHA_MIN)  # (N, C)

        S    = alpha.sum(dim=1, keepdim=True)               # (N, 1)
        prob = alpha / S                                     # (N, C)

        # Vacuity: u = K / S  ∈ (0, 1]  — high when evidence is low
        vacuity = self.num_classes / S                       # (N, 1)

        # Entropy of the expected distribution
        entropy = -(prob * torch.log(prob + 1e-10)).sum(dim=1, keepdim=True)  # (N, 1)

        return {
            'evidence':    evidence,
            'alpha':       alpha,
            'S':           S,
            'prob':        prob,
            'logits':      torch.log(prob + 1e-10),  # log-probability for CE compat
            'uncertainty': vacuity,
            'vacuity':     vacuity,
            'entropy':     entropy,
        }


class StandardClassifier(nn.Module):
    """
    Baseline classification head — no uncertainty quantification.
    """

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes),
        )
        logger.info(
            f"StandardClassifier initialized: {input_dim} → {num_classes} classes"
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (N, input_dim)

        Returns dict with:
            logits (N, C)
            prob   (N, C)
        """
        logits = self.classifier(x)
        return {
            'logits': logits,
            'prob':   F.softmax(logits, dim=-1),
        }


class ExplanationGenerator(nn.Module):
    """
    Optional seq2seq head that generates a natural-language explanation
    conditioned on the text encoder's hidden states.

    Disabled by default (config.use_explanation_head = False).
    """

    def __init__(self, config, tokenizer_vocab_size: int):
        super().__init__()
        self.config = config

        if not config.use_explanation_head:
            logger.info("Explanation head disabled")
            self.decoder = None
            return

        try:
            decoder_config = AutoConfig.from_pretrained(config.text_encoder_name)
            decoder_config.is_decoder          = True
            decoder_config.add_cross_attention = True
            decoder_config.num_hidden_layers   = config.decoder_num_layers
            decoder_config.hidden_size         = config.decoder_hidden_dim

            self.decoder           = AutoModel.from_config(decoder_config)
            self.output_projection = nn.Linear(
                decoder_config.hidden_size, tokenizer_vocab_size
            )
            self.layer_norm = nn.LayerNorm(decoder_config.hidden_size)

            logger.info(
                f"ExplanationGenerator initialized: "
                f"{config.decoder_num_layers} layers, "
                f"hidden={config.decoder_hidden_dim}, "
                f"vocab={tokenizer_vocab_size}"
            )

        except Exception as exc:
            logger.warning(f"ExplanationGenerator init failed: {exc}")
            logger.warning("Falling back to disabled explanation head.")
            self.decoder = None

    # ------------------------------------------------------------------

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_hidden_states: (N, src_len, hidden)
            encoder_attention_mask:(N, src_len)
            decoder_input_ids:     (N, tgt_len)
            decoder_attention_mask:(N, tgt_len)

        Returns:
            logits: (N, tgt_len, vocab_size)
        """
        if self.decoder is None:
            N, T = decoder_input_ids.shape
            return torch.zeros(N, T, 1000, device=decoder_input_ids.device)

        out = self.decoder(
            input_ids               = decoder_input_ids,
            attention_mask          = decoder_attention_mask,
            encoder_hidden_states   = encoder_hidden_states,
            encoder_attention_mask  = encoder_attention_mask,
            return_dict             = True,
        )
        return self.output_projection(self.layer_norm(out.last_hidden_state))

    # ------------------------------------------------------------------

    def generate(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        tokenizer,
        max_length: int = 128,
        temperature: float = 1.0,
    ) -> Optional[torch.Tensor]:
        """
        Greedy decoding.  Returns (N, gen_len) token IDs or None if disabled.
        """
        if self.decoder is None:
            logger.warning("Explanation generator is disabled.")
            return None

        device = encoder_hidden_states.device
        N      = encoder_hidden_states.size(0)

        ids = torch.full((N, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            logits     = self.forward(
                encoder_hidden_states  = encoder_hidden_states,
                encoder_attention_mask = encoder_attention_mask,
                decoder_input_ids      = ids,
                decoder_attention_mask = torch.ones_like(ids),
            )
            next_tok   = torch.argmax(logits[:, -1, :] / temperature, dim=-1, keepdim=True)
            ids        = torch.cat([ids, next_tok], dim=1)

            if (next_tok == tokenizer.eos_token_id).all():
                break

        return ids