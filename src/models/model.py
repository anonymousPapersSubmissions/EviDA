# src/models/model.py
"""
Enhanced Multimodal Fake News Detector with Domain Adaptation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

from .encoders import TextEncoder, VisionEncoder
from .fusion import MultimodalFusion
from .classifier import EvidentialClassifierV2, StandardClassifier, ExplanationGenerator
from .domain_adaptation import (
    GradientReversalLayer,
    DomainDiscriminator,
    DomainSpecificBatchNorm,
    LambdaScheduler,
)

logger = logging.getLogger(__name__)


class MultimodalFakeNewsDetectorV2(nn.Module):
    """
    End-to-end multimodal fake-news detector.

    Components
    ----------
    1. TextEncoder          — XLM-RoBERTa (or similar)
    2. VisionEncoder        — Swin Transformer (timm)
    3. DomainSpecificBatchNorm — optional per-domain LayerNorm
    4. MultimodalFusion     — bidirectional cross-modal attention
    5. EvidentialClassifierV2 / StandardClassifier
    6. GradientReversalLayer + DomainDiscriminator  — optional DANN
    7. ExplanationGenerator — optional seq2seq head
    """

    def __init__(
        self,
        config,
        tokenizer_vocab_size: int,
        source_to_id: Dict[str, int],
    ):
        super().__init__()
        self.config       = config
        self.source_to_id = source_to_id
        self.num_domains  = len(source_to_id)

        logger.info("Initializing MultimodalFakeNewsDetectorV2 …")
        logger.info(f"  Domains ({self.num_domains}): {list(source_to_id.keys())}")

        # ── Encoders ──────────────────────────────────────────────────────
        self.text_encoder   = TextEncoder(config)
        self.vision_encoder = VisionEncoder(config)

        # ── Domain-specific normalisation (optional) ──────────────────────
        if config.domain_adaptation.use_domain_specific_bn:
            self.text_domain_bn = DomainSpecificBatchNorm(
                config.text_output_dim, self.num_domains
            )
            self.vision_domain_bn = DomainSpecificBatchNorm(
                config.vision_output_dim, self.num_domains
            )
            logger.info("Domain-Specific BatchNorm enabled")
        else:
            self.text_domain_bn   = None
            self.vision_domain_bn = None

        # ── Fusion ────────────────────────────────────────────────────────
        self.fusion = MultimodalFusion(config)

        # ── Classification head ───────────────────────────────────────────
        num_classes = getattr(config, 'num_classes', 2)
        if config.evidential.use_evidential:
            self.classifier = EvidentialClassifierV2(
                input_dim   = config.fusion_dim,
                num_classes = num_classes,
            )
            logger.info("Evidential Classifier enabled")
        else:
            self.classifier = StandardClassifier(
                input_dim   = config.fusion_dim,
                num_classes = num_classes,
                dropout     = config.dropout_rate,
            )
            logger.info("Standard Classifier enabled")

        # ── Domain adversarial components (optional) ──────────────────────
        if config.domain_adaptation.use_domain_adversarial:
            self.gradient_reversal = GradientReversalLayer(
                lambda_=config.domain_adaptation.gradient_reversal_lambda
            )
            self.domain_discriminator = DomainDiscriminator(
                input_dim  = config.fusion_dim,
                num_domains = self.num_domains,
                hidden_dim = config.domain_adaptation.discriminator_hidden_dim,
                num_layers = config.domain_adaptation.discriminator_num_layers,
                dropout    = config.domain_adaptation.discriminator_dropout,
            )
            logger.info("Domain Adversarial Training enabled")
        else:
            self.gradient_reversal    = None
            self.domain_discriminator = None

        # ── Explanation generator (optional) ──────────────────────────────
        if config.use_explanation_head:
            self.explanation_generator = ExplanationGenerator(
                config               = config,
                tokenizer_vocab_size = tokenizer_vocab_size,
            )
            logger.info("Explanation Generator enabled")
        else:
            self.explanation_generator = None

        self._init_weights()
        logger.info("Model initialization complete.")

    # ──────────────────────────────────────────────────────────────────────

    def _init_weights(self):
        """Xavier-init all new Linear layers (encoders keep pre-trained weights)."""
        for module in filter(None, [
            self.fusion,
            self.classifier,
            self.domain_discriminator,
        ]):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        if (
            self.explanation_generator is not None
            and self.explanation_generator.decoder is not None
            and hasattr(self.explanation_generator, 'output_projection')
        ):
            nn.init.xavier_uniform_(self.explanation_generator.output_projection.weight)
            nn.init.zeros_(self.explanation_generator.output_projection.bias)

    # ──────────────────────────────────────────────────────────────────────

    def set_gradient_reversal_lambda(self, lambda_: float):
        """Update gradient-reversal λ (called by LambdaScheduler each step)."""
        if self.gradient_reversal is not None:
            self.gradient_reversal.set_lambda(lambda_)

    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
        sources: Optional[List[str]] = None,
        explanation_ids: Optional[torch.Tensor] = None,
        explanation_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids:        (N, seq_len)
            attention_mask:   (N, seq_len)
            images:           (N, 3, H, W)
            sources:          list of N source-name strings (optional)
            explanation_ids:  (N, exp_len) decoder input IDs (optional)
            explanation_mask: (N, exp_len) decoder attention mask (optional)
            return_features:  if True, include intermediate tensors in output

        Returns dict with:
            classification    — classifier output dict
            fused_features    — (N, fusion_dim)
            domain_logits     — (N, num_domains)  if domain adversarial enabled
            explanation_logits— (N, exp_len, vocab) if explanation head enabled
            text_features, vision_features, text_sequence  — if return_features=True
        """
        # ── Domain IDs ────────────────────────────────────────────────────
        if sources is not None:
            domain_ids = torch.tensor(
                [self.source_to_id.get(s, 0) for s in sources],
                dtype=torch.long, device=input_ids.device,
            )
        else:
            domain_ids = torch.zeros(
                input_ids.size(0), dtype=torch.long, device=input_ids.device
            )

        # ── Encode ────────────────────────────────────────────────────────
        text_pooled, text_sequence = self.text_encoder(input_ids, attention_mask)
        vision_features            = self.vision_encoder(images)

        # ── Domain-specific normalisation ─────────────────────────────────
        if self.text_domain_bn is not None:
            text_pooled     = self.text_domain_bn(text_pooled, domain_ids)
        if self.vision_domain_bn is not None:
            vision_features = self.vision_domain_bn(vision_features, domain_ids)

        # ── Fusion ────────────────────────────────────────────────────────
        fused = self.fusion(text_pooled, vision_features)   # (N, fusion_dim)

        # ── Classification ────────────────────────────────────────────────
        outputs: Dict[str, torch.Tensor] = {
            'classification': self.classifier(fused),
            'fused_features': fused,
        }

        # ── Domain prediction (gradient reversal) ─────────────────────────
        if self.domain_discriminator is not None:
            outputs['domain_logits'] = self.domain_discriminator(
                self.gradient_reversal(fused)
            )

        # ── Explanation generation ─────────────────────────────────────────
        if self.explanation_generator is not None and explanation_ids is not None:
            outputs['explanation_logits'] = self.explanation_generator(
                encoder_hidden_states  = text_sequence,
                encoder_attention_mask = attention_mask,
                decoder_input_ids      = explanation_ids,
                decoder_attention_mask = explanation_mask,
            )

        # ── Optional intermediate features ────────────────────────────────
        if return_features:
            outputs['text_features']   = text_pooled
            outputs['vision_features'] = vision_features
            outputs['text_sequence']   = text_sequence

        return outputs

    # ──────────────────────────────────────────────────────────────────────

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)