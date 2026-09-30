"""
Loss functions for Cross-Domain Fake News Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

# Minimum value for Dirichlet alpha parameters.
# alpha = evidence + 1 via softplus, so the true floor is ~1.0,
# but we clamp to ALPHA_MIN before every lgamma / digamma call
# as a safety net against numerical edge cases on CUDA.
ALPHA_MIN = 1e-6


class EvidentialLossV2(nn.Module):
    """
    Enhanced Evidential Loss with domain-specific priors and annealing.

    Based on "Evidential Deep Learning to Quantify Classification Uncertainty"
    (Sensoy et al., NeurIPS 2018) with extensions for domain adaptation.

    Key fixes vs. the original implementation
    ------------------------------------------
    1. alpha is clamped to >= ALPHA_MIN before every call to
       torch.lgamma / torch.digamma — those functions return NaN / raise a
       CUDA "invalid argument" error for inputs <= 0.
    2. The per-sample Python loop over domain_names has been replaced with
       a fully-vectorised batched KL computation.  This eliminates scalar
       0-dim tensor operations (another source of the CUDA error) and is
       also ~N times faster for large batches.
    3. S_prior is always kept as a (N,1) tensor, never a 0-dim scalar.
    """

    def __init__(
        self,
        num_classes: int,
        kl_weight: float = 0.01,
        annealing_start: int = 10,
        annealing_step: int = 10,
        use_domain_priors: bool = True,
        prior_concentration: float = 1.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.kl_weight = kl_weight
        self.annealing_start = annealing_start
        self.annealing_step = annealing_step
        self.use_domain_priors = use_domain_priors
        self.prior_concentration = prior_concentration

        self.current_epoch = 0

        # Domain-specific learnable priors: name -> Parameter of shape (C,)
        if use_domain_priors:
            self.domain_priors = nn.ParameterDict()

    def set_epoch(self, epoch: int):
        """Update current epoch for KL annealing."""
        self.current_epoch = epoch

    def add_domain_prior(self, domain_name: str, num_classes: int):
        """Register a learnable prior for *domain_name* if not already present."""
        if self.use_domain_priors and domain_name not in self.domain_priors:
            self.domain_priors[domain_name] = nn.Parameter(
                torch.ones(num_classes) * self.prior_concentration
            )

    def get_annealing_coefficient(self) -> float:
        """Linearly ramp KL weight from 0 → 1 over *annealing_step* epochs."""
        if self.current_epoch < self.annealing_start:
            return 0.0
        return min(
            1.0,
            (self.current_epoch - self.annealing_start) / max(self.annealing_step, 1)
        )

    # ------------------------------------------------------------------
    # KL divergence between two Dirichlet distributions (batched)
    # KL[ Dir(alpha) || Dir(prior_alpha) ]
    # ------------------------------------------------------------------
    @staticmethod
    def _dirichlet_kl(
        alpha: torch.Tensor,       # (N, C)  — already clamped
        prior_alpha: torch.Tensor  # (N, C)  — already clamped
    ) -> torch.Tensor:             # (N,)
        """
        Closed-form KL divergence between two Dirichlet distributions,
        computed in a fully vectorised manner with no Python loops.

        KL = lgamma(S) - lgamma(S0)
             - sum_c [ lgamma(a_c) - lgamma(a0_c) ]
             + sum_c [ (a_c - a0_c) * (digamma(a_c) - digamma(S)) ]

        where S  = sum(alpha,  dim=1)
              S0 = sum(prior_alpha, dim=1)

        IMPORTANT: torch.lgamma and torch.digamma are NOT supported for
        float16 or bfloat16 on CUDA — they raise "invalid argument".
        We cast to float32 for the gamma ops, then cast the result back
        to the original dtype so the rest of the graph is unaffected.
        """
        orig_dtype = alpha.dtype

        # Cast to float32 — required for lgamma/digamma on CUDA
        a  = alpha.float().clamp(min=ALPHA_MIN)
        a0 = prior_alpha.float().clamp(min=ALPHA_MIN)

        S  = a.sum(dim=1, keepdim=True)   # (N, 1)
        S0 = a0.sum(dim=1, keepdim=True)  # (N, 1)

        kl = (
            torch.lgamma(S)  - torch.lgamma(S0)
            - (torch.lgamma(a).sum(dim=1, keepdim=True)
               - torch.lgamma(a0).sum(dim=1, keepdim=True))
            + ((a - a0) * (torch.digamma(a) - torch.digamma(S))).sum(dim=1, keepdim=True)
        )  # (N, 1)

        return kl.squeeze(1).to(orig_dtype)  # (N,)

    def forward(
        self,
        alpha: torch.Tensor,
        targets: torch.Tensor,
        domain_ids: Optional[torch.Tensor] = None,
        domain_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            alpha:        (N, C) Dirichlet parameters (evidence + 1, all > 0).
            targets:      (N,)   integer class labels in [0, C).
            domain_ids:   (N,)   integer domain indices (unused here, kept for
                                 API compatibility with CombinedLossV2).
            domain_names: list of N domain-name strings; used to look up
                          per-domain learnable priors.

        Returns:
            dict with keys: total, ce, mse, kl, annealing_coef
        """
        device = alpha.device
        N = alpha.size(0)

        # ── Cast to float32 + clamp ────────────────────────────────────────
        # torch.lgamma / digamma raise "CUDA invalid argument" for fp16/bf16.
        # We promote once here so every downstream op in this forward is fp32.
        alpha = torch.clamp(alpha.float(), min=ALPHA_MIN)

        # One-hot targets
        y = F.one_hot(targets, self.num_classes).float()  # (N, C)

        # Dirichlet strength & expected probability
        S    = alpha.sum(dim=1, keepdim=True)   # (N, 1)
        prob = alpha / S                         # (N, C)

        # ── Loss 1: cross-entropy on expected probabilities ────────────────
        ce_loss = -torch.sum(y * torch.log(prob + 1e-10), dim=1)  # (N,)

        # ── Loss 2: MSE (training-stability regulariser) ───────────────────
        mse_loss = torch.sum((y - prob) ** 2, dim=1)  # (N,)

        # ── Loss 3: KL divergence ──────────────────────────────────────────
        if self.use_domain_priors and domain_names is not None:
            # Ensure all encountered domains have a prior registered.
            for dn in set(domain_names):
                self.add_domain_prior(dn, self.num_classes)

            # Build (N, C) prior tensor — vectorised, no per-sample Python loop.
            prior_alpha = torch.stack(
                [self.domain_priors[dn].to(device) for dn in domain_names],
                dim=0
            )  # (N, C)
        else:
            # Uniform prior broadcast to (N, C)
            prior_alpha = torch.full(
                (N, self.num_classes),
                self.prior_concentration,
                dtype=alpha.dtype,
                device=device
            )

        # Clamp prior as well before lgamma / digamma
        prior_alpha = torch.clamp(prior_alpha, min=ALPHA_MIN)

        kl_loss = self._dirichlet_kl(alpha, prior_alpha)  # (N,)

        # ── Annealing ─────────────────────────────────────────────────────
        annealing_coef = self.get_annealing_coefficient()

        total_loss = torch.mean(
            ce_loss
            + 0.1 * mse_loss
            + annealing_coef * self.kl_weight * kl_loss
        )

        return {
            'total':          total_loss,
            'ce':             torch.mean(ce_loss),
            'mse':            torch.mean(mse_loss),
            'kl':             torch.mean(kl_loss),
            'annealing_coef': annealing_coef,
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss    = F.cross_entropy(inputs, targets, reduction='none')
        p_t        = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples.
    CB = (1 - beta) / (1 - beta^n)
    """

    def __init__(
        self,
        samples_per_class: List[int],
        beta: float = 0.9999,
        gamma: float = 2.0,
        loss_type: str = 'focal'
    ):
        super().__init__()
        self.gamma     = gamma
        self.loss_type = loss_type

        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights       = (1.0 - beta) / np.array(effective_num)
        weights       = weights / weights.sum() * len(samples_per_class)

        self.weights = torch.tensor(weights, dtype=torch.float32)
        logger.info(f"Class-balanced weights: {self.weights}")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = self.weights.to(inputs.device)[targets]

        if self.loss_type == 'focal':
            ce_loss    = F.cross_entropy(inputs, targets, reduction='none')
            p_t        = torch.exp(-ce_loss)
            focal_loss = (1 - p_t) ** self.gamma * ce_loss
            return (weights * focal_loss).mean()

        return F.cross_entropy(inputs, targets, weight=self.weights.to(inputs.device))


class DomainAdversarialLoss(nn.Module):
    """
    Domain Adversarial Loss — encourages domain-invariant features.
    """

    def __init__(self, num_domains: int):
        super().__init__()
        self.num_domains = num_domains
        self.criterion   = nn.CrossEntropyLoss()

    def forward(
        self,
        domain_logits: torch.Tensor,
        domain_labels: torch.Tensor
    ) -> torch.Tensor:
        return self.criterion(domain_logits, domain_labels)


class CombinedLossV2(nn.Module):
    """
    Combined Loss for Cross-Domain Fake News Detection.

    Combines:
      - Classification loss  (Evidential or standard CE / Focal)
      - Domain adversarial loss
      - Explanation loss  (disabled by default)
    """

    def __init__(
        self,
        config,
        num_domains: int,
        class_weights: Optional[torch.Tensor] = None,
        source_weights: Optional[Dict[str, float]] = None,
        classification_weight: float = 1.0,
        domain_weight: float = 0.1
    ):
        super().__init__()
        self.config                = config
        self.num_domains           = num_domains
        self.source_weights        = source_weights
        self.classification_weight = classification_weight
        self.domain_weight         = domain_weight

        # ── Classification loss ───────────────────────────────────────────
        if config.evidential.use_evidential:
            self.classification_loss = EvidentialLossV2(
                num_classes          = getattr(config, 'num_classes', 2),
                kl_weight            = config.evidential.kl_weight,
                annealing_start      = config.evidential.annealing_start,
                annealing_step       = config.evidential.annealing_step,
                use_domain_priors    = config.evidential.use_domain_specific_priors,
                prior_concentration  = config.evidential.prior_concentration,
            )
            logger.info("Using Enhanced Evidential Loss")

        elif config.imbalance_config.class_balance_strategy == 'focal_loss':
            self.classification_loss = FocalLoss(
                alpha = config.imbalance_config.focal_loss_alpha,
                gamma = config.imbalance_config.focal_loss_gamma,
            )
            logger.info("Using Focal Loss")

        elif class_weights is not None:
            self.classification_loss = nn.CrossEntropyLoss(weight=class_weights)
            logger.info(f"Using weighted CrossEntropy: {class_weights}")

        else:
            self.classification_loss = nn.CrossEntropyLoss()
            logger.info("Using standard CrossEntropy")

        # ── Domain adversarial loss ───────────────────────────────────────
        if config.domain_adaptation.use_domain_adversarial:
            self.domain_loss = DomainAdversarialLoss(num_domains)
            logger.info("Using Domain Adversarial Loss")
        else:
            self.domain_loss = None

        # ── Explanation loss (disabled by default) ────────────────────────
        self.explanation_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def set_epoch(self, epoch: int):
        """Propagate epoch to evidential loss for KL annealing."""
        if isinstance(self.classification_loss, EvidentialLossV2):
            self.classification_loss.set_epoch(epoch)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        domain_labels: torch.Tensor,
        sources: Optional[List[str]] = None,
        explanation_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs:         Model output dict (must contain 'classification').
            labels:          (N,) ground-truth class labels.
            domain_labels:   (N,) domain indices.
            sources:         List of N source-name strings (for domain priors).
            explanation_ids: Not used; kept for API compatibility.

        Returns:
            dict of named scalar losses plus 'total'.
        """
        losses: Dict[str, torch.Tensor] = {}

        # ── Classification ────────────────────────────────────────────────
        if isinstance(self.classification_loss, EvidentialLossV2):
            cls_losses = self.classification_loss(
                alpha        = outputs['classification']['alpha'],
                targets      = labels,
                domain_ids   = domain_labels,
                domain_names = sources,
            )
            losses['classification'] = cls_losses['total'] * self.classification_weight
            losses['cls_ce']         = cls_losses['ce']
            losses['cls_mse']        = cls_losses['mse']
            losses['cls_kl']         = cls_losses['kl']
            losses['annealing']      = cls_losses['annealing_coef']

        else:
            logits = outputs['classification']['logits']

            if self.source_weights is not None and sources is not None:
                sw = torch.tensor(
                    [self.source_weights.get(s, 1.0) for s in sources],
                    device=logits.device, dtype=torch.float32
                )
                cls_loss = (F.cross_entropy(logits, labels, reduction='none') * sw).mean()
            else:
                cls_loss = self.classification_loss(logits, labels)

            losses['classification'] = cls_loss * self.classification_weight

        # ── Domain adversarial ────────────────────────────────────────────
        if self.domain_loss is not None and 'domain_logits' in outputs:
            losses['domain'] = (
                self.domain_loss(outputs['domain_logits'], domain_labels)
                * self.domain_weight
            )

        # ── Total (exclude the scalar annealing coefficient) ──────────────
        losses['total'] = sum(
            v for k, v in losses.items() if k != 'annealing'
        )

        return losses