# src/models/domain_adaptation.py
"""
Domain Adaptation Components for Cross-Domain Fake News Detection

- GradientReversalLayer  : reverses gradients from the domain discriminator
- DomainDiscriminator    : predicts source domain from fused features
- DomainSpecificBatchNorm: per-domain LayerNorm (robust to batch size = 1)
- LambdaScheduler        : schedules the gradient-reversal strength λ
- MMD                    : Maximum Mean Discrepancy (alternative to adversarial loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Gradient Reversal
# ──────────────────────────────────────────────────────────────────────────────

class GradientReversalFunction(Function):
    """
    Custom autograd function that acts as the identity in the forward pass
    and multiplies the upstream gradient by −λ in the backward pass.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Return negated gradient for x; None for lambda_ (not a tensor)
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """
    Thin nn.Module wrapper around GradientReversalFunction.

    Usage::

        grl = GradientReversalLayer(lambda_=1.0)
        domain_features = grl(fused_features)   # forward = identity
        # gradients flowing back are multiplied by -lambda_
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        """Update λ in-place (called by LambdaScheduler each step)."""
        self.lambda_ = lambda_


# ──────────────────────────────────────────────────────────────────────────────
# Domain Discriminator
# ──────────────────────────────────────────────────────────────────────────────

class DomainDiscriminator(nn.Module):
    """
    MLP that predicts which domain a sample comes from.

    When placed after a GradientReversalLayer the combined effect encourages
    the upstream encoder to produce domain-invariant representations.

    Uses LayerNorm instead of BatchNorm so that a batch (or domain sub-batch)
    of size 1 never triggers the "expected more than 1 value per channel" error.
    """

    def __init__(
        self,
        input_dim: int,
        num_domains: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_domains = num_domains

        layers: list = []

        # ── Input layer ───────────────────────────────────────────────────
        layers += [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]

        # ── Hidden layers ─────────────────────────────────────────────────
        for _ in range(max(num_layers - 2, 0)):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]

        # ── Output layer ──────────────────────────────────────────────────
        layers.append(nn.Linear(hidden_dim, num_domains))

        self.discriminator = nn.Sequential(*layers)
        logger.info(
            f"DomainDiscriminator: {input_dim} → {hidden_dim} "
            f"(×{num_layers - 1}) → {num_domains} domains"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, input_dim)

        Returns:
            domain_logits: (N, num_domains)
        """
        return self.discriminator(x)


# ──────────────────────────────────────────────────────────────────────────────
# Domain-Specific Normalisation
# ──────────────────────────────────────────────────────────────────────────────

class DomainSpecificBatchNorm(nn.Module):
    """
    Maintains a separate LayerNorm for each domain.

    LayerNorm is used instead of BatchNorm1d so that a group containing only
    a single sample does not cause "Expected more than 1 value per channel".

    Reference: *Domain-Specific Batch Normalization for Unsupervised Domain
    Adaptation* (Chang et al., CVPR 2019).
    """

    def __init__(
        self,
        num_features: int,
        num_domains: int,
        eps: float = 1e-5,
        momentum: float = 0.1,   # kept for API compatibility; LayerNorm has no momentum
    ):
        super().__init__()
        self.num_domains  = num_domains
        self.num_features = num_features

        self.bns = nn.ModuleList(
            [nn.LayerNorm(num_features, eps=eps) for _ in range(num_domains)]
        )
        logger.info(
            f"DomainSpecificBatchNorm: {num_features} features, {num_domains} domains"
        )

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          (N, num_features)
            domain_ids: (N,) integer domain indices in [0, num_domains)

        Returns:
            (N, num_features) normalised features
        """
        output = torch.zeros_like(x)
        for d in range(self.num_domains):
            mask = domain_ids == d
            if mask.any():
                output[mask] = self.bns[d](x[mask])
        return output


# ──────────────────────────────────────────────────────────────────────────────
# Lambda Scheduler
# ──────────────────────────────────────────────────────────────────────────────

class LambdaScheduler:
    """
    Schedules the gradient-reversal strength λ during training.

    Schedules
    ---------
    constant    : λ = max_lambda  (no warm-up)
    linear      : λ grows linearly from 0 → max_lambda over *total_steps*
    exponential : λ follows the DANN exponential schedule
                  λ = max_lambda * (2 / (1 + exp(-10p)) − 1),  p ∈ [0, 1]
    """

    def __init__(
        self,
        schedule_type: str = 'linear',
        max_lambda: float = 1.0,
        total_steps: int = 10_000,
    ):
        self.schedule_type = schedule_type
        self.max_lambda    = max_lambda
        self.total_steps   = total_steps
        self.current_step  = 0

        logger.info(
            f"LambdaScheduler: type={schedule_type}, "
            f"max_lambda={max_lambda}, total_steps={total_steps}"
        )

    def step(self):
        """Advance the step counter by one."""
        self.current_step += 1

    def get_lambda(self) -> float:
        """Return the current λ value."""
        if self.schedule_type == 'constant':
            return self.max_lambda

        p = min(self.current_step / max(self.total_steps, 1), 1.0)

        if self.schedule_type == 'linear':
            return self.max_lambda * p

        if self.schedule_type == 'exponential':
            return self.max_lambda * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

        logger.warning(
            f"Unknown LambdaScheduler type '{self.schedule_type}'; "
            "falling back to constant."
        )
        return self.max_lambda

    def reset(self):
        """Reset step counter to 0."""
        self.current_step = 0


# ──────────────────────────────────────────────────────────────────────────────
# Maximum Mean Discrepancy (MMD)
# ──────────────────────────────────────────────────────────────────────────────

class MMD(nn.Module):
    """
    Maximum Mean Discrepancy with multi-scale RBF kernel.

    Alternative (or complement) to adversarial training for aligning
    source and target domain distributions in feature space.
    """

    def __init__(
        self,
        kernel_type: str = 'rbf',
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
    ):
        super().__init__()
        self.kernel_type = kernel_type
        self.kernel_mul  = kernel_mul
        self.kernel_num  = kernel_num

    def _gaussian_kernel(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        fix_sigma: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute the multi-scale RBF kernel matrix for source ∪ target."""
        n      = source.size(0) + target.size(0)
        total  = torch.cat([source, target], dim=0)  # (n, D)

        # Pairwise squared L2 distances
        sq     = (total.unsqueeze(0) - total.unsqueeze(1)).pow(2).sum(2)  # (n, n)

        if fix_sigma is not None:
            bandwidth = fix_sigma
        else:
            bandwidth = sq.data.sum() / (n ** 2 - n)

        bandwidth  = bandwidth / (self.kernel_mul ** (self.kernel_num // 2))
        bandwidths = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]

        return sum(torch.exp(-sq / bw) for bw in bandwidths)  # (n, n)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source: (n_s, D) source-domain features
            target: (n_t, D) target-domain features

        Returns:
            Scalar MMD² loss.
        """
        n  = source.size(0)
        K  = self._gaussian_kernel(source, target)

        XX = K[:n, :n]
        YY = K[n:, n:]
        XY = K[:n, n:]
        YX = K[n:, :n]

        return torch.mean(XX + YY - XY - YX)