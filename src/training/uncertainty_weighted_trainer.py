"""
Uncertainty-Weighted Domain Adversarial Training
CORE INNOVATION: Using evidential uncertainty to guide domain adaptation
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import logging
from typing import Dict
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from .trainer import EnhancedTrainer
from src.utils.metrics import MetricsTracker

logger = logging.getLogger(__name__)


class UncertaintyWeightedTrainer(EnhancedTrainer):
    """
    Enhanced trainer with uncertainty-weighted domain adversarial loss.

    INNOVATION: Uses evidential uncertainty to focus domain adaptation
    on samples where the model is uncertain about domain classification.

    Standard:  loss = cls_loss + λ * domain_loss
    Ours:      loss = cls_loss + λ * uncertainty_weight * domain_loss

    where uncertainty_weight = 1 + α * uncertainty_score
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config,
        model_config,
        device: torch.device,
        balance_info: Dict,
        source_to_id: Dict[str, int],
        uncertainty_weighting: str = 'adaptive',
        uncertainty_alpha: float = 0.5
    ):
        super().__init__(
            model, train_loader, val_loader, config, model_config,
            device, balance_info, source_to_id
        )

        self.uncertainty_weighting = uncertainty_weighting
        self.uncertainty_alpha     = uncertainty_alpha

        if uncertainty_weighting == 'adaptive':
            self.adaptive_alpha = nn.Parameter(
                torch.tensor(uncertainty_alpha, device=device)
            )
            self.optimizer.add_param_group(
                {'params': [self.adaptive_alpha], 'lr': 1e-3}
            )
            logger.info(
                f"Using ADAPTIVE uncertainty weighting "
                f"(initial α={uncertainty_alpha}, device={device})"
            )
        else:
            self.adaptive_alpha = None
            logger.info(
                f"Using {uncertainty_weighting.upper()} uncertainty weighting "
                f"(α={uncertainty_alpha})"
            )

        self.uncertainty_stats = {
            'train_uncertainties': [],
            'domain_weights':      [],
            'alpha_history':       [],
        }

    def compute_uncertainty_weights(
        self,
        uncertainties: torch.Tensor,
        domain_labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.uncertainty_weighting == 'none':
            return torch.ones_like(uncertainties)
        elif self.uncertainty_weighting == 'static':
            return 1.0 + self.uncertainty_alpha * uncertainties
        elif self.uncertainty_weighting == 'adaptive':
            alpha = torch.sigmoid(self.adaptive_alpha)
            return 1.0 + alpha * uncertainties
        elif self.uncertainty_weighting == 'threshold':
            threshold = 0.3
            return torch.where(
                uncertainties > threshold,
                torch.full_like(uncertainties, 1.0 + self.uncertainty_alpha),
                torch.ones_like(uncertainties),
            )
        else:
            raise ValueError(f"Unknown uncertainty_weighting: '{self.uncertainty_weighting}'")

    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch with uncertainty-weighted domain adversarial training."""
        self.model.train()

        metrics_tracker      = MetricsTracker()
        epoch_uncertainties  = []
        epoch_domain_weights = []
        domain_correct       = 0
        domain_total         = 0

        self.criterion.set_epoch(self.epoch)

        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.epoch + 1}/{self.config.epochs} [UW-Train]'
        )

        for batch_idx, batch in enumerate(pbar):

            input_ids      = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            images         = batch['image'].to(self.device)
            labels         = batch['label'].to(self.device)
            sources        = batch.get('source', None)

            if sources is not None:
                domain_labels = torch.tensor(
                    [self.source_to_id.get(s, 0) for s in sources],
                    dtype=torch.long, device=self.device,
                )
            else:
                domain_labels = torch.zeros(
                    labels.size(0), dtype=torch.long, device=self.device
                )

            if self.lambda_scheduler is not None:
                current_lambda = self.lambda_scheduler.get_lambda()
                self.model.set_gradient_reversal_lambda(current_lambda)
                self.lambda_scheduler.step()
            else:
                current_lambda = 0.0

            if self.config.use_amp:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask,
                        images=images, sources=sources,
                    )
                    uncertainties       = self._get_uncertainties(outputs, labels)
                    uncertainty_weights = self.compute_uncertainty_weights(uncertainties, domain_labels)
                    losses = self.criterion(outputs, labels, domain_labels, sources)
                    losses = self._apply_uncertainty_weighting(losses, outputs, domain_labels, uncertainty_weights)
                    loss   = losses['total']

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    images=images, sources=sources,
                )
                uncertainties       = self._get_uncertainties(outputs, labels)
                uncertainty_weights = self.compute_uncertainty_weights(uncertainties, domain_labels)
                losses = self.criterion(outputs, labels, domain_labels, sources)
                losses = self._apply_uncertainty_weighting(losses, outputs, domain_labels, uncertainty_weights)
                loss   = losses['total']

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()

            self.scheduler.step()

            # ── Metrics tracker (gives acc, f1, precision, recall, auc) ──
            probs = outputs['classification']['prob'].detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)
            metrics_tracker.update(
                predictions   = preds,
                labels        = labels.cpu().numpy(),
                probabilities = probs,
                sources       = sources,
                loss          = loss.item(),
            )

            if 'domain_logits' in outputs:
                domain_preds    = torch.argmax(outputs['domain_logits'], dim=1)
                domain_correct += (domain_preds == domain_labels).sum().item()
                domain_total   += domain_labels.size(0)

            epoch_uncertainties.extend(uncertainties.detach().cpu().float().numpy())
            epoch_domain_weights.extend(uncertainty_weights.detach().cpu().float().numpy())

            # ── Progress bar ──────────────────────────────────────────────
            postfix = {'loss': f'{loss.item():.4f}', 'unc': f'{uncertainties.mean().item():.3f}'}
            if self.lambda_scheduler is not None:
                postfix['λ'] = f'{current_lambda:.3f}'
            if self.adaptive_alpha is not None:
                postfix['α'] = f'{torch.sigmoid(self.adaptive_alpha).item():.3f}'
            pbar.set_postfix(postfix)

            # ── Tensorboard (batch-level) ─────────────────────────────────
            if self.global_step % self.config.log_interval == 0:
                self.writer.add_scalar('train/loss',              loss.item(),                       self.global_step)
                self.writer.add_scalar('train/cls_loss',          losses['classification'].item(),   self.global_step)
                self.writer.add_scalar('train/avg_uncertainty',   uncertainties.mean().item(),       self.global_step)
                self.writer.add_scalar('train/avg_domain_weight', uncertainty_weights.mean().item(), self.global_step)
                if 'domain' in losses:
                    self.writer.add_scalar('train/domain_loss', losses['domain'].item(), self.global_step)
                if self.adaptive_alpha is not None:
                    self.writer.add_scalar('train/adaptive_alpha', torch.sigmoid(self.adaptive_alpha).item(), self.global_step)
                if self.lambda_scheduler is not None:
                    self.writer.add_scalar('train/lambda', current_lambda, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1

        # ── Epoch-level metrics ───────────────────────────────────────────
        epoch_metrics = metrics_tracker.compute_metrics()

        if domain_total > 0:
            epoch_metrics['domain_accuracy'] = domain_correct / domain_total

        avg_unc    = float(np.mean(epoch_uncertainties))
        avg_weight = float(np.mean(epoch_domain_weights))
        epoch_metrics['avg_uncertainty']   = avg_unc
        epoch_metrics['avg_domain_weight'] = avg_weight

        self.uncertainty_stats['train_uncertainties'].append(avg_unc)
        self.uncertainty_stats['domain_weights'].append(avg_weight)
        if self.adaptive_alpha is not None:
            self.uncertainty_stats['alpha_history'].append(torch.sigmoid(self.adaptive_alpha).item())

        for key, value in epoch_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'epoch/train_{key}', value, self.epoch)

        return epoch_metrics

    def _get_uncertainties(self, outputs, labels):
        if 'uncertainty' in outputs['classification']:
            return outputs['classification']['uncertainty'].squeeze().float()
        return torch.zeros(labels.size(0), device=self.device)

    def _apply_uncertainty_weighting(self, losses, outputs, domain_labels, uncertainty_weights):
        if 'domain' not in losses or self.uncertainty_weighting == 'none':
            return losses

        weighted_domain_loss = (
            nn.functional.cross_entropy(
                outputs['domain_logits'], domain_labels, reduction='none'
            ) * uncertainty_weights
        ).mean()

        losses = dict(losses)
        losses['domain'] = weighted_domain_loss * self.config.domain_weight
        losses['total']  = losses['classification'] + losses['domain']
        return losses