import sys
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional
from torch.utils.tensorboard import SummaryWriter

from src.losses.losses import CombinedLossV2
from src.utils.metrics import MetricsTracker
from src.models.domain_adaptation import LambdaScheduler
from src.training.meta_learning import MetaLearningTrainer
from src.utils.logger import setup_metrics_logger

logger = logging.getLogger(__name__)
metrics_logger = None  # initialised in train() once save_dir is known



class EnhancedTrainer:
    """
    Enhanced Trainer for Cross-Domain Fake News Detection.

    Features:
    1. Domain Adversarial Training
    2. Evidential Deep Learning
    3. Meta-Learning (MAML) — optional
    4. Mixed Precision Training
    5. Comprehensive Logging

    Checkpoint policy
    -----------------
    Only the single best model (by F1 / accuracy) is kept on disk as
    best_model.pt.  Per-epoch checkpoints are NOT saved — they accumulate
    quickly and are rarely needed.  If you need to resume mid-training,
    the best checkpoint is always available.
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
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.model_config = model_config
        self.device       = device
        self.balance_info = balance_info
        self.source_to_id = source_to_id
        self.num_domains  = len(source_to_id)

        self.optimizer  = self._setup_optimizer()
        self.scheduler  = self._setup_scheduler()
        self.criterion  = self._setup_loss()

        # Lambda scheduler for gradient reversal
        if model_config.domain_adaptation.use_domain_adversarial:
            total_steps = len(train_loader) * config.epochs
            self.lambda_scheduler = LambdaScheduler(
                schedule_type = model_config.domain_adaptation.lambda_schedule,
                max_lambda    = model_config.domain_adaptation.max_lambda,
                total_steps   = total_steps,
            )
        else:
            self.lambda_scheduler = None

        # Meta-learning (optional)
        if config.meta_learning.use_meta_learning:
            self.meta_trainer = MetaLearningTrainer(
                model         = model,
                train_dataset = train_loader.dataset,
                config        = config,
                device        = device,
            )
            self.meta_optimizer = self._setup_meta_optimizer()
            logger.info('Meta-learning enabled')
        else:
            self.meta_trainer   = None
            self.meta_optimizer = None

        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None

        # Tensorboard
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(config.log_dir)

        # State
        self.best_f1         = 0.0
        self.best_loss       = float('inf')
        self.patience_counter = 0
        self.global_step     = 0
        self.epoch           = 0

        Path(config.save_dir).mkdir(parents=True, exist_ok=True)

        logger.info('='*80)
        logger.info('Enhanced Trainer Initialized')
        logger.info('='*80)
        logger.info(f'  Total parameters:     {model.get_num_parameters():,}')
        logger.info(f'  Trainable parameters: {model.get_trainable_parameters():,}')
        logger.info(f'  Number of domains:    {self.num_domains}')
        logger.info(f'  Domains:              {list(source_to_id.keys())}')
        logger.info(f'  Meta-learning:        {config.meta_learning.use_meta_learning}')
        logger.info(f'  Domain adversarial:   {model_config.domain_adaptation.use_domain_adversarial}')
        logger.info(f'  Evidential learning:  {model_config.evidential.use_evidential}')
        logger.info(f'  Checkpoint policy:    best model only (best_model.pt)')
        logger.info('='*80)

    # ──────────────────────────────────────────────────────────────────────
    # Setup helpers
    # ──────────────────────────────────────────────────────────────────────

    def _setup_optimizer(self) -> AdamW:
        """AdamW with layer-wise learning rates."""
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']

        groups = [
            # Encoders — 10× lower LR (pretrained backbone)
            {
                'params': [p for n, p in self.model.named_parameters()
                           if 'encoder' in n and not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay,
                'lr': self.config.learning_rate * 0.1,
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                           if 'encoder' in n and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.config.learning_rate * 0.1,
            },
            # Domain discriminator — 2× higher LR
            {
                'params': [p for n, p in self.model.named_parameters()
                           if 'domain_discriminator' in n
                           and not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay,
                'lr': self.config.learning_rate * 2.0,
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                           if 'domain_discriminator' in n
                           and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.config.learning_rate * 2.0,
            },
            # Everything else — normal LR
            {
                'params': [p for n, p in self.model.named_parameters()
                           if 'encoder' not in n
                           and 'domain_discriminator' not in n
                           and not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay,
                'lr': self.config.learning_rate,
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                           if 'encoder' not in n
                           and 'domain_discriminator' not in n
                           and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.config.learning_rate,
            },
        ]
        return AdamW(groups)

    def _setup_meta_optimizer(self) -> AdamW:
        return AdamW(
            self.model.parameters(),
            lr           = self.config.learning_rate,
            weight_decay = self.config.weight_decay,
        )

    def _setup_scheduler(self):
        total_steps = len(self.train_loader) * self.config.epochs
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps  = self.config.warmup_steps,
            num_training_steps = total_steps,
        )

    def _setup_loss(self):
        class_weights = None
        if self.balance_info and 'class_weights' in self.balance_info:
            w = self.balance_info['class_weights']
            if w is not None:
                if isinstance(w, torch.Tensor):
                    class_weights = w.to(self.device)
                elif isinstance(w, dict):
                    class_weights = torch.tensor(
                        [w[i] for i in range(len(w))],
                        device=self.device, dtype=torch.float32,
                    )
                else:
                    class_weights = torch.tensor(
                        w, device=self.device, dtype=torch.float32
                    )

        source_weights = (
            self.balance_info.get('source_weights')
            if self.balance_info else None
        )

        return CombinedLossV2(
            config                = self.model_config,
            num_domains           = len(self.source_to_id),
            class_weights         = class_weights,
            source_weights        = source_weights,
            classification_weight = getattr(self.config, 'classification_weight', 1.0),
            domain_weight         = getattr(self.config, 'domain_weight', 0.1),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Meta-learning phase
    # ──────────────────────────────────────────────────────────────────────

    def meta_learning_phase(self):
        if self.meta_trainer is None:
            return

        logger.info('\n' + '='*80)
        logger.info('STARTING META-LEARNING PHASE')
        logger.info('='*80)

        for epoch in range(self.config.meta_learning.meta_learning_epochs):
            logger.info(
                f'\nMeta-Learning Epoch '
                f'{epoch + 1}/{self.config.meta_learning.meta_learning_epochs}'
            )
            metrics = self.meta_trainer.train_epoch(self.meta_optimizer)
            logger.info(
                f'  Meta Loss: {metrics["meta_loss"]:.4f}, '
                f'Support Loss: {metrics["support_loss"]:.4f}'
            )
            self.writer.add_scalar('meta/meta_loss',    metrics['meta_loss'],    epoch)
            self.writer.add_scalar('meta/support_loss', metrics['support_loss'], epoch)

        logger.info('='*80)
        logger.info('META-LEARNING PHASE COMPLETED')
        logger.info('='*80 + '\n')

    # ──────────────────────────────────────────────────────────────────────
    # Train epoch
    # ──────────────────────────────────────────────────────────────────────

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with domain adaptation."""
        self.model.train()
        metrics_tracker = MetricsTracker()

        self.criterion.set_epoch(self.epoch)

        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.epoch + 1}/{self.config.epochs} [Train]',
        )

        domain_correct = 0
        domain_total   = 0

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
                    losses = self.criterion(outputs, labels, domain_labels, sources)
                    loss   = losses['total']

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    images=images, sources=sources,
                )
                losses = self.criterion(outputs, labels, domain_labels, sources)
                loss   = losses['total']

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
                self.optimizer.step()

            self.scheduler.step()

            probs = outputs['classification']['prob'].detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)
            metrics_tracker.update(
                predictions  = preds,
                labels       = labels.cpu().numpy(),
                probabilities = probs,
                sources      = sources,
                loss         = loss.item(),
            )

            if 'domain_logits' in outputs:
                domain_preds    = torch.argmax(outputs['domain_logits'], dim=1)
                domain_correct += (domain_preds == domain_labels).sum().item()
                domain_total   += domain_labels.size(0)

            postfix = {'loss': f'{loss.item():.4f}'}
            if self.lambda_scheduler is not None:
                postfix['λ'] = f'{current_lambda:.3f}'
            if 'uncertainty' in outputs['classification']:
                postfix['unc'] = (
                    f'{outputs["classification"]["uncertainty"].mean().item():.3f}'
                )
            pbar.set_postfix(postfix)

            if self.global_step % self.config.log_interval == 0:
                self.writer.add_scalar('train/loss',     loss.item(),                     self.global_step)
                self.writer.add_scalar('train/cls_loss', losses['classification'].item(), self.global_step)
                if 'domain' in losses:
                    self.writer.add_scalar('train/domain_loss', losses['domain'].item(), self.global_step)
                if 'cls_kl' in losses:
                    self.writer.add_scalar('train/kl_loss',   losses['cls_kl'].item(),        self.global_step)
                    self.writer.add_scalar('train/annealing',  losses.get('annealing', 0),     self.global_step)
                if self.lambda_scheduler is not None:
                    self.writer.add_scalar('train/lambda', current_lambda, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1

        epoch_metrics = metrics_tracker.compute_metrics()

        if domain_total > 0:
            domain_acc = domain_correct / domain_total
            epoch_metrics['domain_accuracy'] = domain_acc

        # Log every train metric for this epoch to TensorBoard
        for key, value in epoch_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'epoch/train_{key}', value, self.epoch)

        return epoch_metrics

    # ──────────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        metrics_tracker = MetricsTracker()

        uncertainties           = []
        correct_uncertainties   = []
        incorrect_uncertainties = []

        pbar = tqdm(
            self.val_loader,
            desc=f'Epoch {self.epoch + 1}/{self.config.epochs} [Val]',
        )

        for batch in pbar:
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

            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                images=images, sources=sources,
            )
            losses = self.criterion(outputs, labels, domain_labels, sources)
            loss   = losses['total']

            probs = outputs['classification']['prob'].cpu().numpy()
            preds = np.argmax(probs, axis=1)
            metrics_tracker.update(
                predictions  = preds,
                labels       = labels.cpu().numpy(),
                probabilities = probs,
                sources      = sources,
                loss         = loss.item(),
            )

            if 'uncertainty' in outputs['classification']:
                unc          = outputs['classification']['uncertainty'].cpu().numpy().flatten()
                correct_mask = (preds == labels.cpu().numpy())
                uncertainties.extend(unc)
                correct_uncertainties.extend(unc[correct_mask])
                incorrect_uncertainties.extend(unc[~correct_mask])

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        val_metrics = metrics_tracker.compute_metrics()

        if uncertainties:
            val_metrics['avg_uncertainty'] = np.mean(uncertainties)
            val_metrics['std_uncertainty'] = np.std(uncertainties)
            if correct_uncertainties:
                val_metrics['correct_uncertainty']   = np.mean(correct_uncertainties)
            if incorrect_uncertainties:
                val_metrics['incorrect_uncertainty'] = np.mean(incorrect_uncertainties)

            logger.info('\nUncertainty Statistics:')
            logger.info(f'  Average: {val_metrics["avg_uncertainty"]:.4f}')
            if correct_uncertainties and incorrect_uncertainties:
                logger.info(f'  Correct:   {val_metrics["correct_uncertainty"]:.4f}')
                logger.info(f'  Incorrect: {val_metrics["incorrect_uncertainty"]:.4f}')

        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'val/{key}', value, self.epoch)

        logger.info('\n' + metrics_tracker.get_classification_report())
        logger.info(f'\nConfusion Matrix:\n{metrics_tracker.get_confusion_matrix()}')

        return val_metrics

    # ──────────────────────────────────────────────────────────────────────
    # Checkpointing  —  best model only
    # ──────────────────────────────────────────────────────────────────────

    def save_best_checkpoint(self, metrics: Dict[str, float]):
        """
        Overwrite best_model.pt with the current model state.

        Only called when a new best validation score is achieved, so disk
        usage stays constant regardless of the number of epochs.
        """
        best_path = Path(self.config.save_dir) / 'best_model.pt'

        checkpoint = {
            'epoch':              self.epoch,
            'global_step':        self.global_step,
            'model_state_dict':   self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics':            metrics,
            'config':             self.config,
            'model_config':       self.model_config,
            'best_f1':            self.best_f1,
            'source_to_id':       self.source_to_id,
        }

        torch.save(checkpoint, best_path)
        logger.info(
            f'✓ New best model saved → {best_path}  '
            f'(F1={metrics.get("f1", 0):.4f}, epoch={self.epoch + 1})'
        )

    # ──────────────────────────────────────────────────────────────────────
    # Full training loop
    # ──────────────────────────────────────────────────────────────────────

    def train(self):
        """Full training loop — saves only the best checkpoint."""
        global metrics_logger

        # ── Metrics log file (one clean line per epoch) ───────────────────
        metrics_log_path = str(Path(self.config.log_dir) / 'metrics.log')
        metrics_logger   = setup_metrics_logger(metrics_log_path)
        # Write header
        metrics_logger.info(
            f"{'Epoch':>5}  {'Split':>5}  "
            f"{'Loss':>7}  {'Acc':>7}  {'F1':>7}  "
            f"{'Prec':>7}  {'Recall':>7}  {'AUC':>7}  "
            f"{'DomAcc':>7}  {'Unc':>7}  {'BestF1':>7}"
        )
        metrics_logger.info('-' * 90)

        print('\n' + '='*80)
        print('STARTING TRAINING')
        print('='*80)

        if self.config.meta_learning.use_meta_learning:
            self.meta_learning_phase()

        print('\n' + '='*80)
        print('STANDARD TRAINING PHASE')
        print('='*80 + '\n')

        for epoch in range(self.config.epochs):
            self.epoch = epoch

            train_metrics = self.train_epoch()

            # ── Terminal: per-epoch train summary ─────────────────────────
            sep = '─' * 80
            print(f'\n{sep}')
            print(f'Epoch {epoch + 1:>3}/{self.config.epochs}  [TRAIN]')
            print(
                f'  Loss: {train_metrics.get("loss", 0):.4f}  |  '
                f'Acc: {train_metrics.get("accuracy", 0):.4f}  |  '
                f'F1: {train_metrics.get("f1", 0):.4f}  |  '
                f'Prec: {train_metrics.get("precision", 0):.4f}  |  '
                f'Recall: {train_metrics.get("recall", 0):.4f}  |  '
                f'AUC: {train_metrics.get("auc", 0):.4f}'
            )
            if 'domain_accuracy' in train_metrics:
                print(f'  Domain Acc: {train_metrics["domain_accuracy"]:.4f}')
            if 'avg_uncertainty' in train_metrics:
                print(f'  Avg Uncertainty: {train_metrics["avg_uncertainty"]:.4f}')
            print(
                f'  LR: {self.optimizer.param_groups[0]["lr"]:.2e}  |  '
                f'Step: {self.global_step}'
            )

            # ── Metrics file: train row ───────────────────────────────────
            metrics_logger.info(
                f'{epoch + 1:>5}  {"train":>5}  '
                f'{train_metrics.get("loss", 0):>7.4f}  '
                f'{train_metrics.get("accuracy", 0):>7.4f}  '
                f'{train_metrics.get("f1", 0):>7.4f}  '
                f'{train_metrics.get("precision", 0):>7.4f}  '
                f'{train_metrics.get("recall", 0):>7.4f}  '
                f'{train_metrics.get("auc", 0):>7.4f}  '
                f'{train_metrics.get("domain_accuracy", float("nan")):>7.4f}  '
                f'{train_metrics.get("avg_uncertainty", float("nan")):>7.4f}  '
                f'{self.best_f1:>7.4f}'
            )

            if (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self.validate()

                # ── Terminal: per-epoch val summary ───────────────────────
                print(f'Epoch {epoch + 1:>3}/{self.config.epochs}  [VAL]')
                print(
                    f'  Loss: {val_metrics.get("loss", 0):.4f}  |  '
                    f'Acc: {val_metrics.get("accuracy", 0):.4f}  |  '
                    f'F1: {val_metrics.get("f1", 0):.4f}  |  '
                    f'Prec: {val_metrics.get("precision", 0):.4f}  |  '
                    f'Recall: {val_metrics.get("recall", 0):.4f}  |  '
                    f'AUC: {val_metrics.get("auc", 0):.4f}'
                )
                if 'avg_uncertainty' in val_metrics:
                    print(
                        f'  Uncertainty avg/std: '
                        f'{val_metrics["avg_uncertainty"]:.4f} / '
                        f'{val_metrics.get("std_uncertainty", 0):.4f}  |  '
                        f'correct: {val_metrics.get("correct_uncertainty", 0):.4f}  '
                        f'incorrect: {val_metrics.get("incorrect_uncertainty", 0):.4f}'
                    )

                # ── Best-model tracking ────────────────────────────────────
                current_metric = val_metrics.get('f1', val_metrics.get('accuracy', 0.0))

                if current_metric > self.best_f1:
                    self.best_f1 = current_metric
                    self.patience_counter = 0
                    self.save_best_checkpoint(val_metrics)
                    print(f'  ✓ New best model  (F1={self.best_f1:.4f})')
                else:
                    self.patience_counter += 1
                    print(
                        f'  No improvement — patience '
                        f'{self.patience_counter}/{self.config.patience}  '
                        f'(best F1: {self.best_f1:.4f})'
                    )

                # ── Metrics file: val row ─────────────────────────────────
                metrics_logger.info(
                    f'{epoch + 1:>5}  {"val":>5}  '
                    f'{val_metrics.get("loss", 0):>7.4f}  '
                    f'{val_metrics.get("accuracy", 0):>7.4f}  '
                    f'{val_metrics.get("f1", 0):>7.4f}  '
                    f'{val_metrics.get("precision", 0):>7.4f}  '
                    f'{val_metrics.get("recall", 0):>7.4f}  '
                    f'{val_metrics.get("auc", 0):>7.4f}  '
                    f'{"nan":>7}  '
                    f'{val_metrics.get("avg_uncertainty", float("nan")):>7.4f}  '
                    f'{self.best_f1:>7.4f}'
                )

                # ── TensorBoard ───────────────────────────────────────────
                self.writer.add_scalars('epoch/loss',     {'train': train_metrics.get('loss', 0),     'val': val_metrics.get('loss', 0)},     epoch)
                self.writer.add_scalars('epoch/f1',       {'train': train_metrics.get('f1', 0),       'val': val_metrics.get('f1', 0)},       epoch)
                self.writer.add_scalars('epoch/accuracy', {'train': train_metrics.get('accuracy', 0), 'val': val_metrics.get('accuracy', 0)}, epoch)
                self.writer.add_scalars('epoch/auc',      {'train': train_metrics.get('auc', 0),      'val': val_metrics.get('auc', 0)},      epoch)
                if 'avg_uncertainty' in val_metrics:
                    self.writer.add_scalar('epoch/val_uncertainty', val_metrics['avg_uncertainty'], epoch)

                print(sep)

                if self.patience_counter >= self.config.patience:
                    print(f'\n⚠  Early stopping triggered after {epoch + 1} epochs')
                    metrics_logger.info(f'\nEarly stopping at epoch {epoch + 1}')
                    break

        print('\n' + '='*80)
        print(f'✓ TRAINING COMPLETED  Best F1: {self.best_f1:.4f}')
        print(f'  Metrics log → {metrics_log_path}')
        print('='*80)

        self.writer.close()