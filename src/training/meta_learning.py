"""
Meta-Learning (MAML) for Few-Shot Domain Adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
import copy
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TaskSampler(Sampler):
    """
    Sample tasks for meta-learning
    
    Each task is a N-way K-shot classification problem constructed from
    one or more data sources.
    
    Task structure:
        - Support set: K examples per class (for adaptation)
        - Query set: Q examples per class (for evaluation)
    """
    
    def __init__(
        self,
        dataset,
        n_way: int,
        k_shot: int,
        q_queries: int,
        num_tasks: int,
        min_sources_per_task: int = 1,
        max_sources_per_task: int = 2
    ):
        """
        Args:
            dataset: Dataset to sample from
            n_way: Number of classes per task (typically 2 for binary)
            k_shot: Number of support examples per class
            q_queries: Number of query examples per class
            num_tasks: Number of tasks to sample per epoch
            min_sources_per_task: Minimum number of sources per task
            max_sources_per_task: Maximum number of sources per task
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.num_tasks = num_tasks
        self.min_sources = min_sources_per_task
        self.max_sources = max_sources_per_task
        
        # Organize samples by source and class
        self.source_class_indices = self._organize_data()
        self.sources = list(self.source_class_indices.keys())
        
        logger.info(f"TaskSampler initialized:")
        logger.info(f"  {n_way}-way {k_shot}-shot with {q_queries} queries")
        logger.info(f"  {num_tasks} tasks per epoch")
        logger.info(f"  {len(self.sources)} sources: {self.sources}")
    
    def _organize_data(self):
        """Organize dataset indices by source and class"""
        source_class_indices = defaultdict(lambda: defaultdict(list))
        
        for idx in range(len(self.dataset)):
            sample = self.dataset.data.iloc[idx]
            source = sample['source']
            
            # Normalize label to 0/1
            label = sample['label']
            if label in ['fake', '1', 1]:
                label = 1
            else:
                label = 0
            
            source_class_indices[source][label].append(idx)
        
        # Log statistics
        for source, class_dict in source_class_indices.items():
            logger.info(f"  {source}: Real={len(class_dict.get(0, []))}, Fake={len(class_dict.get(1, []))}")
        
        return dict(source_class_indices)
    
    def sample_task(self):
        """
        Sample one task
        
        Returns:
            support_indices: Indices for support set
            query_indices: Indices for query set
        """
        # Sample number of sources for this task
        num_sources = np.random.randint(self.min_sources, self.max_sources + 1)
        
        # Sample sources
        task_sources = np.random.choice(self.sources, size=num_sources, replace=False)
        
        # For binary classification, always use both classes
        classes = [0, 1]
        
        support_indices = []
        query_indices = []
        
        for source in task_sources:
            for cls in classes:
                available_indices = self.source_class_indices[source][cls]
                
                if len(available_indices) == 0:
                    continue
                
                # Total samples needed
                total_needed = self.k_shot + self.q_queries
                
                if len(available_indices) < total_needed:
                    # Sample with replacement
                    sampled = np.random.choice(
                        available_indices,
                        size=total_needed,
                        replace=True
                    )
                else:
                    # Sample without replacement
                    sampled = np.random.choice(
                        available_indices,
                        size=total_needed,
                        replace=False
                    )
                
                support_indices.extend(sampled[:self.k_shot])
                query_indices.extend(sampled[self.k_shot:])
        
        return support_indices, query_indices
    
    def __iter__(self):
        """Generate tasks"""
        for _ in range(self.num_tasks):
            support_indices, query_indices = self.sample_task()
            yield support_indices, query_indices
    
    def __len__(self):
        return self.num_tasks


class MAML:
    """
    Model-Agnostic Meta-Learning (MAML)
    
    Learns model initialization that can quickly adapt to new tasks/domains
    with few gradient steps.
    
    Reference: "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
    (Finn et al., ICML 2017)
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        device: torch.device = torch.device('cuda')
    ):
        """
        Args:
            model: Model to meta-learn
            inner_lr: Learning rate for inner loop (task adaptation)
            inner_steps: Number of gradient steps in inner loop
            device: Device to run on
        """
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device
        
        logger.info(f"MAML initialized:")
        logger.info(f"  Inner LR: {inner_lr}")
        logger.info(f"  Inner steps: {inner_steps}")
    
    def inner_loop(
        self,
        support_data: Dict[str, torch.Tensor],
        fast_weights: Optional[Dict] = None
    ) -> Tuple[Dict, torch.Tensor]:
        """
        Perform inner loop adaptation on support set
        
        Args:
            support_data: Support set batch (dict with inputs and labels)
            fast_weights: Current task-specific weights (None = use model weights)
        
        Returns:
            updated_weights: Updated task-specific weights
            support_loss: Loss on support set
        """
        if fast_weights is None:
            # Initialize with current model parameters
            fast_weights = {
                name: param.clone()
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }
        
        # Forward pass with current weights
        support_loss = self._compute_loss(support_data, fast_weights)
        
        # Compute gradients with respect to fast weights
        grads = torch.autograd.grad(
            support_loss,
            fast_weights.values(),
            create_graph=True,  # Need second-order gradients for meta-update
            allow_unused=True
        )
        
        # Update weights using gradient descent
        updated_weights = {}
        for (name, param), grad in zip(fast_weights.items(), grads):
            if grad is not None:
                updated_weights[name] = param - self.inner_lr * grad
            else:
                updated_weights[name] = param
        
        return updated_weights, support_loss
    
    # def _compute_loss(
    #     self,
    #     batch: Dict[str, torch.Tensor],
    #     weights: Dict
    # ) -> torch.Tensor:
    #     """
    #     Compute loss using given weights (functional forward pass)
        
    #     Args:
    #         batch: Data batch
    #         weights: Model weights to use
        
    #     Returns:
    #         loss: Computed loss
    #     """
    #     # Temporarily replace model parameters
    #     original_params = {}
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad:
    #             original_params[name] = param.data.clone()
    #             if name in weights:
    #                 param.data = weights[name]
        
    #     try:
    #         # Forward pass
    #         outputs = self.model(
    #             input_ids=batch['input_ids'],
    #             attention_mask=batch['attention_mask'],
    #             images=batch['image'],
    #             sources=batch.get('source', None)
    #         )
            
    #         # Compute loss (simple cross-entropy for inner loop)
    #         logits = outputs['classification']['logits']
    #         loss = F.cross_entropy(logits, batch['label'])
        
    #     finally:
    #         # Restore original parameters
    #         for name, param in self.model.named_parameters():
    #             if name in original_params:
    #                 param.data = original_params[name]
        
    #     return loss

    def _compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        weights: Dict
    ) -> torch.Tensor:
        """
        Compute loss using given weights (functional forward pass)
        
        Args:
            batch: Data batch
            weights: Model weights to use
        
        Returns:
            loss: Computed loss
        """
        # Temporarily replace model parameters
        original_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                original_params[name] = param.data.clone()
                if name in weights:
                    param.data = weights[name]
        
        try:
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                images=batch['image'],
                sources=batch.get('source', None)
            )
            
            # Compute loss (simple cross-entropy for inner loop)
            logits = outputs['classification']['logits']
            
            # ✅ FIX: Ensure labels are tensors
            labels = batch['label']
            if isinstance(labels, list):
                labels = torch.tensor(labels, dtype=torch.long, device=logits.device)
            elif not isinstance(labels, torch.Tensor):
                labels = torch.as_tensor(labels, dtype=torch.long, device=logits.device)
            
            # Ensure labels are on the same device
            if labels.device != logits.device:
                labels = labels.to(logits.device)
            
            loss = F.cross_entropy(logits, labels)
        
        finally:
            # Restore original parameters
            for name, param in self.model.named_parameters():
                if name in original_params:
                    param.data = original_params[name]
        
        return loss


    def meta_update(
        self,
        task_batch: List[Tuple[Dict, Dict]],
        meta_optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Perform meta-update across multiple tasks
        
        Args:
            task_batch: List of (support_batch, query_batch) tuples
            meta_optimizer: Optimizer for meta-parameters
        
        Returns:
            metrics: Dictionary of training metrics
        """
        meta_optimizer.zero_grad()
        
        total_query_loss = 0.0
        total_support_loss = 0.0
        num_tasks = len(task_batch)
        
        for support_batch, query_batch in task_batch:
            # Move to device
            support_batch = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in support_batch.items()
            }
            query_batch = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in query_batch.items()
            }
            
            # Inner loop: adapt on support set
            fast_weights = None
            for step in range(self.inner_steps):
                fast_weights, support_loss = self.inner_loop(support_batch, fast_weights)
            
            total_support_loss += support_loss.item()
            
            # Evaluate on query set with adapted weights
            query_loss = self._compute_loss(query_batch, fast_weights)
            total_query_loss += query_loss.item()
            
            # Backprop through query loss (meta-gradient)
            query_loss = query_loss / num_tasks
            query_loss.backward()
        
        # Meta-update: update model initialization
        meta_optimizer.step()
        
        return {
            'meta_train_loss': total_query_loss / num_tasks,
            'meta_support_loss': total_support_loss / num_tasks
        }


class MetaLearningTrainer:
    """
    Trainer for meta-learning phase
    
    Handles task sampling, batch creation, and meta-training loop.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        config,
        device: torch.device
    ):
        """
        Args:
            model: Model to meta-train
            train_dataset: Training dataset
            config: Training configuration
            device: Device to run on
        """
        self.model = model
        self.train_dataset = train_dataset
        self.config = config.meta_learning
        self.device = device
        
        # Initialize MAML
        self.maml = MAML(
            model=model,
            inner_lr=config.meta_learning.inner_lr,
            inner_steps=config.meta_learning.inner_steps,
            device=device
        )
        
        # Task sampler
        self.task_sampler = TaskSampler(
            dataset=train_dataset,
            n_way=config.meta_learning.num_classes_per_task,
            k_shot=config.meta_learning.support_samples_per_class,
            q_queries=config.meta_learning.query_samples_per_class,
            num_tasks=100,  # Tasks per epoch
            min_sources_per_task=config.meta_learning.min_sources_per_task,
            max_sources_per_task=config.meta_learning.max_sources_per_task
        )
        
        logger.info("MetaLearningTrainer initialized")
    
    # def create_batch_from_indices(self, indices: List[int]) -> Dict:
    #     """Create a batch from dataset indices"""
    #     batch = defaultdict(list)
        
    #     for idx in indices:
    #         sample = self.train_dataset[idx]
    #         for key, value in sample.items():
    #             batch[key].append(value)
        
    #     # Stack tensors
    #     for key in list(batch.keys()):
    #         if len(batch[key]) > 0 and torch.is_tensor(batch[key][0]):
    #             batch[key] = torch.stack(batch[key])
    #         # Keep lists for non-tensors (e.g., source names)
        
    #     return dict(batch)


    def create_batch_from_indices(self, indices: List[int]) -> Dict:
        """Create a batch from dataset indices"""
        batch = defaultdict(list)
        
        for idx in indices:
            sample = self.train_dataset[idx]
            for key, value in sample.items():
                batch[key].append(value)
        
        # Stack tensors
        for key in list(batch.keys()):
            if len(batch[key]) > 0 and torch.is_tensor(batch[key][0]):
                batch[key] = torch.stack(batch[key])
            elif key == 'label':  # ✅ FIX: Convert labels to tensor
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            # Keep lists for non-tensors (e.g., source names)
        
        return dict(batch)
    
    def train_epoch(self, meta_optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Train one meta-learning epoch
        
        Args:
            meta_optimizer: Optimizer for meta-updates
        
        Returns:
            metrics: Training metrics
        """
        self.model.train()
        
        total_meta_loss = 0.0
        total_support_loss = 0.0
        num_batches = 0
        
        # Sample tasks
        task_iter = iter(self.task_sampler)
        
        meta_batch = []
        pbar = tqdm(task_iter, total=len(self.task_sampler), desc='Meta-Learning')
        
        for i, (support_indices, query_indices) in enumerate(pbar):
            # Create batches
            support_batch = self.create_batch_from_indices(support_indices)
            query_batch = self.create_batch_from_indices(query_indices)
            
            meta_batch.append((support_batch, query_batch))
            
            # Perform meta-update when we have enough tasks
            if len(meta_batch) == self.config.meta_batch_size:
                metrics = self.maml.meta_update(meta_batch, meta_optimizer)
                
                total_meta_loss += metrics['meta_train_loss']
                total_support_loss += metrics['meta_support_loss']
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'meta_loss': f"{metrics['meta_train_loss']:.4f}",
                    'support_loss': f"{metrics['meta_support_loss']:.4f}"
                })
                
                meta_batch = []
        
        # Handle remaining tasks
        if meta_batch:
            metrics = self.maml.meta_update(meta_batch, meta_optimizer)
            total_meta_loss += metrics['meta_train_loss']
            total_support_loss += metrics['meta_support_loss']
            num_batches += 1
        
        return {
            'meta_loss': total_meta_loss / max(num_batches, 1),
            'support_loss': total_support_loss / max(num_batches, 1)
        }