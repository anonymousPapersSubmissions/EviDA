# config/config.py
"""
Configuration for Cross-Domain Fake News Detection
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
import torch


@dataclass
class DomainAdaptationConfig:
    """Configuration for domain adaptation"""
    
    # Domain adversarial training
    use_domain_adversarial: bool = True
    adversarial_loss_weight: float = 0.1
    gradient_reversal_lambda: float = 1.0
    discriminator_hidden_dim: int = 256
    discriminator_num_layers: int = 3
    discriminator_dropout: float = 0.3
    
    # Gradient reversal schedule
    use_gradual_lambda: bool = True
    lambda_schedule: Literal['constant', 'linear', 'exponential'] = 'linear'
    max_lambda: float = 1.0
    
    # Domain-specific batch normalization
    use_domain_specific_bn: bool = True


@dataclass
class EvidentialConfig:
    """Configuration for evidential deep learning"""
    
    use_evidential: bool = True
    num_classes: int = 2
    
    # Evidential loss parameters
    kl_weight: float = 0.01
    annealing_start: int = 10
    annealing_step: int = 10
    
    # Uncertainty thresholds
    uncertainty_threshold: float = 0.5
    
    # Domain-specific priors
    use_domain_specific_priors: bool = True
    prior_concentration: float = 1.0


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning (MAML)"""
    
    use_meta_learning: bool = True
    
    # MAML parameters
    inner_lr: float = 0.01
    inner_steps: int = 5
    meta_batch_size: int = 4
    
    # Task construction
    support_samples_per_class: int = 5  # K-shot
    query_samples_per_class: int = 10
    num_classes_per_task: int = 2  # Binary
    
    # Source sampling for tasks
    min_sources_per_task: int = 1
    max_sources_per_task: int = 2
    
    # Meta-learning schedule
    meta_learning_epochs: int = 5
    finetune_epochs: int = 25


@dataclass
class ImbalanceConfig:
    """Configuration for handling imbalance"""
    
    # Class imbalance
    class_balance_strategy: Literal[
        'none', 'oversample', 'undersample', 'focal_loss', 'class_weights'
    ] = 'class_weights'
    oversample_minority: bool = True
    undersample_majority: bool = False
    target_balance_ratio: float = 0.4
    
    # Source imbalance
    source_balance_strategy: Literal[
        'none', 'weighted_sampling', 'balanced_batch', 'curriculum'
    ] = 'balanced_batch'
    min_source_samples_per_batch: int = 1
    source_sampling_weights: Optional[Dict[str, float]] = None
    
    # Focal loss parameters
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    
    # Class weights
    use_inverse_freq_weights: bool = True
    manual_class_weights: Optional[Dict[int, float]] = None
    
    # Source weights
    use_inverse_source_weights: bool = True
    manual_source_weights: Optional[Dict[str, float]] = None


@dataclass
class DataConfig:
    """Configuration for dataset handling"""
    max_text_length: int = 512
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    
    # Multi-source dataset paths
    dataset_sources: Dict[str, str] = field(default_factory=lambda: {
        'twitter': 'data/twitter/',
        'french': 'data/french/',
        'weibo': 'data/weibo/',
        'fakeddit': 'data/fakeddit/',
        'reddit': 'data/reddit/',
        'facebook': 'data/facebook/',
        'airbnb': 'data/airbnb/',
        'politics': 'data/politics/',
        'education': 'data/education/'
    })
    
    # Language support
    supported_languages: List[str] = field(default_factory=lambda: [
        'en', 'zh', 'es', 'fr', 'de', 'ar', 'hi', 'ja', 'ko'
    ])
    
    # Imbalance handling
    imbalance_config: ImbalanceConfig = field(default_factory=ImbalanceConfig)


@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    
    # Text encoder
    text_encoder_name: str = 'xlm-roberta-base'
    text_hidden_dim: int = 768
    text_intermediate_dim: int = 1024
    text_output_dim: int = 512
    
    # Vision encoder
    vision_encoder_name: str = 'microsoft/swin-base-patch4-window7-224'
    vision_hidden_dim: int = 1024
    vision_output_dim: int = 512
    
    # Fusion and classification
    fusion_dim: int = 768
    num_classes: int = 2
    dropout_rate: float = 0.3
    
    # Cross-attention
    num_attention_heads: int = 8
    
    # Explainability generator (disabled for this paper)
    use_explanation_head: bool = False
    explanation_max_length: int = 128
    decoder_hidden_dim: int = 768
    decoder_num_layers: int = 4
    
    # Domain adaptation
    domain_adaptation: DomainAdaptationConfig = field(default_factory=DomainAdaptationConfig)
    
    # Evidential learning
    evidential: EvidentialConfig = field(default_factory=EvidentialConfig)
    
    # Deprecated
    use_evidential_learning: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training"""
    
    # Training parameters
    epochs: int = 100
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    
    # Mixed precision training
    use_amp: bool = True
    
    # Loss weights
    classification_weight: float = 1.0
    explanation_weight: float = 0.0
    uncertainty_weight: float = 0.1
    domain_weight: float = 0.1
    
    # Early stopping
    patience: int = 5
    min_delta: float = 0.001
    
    # Checkpointing
    save_dir: str = 'checkpoints/'
    log_dir: str = 'logs/'
    
    # Logging
    log_interval: int = 50
    eval_interval: int = 1
    
    # Imbalance config
    imbalance_config: ImbalanceConfig = field(default_factory=ImbalanceConfig)
    
    # Meta-learning
    meta_learning: MetaLearningConfig = field(default_factory=MetaLearningConfig)
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'