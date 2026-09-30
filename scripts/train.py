"""
Main Training Script for Cross-Domain Fake News Detection
with Adversarial Source Alignment and Evidential Deep Learning
WITH UNCERTAINTY-WEIGHTED DOMAIN ADVERSARIAL TRAINING
"""

import torch
import argparse
import sys
from pathlib import Path
import pandas as pd
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import DataConfig, ModelConfig, TrainingConfig
from src.data.dataset import create_dataloaders
from src.models.model import MultimodalFakeNewsDetectorV2
from src.training.trainer import EnhancedTrainer
from src.training.uncertainty_weighted_trainer import UncertaintyWeightedTrainer
from src.utils.logger import setup_logger
from transformers import AutoTokenizer

logger = setup_logger('training', 'logs/training.log')


def save_hf_artifacts(model, tokenizer, save_dir: str):
    """
    Save HuggingFace tokenizer and text-encoder backbone to disk.

    Produces the files needed for fully offline inference / evaluation:
        tokenizer.json          tokenizer_config.json   spiece.model
        special_tokens_map.json model.safetensors
        generation_config.json  config.json

    Call this once after the model is first initialised (the backbone
    weights are identical to the pretrained checkpoint at this point).
    The trained weights are saved separately via the normal checkpoint
    mechanism; load_state_dict() overwrites the backbone at inference time.

    Args:
        model:     Initialised MultimodalFakeNewsDetectorV2.
        tokenizer: AutoTokenizer used for training.
        save_dir:  Root experiment directory (hf_artifacts/ is created inside).
    """
    hf_dir = Path(save_dir) / 'hf_artifacts'
    hf_dir.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    # Saves: tokenizer.json, tokenizer_config.json, spiece.model,
    #        special_tokens_map.json
    tokenizer.save_pretrained(str(hf_dir))
    logger.info(f'  Tokenizer artifacts saved.')

    # ── Text encoder backbone ─────────────────────────────────────────────
    # Saves: config.json, model.safetensors (and generation_config.json
    # if the model has a generative config)
    model.text_encoder.encoder.save_pretrained(str(hf_dir))
    logger.info(f'  Backbone artifacts saved.')

    logger.info(f'HuggingFace artifacts → {hf_dir}')
    logger.info(
        '  Use --hf_dir ' + str(hf_dir) +
        ' with inference.py / evaluate.py for offline use.'
    )


def load_datasets(data_config: DataConfig):
    """Load all datasets from different sources"""
    train_dfs = {}
    val_dfs = {}
    test_dfs = {}
    
    for source, path in data_config.dataset_sources.items():
        source_path = Path(path)
        
        if not source_path.exists():
            logger.warning(f"Source path does not exist: {source_path}")
            continue
        
        train_csv = source_path / 'train.csv'
        val_csv   = source_path / 'val.csv'
        test_csv  = source_path / 'test.csv'
        
        if train_csv.exists():
            train_dfs[source] = pd.read_csv(train_csv)
            logger.info(f'✓ Loaded {len(train_dfs[source])} training samples from {source}')
        else:
            logger.warning(f"✗ Training CSV not found: {train_csv}")
        
        if val_csv.exists():
            val_dfs[source] = pd.read_csv(val_csv)
            logger.info(f'✓ Loaded {len(val_dfs[source])} validation samples from {source}')
        else:
            logger.warning(f"✗ Validation CSV not found: {val_csv}")
        
        if test_csv.exists():
            test_dfs[source] = pd.read_csv(test_csv)
            logger.info(f'✓ Loaded {len(test_dfs[source])} test samples from {source}')
    
    if not train_dfs:
        raise ValueError("No training data found! Please check your data directory.")
    if not val_dfs:
        raise ValueError("No validation data found! Please check your data directory.")
    
    return train_dfs, val_dfs, test_dfs if test_dfs else None


def main():
    parser = argparse.ArgumentParser(
        description='Train Cross-Domain Fake News Detector with Adversarial Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data_dir',    type=str, default='data')
    parser.add_argument('--batch_size',  type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model arguments
    parser.add_argument('--text_encoder',   type=str, default='xlm-roberta-base')
    parser.add_argument('--vision_encoder', type=str,
                        default='swin_base_patch4_window7_224')
    
    # Domain adaptation arguments
    parser.add_argument('--use_domain_adversarial', action='store_true', default=True)
    parser.add_argument('--no_domain_adversarial',  dest='use_domain_adversarial',
                        action='store_false')
    parser.add_argument('--adversarial_weight',       type=float, default=0.1)
    parser.add_argument('--gradient_reversal_lambda', type=float, default=1.0)
    parser.add_argument('--lambda_schedule', type=str,  default='linear',
                        choices=['constant', 'linear', 'exponential'])
    parser.add_argument('--use_domain_bn', action='store_true', default=True)
    
    # Evidential learning arguments
    parser.add_argument('--use_evidential', action='store_true', default=True)
    parser.add_argument('--no_evidential',  dest='use_evidential', action='store_false')
    parser.add_argument('--kl_weight',       type=float, default=0.01)
    parser.add_argument('--annealing_start', type=int,   default=10)
    parser.add_argument('--use_domain_priors', action='store_true', default=True)
    
    # Meta-learning arguments
    parser.add_argument('--use_meta_learning', action='store_true', default=False)
    parser.add_argument('--meta_epochs',  type=int,   default=5)
    parser.add_argument('--inner_lr',     type=float, default=0.01)
    parser.add_argument('--inner_steps',  type=int,   default=5)
    parser.add_argument('--k_shot',       type=int,   default=5)
    
    # Uncertainty-weighted domain adversarial training
    parser.add_argument('--uncertainty_weighting', type=str, default='none',
                        choices=['none', 'static', 'adaptive', 'threshold'])
    parser.add_argument('--uncertainty_alpha',     type=float, default=0.5)
    parser.add_argument('--uncertainty_threshold', type=float, default=0.3)
    
    # Training arguments
    parser.add_argument('--epochs',         type=int,   default=30)
    parser.add_argument('--lr',             type=float, default=2e-5)
    parser.add_argument('--weight_decay',   type=float, default=0.01)
    parser.add_argument('--warmup_steps',   type=int,   default=500)
    parser.add_argument('--patience',       type=int,   default=5)
    parser.add_argument('--gradient_clip',  type=float, default=1.0)
    
    # Imbalance handling
    parser.add_argument('--class_balance',  type=str, default='class_weights',
                        choices=['none', 'oversample', 'undersample',
                                 'focal_loss', 'class_weights'])
    parser.add_argument('--source_balance', type=str, default='balanced_batch',
                        choices=['none', 'weighted_sampling', 'balanced_batch'])
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir',  type=str, default='logs')
    
    # Resume / experiment
    parser.add_argument('--resume',   type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--experiment_mode', action='store_true')
    
    args = parser.parse_args()
    
    if args.exp_name:
        args.save_dir = f"{args.save_dir}/{args.exp_name}"
        args.log_dir  = f"{args.log_dir}/{args.exp_name}"
    
    # ── Log configuration ─────────────────────────────────────────────────
    logger.info("="*80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Data directory:        {args.data_dir}")
    logger.info(f"Batch size:            {args.batch_size}")
    logger.info(f"Epochs:                {args.epochs}")
    logger.info(f"Learning rate:         {args.lr}")
    logger.info(f"Text encoder:          {args.text_encoder}")
    logger.info(f"Vision encoder:        {args.vision_encoder}")
    logger.info(f"Domain adversarial:    {args.use_domain_adversarial}")
    logger.info(f"Evidential learning:   {args.use_evidential}")
    logger.info(f"Meta-learning:         {args.use_meta_learning}")
    logger.info(f"Uncertainty weighting: {args.uncertainty_weighting}")
    if args.uncertainty_weighting != 'none':
        logger.info(f"  └─ Alpha: {args.uncertainty_alpha}")
        if args.uncertainty_weighting == 'threshold':
            logger.info(f"  └─ Threshold: {args.uncertainty_threshold}")
    logger.info(f"Class balancing:       {args.class_balance}")
    logger.info(f"Source balancing:      {args.source_balance}")
    logger.info("="*80 + "\n")
    
    # ── Configs ───────────────────────────────────────────────────────────
    data_config = DataConfig()
    data_config.batch_size  = args.batch_size
    data_config.num_workers = args.num_workers
    data_config.imbalance_config.class_balance_strategy  = args.class_balance
    data_config.imbalance_config.source_balance_strategy = args.source_balance
    
    model_config = ModelConfig()
    model_config.text_encoder_name   = args.text_encoder
    model_config.vision_encoder_name = args.vision_encoder
    model_config.use_explanation_head = False

    model_config.domain_adaptation.use_domain_adversarial    = args.use_domain_adversarial
    model_config.domain_adaptation.adversarial_loss_weight   = args.adversarial_weight
    model_config.domain_adaptation.gradient_reversal_lambda  = args.gradient_reversal_lambda
    model_config.domain_adaptation.lambda_schedule           = args.lambda_schedule
    model_config.domain_adaptation.use_domain_specific_bn    = args.use_domain_bn

    model_config.evidential.use_evidential              = args.use_evidential
    model_config.evidential.kl_weight                   = args.kl_weight
    model_config.evidential.annealing_start             = args.annealing_start
    model_config.evidential.use_domain_specific_priors  = args.use_domain_priors

    training_config = TrainingConfig()
    training_config.epochs          = args.epochs
    training_config.learning_rate   = args.lr
    training_config.weight_decay    = args.weight_decay
    training_config.warmup_steps    = args.warmup_steps
    training_config.patience        = args.patience
    training_config.gradient_clip   = args.gradient_clip
    training_config.save_dir        = args.save_dir
    training_config.log_dir         = args.log_dir
    training_config.imbalance_config = data_config.imbalance_config
    training_config.domain_weight   = args.adversarial_weight

    training_config.meta_learning.use_meta_learning          = args.use_meta_learning
    training_config.meta_learning.meta_learning_epochs       = args.meta_epochs
    training_config.meta_learning.inner_lr                   = args.inner_lr
    training_config.meta_learning.inner_steps                = args.inner_steps
    training_config.meta_learning.support_samples_per_class  = args.k_shot
    
    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    if torch.cuda.is_available():
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
        logger.info(
            f'GPU Memory: '
            f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB'
        )
    logger.info('')
    
    # ── Data ──────────────────────────────────────────────────────────────
    logger.info('Loading datasets...')
    train_dfs, val_dfs, test_dfs = load_datasets(data_config)
    logger.info('')

    all_sources  = set(train_dfs.keys()) | set(val_dfs.keys())
    if test_dfs:
        all_sources |= set(test_dfs.keys())
    source_to_id = {s: i for i, s in enumerate(sorted(all_sources))}

    logger.info('Source → ID mapping:')
    for source, idx in sorted(source_to_id.items(), key=lambda x: x[1]):
        logger.info(f'  {source}: {idx}')
    logger.info('')

    logger.info('Creating dataloaders...')
    train_loader, val_loader, test_loader, balance_info = create_dataloaders(
        data_config, train_dfs, val_dfs, test_dfs
    )
    logger.info('')
    
    # ── Tokenizer ─────────────────────────────────────────────────────────
    logger.info(f'Loading tokenizer: {model_config.text_encoder_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_config.text_encoder_name)
    logger.info(f'Vocabulary size: {len(tokenizer):,}')
    logger.info('')
    
    # ── Model ─────────────────────────────────────────────────────────────
    logger.info('Initializing model...')
    model = MultimodalFakeNewsDetectorV2(
        config               = model_config,
        tokenizer_vocab_size = len(tokenizer),
        source_to_id         = source_to_id,
    )
    model = model.to(device)
    logger.info(f'Total parameters:     {model.get_num_parameters():,}')
    logger.info(f'Trainable parameters: {model.get_trainable_parameters():,}')
    logger.info('')

    # ── Save HuggingFace artifacts (once, before training) ────────────────
    # Writes tokenizer.json, tokenizer_config.json, spiece.model,
    # special_tokens_map.json, config.json, model.safetensors,
    # and generation_config.json (if applicable) to
    #   <save_dir>/hf_artifacts/
    # These files are used by inference.py and evaluate.py with --hf_dir
    # for fully offline deployment without re-downloading from the Hub.
    logger.info('Saving HuggingFace artifacts for offline deployment...')
    try:
        save_hf_artifacts(model, tokenizer, args.save_dir)
    except Exception as e:
        # Non-fatal — training can still proceed; warn and continue.
        logger.warning(f'Could not save HuggingFace artifacts: {e}')
    logger.info('')
    
    # ── Resume from checkpoint ────────────────────────────────────────────
    if args.resume:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info(f"  Resumed from epoch {ckpt['epoch']}")
        logger.info(f"  Previous best F1: {ckpt.get('best_f1', 0):.4f}")
        logger.info('')
    
    # ── Trainer ───────────────────────────────────────────────────────────
    logger.info('Initializing trainer...')

    if args.uncertainty_weighting != 'none':
        trainer = UncertaintyWeightedTrainer(
            model                = model,
            train_loader         = train_loader,
            val_loader           = val_loader,
            config               = training_config,
            model_config         = model_config,
            device               = device,
            balance_info         = balance_info,
            source_to_id         = source_to_id,
            uncertainty_weighting = args.uncertainty_weighting,
            uncertainty_alpha    = args.uncertainty_alpha,
        )
        logger.info("="*80)
        logger.info("✓ USING UNCERTAINTY-WEIGHTED DOMAIN ADVERSARIAL TRAINING")
        logger.info("="*80)
        logger.info(f"  Strategy: {args.uncertainty_weighting.upper()}")
        logger.info(f"  Alpha:    {args.uncertainty_alpha}")
        if args.uncertainty_weighting == 'threshold':
            logger.info(f"  Threshold: {args.uncertainty_threshold}")
        logger.info("="*80 + "\n")
    else:
        trainer = EnhancedTrainer(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            config       = training_config,
            model_config = model_config,
            device       = device,
            balance_info = balance_info,
            source_to_id = source_to_id,
        )
        logger.info("="*80)
        logger.info("✓ USING STANDARD DOMAIN ADVERSARIAL TRAINING (Baseline)")
        logger.info("="*80 + "\n")
    
    # ── Training ──────────────────────────────────────────────────────────
    logger.info('\n' + '='*80)
    logger.info('STARTING TRAINING')
    logger.info('='*80)
    logger.info(f"Experiment:            {args.exp_name or 'default'}")
    logger.info(f"Uncertainty weighting: {args.uncertainty_weighting}")
    logger.info('='*80 + '\n')
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info('\n\n' + '='*80)
        logger.info('Training interrupted by user')
        logger.info('='*80)
    except Exception as e:
        logger.error(f'\n\nTraining failed with error: {e}')
        import traceback
        traceback.print_exc()
        raise
    
    # ── Done ──────────────────────────────────────────────────────────────
    logger.info('\n' + '='*80)
    logger.info('TRAINING COMPLETED SUCCESSFULLY!')
    logger.info('='*80)
    logger.info(f'Best model:       {Path(args.save_dir) / "best_model.pt"}')
    logger.info(f'HF artifacts:     {Path(args.save_dir) / "hf_artifacts/"}')
    logger.info(f'Logs:             {args.log_dir}')

    if args.uncertainty_weighting != 'none' and hasattr(trainer, 'uncertainty_stats'):
        stats = trainer.uncertainty_stats
        logger.info('\nUncertainty Statistics:')
        if stats['train_uncertainties']:
            logger.info(f'  Avg uncertainty:  {stats["train_uncertainties"][-1]:.4f}')
            logger.info(f'  Avg domain weight:{stats["domain_weights"][-1]:.4f}')
        if stats['alpha_history'] and args.uncertainty_weighting == 'adaptive':
            logger.info(f'  Final learned α:  {stats["alpha_history"][-1]:.4f}')

    logger.info('='*80 + '\n')


if __name__ == '__main__':
    main()