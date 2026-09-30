# scripts/run_experiments.py
"""
Comprehensive Experimental Suite
=================================
Runs all experiments to validate innovation claims by loading
best_model.pt and evaluating per-domain on the test set.

Experiments
-----------
3. Uncertainty-Weighted Training comparison   (none / static / adaptive / threshold)
4. Component Ablation                         (baseline → full model)
5. SOTA Comparison                            (our model vs reported baselines)

All experiments share the same evaluation pipeline:
  load checkpoint → build model → evaluate per domain → save JSON + CSV
"""

import argparse
import sys
import json
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from config.config import DataConfig, ModelConfig, TrainingConfig
from src.data.dataset import MultiSourceDataset, create_dataloaders
from src.models.model import MultimodalFakeNewsDetectorV2
from src.training.uncertainty_weighted_trainer import UncertaintyWeightedTrainer
from src.utils.logger import setup_logger

logger = setup_logger('experiments', 'logs/experiments.log')


# ──────────────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_model_from_checkpoint(checkpoint_path: str, hf_dir: str, device: torch.device):
    """Load model + tokenizer from a checkpoint file."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config  = checkpoint['model_config']
    source_to_id  = checkpoint['source_to_id']

    # Resolve HF source (local dir or Hub)
    if hf_dir:
        hf_source = str(Path(hf_dir).resolve())
        logger.info(f"Using local HuggingFace files: {hf_source}")
        model_config.text_encoder_name = hf_source
    else:
        hf_source = model_config.text_encoder_name
        logger.info(f"Loading from Hub: {hf_source}")

    tokenizer = AutoTokenizer.from_pretrained(hf_source)

    model = MultimodalFakeNewsDetectorV2(
        config               = model_config,
        tokenizer_vocab_size = len(tokenizer),
        source_to_id         = source_to_id,
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device)
    model.eval()

    logger.info(
        f"  Checkpoint epoch {checkpoint['epoch']}  "
        f"val F1={checkpoint['metrics'].get('f1', 0):.4f}"
    )
    return model, tokenizer, model_config, source_to_id, checkpoint


def load_test_data(data_dir: str, data_config: DataConfig):
    """Return {source: DataFrame} for every source that has a test.csv."""
    test_dfs = {}
    for source, path in data_config.dataset_sources.items():
        test_csv = Path(path) / 'test.csv'
        if test_csv.exists():
            test_dfs[source] = pd.read_csv(test_csv)
            logger.info(f"  ✓ {source}: {len(test_dfs[source])} test samples")
        else:
            logger.warning(f"  ✗ {source}: test.csv not found ({test_csv})")
    if not test_dfs:
        raise ValueError("No test data found. Check --data_dir.")
    return test_dfs


def build_domain_loaders(test_dfs, data_config, tokenizer, batch_size, num_workers):
    """
    Build one DataLoader per domain so we can get per-domain metrics.
    Returns {domain: DataLoader}.
    """
    loaders = {}
    for source, df in test_dfs.items():
        single = {source: df}
        dataset = MultiSourceDataset(
            single, data_config, tokenizer,
            mode='test', include_explanations=False,
        )
        loaders[source] = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = True,
        )
    return loaders


@torch.no_grad()
def evaluate_domain(model, loader, source_to_id, device) -> dict:
    """
    Evaluate model on a single domain loader.
    Returns dict with accuracy, f1, auc, and per-sample arrays.
    """
    all_preds  = []
    all_labels = []
    all_probs  = []

    for batch in tqdm(loader, desc="  evaluating", leave=False):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images         = batch['image'].to(device)
        labels         = batch['label']
        sources        = batch.get('source', None)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            sources=sources,
        )

        probs = outputs['classification']['prob'].cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs[:, 1].tolist())   # prob of class 1 for AUC

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    acc = float(accuracy_score(all_labels, all_preds))
    f1  = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))
    try:
        auc = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        auc = float('nan')

    return {'accuracy': acc, 'f1': f1, 'auc': auc, 'n_samples': len(all_labels)}


def evaluate_all_domains(model, domain_loaders, source_to_id, device) -> dict:
    """Run evaluate_domain for every domain. Returns nested dict."""
    results = {}
    for domain, loader in domain_loaders.items():
        logger.info(f"  Domain: {domain}")
        results[domain] = evaluate_domain(model, loader, source_to_id, device)
        m = results[domain]
        print(
            f"    {domain:<20}  "
            f"Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}  "
            f"n={m['n_samples']}"
        )

    # Cross-domain averages (exclude nan AUC domains)
    accs = [v['accuracy'] for v in results.values()]
    f1s  = [v['f1']       for v in results.values()]
    aucs = [v['auc']      for v in results.values() if not np.isnan(v['auc'])]

    results['__avg__'] = {
        'accuracy':  float(np.mean(accs)),
        'f1':        float(np.mean(f1s)),
        'auc':       float(np.mean(aucs)) if aucs else float('nan'),
        'n_samples': sum(v['n_samples'] for v in results.values()),
    }
    m = results['__avg__']
    print(
        f"\n    {'CROSS-DOMAIN AVG':<20}  "
        f"Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}"
    )
    return results


def save_results(results: dict, output_path: Path):
    """Save results dict as pretty JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Results saved → {output_path}")



def train_from_scratch(
    exp_name: str,
    args,
    model_config_overrides: dict,
    uncertainty_weighting: str = 'adaptive',
    device: torch.device = None,
) -> str:
    """
    Train a model from scratch with the given config overrides.
    Returns the path to best_model.pt.

    model_config_overrides keys:
        use_domain_adversarial, use_evidential, use_meta_learning
    uncertainty_weighting: none | static | adaptive | threshold
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = str(Path('checkpoints') / exp_name)
    log_dir  = str(Path('logs') / exp_name)

    print(f'\n  Training {exp_name} from scratch...')
    logger.info(f'Training config: {exp_name}  overrides={model_config_overrides}')

    data_config              = DataConfig()
    data_config.data_dir     = args.data_dir
    model_config             = ModelConfig()
    training_config          = TrainingConfig()
    training_config.save_dir = save_dir
    training_config.log_dir  = log_dir

    # Apply component-level overrides
    for key, val in model_config_overrides.items():
        if hasattr(model_config.domain_adaptation, key):
            setattr(model_config.domain_adaptation, key, val)
        elif hasattr(model_config.evidential, key):
            setattr(model_config.evidential, key, val)
        elif hasattr(training_config.meta_learning, key):
            setattr(training_config.meta_learning, key, val)
        else:
            logger.warning(f'  Unknown override key: {key}')

    # Shorten run for quick_test mode
    if getattr(args, 'quick_test', False):
        training_config.epochs   = 5
        training_config.patience = 3

    # Tokenizer
    if args.hf_dir:
        model_config.text_encoder_name = str(Path(args.hf_dir).resolve())
    tokenizer = AutoTokenizer.from_pretrained(model_config.text_encoder_name)

    # Data
    train_loader, val_loader, balance_info, source_to_id = create_dataloaders(
        data_config, tokenizer
    )

    # Model
    model = MultimodalFakeNewsDetectorV2(
        config               = model_config,
        tokenizer_vocab_size = len(tokenizer),
        source_to_id         = source_to_id,
    ).to(device)

    # Trainer
    trainer = UncertaintyWeightedTrainer(
        model                 = model,
        train_loader          = train_loader,
        val_loader            = val_loader,
        config                = training_config,
        model_config          = model_config,
        device                = device,
        balance_info          = balance_info,
        source_to_id          = source_to_id,
        uncertainty_weighting = uncertainty_weighting,
    )
    trainer.train()

    best_ckpt = str(Path(save_dir) / 'best_model.pt')
    logger.info(f'  Saved -> {best_ckpt}')
    return best_ckpt


def results_to_table(results: dict, exp_name: str) -> pd.DataFrame:
    """Convert nested results dict to a flat DataFrame row per domain."""
    rows = []
    for domain, metrics in results.items():
        rows.append({
            'experiment': exp_name,
            'domain':     domain,
            'accuracy':   metrics.get('accuracy', float('nan')),
            'f1':         metrics.get('f1',       float('nan')),
            'auc':        metrics.get('auc',       float('nan')),
            'n_samples':  metrics.get('n_samples', 0),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 3 — Uncertainty-Weighted Training comparison
# ──────────────────────────────────────────────────────────────────────────────

def experiment_3_uncertainty_weighting(args, domain_loaders, device):
    """
    Load the checkpoint for each weighting strategy and evaluate per domain.
    Expects checkpoints at:
        checkpoints/<exp_name>_none/best_model.pt
        checkpoints/<exp_name>_static/best_model.pt
        checkpoints/<exp_name>_adaptive/best_model.pt
        checkpoints/<exp_name>_threshold/best_model.pt

    Falls back to args.checkpoint if a strategy-specific checkpoint is missing,
    so you can still run the experiment with a single checkpoint.
    """
    print('\n' + '='*80)
    print('EXPERIMENT 3: Uncertainty-Weighted Training Comparison')
    print('='*80)

    strategies = ['none', 'static', 'adaptive', 'threshold']
    all_results = {}
    all_tables  = []

    for strategy in strategies:
        print(f'\n── Strategy: {strategy} ──')

        strategy_ckpt = Path('checkpoints') / f'exp3_{strategy}' / 'best_model.pt'

        if args.train_configs:
            # Train from scratch for this weighting strategy
            ckpt_path = train_from_scratch(
                exp_name              = f'exp3_{strategy}',
                args                  = args,
                model_config_overrides = {
                    'use_domain_adversarial': True,
                    'use_evidential':         True,
                },
                uncertainty_weighting = strategy,
                device                = device,
            )
        elif strategy_ckpt.exists():
            ckpt_path = str(strategy_ckpt)
        else:
            logger.warning(
                f"  No checkpoint for strategy '{strategy}' at {strategy_ckpt}. "
                f"Falling back to {args.checkpoint}."
            )
            ckpt_path = args.checkpoint

        model, tokenizer, model_config, source_to_id, ckpt = load_model_from_checkpoint(
            ckpt_path, args.hf_dir, device
        )

        results = evaluate_all_domains(model, domain_loaders, source_to_id, device)
        all_results[strategy] = results
        all_tables.append(results_to_table(results, exp_name=strategy))

        del model

    # Save
    output_dir = Path(args.output_dir) / 'experiment_3'
    save_results(all_results, output_dir / 'results.json')

    summary_df = pd.concat(all_tables, ignore_index=True)
    summary_df.to_csv(output_dir / 'summary.csv', index=False)
    logger.info(f"  Summary table → {output_dir / 'summary.csv'}")

    # Print compact comparison table
    print('\nSummary (cross-domain averages):')
    print(f"  {'Strategy':<12}  {'Acc':>7}  {'F1':>7}  {'AUC':>7}")
    print('  ' + '-'*38)
    for strategy, res in all_results.items():
        avg = res.get('__avg__', {})
        print(
            f"  {strategy:<12}  "
            f"{avg.get('accuracy', float('nan')):>7.4f}  "
            f"{avg.get('f1', float('nan')):>7.4f}  "
            f"{avg.get('auc', float('nan')):>7.4f}"
        )

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 4 — Component Ablation
# ──────────────────────────────────────────────────────────────────────────────

def experiment_4_ablation(args, domain_loaders, device):
    """
    Load the checkpoint for each ablation configuration and evaluate per domain.
    Expects checkpoints at:
        checkpoints/ablation_baseline/best_model.pt
        checkpoints/ablation_domain_adv/best_model.pt
        checkpoints/ablation_evidential/best_model.pt
        checkpoints/ablation_meta/best_model.pt
        checkpoints/ablation_full/best_model.pt  (or args.checkpoint)
    """
    print('\n' + '='*80)
    print('EXPERIMENT 4: Component Ablation Study')
    print('='*80)

    configurations = {
        'baseline':    'Baseline (no domain adapt, no evidential)',
        'domain_adv':  '+Domain Adversarial',
        'evidential':  '+Evidential Learning',
        'meta':        '+Meta-Learning',
        'full':        'Full Model',
    }

    all_results = {}
    all_tables  = []

    # Component flag sets for each ablation config
    component_flags = {
        'baseline':   {'use_domain_adversarial': False, 'use_evidential': False, 'use_meta_learning': False},
        'domain_adv': {'use_domain_adversarial': True,  'use_evidential': False, 'use_meta_learning': False},
        'evidential': {'use_domain_adversarial': False, 'use_evidential': True,  'use_meta_learning': False},
        'meta':       {'use_domain_adversarial': False, 'use_evidential': False, 'use_meta_learning': True},
        'full':       {'use_domain_adversarial': True,  'use_evidential': True,  'use_meta_learning': True},
    }

    for config_name, config_desc in configurations.items():
        print(f'\n── Config: {config_name}  ({config_desc}) ──')

        ablation_ckpt = Path('checkpoints') / f'ablation_{config_name}' / 'best_model.pt'
        # 'full' can also use the main provided checkpoint
        if config_name == 'full' and not ablation_ckpt.exists():
            ablation_ckpt = Path(args.checkpoint)

        if args.train_configs:
            ckpt_path = train_from_scratch(
                exp_name               = f'ablation_{config_name}',
                args                   = args,
                model_config_overrides = component_flags[config_name],
                uncertainty_weighting  = 'adaptive',
                device                 = device,
            )
        elif ablation_ckpt.exists():
            ckpt_path = str(ablation_ckpt)
        else:
            logger.warning(
                f"  No checkpoint for '{config_name}' at {ablation_ckpt}. Skipping. "
                f"Run with --train_configs to train it."
            )
            all_results[config_name] = {'error': 'checkpoint not found — use --train_configs'}
            continue

        model, tokenizer, model_config, source_to_id, ckpt = load_model_from_checkpoint(
            ckpt_path, args.hf_dir, device
        )

        results = evaluate_all_domains(model, domain_loaders, source_to_id, device)
        all_results[config_name] = results
        all_tables.append(results_to_table(results, exp_name=config_name))

        del model

    # Save
    output_dir = Path(args.output_dir) / 'experiment_4'
    save_results(all_results, output_dir / 'results.json')

    if all_tables:
        summary_df = pd.concat(all_tables, ignore_index=True)
        summary_df.to_csv(output_dir / 'summary.csv', index=False)
        logger.info(f"  Summary table → {output_dir / 'summary.csv'}")

    # Print compact comparison
    print('\nSummary (cross-domain averages):')
    print(f"  {'Config':<14}  {'Acc':>7}  {'F1':>7}  {'AUC':>7}")
    print('  ' + '-'*40)
    for config_name, res in all_results.items():
        if 'error' in res:
            print(f"  {config_name:<14}  {'N/A':>7}  {'N/A':>7}  {'N/A':>7}  ← checkpoint missing")
            continue
        avg = res.get('__avg__', {})
        print(
            f"  {config_name:<14}  "
            f"{avg.get('accuracy', float('nan')):>7.4f}  "
            f"{avg.get('f1', float('nan')):>7.4f}  "
            f"{avg.get('auc', float('nan')):>7.4f}"
        )

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 5 — SOTA Comparison
# ──────────────────────────────────────────────────────────────────────────────

def experiment_5_sota_comparison(args, domain_loaders, device):
    """
    Evaluate our best model and compare against published SOTA numbers.
    SOTA numbers are from published papers (hardcoded as reported values).
    """
    print('\n' + '='*80)
    print('EXPERIMENT 5: SOTA Comparison')
    print('='*80)

    # Published results from papers (accuracy, f1, auc where available)
    # Update these with actual reported numbers for your target datasets
    sota_baselines = {
        'EANN (2018)':    {'accuracy': 0.827, 'f1': 0.826, 'auc': float('nan')},
        'SAFE (2019)':    {'accuracy': 0.848, 'f1': 0.847, 'auc': float('nan')},
        'MCAN (2023)':    {'accuracy': 0.899, 'f1': 0.898, 'auc': float('nan')},
        'CAFE (2022)':    {'accuracy': 0.881, 'f1': 0.880, 'auc': float('nan')},
    }

    print('\nEvaluating our model...')
    model, tokenizer, model_config, source_to_id, ckpt = load_model_from_checkpoint(
        args.checkpoint, args.hf_dir, device
    )
    our_results = evaluate_all_domains(model, domain_loaders, source_to_id, device)
    del model

    our_avg = our_results.get('__avg__', {})

    # Save full per-domain results
    output_dir = Path(args.output_dir) / 'experiment_5'
    save_results({'ours': our_results, 'sota_baselines': sota_baselines},
                 output_dir / 'results.json')

    # Print comparison table
    print('\nSOTA Comparison (cross-domain average):')
    print(f"  {'Method':<22}  {'Acc':>7}  {'F1':>7}  {'AUC':>7}")
    print('  ' + '-'*46)
    for method, metrics in sota_baselines.items():
        auc_str = f"{metrics['auc']:>7.4f}" if not np.isnan(metrics['auc']) else f"{'N/A':>7}"
        print(f"  {method:<22}  {metrics['accuracy']:>7.4f}  {metrics['f1']:>7.4f}  {auc_str}")
    print('  ' + '-'*46)
    print(
        f"  {'Ours (full model)':<22}  "
        f"{our_avg.get('accuracy', float('nan')):>7.4f}  "
        f"{our_avg.get('f1', float('nan')):>7.4f}  "
        f"{our_avg.get('auc', float('nan')):>7.4f}  ← best_model.pt"
    )

    return {'ours': our_results, 'sota': sota_baselines}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive experiments — evaluate best_model.pt per domain',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint',   type=str, required=True,
                        help='Path to best_model.pt (used as fallback for all experiments)')
    parser.add_argument('--hf_dir',       type=str, default=None,
                        help='Local HuggingFace files dir (offline mode)')
    parser.add_argument('--data_dir',     type=str, default='data',
                        help='Root data directory')
    parser.add_argument('--output_dir',   type=str, default='experiments',
                        help='Directory to save all experiment results')
    parser.add_argument('--batch_size',   type=int, default=16)
    parser.add_argument('--num_workers',  type=int, default=4)
    parser.add_argument('--experiments',  nargs='+',
                        choices=['3', '4', '5', 'all'], default=['all'],
                        help='Which experiments to run')
    parser.add_argument('--train_configs', action='store_true', default=False,
                        help='Train each config from scratch instead of loading '
                             'existing checkpoints. Default: False (eval only).')
    parser.add_argument('--quick_test', action='store_true', default=False,
                        help='When --train_configs is set, limit to 5 epochs '
                             'per config for a fast sanity check.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load test data once (shared across all experiments) ───────────────
    print('\nLoading test datasets...')
    data_config             = DataConfig()
    data_config.data_dir    = args.data_dir
    data_config.batch_size  = args.batch_size
    data_config.num_workers = args.num_workers

    test_dfs = load_test_data(args.data_dir, data_config)

    # We need a tokenizer to build datasets — load from checkpoint
    _, tokenizer, _, _, _ = load_model_from_checkpoint(
        args.checkpoint, args.hf_dir, device
    )

    domain_loaders = build_domain_loaders(
        test_dfs, data_config, tokenizer,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f'Domains available: {list(domain_loaders.keys())}\n')

    # ── Run requested experiments ─────────────────────────────────────────
    to_run = args.experiments if 'all' not in args.experiments else ['3', '4', '5']

    if '3' in to_run:
        experiment_3_uncertainty_weighting(args, domain_loaders, device)

    if '4' in to_run:
        experiment_4_ablation(args, domain_loaders, device)

    if '5' in to_run:
        experiment_5_sota_comparison(args, domain_loaders, device)

    print('\n' + '='*80)
    print(f'ALL EXPERIMENTS COMPLETE  →  {args.output_dir}/')
    print('='*80)


if __name__ == '__main__':
    main()