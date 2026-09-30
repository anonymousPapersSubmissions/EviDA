# scripts/evaluate.py
"""
Comprehensive Evaluation Script with Domain Shift Analysis.

Deployment usage
----------------
Pass --hf_dir to the directory containing your saved HuggingFace files:
    tokenizer.json  tokenizer_config.json  spiece.model
    special_tokens_map.json  model.safetensors
    generation_config.json  config.json

This allows fully offline evaluation without downloading from the Hub.
If --hf_dir is omitted the Hub ID stored in the checkpoint is used
(original behaviour, works fine during development).
"""

import torch
import argparse
import sys
from pathlib import Path
import pandas as pd
import logging

sys.path.append(str(Path(__file__).parent.parent))

from config.config import DataConfig
from src.data.dataset import MultiSourceDataset
from src.models.model import MultimodalFakeNewsDetectorV2
from src.training.domain_evaluator import DomainShiftEvaluator
from src.utils.logger import setup_logger
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

logger = setup_logger('evaluation', 'logs/evaluation.log')


def load_test_datasets(data_config: DataConfig):
    """Load test datasets from all sources."""
    test_dfs = {}

    for source, path in data_config.dataset_sources.items():
        test_csv = Path(path) / 'test.csv'

        if test_csv.exists():
            test_dfs[source] = pd.read_csv(test_csv)
            logger.info(
                f'✓ Loaded {len(test_dfs[source])} test samples from {source}'
            )
        else:
            logger.warning(f'✗ Test CSV not found: {test_csv}')

    if not test_dfs:
        raise ValueError('No test data found! Please check your data directory.')

    return test_dfs


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Cross-Domain Fake News Detector with Domain Shift Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint',  type=str, required=True,
                        help='Path to model checkpoint (best_model.pt)')
    parser.add_argument('--hf_dir', type=str, default=None,
                        help='Directory with saved HuggingFace files '
                             '(tokenizer.json, config.json, model.safetensors, …). '
                             'Enables fully offline evaluation. '
                             'Falls back to Hub download when omitted.')
    parser.add_argument('--data_dir',    type=str, default='data',
                        help='Root data directory')
    parser.add_argument('--output_dir',  type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size',  type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    args = parser.parse_args()

    logger.info('='*80)
    logger.info('EVALUATION CONFIGURATION')
    logger.info('='*80)
    logger.info(f'Checkpoint:    {args.checkpoint}')
    logger.info(f'HF directory:  {args.hf_dir or "Hub (online)"}')
    logger.info(f'Data dir:      {args.data_dir}')
    logger.info(f'Output dir:    {args.output_dir}')
    logger.info(f'Batch size:    {args.batch_size}')
    logger.info('='*80 + '\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    if torch.cuda.is_available():
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
    logger.info('')

    # ── Load checkpoint ───────────────────────────────────────────────────
    logger.info(f'Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model_config = checkpoint['model_config']
    source_to_id = checkpoint['source_to_id']

    logger.info(f'  Epoch:   {checkpoint["epoch"]}')
    logger.info(f'  F1:      {checkpoint["metrics"].get("f1", 0):.4f}')
    logger.info(f'  Domains: {list(source_to_id.keys())}')
    logger.info('')

    # ── Resolve text-encoder source ───────────────────────────────────────
    # Override the stored Hub ID with the local directory when provided,
    # so both AutoTokenizer and the TextEncoder backbone load offline.
    if args.hf_dir is not None:
        hf_source = str(Path(args.hf_dir).resolve())
        logger.info(f'Using local HuggingFace files: {hf_source}')
        model_config.text_encoder_name = hf_source
    else:
        hf_source = model_config.text_encoder_name
        logger.info(f'Loading from Hub: {hf_source}')

    # ── Tokenizer ─────────────────────────────────────────────────────────
    logger.info(f'Loading tokenizer from: {hf_source}')
    tokenizer = AutoTokenizer.from_pretrained(hf_source)
    logger.info('')

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info('Initializing model…')
    model = MultimodalFakeNewsDetectorV2(
        config               = model_config,
        tokenizer_vocab_size = len(tokenizer),
        source_to_id         = source_to_id,
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device)
    model.eval()

    logger.info(f'Parameters: {model.get_num_parameters():,}')
    logger.info('')

    # ── Data ──────────────────────────────────────────────────────────────
    data_config             = DataConfig()
    data_config.batch_size  = args.batch_size
    data_config.num_workers = args.num_workers

    logger.info('Loading test datasets…')
    test_dfs = load_test_datasets(data_config)
    logger.info('')

    test_dataset = MultiSourceDataset(
        test_dfs,
        data_config,
        tokenizer,
        mode                 = 'test',
        include_explanations = False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
    )

    logger.info(f'Test samples: {len(test_dataset):,}')
    logger.info('')

    # ── Evaluation ────────────────────────────────────────────────────────
    evaluator = DomainShiftEvaluator(
        model        = model,
        test_loader  = test_loader,
        device       = device,
        source_to_id = source_to_id,
        output_dir   = args.output_dir,
    )

    logger.info('='*80)
    logger.info('STARTING EVALUATION')
    logger.info('='*80 + '\n')

    try:
        metrics = evaluator.evaluate()
    except Exception as e:
        logger.error(f'Evaluation failed: {e}')
        raise

    logger.info('\n' + '='*80)
    logger.info('EVALUATION COMPLETE')
    logger.info('='*80)
    logger.info(f'Results saved to: {args.output_dir}')
    logger.info('Generated files:')
    for f in [
        'overall_metrics.json', 'domain_metrics.json',
        'uncertainty_analysis.json', 'predictions.csv', 'errors.csv',
        'domain_comparison.png', 'uncertainty_distribution.png',
        'uncertainty_calibration.png',
    ]:
        logger.info(f'  - {f}')
    logger.info('='*80 + '\n')


if __name__ == '__main__':
    main()