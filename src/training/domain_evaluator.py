"""
Comprehensive Domain Shift Evaluation
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from ..utils.metrics import MetricsTracker

logger = logging.getLogger(__name__)


class DomainShiftEvaluator:
    """
    Evaluator with comprehensive domain shift analysis
    
    Provides:
    - Per-domain performance metrics
    - Uncertainty analysis by domain
    - Domain confusion analysis
    - Calibration analysis
    """
    
    def __init__(
        self,
        model: nn.Module,
        test_loader,
        device: torch.device,
        source_to_id: Dict[str, int],
        output_dir: str = 'evaluation_results'
    ):
        """
        Args:
            model: Trained model
            test_loader: Test data loader
            device: Device to run on
            source_to_id: Mapping from source names to IDs
            output_dir: Directory to save results
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.source_to_id = source_to_id
        self.id_to_source = {v: k for k, v in source_to_id.items()}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DomainShiftEvaluator initialized")
        logger.info(f"  Domains: {list(source_to_id.keys())}")
        logger.info(f"  Output: {self.output_dir}")
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Run comprehensive evaluation
        
        Returns:
            Overall metrics dictionary
        """
        self.model.eval()
        
        # Trackers
        overall_tracker = MetricsTracker()
        domain_trackers = {
            source: MetricsTracker()
            for source in self.source_to_id.keys()
        }
        
        # Uncertainty analysis
        uncertainty_by_domain = defaultdict(list)
        uncertainty_correct = defaultdict(list)
        uncertainty_incorrect = defaultdict(list)
        
        # Predictions storage
        all_predictions = []
        
        logger.info("Running evaluation...")
        pbar = tqdm(self.test_loader, desc='Evaluating')
        
        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            sources = batch.get('source', None)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                sources=sources,
                return_features=True
            )
            
            # Get predictions
            probs = outputs['classification']['prob'].cpu().numpy()
            preds = np.argmax(probs, axis=1)
            labels_np = labels.cpu().numpy()
            
            # Get uncertainty
            if 'uncertainty' in outputs['classification']:
                uncertainties = outputs['classification']['uncertainty'].cpu().numpy().flatten()
            else:
                uncertainties = np.zeros(len(preds))
            
            # Track overall
            overall_tracker.update(
                predictions=preds,
                labels=labels_np,
                probabilities=probs,
                sources=sources
            )
            
            # Track per-domain
            for i, source in enumerate(sources):
                domain_trackers[source].update(
                    predictions=[preds[i]],
                    labels=[labels_np[i]],
                    probabilities=[probs[i]]
                )
                
                # Uncertainty tracking
                uncertainty_by_domain[source].append(uncertainties[i])
                
                if preds[i] == labels_np[i]:
                    uncertainty_correct[source].append(uncertainties[i])
                else:
                    uncertainty_incorrect[source].append(uncertainties[i])
            
            # Store predictions
            for i in range(len(preds)):
                pred_dict = {
                    'prediction': int(preds[i]),
                    'true_label': int(labels_np[i]),
                    'prob_real': float(probs[i, 0]),
                    'prob_fake': float(probs[i, 1]),
                    'uncertainty': float(uncertainties[i]),
                    'source': sources[i],
                    'correct': bool(preds[i] == labels_np[i])
                }
                all_predictions.append(pred_dict)
        
        # Compute metrics
        overall_metrics = overall_tracker.compute_metrics()
        domain_metrics = {
            source: tracker.compute_metrics()
            for source, tracker in domain_trackers.items()
        }
        
        # Print and save results
        self._print_results(overall_metrics, domain_metrics, uncertainty_by_domain)
        self._save_results(
            overall_metrics, domain_metrics,
            uncertainty_by_domain, uncertainty_correct, uncertainty_incorrect,
            all_predictions
        )
        self._generate_visualizations(
            domain_metrics, uncertainty_by_domain,
            uncertainty_correct, uncertainty_incorrect
        )
        
        return overall_metrics
    
    def _print_results(
        self,
        overall_metrics: Dict,
        domain_metrics: Dict[str, Dict],
        uncertainty_by_domain: Dict
    ):
        """Print results to console"""
        logger.info("\n" + "="*80)
        logger.info("OVERALL RESULTS")
        logger.info("="*80)
        for key, value in overall_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"{key:25s}: {value:.4f}")
        
        logger.info("\n" + "="*80)
        logger.info("PER-DOMAIN RESULTS")
        logger.info("="*80)
        
        for source in sorted(domain_metrics.keys()):
            metrics = domain_metrics[source]
            unc = uncertainty_by_domain.get(source, [])
            avg_unc = np.mean(unc) if unc else 0
            
            logger.info(f"\n{source}:")
            logger.info(f"  Accuracy:    {metrics.get('accuracy', 0):.4f}")
            logger.info(f"  F1:          {metrics.get('f1', 0):.4f}")
            logger.info(f"  Precision:   {metrics.get('precision', 0):.4f}")
            logger.info(f"  Recall:      {metrics.get('recall', 0):.4f}")
            logger.info(f"  Uncertainty: {avg_unc:.4f}")
    
    def _save_results(
        self,
        overall_metrics: Dict,
        domain_metrics: Dict,
        uncertainty_by_domain: Dict,
        uncertainty_correct: Dict,
        uncertainty_incorrect: Dict,
        predictions: List[Dict]
    ):
        """Save all results to files"""
        # Overall metrics
        with open(self.output_dir / 'overall_metrics.json', 'w') as f:
            json.dump(overall_metrics, f, indent=2)
        
        # Domain metrics
        domain_metrics_clean = {}
        for source, metrics in domain_metrics.items():
            domain_metrics_clean[source] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in metrics.items()
                if isinstance(v, (int, float, np.number))
            }
        
        with open(self.output_dir / 'domain_metrics.json', 'w') as f:
            json.dump(domain_metrics_clean, f, indent=2)
        
        # Uncertainty analysis
        uncertainty_analysis = {}
        for source in uncertainty_by_domain.keys():
            uncertainty_analysis[source] = {
                'overall_mean': float(np.mean(uncertainty_by_domain[source])),
                'overall_std': float(np.std(uncertainty_by_domain[source])),
                'correct_mean': float(np.mean(uncertainty_correct.get(source, [0]))),
                'incorrect_mean': float(np.mean(uncertainty_incorrect.get(source, [0])))
            }
        
        with open(self.output_dir / 'uncertainty_analysis.json', 'w') as f:
            json.dump(uncertainty_analysis, f, indent=2)
        
        # Predictions
        df = pd.DataFrame(predictions)
        df.to_csv(self.output_dir / 'predictions.csv', index=False)
        
        # Errors
        errors = df[df['correct'] == False]
        errors.to_csv(self.output_dir / 'errors.csv', index=False)
        
        logger.info(f"\nSaved results to {self.output_dir}")
    
    def _generate_visualizations(
        self,
        domain_metrics: Dict,
        uncertainty_by_domain: Dict,
        uncertainty_correct: Dict,
        uncertainty_incorrect: Dict
    ):
        """Generate visualization plots"""
        # 1. Domain performance comparison
        self._plot_domain_comparison(domain_metrics)
        
        # 2. Uncertainty distributions
        self._plot_uncertainty_distribution(uncertainty_by_domain)
        
        # 3. Uncertainty calibration
        self._plot_uncertainty_calibration(uncertainty_correct, uncertainty_incorrect)
    
    def _plot_domain_comparison(self, domain_metrics: Dict):
        """Plot performance across domains"""
        metrics_to_plot = ['accuracy', 'f1', 'precision', 'recall']
        sources = sorted(domain_metrics.keys())
        
        data = {m: [domain_metrics[s].get(m, 0) for s in sources] for m in metrics_to_plot}
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            bars = ax.bar(sources, data[metric], color='steelblue', alpha=0.7)
            
            # Highlight best/worst
            values = data[metric]
            if values:
                best_idx = np.argmax(values)
                worst_idx = np.argmin(values)
                bars[best_idx].set_color('green')
                bars[worst_idx].set_color('red')
            
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} by Domain')
            ax.set_ylim([0, 1])
            ax.grid(alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'domain_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_uncertainty_distribution(self, uncertainty_by_domain: Dict):
        """Plot uncertainty distributions"""
        sources = sorted(uncertainty_by_domain.keys())
        data = [uncertainty_by_domain[s] for s in sources]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bp = ax.boxplot(data, labels=sources, patch_artist=True, showfliers=False)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_ylabel('Uncertainty')
        ax.set_title('Uncertainty Distribution by Domain')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'uncertainty_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_uncertainty_calibration(
        self,
        uncertainty_correct: Dict,
        uncertainty_incorrect: Dict
    ):
        """Plot uncertainty calibration"""
        sources = sorted(set(uncertainty_correct.keys()) | set(uncertainty_incorrect.keys()))
        
        correct_means = [np.mean(uncertainty_correct.get(s, [0])) for s in sources]
        incorrect_means = [np.mean(uncertainty_incorrect.get(s, [0])) for s in sources]
        
        x = np.arange(len(sources))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, correct_means, width, label='Correct', color='green', alpha=0.7)
        ax.bar(x + width/2, incorrect_means, width, label='Incorrect', color='red', alpha=0.7)
        
        ax.set_ylabel('Average Uncertainty')
        ax.set_title('Uncertainty Calibration')
        ax.set_xticks(x)
        ax.set_xticklabels(sources, rotation=45, ha='right')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'uncertainty_calibration.png', dpi=300, bbox_inches='tight')
        plt.close()