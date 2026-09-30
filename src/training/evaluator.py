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
from sklearn.metrics import confusion_matrix, roc_curve, auc

from ..utils.metrics import MetricsTracker

logger = logging.getLogger(__name__)


class Evaluator:
    """Comprehensive model evaluation"""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader,
        device: torch.device,
        output_dir: str = 'evaluation_results'
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Evaluator initialized. Results will be saved to {self.output_dir}")
    
    @torch.no_grad()
    def evaluate(self, save_predictions: bool = True) -> Dict[str, float]:
        """
        Comprehensive evaluation
        
        Args:
            save_predictions: Whether to save predictions to file
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        metrics_tracker = MetricsTracker()
        all_predictions = []
        all_metadata = []
        
        logger.info("Running evaluation...")
        pbar = tqdm(self.test_loader, desc='Evaluating')
        
        for batch_idx, batch in enumerate(pbar):
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
                images=images
            )
            
            # Get predictions
            probs = outputs['classification']['prob'].cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            # Track metrics
            metrics_tracker.update(
                predictions=preds,
                labels=labels.cpu().numpy(),
                probabilities=probs,
                sources=sources
            )
            
            # Store predictions and metadata
            if save_predictions:
                batch_size = labels.size(0)
                for i in range(batch_size):
                    pred_dict = {
                        'prediction': int(preds[i]),
                        'true_label': int(labels[i].cpu().item()),
                        'prob_real': float(probs[i, 0]),
                        'prob_fake': float(probs[i, 1]),
                        'source': sources[i] if sources else 'unknown'
                    }
                    
                    # Add uncertainty if available
                    if 'uncertainty' in outputs['classification']:
                        uncertainty = outputs['classification']['uncertainty'].cpu().numpy()
                        pred_dict['uncertainty'] = float(uncertainty[i])
                    
                    all_predictions.append(pred_dict)
        
        # Compute overall metrics
        metrics = metrics_tracker.compute_metrics()
        
        # Log results
        logger.info("="*80)
        logger.info("EVALUATION RESULTS")
        logger.info("="*80)
        for key, value in metrics.items():
            logger.info(f"{key:20s}: {value:.4f}")
        
        # Print classification report
        logger.info("\nClassification Report:")
        logger.info(metrics_tracker.get_classification_report())
        
        # Get confusion matrix
        cm = metrics_tracker.get_confusion_matrix()
        logger.info(f"\nConfusion Matrix:\n{cm}")
        
        # Save results
        self._save_metrics(metrics)
        self._plot_confusion_matrix(cm)
        
        if save_predictions:
            self._save_predictions(all_predictions)
        
        # Plot ROC curve if probabilities available
        if metrics_tracker.probabilities:
            self._plot_roc_curve(
                metrics_tracker.labels,
                np.array(metrics_tracker.probabilities)
            )
        
        # Per-source analysis
        if metrics_tracker.sources:
            self._analyze_per_source(metrics_tracker)
        
        logger.info(f"\nResults saved to {self.output_dir}")
        logger.info("="*80)
        
        return metrics
    
    def _save_metrics(self, metrics: Dict[str, float]):
        """Save metrics to JSON"""
        output_file = self.output_dir / 'metrics.json'
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {output_file}")
    
    def _save_predictions(self, predictions: List[Dict]):
        """Save predictions to CSV"""
        df = pd.DataFrame(predictions)
        output_file = self.output_dir / 'predictions.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"Saved predictions to {output_file}")
        
        # Calculate and save error analysis
        errors = df[df['prediction'] != df['true_label']]
        error_file = self.output_dir / 'errors.csv'
        errors.to_csv(error_file, index=False)
        logger.info(f"Saved {len(errors)} errors to {error_file}")
    
    def _plot_confusion_matrix(self, cm: np.ndarray):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake']
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        output_file = self.output_dir / 'confusion_matrix.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusion matrix to {output_file}")
    
    def _plot_roc_curve(self, labels: List[int], probabilities: np.ndarray):
        """Plot and save ROC curve"""
        # Get probabilities for positive class
        if len(probabilities.shape) == 2:
            probs_positive = probabilities[:, 1]
        else:
            probs_positive = probabilities
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(labels, probs_positive)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        output_file = self.output_dir / 'roc_curve.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved ROC curve to {output_file}")
    
    def _analyze_per_source(self, metrics_tracker: MetricsTracker):
        """Analyze performance per source"""
        sources = metrics_tracker.sources
        labels = np.array(metrics_tracker.labels)
        predictions = np.array(metrics_tracker.predictions)
        
        unique_sources = sorted(set(sources))
        
        per_source_metrics = []
        
        for source in unique_sources:
            source_mask = np.array([s == source for s in sources])
            source_labels = labels[source_mask]
            source_preds = predictions[source_mask]
            
            if len(source_labels) == 0:
                continue
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                'source': source,
                'num_samples': len(source_labels),
                'accuracy': accuracy_score(source_labels, source_preds),
                'precision': precision_score(source_labels, source_preds, average='weighted', zero_division=0),
                'recall': recall_score(source_labels, source_preds, average='weighted', zero_division=0),
                'f1': f1_score(source_labels, source_preds, average='weighted', zero_division=0)
            }
            
            per_source_metrics.append(metrics)
        
        # Save to DataFrame
        df = pd.DataFrame(per_source_metrics)
        output_file = self.output_dir / 'per_source_metrics.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"\nPer-Source Metrics:\n{df.to_string()}")
        logger.info(f"Saved per-source metrics to {output_file}")
        
        # Plot per-source performance
        self._plot_per_source_metrics(df)
    
    def _plot_per_source_metrics(self, df: pd.DataFrame):
        """Plot per-source performance comparison"""
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            df_sorted = df.sort_values(metric, ascending=True)
            ax.barh(df_sorted['source'], df_sorted[metric])
            ax.set_xlabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} by Source')
            ax.set_xlim([0, 1])
            ax.grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        output_file = self.output_dir / 'per_source_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved per-source comparison to {output_file}")