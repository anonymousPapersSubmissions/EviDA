import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import logging

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track and compute metrics during training and evaluation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked values"""
        self.predictions = []
        self.labels = []
        self.probabilities = []
        self.sources = []
        self.losses = []
    
    def update(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        sources: Optional[List[str]] = None,
        loss: Optional[float] = None
    ):
        """Update tracked values"""
        # Convert to list if needed
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        
        self.predictions.extend(predictions)
        self.labels.extend(labels)
        
        if probabilities is not None:
            if isinstance(probabilities, np.ndarray):
                probabilities = probabilities.tolist()
            self.probabilities.extend(probabilities)
        
        if sources is not None:
            self.sources.extend(sources)
        
        if loss is not None:
            self.losses.append(loss)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics with guaranteed safety"""
        # Return defaults if no data
        if not self.predictions or not self.labels:
            logger.warning("No predictions/labels to compute metrics")
            return {
                'loss': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auc': 0.0
            }
        
        # Convert to numpy arrays
        preds = np.array(self.predictions)
        labels = np.array(self.labels)
        
        # Ensure same length
        min_len = min(len(preds), len(labels))
        preds = preds[:min_len]
        labels = labels[:min_len]
        
        # Compute core metrics
        metrics = {}
        
        try:
            metrics['accuracy'] = accuracy_score(labels, preds)
        except Exception as e:
            logger.error(f"Error computing accuracy: {e}")
            metrics['accuracy'] = 0.0
        
        try:
            metrics['precision'] = precision_score(labels, preds, average='weighted', zero_division=0)
        except Exception as e:
            logger.error(f"Error computing precision: {e}")
            metrics['precision'] = 0.0
        
        try:
            metrics['recall'] = recall_score(labels, preds, average='weighted', zero_division=0)
        except Exception as e:
            logger.error(f"Error computing recall: {e}")
            metrics['recall'] = 0.0
        
        try:
            metrics['f1'] = f1_score(labels, preds, average='weighted', zero_division=0)
        except Exception as e:
            logger.error(f"Error computing F1: {e}")
            metrics['f1'] = 0.0
        
        # Macro metrics
        try:
            metrics['precision_macro'] = precision_score(labels, preds, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(labels, preds, average='macro', zero_division=0)
            metrics['f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
        except Exception as e:
            logger.error(f"Error computing macro metrics: {e}")
            metrics['precision_macro'] = 0.0
            metrics['recall_macro'] = 0.0
            metrics['f1_macro'] = 0.0
        
        # AUC if probabilities available
        if self.probabilities:
            try:
                probs = np.array(self.probabilities)
                probs = probs[:min_len]  # Match length
                
                if len(probs.shape) == 1:
                    metrics['auc'] = roc_auc_score(labels, probs)
                else:
                    # Use positive class probabilities
                    metrics['auc'] = roc_auc_score(labels, probs[:, 1])
            except Exception as e:
                logger.warning(f"Could not compute AUC: {e}")
                metrics['auc'] = 0.0
        else:
            metrics['auc'] = 0.0
        
        # Average loss
        if self.losses:
            metrics['loss'] = float(np.mean(self.losses))
        else:
            metrics['loss'] = 0.0
        
        # Per-class metrics
        for i, class_name in enumerate(['real', 'fake']):
            try:
                class_mask = (labels == i)
                if class_mask.sum() > 0:
                    class_preds = preds[class_mask]
                    class_labels = labels[class_mask]
                    
                    metrics[f'{class_name}_precision'] = precision_score(
                        class_labels, class_preds, 
                        average='binary', pos_label=i, zero_division=0
                    )
                    metrics[f'{class_name}_recall'] = recall_score(
                        class_labels, class_preds, 
                        average='binary', pos_label=i, zero_division=0
                    )
                    metrics[f'{class_name}_f1'] = f1_score(
                        class_labels, class_preds, 
                        average='binary', pos_label=i, zero_division=0
                    )
                else:
                    metrics[f'{class_name}_precision'] = 0.0
                    metrics[f'{class_name}_recall'] = 0.0
                    metrics[f'{class_name}_f1'] = 0.0
            except Exception as e:
                logger.warning(f"Error computing {class_name} metrics: {e}")
                metrics[f'{class_name}_precision'] = 0.0
                metrics[f'{class_name}_recall'] = 0.0
                metrics[f'{class_name}_f1'] = 0.0
        
        # Per-source metrics if available
        if self.sources and len(self.sources) >= min_len:
            try:
                sources_array = np.array(self.sources[:min_len])
                unique_sources = set(sources_array)
                
                for source in unique_sources:
                    source_mask = (sources_array == source)
                    if source_mask.sum() > 0:
                        source_preds = preds[source_mask]
                        source_labels = labels[source_mask]
                        
                        metrics[f'{source}_accuracy'] = accuracy_score(
                            source_labels, source_preds
                        )
                        metrics[f'{source}_f1'] = f1_score(
                            source_labels, source_preds, 
                            average='weighted', zero_division=0
                        )
            except Exception as e:
                logger.warning(f"Error computing per-source metrics: {e}")
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        if not self.predictions or not self.labels:
            return np.array([[0, 0], [0, 0]])
        
        try:
            preds = np.array(self.predictions)
            labels = np.array(self.labels)
            min_len = min(len(preds), len(labels))
            return confusion_matrix(labels[:min_len], preds[:min_len])
        except Exception as e:
            logger.error(f"Error computing confusion matrix: {e}")
            return np.array([[0, 0], [0, 0]])
    
    def get_classification_report(self) -> str:
        """Get detailed classification report"""
        if not self.predictions or not self.labels:
            return "No predictions available"
        
        try:
            preds = np.array(self.predictions)
            labels = np.array(self.labels)
            min_len = min(len(preds), len(labels))
            
            return classification_report(
                labels[:min_len],
                preds[:min_len],
                target_names=['Real', 'Fake'],
                digits=4,
                zero_division=0
            )
        except Exception as e:
            logger.error(f"Error generating classification report: {e}")
            return f"Error: {e}"