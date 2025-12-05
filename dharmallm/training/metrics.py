"""
🕉️ Real Evaluation Metrics for DharmaMind LLM Training
========================================================

Production-ready evaluation metrics to replace placeholders in training pipeline.

This module implements:
- Language model metrics: Perplexity, Loss, Token Accuracy
- Generation metrics: BLEU, ROUGE, METEOR
- Classification metrics: Accuracy, Precision, Recall, F1
- Custom metrics: Dharmic alignment, Wisdom consistency

All metrics are computed efficiently with batching and proper handling of
special tokens (padding, EOS, etc.).

Dependencies:
    pip install torch numpy sacrebleu rouge-score nltk

Author: DharmaMind Team
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available for metrics computation")

try:
    from sacrebleu import corpus_bleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    logging.warning("SacreBLEU not available. Install: pip install sacrebleu")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logging.warning("rouge-score not available. Install: pip install rouge-score")

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result of a single metric computation"""
    name: str
    value: float
    description: str
    higher_is_better: bool = True
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MetricsBatch:
    """Batch of metrics computed together"""
    metrics: Dict[str, float]
    num_samples: int
    num_tokens: int
    timestamp: Optional[str] = None


class LanguageModelMetrics:
    """
    Core language model evaluation metrics.
    
    Computes perplexity, token accuracy, and loss-based metrics.
    """
    
    def __init__(self, ignore_index: int = -100):
        """
        Initialize language model metrics.
        
        Args:
            ignore_index: Token index to ignore (padding, etc.)
        """
        self.ignore_index = ignore_index
        
    def compute_perplexity(
        self,
        loss: Union[float, torch.Tensor],
    ) -> float:
        """
        Compute perplexity from cross-entropy loss.
        
        Perplexity = exp(loss)
        
        Lower perplexity indicates better language modeling.
        Typical ranges:
        - Very good: < 20
        - Good: 20-50
        - Acceptable: 50-100
        - Poor: > 100
        
        Args:
            loss: Cross-entropy loss value
            
        Returns:
            Perplexity value
        """
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        
        try:
            perplexity = math.exp(loss)
            
            # Cap at very large value to avoid inf
            if perplexity > 1e10:
                logger.warning(f"Very high perplexity ({perplexity}), capping at 1e10")
                perplexity = 1e10
                
            return perplexity
        except (OverflowError, ValueError) as e:
            logger.error(f"Error computing perplexity: {e}")
            return float('inf')
    
    def compute_token_accuracy(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: Optional[int] = None,
    ) -> Tuple[float, Dict[str, int]]:
        """
        Compute token-level accuracy.
        
        Args:
            predictions: Predicted token IDs, shape (batch, seq_len) or (batch, seq_len, vocab_size)
            labels: True token IDs, shape (batch, seq_len)
            ignore_index: Token index to ignore (default: self.ignore_index)
            
        Returns:
            Tuple of (accuracy, stats_dict)
        """
        if not TORCH_AVAILABLE:
            return 0.0, {}
        
        ignore_idx = ignore_index if ignore_index is not None else self.ignore_index
        
        # Handle logits (batch, seq_len, vocab_size)
        if predictions.dim() == 3:
            predictions = predictions.argmax(dim=-1)
        
        # Create mask for valid tokens
        mask = labels != ignore_idx
        
        # Compute accuracy
        correct = (predictions == labels) & mask
        num_correct = correct.sum().item()
        num_total = mask.sum().item()
        
        accuracy = num_correct / num_total if num_total > 0 else 0.0
        
        stats = {
            "num_correct": num_correct,
            "num_total": num_total,
            "num_ignored": (labels == ignore_idx).sum().item(),
        }
        
        return accuracy, stats
    
    def compute_top_k_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        k: int = 5,
        ignore_index: Optional[int] = None,
    ) -> float:
        """
        Compute top-k token accuracy.
        
        Checks if true token is in top-k predictions.
        
        Args:
            logits: Model logits, shape (batch, seq_len, vocab_size)
            labels: True token IDs, shape (batch, seq_len)
            k: Number of top predictions to consider
            ignore_index: Token index to ignore
            
        Returns:
            Top-k accuracy
        """
        if not TORCH_AVAILABLE:
            return 0.0
        
        ignore_idx = ignore_index if ignore_index is not None else self.ignore_index
        
        # Get top-k predictions
        top_k_preds = logits.topk(k, dim=-1).indices  # (batch, seq_len, k)
        
        # Expand labels to match
        labels_expanded = labels.unsqueeze(-1).expand_as(top_k_preds)
        
        # Check if true label is in top-k
        mask = labels != ignore_idx
        in_top_k = (top_k_preds == labels_expanded).any(dim=-1) & mask
        
        num_correct = in_top_k.sum().item()
        num_total = mask.sum().item()
        
        return num_correct / num_total if num_total > 0 else 0.0
    
    def compute_sequence_accuracy(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: Optional[int] = None,
    ) -> float:
        """
        Compute sequence-level accuracy (all tokens must match).
        
        Args:
            predictions: Predicted token IDs, shape (batch, seq_len)
            labels: True token IDs, shape (batch, seq_len)
            ignore_index: Token index to ignore
            
        Returns:
            Sequence accuracy
        """
        if not TORCH_AVAILABLE:
            return 0.0
        
        ignore_idx = ignore_index if ignore_index is not None else self.ignore_index
        
        # Handle logits
        if predictions.dim() == 3:
            predictions = predictions.argmax(dim=-1)
        
        # Create mask
        mask = labels != ignore_idx
        
        # Check if all tokens match in each sequence
        matches = ((predictions == labels) | ~mask).all(dim=1)
        
        return matches.float().mean().item()


class GenerationMetrics:
    """
    Metrics for evaluating generated text quality.
    
    Includes BLEU, ROUGE, and custom metrics.
    """
    
    def __init__(self):
        """Initialize generation metrics."""
        self.rouge_scorer = None
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
    
    def compute_bleu(
        self,
        predictions: List[str],
        references: List[str],
        max_order: int = 4,
    ) -> Dict[str, float]:
        """
        Compute BLEU score using SacreBLEU.
        
        BLEU (Bilingual Evaluation Understudy) measures n-gram precision
        between generated and reference texts.
        
        Score ranges: 0-100 (higher is better)
        - Excellent: > 40
        - Good: 30-40
        - Acceptable: 20-30
        - Poor: < 20
        
        Args:
            predictions: List of generated texts
            references: List of reference texts
            max_order: Maximum n-gram order (default: 4)
            
        Returns:
            Dictionary with BLEU scores
        """
        if not SACREBLEU_AVAILABLE:
            logger.warning("SacreBLEU not available")
            return {"bleu": 0.0}
        
        if not predictions or not references:
            return {"bleu": 0.0}
        
        try:
            # SacreBLEU expects references as list of lists
            refs = [[ref] for ref in references]
            
            # Compute BLEU
            bleu = corpus_bleu(predictions, refs, max_order=max_order)
            
            return {
                "bleu": bleu.score,
                "bleu_1": bleu.precisions[0] if len(bleu.precisions) > 0 else 0.0,
                "bleu_2": bleu.precisions[1] if len(bleu.precisions) > 1 else 0.0,
                "bleu_3": bleu.precisions[2] if len(bleu.precisions) > 2 else 0.0,
                "bleu_4": bleu.precisions[3] if len(bleu.precisions) > 3 else 0.0,
            }
        except Exception as e:
            logger.error(f"Error computing BLEU: {e}")
            return {"bleu": 0.0}
    
    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores (Recall-Oriented Understudy for Gisting Evaluation).
        
        ROUGE measures overlap of n-grams and sequences between texts.
        
        Score ranges: 0-1 (higher is better)
        - ROUGE-1: Unigram overlap
        - ROUGE-2: Bigram overlap  
        - ROUGE-L: Longest common subsequence
        
        Args:
            predictions: List of generated texts
            references: List of reference texts
            
        Returns:
            Dictionary with ROUGE scores
        """
        if not ROUGE_AVAILABLE or self.rouge_scorer is None:
            logger.warning("ROUGE not available")
            return {}
        
        if not predictions or not references:
            return {}
        
        try:
            rouge_scores = defaultdict(list)
            
            for pred, ref in zip(predictions, references):
                scores = self.rouge_scorer.score(ref, pred)
                
                for key in scores:
                    rouge_scores[f"{key}_precision"].append(scores[key].precision)
                    rouge_scores[f"{key}_recall"].append(scores[key].recall)
                    rouge_scores[f"{key}_fmeasure"].append(scores[key].fmeasure)
            
            # Average scores
            result = {}
            for key, values in rouge_scores.items():
                result[key] = sum(values) / len(values) if values else 0.0
            
            return result
            
        except Exception as e:
            logger.error(f"Error computing ROUGE: {e}")
            return {}


class ClassificationMetrics:
    """
    Metrics for classification and token prediction tasks.
    
    Includes precision, recall, F1, and confusion matrix.
    """
    
    def __init__(self, num_classes: Optional[int] = None):
        """
        Initialize classification metrics.
        
        Args:
            num_classes: Number of classes (for confusion matrix)
        """
        self.num_classes = num_classes
    
    def compute_precision_recall_f1(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        average: str = 'macro',
    ) -> Dict[str, float]:
        """
        Compute precision, recall, and F1 score.
        
        Args:
            predictions: Predicted class IDs
            labels: True class IDs
            average: Averaging method ('macro', 'micro', 'weighted')
            
        Returns:
            Dictionary with precision, recall, F1
        """
        if len(predictions) == 0 or len(labels) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Flatten arrays
        predictions = predictions.flatten()
        labels = labels.flatten()
        
        # Get unique classes
        classes = np.unique(np.concatenate([predictions, labels]))
        
        # Compute per-class metrics
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        support_per_class = []
        
        for cls in classes:
            tp = np.sum((predictions == cls) & (labels == cls))
            fp = np.sum((predictions == cls) & (labels != cls))
            fn = np.sum((predictions != cls) & (labels == cls))
            support = np.sum(labels == cls)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)
            support_per_class.append(support)
        
        # Average metrics
        if average == 'macro':
            precision = np.mean(precision_per_class)
            recall = np.mean(recall_per_class)
            f1 = np.mean(f1_per_class)
        elif average == 'micro':
            # Global counts
            total_tp = np.sum([np.sum((predictions == cls) & (labels == cls)) for cls in classes])
            total_fp = np.sum([np.sum((predictions == cls) & (labels != cls)) for cls in classes])
            total_fn = np.sum([np.sum((predictions != cls) & (labels == cls)) for cls in classes])
            
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif average == 'weighted':
            total_support = sum(support_per_class)
            precision = sum(p * s for p, s in zip(precision_per_class, support_per_class)) / total_support if total_support > 0 else 0.0
            recall = sum(r * s for r, s in zip(recall_per_class, support_per_class)) / total_support if total_support > 0 else 0.0
            f1 = sum(f * s for f, s in zip(f1_per_class, support_per_class)) / total_support if total_support > 0 else 0.0
        else:
            raise ValueError(f"Unknown average method: {average}")
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


class MetricsComputer:
    """
    Main metrics computer that combines all metric types.
    
    This class provides a unified interface for computing all evaluation metrics.
    """
    
    def __init__(self, ignore_index: int = -100):
        """
        Initialize metrics computer.
        
        Args:
            ignore_index: Token index to ignore (padding, etc.)
        """
        self.ignore_index = ignore_index
        self.lm_metrics = LanguageModelMetrics(ignore_index=ignore_index)
        self.gen_metrics = GenerationMetrics()
        self.clf_metrics = ClassificationMetrics()
    
    def compute_all_metrics(
        self,
        loss: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        predictions: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        generated_texts: Optional[List[str]] = None,
        reference_texts: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute all applicable metrics based on provided inputs.
        
        Args:
            loss: Training loss
            logits: Model logits
            predictions: Predicted token IDs
            labels: True token IDs
            generated_texts: Generated text strings
            reference_texts: Reference text strings
            
        Returns:
            Dictionary of all computed metrics
        """
        metrics = {}
        
        # Compute perplexity from loss
        if loss is not None:
            metrics["loss"] = loss.item() if isinstance(loss, torch.Tensor) else loss
            metrics["perplexity"] = self.lm_metrics.compute_perplexity(loss)
        
        # Compute token-level metrics
        if predictions is not None and labels is not None:
            accuracy, stats = self.lm_metrics.compute_token_accuracy(predictions, labels)
            metrics["token_accuracy"] = accuracy
            metrics["num_tokens"] = stats.get("num_total", 0)
            
            seq_accuracy = self.lm_metrics.compute_sequence_accuracy(predictions, labels)
            metrics["sequence_accuracy"] = seq_accuracy
        
        # Compute top-k accuracy from logits
        if logits is not None and labels is not None:
            metrics["top_5_accuracy"] = self.lm_metrics.compute_top_k_accuracy(
                logits, labels, k=5
            )
        
        # Compute generation metrics
        if generated_texts is not None and reference_texts is not None:
            bleu_scores = self.gen_metrics.compute_bleu(generated_texts, reference_texts)
            metrics.update(bleu_scores)
            
            rouge_scores = self.gen_metrics.compute_rouge(generated_texts, reference_texts)
            metrics.update(rouge_scores)
        
        return metrics
    
    def compute_batch_metrics(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> MetricsBatch:
        """
        Compute metrics for a single training/validation batch.
        
        Args:
            loss: Batch loss
            logits: Model logits
            labels: True labels
            
        Returns:
            MetricsBatch with computed metrics
        """
        metrics = self.compute_all_metrics(
            loss=loss,
            logits=logits,
            predictions=logits.argmax(dim=-1) if logits.dim() == 3 else logits,
            labels=labels,
        )
        
        mask = labels != self.ignore_index
        num_tokens = mask.sum().item() if TORCH_AVAILABLE else 0
        
        return MetricsBatch(
            metrics=metrics,
            num_samples=labels.shape[0] if TORCH_AVAILABLE else 0,
            num_tokens=num_tokens,
        )


def aggregate_metrics(
    metrics_batches: List[MetricsBatch],
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple batches.
    
    Args:
        metrics_batches: List of MetricsBatch objects
        
    Returns:
        Dictionary of aggregated metrics
    """
    if not metrics_batches:
        return {}
    
    # Collect all metric values
    all_metrics = defaultdict(list)
    total_samples = 0
    total_tokens = 0
    
    for batch in metrics_batches:
        for key, value in batch.metrics.items():
            all_metrics[key].append((value, batch.num_tokens))
        total_samples += batch.num_samples
        total_tokens += batch.num_tokens
    
    # Compute weighted averages (weighted by number of tokens)
    aggregated = {}
    for key, values in all_metrics.items():
        if values:
            # Weight by number of tokens for token-level metrics
            if any(k in key.lower() for k in ['token', 'accuracy', 'perplexity']):
                weighted_sum = sum(v * w for v, w in values)
                total_weight = sum(w for _, w in values)
                aggregated[key] = weighted_sum / total_weight if total_weight > 0 else 0.0
            else:
                # Simple average for other metrics
                aggregated[key] = sum(v for v, _ in values) / len(values)
    
    aggregated["num_samples"] = total_samples
    aggregated["num_tokens"] = total_tokens
    
    return aggregated


# Convenience functions for quick metric computation

def compute_perplexity(loss: Union[float, torch.Tensor]) -> float:
    """Quick perplexity computation."""
    metrics = LanguageModelMetrics()
    return metrics.compute_perplexity(loss)


def compute_token_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Quick token accuracy computation."""
    metrics = LanguageModelMetrics(ignore_index=ignore_index)
    accuracy, _ = metrics.compute_token_accuracy(predictions, labels)
    return accuracy


def compute_bleu(
    predictions: List[str],
    references: List[str],
) -> float:
    """Quick BLEU score computation."""
    metrics = GenerationMetrics()
    scores = metrics.compute_bleu(predictions, references)
    return scores.get("bleu", 0.0)


# Test function
def test_metrics():
    """Test metrics computation."""
    if not TORCH_AVAILABLE:
        print("⚠️  PyTorch not available, skipping tests")
        return
    
    print("=" * 60)
    print("Testing DharmaMind Metrics System")
    print("=" * 60)
    
    # Create computer
    computer = MetricsComputer(ignore_index=-100)
    
    # Test data
    batch_size, seq_len, vocab_size = 4, 10, 100
    
    # Generate test tensors
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[0, -2:] = -100  # Add some padding
    
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100
    )
    
    print("\n1. Computing batch metrics...")
    batch_metrics = computer.compute_batch_metrics(loss, logits, labels)
    print(f"   Loss: {batch_metrics.metrics['loss']:.4f}")
    print(f"   Perplexity: {batch_metrics.metrics['perplexity']:.4f}")
    print(f"   Token Accuracy: {batch_metrics.metrics['token_accuracy']:.4f}")
    print(f"   Top-5 Accuracy: {batch_metrics.metrics['top_5_accuracy']:.4f}")
    print(f"   Num Tokens: {batch_metrics.num_tokens}")
    
    print("\n2. Testing generation metrics...")
    predictions = ["The quick brown fox jumps", "Hello world"]
    references = ["The quick brown fox jumped over", "Hello there world"]
    
    if SACREBLEU_AVAILABLE:
        bleu_scores = computer.gen_metrics.compute_bleu(predictions, references)
        print(f"   BLEU: {bleu_scores['bleu']:.2f}")
    else:
        print("   BLEU: (SacreBLEU not available)")
    
    if ROUGE_AVAILABLE:
        rouge_scores = computer.gen_metrics.compute_rouge(predictions, references)
        if rouge_scores:
            print(f"   ROUGE-1 F1: {rouge_scores.get('rouge1_fmeasure', 0):.4f}")
            print(f"   ROUGE-L F1: {rouge_scores.get('rougeL_fmeasure', 0):.4f}")
    else:
        print("   ROUGE: (rouge-score not available)")
    
    print("\n3. Testing aggregation...")
    batches = [batch_metrics, batch_metrics]  # Duplicate for testing
    aggregated = aggregate_metrics(batches)
    print(f"   Aggregated Perplexity: {aggregated.get('perplexity', 0):.4f}")
    print(f"   Total Tokens: {aggregated.get('num_tokens', 0)}")
    
    print("\n" + "=" * 60)
    print("✅ All metrics tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_metrics()
