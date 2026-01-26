"""Evaluation metrics for video dialogue systems."""

import re
import string
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import rouge

try:
    from rouge import Rouge
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge package not available. Install with: pip install rouge")

try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    print("Warning: bert-score package not available. Install with: pip install bert-score")


class VideoDialogueEvaluator:
    """Evaluator for video dialogue systems."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.rouge = Rouge() if ROUGE_AVAILABLE else None
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def evaluate_batch(
        self,
        predictions: List[str],
        references: List[str],
        questions: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate a batch of predictions.
        
        Args:
            predictions: List of predicted answers.
            references: List of reference answers.
            questions: Optional list of questions for context.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        metrics = {}
        
        # Basic metrics
        metrics.update(self._compute_exact_match(predictions, references))
        metrics.update(self._compute_f1_score(predictions, references))
        
        # Text generation metrics
        metrics.update(self._compute_bleu_scores(predictions, references))
        metrics.update(self._compute_meteor_score(predictions, references))
        
        if ROUGE_AVAILABLE:
            metrics.update(self._compute_rouge_scores(predictions, references))
        
        if BERT_SCORE_AVAILABLE:
            metrics.update(self._compute_bert_score(predictions, references))
        
        # Video-specific metrics
        if questions:
            metrics.update(self._compute_video_relevance(predictions, questions))
        
        return metrics
    
    def _compute_exact_match(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute exact match accuracy."""
        exact_matches = 0
        for pred, ref in zip(predictions, references):
            if self._normalize_text(pred) == self._normalize_text(ref):
                exact_matches += 1
        
        return {"exact_match": exact_matches / len(predictions)}
    
    def _compute_f1_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute F1 score between predictions and references."""
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = set(self._tokenize(pred))
            ref_tokens = set(self._tokenize(ref))
            
            if len(pred_tokens) == 0 and len(ref_tokens) == 0:
                f1_scores.append(1.0)
            elif len(pred_tokens) == 0 or len(ref_tokens) == 0:
                f1_scores.append(0.0)
            else:
                precision = len(pred_tokens & ref_tokens) / len(pred_tokens)
                recall = len(pred_tokens & ref_tokens) / len(ref_tokens)
                
                if precision + recall == 0:
                    f1_scores.append(0.0)
                else:
                    f1 = 2 * precision * recall / (precision + recall)
                    f1_scores.append(f1)
        
        return {"f1_score": np.mean(f1_scores)}
    
    def _compute_bleu_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute BLEU scores."""
        bleu_1_scores = []
        bleu_2_scores = []
        bleu_3_scores = []
        bleu_4_scores = []
        
        smoothing = SmoothingFunction().method1
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self._tokenize(pred)
            ref_tokens = self._tokenize(ref)
            
            # BLEU-1
            bleu_1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu_1_scores.append(bleu_1)
            
            # BLEU-2
            bleu_2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu_2_scores.append(bleu_2)
            
            # BLEU-3
            bleu_3 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
            bleu_3_scores.append(bleu_3)
            
            # BLEU-4
            bleu_4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
            bleu_4_scores.append(bleu_4)
        
        return {
            "bleu_1": np.mean(bleu_1_scores),
            "bleu_2": np.mean(bleu_2_scores),
            "bleu_3": np.mean(bleu_3_scores),
            "bleu_4": np.mean(bleu_4_scores)
        }
    
    def _compute_meteor_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute METEOR score."""
        meteor_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self._tokenize(pred)
            ref_tokens = self._tokenize(ref)
            
            meteor = meteor_score([ref_tokens], pred_tokens)
            meteor_scores.append(meteor)
        
        return {"meteor": np.mean(meteor_scores)}
    
    def _compute_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores."""
        if not ROUGE_AVAILABLE:
            return {}
        
        try:
            rouge_scores = self.rouge.get_scores(predictions, references, avg=True)
            
            return {
                "rouge_1": rouge_scores["rouge-1"]["f"],
                "rouge_2": rouge_scores["rouge-2"]["f"],
                "rouge_l": rouge_scores["rouge-l"]["f"]
            }
        except Exception as e:
            print(f"Error computing ROUGE scores: {e}")
            return {}
    
    def _compute_bert_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute BERTScore."""
        if not BERT_SCORE_AVAILABLE:
            return {}
        
        try:
            P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
            return {
                "bert_precision": P.mean().item(),
                "bert_recall": R.mean().item(),
                "bert_f1": F1.mean().item()
            }
        except Exception as e:
            print(f"Error computing BERTScore: {e}")
            return {}
    
    def _compute_video_relevance(self, predictions: List[str], questions: List[str]) -> Dict[str, float]:
        """Compute video relevance metrics."""
        relevance_scores = []
        
        for pred, question in zip(predictions, questions):
            # Simple heuristic: check if prediction contains video-related keywords
            video_keywords = ["video", "scene", "frame", "action", "movement", "visual", "see", "appear"]
            pred_lower = pred.lower()
            question_lower = question.lower()
            
            # Check if prediction addresses the question
            question_words = set(self._tokenize(question_lower))
            pred_words = set(self._tokenize(pred_lower))
            
            # Simple relevance score
            if len(question_words) == 0:
                relevance = 0.5
            else:
                overlap = len(question_words & pred_words) / len(question_words)
                relevance = min(1.0, overlap + 0.3)  # Add base relevance
            
            relevance_scores.append(relevance)
        
        return {"video_relevance": np.mean(relevance_scores)}
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        try:
            return word_tokenize(text.lower())
        except:
            # Fallback to simple tokenization
            return text.lower().split()


class VideoDialogueMetrics:
    """Metrics tracker for video dialogue systems."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics_history = []
        self.current_metrics = {}
    
    def update(self, metrics: Dict[str, float]) -> None:
        """Update current metrics."""
        self.current_metrics.update(metrics)
    
    def log_epoch(self, epoch: int) -> None:
        """Log metrics for an epoch."""
        epoch_metrics = {
            "epoch": epoch,
            **self.current_metrics
        }
        self.metrics_history.append(epoch_metrics)
        self.current_metrics = {}
    
    def get_best_metrics(self, metric_name: str = "f1_score") -> Dict[str, float]:
        """Get the best metrics based on a specific metric."""
        if not self.metrics_history:
            return {}
        
        best_epoch = max(self.metrics_history, key=lambda x: x.get(metric_name, 0))
        return best_epoch
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of all metrics."""
        if not self.metrics_history:
            return {}
        
        summary = {}
        for key in self.metrics_history[0].keys():
            if key != "epoch":
                values = [m[key] for m in self.metrics_history if key in m]
                if values:
                    summary[f"{key}_mean"] = np.mean(values)
                    summary[f"{key}_std"] = np.std(values)
                    summary[f"{key}_max"] = np.max(values)
                    summary[f"{key}_min"] = np.min(values)
        
        return summary


def create_evaluation_report(
    predictions: List[str],
    references: List[str],
    questions: Optional[List[str]] = None,
    model_name: str = "VideoDialogueModel"
) -> str:
    """Create a comprehensive evaluation report.
    
    Args:
        predictions: List of predicted answers.
        references: List of reference answers.
        questions: Optional list of questions.
        model_name: Name of the model being evaluated.
        
    Returns:
        Formatted evaluation report string.
    """
    evaluator = VideoDialogueEvaluator()
    metrics = evaluator.evaluate_batch(predictions, references, questions)
    
    report = f"""
EVALUATION REPORT: {model_name}
{'=' * 50}

BASIC METRICS:
- Exact Match: {metrics.get('exact_match', 0):.4f}
- F1 Score: {metrics.get('f1_score', 0):.4f}

TEXT GENERATION METRICS:
- BLEU-1: {metrics.get('bleu_1', 0):.4f}
- BLEU-2: {metrics.get('bleu_2', 0):.4f}
- BLEU-3: {metrics.get('bleu_3', 0):.4f}
- BLEU-4: {metrics.get('bleu_4', 0):.4f}
- METEOR: {metrics.get('meteor', 0):.4f}
"""
    
    if 'rouge_1' in metrics:
        report += f"""
ROUGE METRICS:
- ROUGE-1: {metrics['rouge_1']:.4f}
- ROUGE-2: {metrics['rouge_2']:.4f}
- ROUGE-L: {metrics['rouge_l']:.4f}
"""
    
    if 'bert_f1' in metrics:
        report += f"""
BERT SCORE METRICS:
- BERT Precision: {metrics['bert_precision']:.4f}
- BERT Recall: {metrics['bert_recall']:.4f}
- BERT F1: {metrics['bert_f1']:.4f}
"""
    
    if 'video_relevance' in metrics:
        report += f"""
VIDEO-SPECIFIC METRICS:
- Video Relevance: {metrics['video_relevance']:.4f}
"""
    
    report += f"""
OVERALL PERFORMANCE:
- Total Samples: {len(predictions)}
- Average Response Length: {np.mean([len(p.split()) for p in predictions]):.2f} words
"""
    
    return report
