"""
Metrics calculation utilities for triage agent evaluation.
"""
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from testing.models import TestResult, TestSummary


class MetricsCalculator:
    """Calculates performance metrics for triage agents."""

    @staticmethod
    def calculate_summary(results: List[TestResult]) -> TestSummary:
        """
        Calculate comprehensive test summary from results.

        Args:
            results: List of test results

        Returns:
            TestSummary with calculated metrics
        """
        if not results:
            raise ValueError("Cannot calculate metrics for empty results")

        agent_name = results[0].agent_name
        total_cases = len(results)
        correct_predictions = sum(1 for r in results if r.correct)
        accuracy = correct_predictions / total_cases

        # Calculate average confidence and processing time
        confidences = [r.confidence for r in results if r.confidence is not None]
        avg_confidence = np.mean(confidences) if confidences else None

        processing_times = [r.processing_time for r in results]
        avg_processing_time = np.mean(processing_times)

        # Get predictions and true values for sklearn metrics
        y_true = [r.expected_esi for r in results]
        y_pred = [r.predicted_esi for r in results]

        # Calculate per-ESI level metrics
        results_by_esi = MetricsCalculator._calculate_per_esi_metrics(y_true, y_pred)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])

        return TestSummary(
            agent_name=agent_name,
            total_cases=total_cases,
            correct_predictions=correct_predictions,
            accuracy=accuracy,
            avg_confidence=avg_confidence,
            avg_processing_time=avg_processing_time,
            results_by_esi=results_by_esi,
            confusion_matrix=cm.tolist()
        )

    @staticmethod
    def _calculate_per_esi_metrics(y_true: List[int], y_pred: List[int]) -> Dict[int, Dict[str, float]]:
        """
        Calculate precision, recall, and F1 score for each ESI level.

        Args:
            y_true: True ESI levels
            y_pred: Predicted ESI levels

        Returns:
            Dict mapping ESI level to metrics dict
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[1, 2, 3, 4, 5], average=None, zero_division=0
        )

        results_by_esi = {}
        for i, esi_level in enumerate([1, 2, 3, 4, 5]):
            results_by_esi[esi_level] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }

        return results_by_esi

    @staticmethod
    def compare_agents(summaries: List[TestSummary]) -> Dict[str, Any]:
        """
        Compare multiple agents and return comparison statistics.

        Args:
            summaries: List of TestSummary objects for different agents

        Returns:
            Dict with comparison results
        """
        if not summaries:
            return {}

        comparison = {
            'agents': [s.agent_name for s in summaries],
            'accuracies': [s.accuracy for s in summaries],
            'avg_processing_times': [s.avg_processing_time for s in summaries],
            'total_cases': summaries[0].total_cases,
            'best_accuracy': max(s.accuracy for s in summaries),
            'fastest_agent': min(summaries, key=lambda s: s.avg_processing_time).agent_name,
            'most_accurate_agent': max(summaries, key=lambda s: s.accuracy).agent_name
        }

        # Calculate confidence scores if available
        confidences = [s.avg_confidence for s in summaries if s.avg_confidence is not None]
        if confidences:
            comparison['avg_confidences'] = confidences
            comparison['most_confident_agent'] = max(
                (s for s in summaries if s.avg_confidence is not None),
                key=lambda s: s.avg_confidence
            ).agent_name

        return comparison

    @staticmethod
    def calculate_esi_level_difficulty(results: List[TestResult]) -> Dict[int, Dict[str, float]]:
        """
        Analyze which ESI levels are most difficult to predict correctly.

        Args:
            results: List of test results

        Returns:
            Dict mapping ESI level to difficulty metrics
        """
        difficulty = {}

        for esi_level in [1, 2, 3, 4, 5]:
            # Get results for this ESI level
            level_results = [r for r in results if r.expected_esi == esi_level]

            if not level_results:
                difficulty[esi_level] = {
                    'total_cases': 0,
                    'accuracy': 0.0,
                    'avg_confidence': None,
                    'most_common_mistake': None
                }
                continue

            total_cases = len(level_results)
            correct_cases = sum(1 for r in level_results if r.correct)
            accuracy = correct_cases / total_cases

            # Calculate average confidence for this level
            confidences = [r.confidence for r in level_results if r.confidence is not None]
            avg_confidence = np.mean(confidences) if confidences else None

            # Find most common mistake
            mistakes = [r.predicted_esi for r in level_results if not r.correct]
            most_common_mistake = None
            if mistakes:
                mistake_counts = {}
                for mistake in mistakes:
                    mistake_counts[mistake] = mistake_counts.get(mistake, 0) + 1
                most_common_mistake = max(mistake_counts.items(), key=lambda x: x[1])[0]

            difficulty[esi_level] = {
                'total_cases': total_cases,
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'most_common_mistake': most_common_mistake
            }

        return difficulty