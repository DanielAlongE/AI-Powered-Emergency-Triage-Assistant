"""
Visualization utilities for test results.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from testing.models import TestResult, TestSummary


class VisualizationHelper:
    """Helper class for creating visualizations of test results."""

    @staticmethod
    def plot_confusion_matrix(summary: TestSummary, title: str = None, figsize: tuple = (8, 6)):
        """
        Plot confusion matrix for agent results.

        Args:
            summary: TestSummary with confusion matrix
            title: Optional title for the plot
            figsize: Figure size tuple
        """
        plt.figure(figsize=figsize)

        # Convert confusion matrix to numpy array
        cm = np.array(summary.confusion_matrix)

        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['ESI 1', 'ESI 2', 'ESI 3', 'ESI 4', 'ESI 5'],
            yticklabels=['ESI 1', 'ESI 2', 'ESI 3', 'ESI 4', 'ESI 5'],
            cbar_kws={'label': 'Number of Cases'}
        )

        plt.title(title or f'Confusion Matrix - {summary.agent_name}')
        plt.xlabel('Predicted ESI Level')
        plt.ylabel('True ESI Level')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_accuracy_comparison(summaries: List[TestSummary], title: str = "Agent Accuracy Comparison"):
        """
        Plot accuracy comparison between multiple agents.

        Args:
            summaries: List of TestSummary objects
            title: Plot title
        """
        plt.figure(figsize=(10, 6))

        agent_names = [s.agent_name for s in summaries]
        accuracies = [s.accuracy for s in summaries]

        bars = plt.bar(agent_names, accuracies, color='skyblue', edgecolor='navy', alpha=0.7)

        # Add value labels on bars
        for bar, accuracy in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{accuracy:.1%}',
                    ha='center', va='bottom', fontweight='bold')

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Agent', fontweight='bold')
        plt.ylabel('Accuracy', fontweight='bold')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_esi_level_performance(summary: TestSummary, metric: str = 'f1_score'):
        """
        Plot performance metrics by ESI level.

        Args:
            summary: TestSummary with per-ESI metrics
            metric: Metric to plot ('precision', 'recall', 'f1_score')
        """
        if metric not in ['precision', 'recall', 'f1_score']:
            raise ValueError("Metric must be one of: precision, recall, f1_score")

        plt.figure(figsize=(10, 6))

        esi_levels = list(summary.results_by_esi.keys())
        values = [summary.results_by_esi[esi][metric] for esi in esi_levels]

        bars = plt.bar([f'ESI {esi}' for esi in esi_levels], values,
                      color='lightcoral', edgecolor='darkred', alpha=0.7)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}',
                    ha='center', va='bottom', fontweight='bold')

        plt.title(f'{metric.replace("_", " ").title()} by ESI Level - {summary.agent_name}',
                 fontsize=14, fontweight='bold')
        plt.xlabel('ESI Level', fontweight='bold')
        plt.ylabel(metric.replace('_', ' ').title(), fontweight='bold')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def create_interactive_comparison(summaries: List[TestSummary]) -> go.Figure:
        """
        Create interactive comparison plot using Plotly.

        Args:
            summaries: List of TestSummary objects

        Returns:
            Plotly Figure object
        """
        # Prepare data
        agents = [s.agent_name for s in summaries]
        accuracies = [s.accuracy for s in summaries]
        avg_times = [s.avg_processing_time for s in summaries]

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Accuracy Comparison', 'Processing Time Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Add accuracy bars
        fig.add_trace(
            go.Bar(
                x=agents,
                y=accuracies,
                name='Accuracy',
                text=[f'{acc:.1%}' for acc in accuracies],
                textposition='auto',
                marker_color='lightblue'
            ),
            row=1, col=1
        )

        # Add processing time bars
        fig.add_trace(
            go.Bar(
                x=agents,
                y=avg_times,
                name='Avg Time (s)',
                text=[f'{time:.3f}s' for time in avg_times],
                textposition='auto',
                marker_color='lightcoral'
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            title_text="Agent Performance Comparison",
            showlegend=False,
            height=500
        )

        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Processing Time (seconds)", row=1, col=2)

        return fig

    @staticmethod
    def plot_error_analysis(results: List[TestResult], show_top_n: int = 20):
        """
        Plot error analysis showing most common mistakes.

        Args:
            results: List of TestResult objects
            show_top_n: Number of top errors to show
        """
        # Get incorrect results
        errors = [r for r in results if not r.correct]

        if not errors:
            print("No errors found - perfect accuracy!")
            return

        # Count error types
        error_types = {}
        for error in errors:
            key = f"Expected ESI {error.expected_esi} → Predicted ESI {error.predicted_esi}"
            error_types[key] = error_types.get(key, 0) + 1

        # Sort by frequency and take top N
        sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:show_top_n]

        if not sorted_errors:
            return

        # Create plot
        plt.figure(figsize=(12, 8))

        error_labels, error_counts = zip(*sorted_errors)

        bars = plt.barh(range(len(error_labels)), error_counts, color='salmon', alpha=0.7)

        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, error_counts)):
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                    str(count), ha='left', va='center', fontweight='bold')

        plt.yticks(range(len(error_labels)), error_labels)
        plt.xlabel('Number of Cases', fontweight='bold')
        plt.title(f'Top {len(sorted_errors)} Most Common Prediction Errors', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def create_results_dataframe(results: List[TestResult]) -> pd.DataFrame:
        """
        Convert test results to pandas DataFrame for analysis.

        Args:
            results: List of TestResult objects

        Returns:
            pandas DataFrame
        """
        return pd.DataFrame([
            {
                'case_id': r.case_id,
                'expected_esi': r.expected_esi,
                'predicted_esi': r.predicted_esi,
                'correct': r.correct,
                'confidence': r.confidence,
                'processing_time': r.processing_time,
                'agent_name': r.agent_name,
                'error_magnitude': abs(r.expected_esi - r.predicted_esi) if not r.correct else 0
            }
            for r in results
        ])

    @staticmethod
    def plot_confidence_calibration(results: List[TestResult]):
        """
        Plot confidence calibration (confidence vs accuracy).

        Args:
            results: List of TestResult objects
        """
        # Filter results with confidence scores
        results_with_conf = [r for r in results if r.confidence is not None]

        if not results_with_conf:
            print("No confidence scores available for calibration plot.")
            return

        # Create confidence bins
        df = VisualizationHelper.create_results_dataframe(results_with_conf)

        # Bin confidence scores
        df['confidence_bin'] = pd.cut(df['confidence'], bins=10, labels=False)
        df['confidence_bin_center'] = pd.cut(df['confidence'], bins=10).apply(lambda x: x.mid)

        # Calculate accuracy for each bin
        calibration_data = df.groupby('confidence_bin_center').agg({
            'correct': 'mean',
            'case_id': 'count'
        }).reset_index()

        plt.figure(figsize=(10, 6))

        # Plot calibration curve
        plt.scatter(calibration_data['confidence_bin_center'],
                   calibration_data['correct'],
                   s=calibration_data['case_id'] * 10,  # Size based on number of samples
                   alpha=0.7, color='blue', label='Actual')

        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect Calibration')

        plt.xlabel('Confidence', fontweight='bold')
        plt.ylabel('Accuracy', fontweight='bold')
        plt.title('Confidence Calibration Plot', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()