"""
Main test runner for evaluating triage agents.
"""
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from agents.base import BaseTriageAgent
from models.esi_assessment import MedicalConversation
from testing.models import TestMedicalConversation, TestResult, TestSummary
from testing.utils.data_loader import TestDataLoader
from testing.utils.metrics import MetricsCalculator


class TriageTestRunner:
    """
    Main test runner for evaluating triage agents against test data.
    """

    def __init__(self, test_data_path: Optional[str | Path] = None, max_workers: int = 4):
        """
        Initialize the test runner.

        Args:
            test_data_path: Path to test data file (uses default if None)
            max_workers: Maximum number of workers for parallel processing
        """
        self.test_data_path = test_data_path or TestDataLoader.get_default_test_file()
        self.max_workers = max_workers
        self.conversations: List[TestMedicalConversation] = []
        self.loaded = False

    def load_test_data(self) -> Dict[str, Any]:
        """
        Load test data from the configured file.

        Returns:
            Dict with loading statistics

        Raises:
            FileNotFoundError: If test data file doesn't exist
            ValueError: If test data format is invalid
        """
        print(f"Loading test data from: {self.test_data_path}")
        self.conversations = TestDataLoader.load_esi_test_cases(self.test_data_path)
        self.loaded = True

        # Validate and get statistics
        stats = TestDataLoader.validate_test_data(self.conversations)
        self._print_data_stats(stats)

        return stats

    def _print_data_stats(self, stats: Dict[str, Any]):
        """Print test data statistics."""
        print("\n=== Test Data Statistics ===")
        print(f"Total cases: {stats['total_cases']}")
        print(f"Valid cases: {stats['valid_cases']}")
        print(f"Cases with ESI levels: {stats['cases_with_esi']}")

        if stats['esi_distribution']:
            print("\nESI Level Distribution:")
            for esi_level in sorted(stats['esi_distribution'].keys()):
                count = stats['esi_distribution'][esi_level]
                print(f"  ESI {esi_level}: {count} cases")

        if stats['case_types']:
            print("\nCase Types:")
            for case_type, count in stats['case_types'].items():
                print(f"  {case_type}: {count} cases")

        print()

    def test_agent(self, agent: BaseTriageAgent, limit: Optional[int] = None) -> List[TestResult]:
        """
        Test a single agent against the loaded test data.

        Args:
            agent: The triage agent to test
            limit: Optional limit on number of test cases to run

        Returns:
            List of TestResult objects

        Raises:
            RuntimeError: If test data hasn't been loaded
        """
        if not self.loaded:
            raise RuntimeError("Test data must be loaded before running tests. Call load_test_data() first.")

        # Filter conversations that have expected ESI levels
        test_conversations = [c for c in self.conversations if c.expected_esi is not None]

        if limit:
            test_conversations = test_conversations[:limit]

        print(f"\n=== Testing Agent: {agent.name} ===")
        print(f"Running {len(test_conversations)} test cases...")

        start_time = time.time()
        results = []

        if self.max_workers == 1:
            # Single-threaded execution
            for i, conversation in enumerate(test_conversations):
                if i % 50 == 0:  # Progress update every 50 cases
                    print(f"Processing case {i+1}/{len(test_conversations)}...")

                result = self._test_single_case(agent, conversation)
                results.append(result)
        else:
            # Multi-threaded execution
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_conversation = {
                    executor.submit(self._test_single_case, agent, conv): conv
                    for conv in test_conversations
                }

                for i, future in enumerate(as_completed(future_to_conversation)):
                    if i % 50 == 0:  # Progress update every 50 cases
                        print(f"Completed {i+1}/{len(test_conversations)} cases...")

                    result = future.result()
                    results.append(result)

        total_time = time.time() - start_time
        accuracy = sum(1 for r in results if r.correct) / len(results)

        print(f"Completed in {total_time:.2f} seconds")
        print(f"Accuracy: {accuracy:.2%} ({sum(1 for r in results if r.correct)}/{len(results)})")

        return results

    def _test_single_case(self, agent: BaseTriageAgent, conversation: TestMedicalConversation) -> TestResult:
        """
        Test a single conversation with an agent.

        Args:
            agent: The triage agent to test
            conversation: The conversation to test

        Returns:
            TestResult object
        """
        # Convert TestMedicalConversation to production MedicalConversation
        production_conversation = MedicalConversation(turns=conversation.turns)
        assessment, processing_time = agent.triage_with_timing(production_conversation)

        return TestResult(
            case_id=conversation.case_id or "unknown",
            expected_esi=conversation.expected_esi,
            predicted_esi=assessment.esi_level,
            confidence=assessment.confidence,
            correct=assessment.esi_level == conversation.expected_esi,
            agent_name=agent.name,
            processing_time=processing_time,
            rationale=assessment.rationale
        )

    def test_multiple_agents(self, agents: List[BaseTriageAgent], limit: Optional[int] = None) -> Dict[str, List[TestResult]]:
        """
        Test multiple agents against the same test data.

        Args:
            agents: List of agents to test
            limit: Optional limit on number of test cases per agent

        Returns:
            Dict mapping agent names to their test results
        """
        if not self.loaded:
            self.load_test_data()

        all_results = {}
        for agent in agents:
            results = self.test_agent(agent, limit)
            all_results[agent.name] = results

        return all_results

    def generate_summary(self, results: List[TestResult]) -> TestSummary:
        """
        Generate a summary of test results.

        Args:
            results: List of test results

        Returns:
            TestSummary object with calculated metrics
        """
        return MetricsCalculator.calculate_summary(results)

    def generate_comparison(self, results_dict: Dict[str, List[TestResult]]) -> Dict[str, Any]:
        """
        Generate a comparison of multiple agents.

        Args:
            results_dict: Dict mapping agent names to their results

        Returns:
            Dict with comparison statistics
        """
        summaries = [self.generate_summary(results) for results in results_dict.values()]
        return MetricsCalculator.compare_agents(summaries)

    def save_results(self, results: List[TestResult], output_path: str | Path):
        """
        Save test results to a JSON file.

        Args:
            results: Test results to save
            output_path: Path to save the results
        """
        import json
        from datetime import datetime

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'timestamp': datetime.now().isoformat(),
            'total_cases': len(results),
            'agent_name': results[0].agent_name if results else 'Unknown',
            'results': [
                {
                    'case_id': r.case_id,
                    'expected_esi': r.expected_esi,
                    'predicted_esi': r.predicted_esi,
                    'correct': r.correct,
                    'confidence': r.confidence,
                    'processing_time': r.processing_time,
                    'rationale': r.rationale
                }
                for r in results
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to: {output_path}")