"""
Utility for loading and parsing test data.
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from models.esi_assessment import ConversationTurn
from testing.models import TestMedicalConversation


class TestDataLoader:
    """Loads and parses test cases from JSON files."""

    @staticmethod
    def load_esi_test_cases(file_path: str | Path) -> List[TestMedicalConversation]:
        """
        Load ESI test cases from JSON file.

        Args:
            file_path: Path to the JSON file containing test cases

        Returns:
            List of TestMedicalConversation objects

        Raises:
            FileNotFoundError: If the test file doesn't exist
            ValueError: If the JSON format is invalid
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Test data file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in test data file: {e}")

        if not isinstance(raw_data, list):
            raise ValueError("Test data must be a JSON array of test cases")

        conversations = []
        for i, case_data in enumerate(raw_data):
            try:
                conversation = TestDataLoader._parse_test_case(case_data, i)
                conversations.append(conversation)
            except Exception as e:
                print(f"Warning: Skipping test case {i}: {e}")

        print(f"Loaded {len(conversations)} test cases from {file_path}")
        return conversations

    @staticmethod
    def _parse_test_case(case_data: Dict[str, Any], case_index: int) -> TestMedicalConversation:
        """
        Parse a single test case from the JSON data.

        Args:
            case_data: Dict containing test case data
            case_index: Index of the case for ID generation

        Returns:
            TestMedicalConversation object
        """
        # Extract expected ESI level
        expected_esi = None
        if 'expected' in case_data and 'esi_level' in case_data['expected']:
            expected_esi = case_data['expected']['esi_level']

        # Extract conversation turns
        conversation_turns = []
        if 'conversation' in case_data:
            for turn_data in case_data['conversation']:
                if 'speaker' in turn_data and 'message' in turn_data:
                    turn = ConversationTurn(
                        speaker=turn_data['speaker'],
                        message=turn_data['message']
                    )
                    conversation_turns.append(turn)

        # Extract case type first
        case_type = case_data.get('type', 'Unknown')

        # Generate descriptive case ID combining type and number
        if 'number' in case_data:
            case_id = f"{case_type}_{case_data['number']}"
        else:
            case_id = f"{case_type}_{case_index}"

        # Extract source case text from meta field
        source_case_text = None
        if 'meta' in case_data and 'source_case_text' in case_data['meta']:
            source_case_text = case_data['meta']['source_case_text']

        return TestMedicalConversation(
            turns=conversation_turns,
            case_id=case_id,
            expected_esi=expected_esi,
            case_type=case_type,
            source_case_text=source_case_text
        )

    @staticmethod
    def get_default_test_file() -> Path:
        """Get the default path to the ESI test cases file."""
        # From testing/utils/data_loader.py, go up to server level, then to tests/data
        return Path(__file__).parent.parent.parent.parent / "tests" / "data" / "esi_test_cases.json"

    @staticmethod
    def validate_test_data(conversations: List[TestMedicalConversation]) -> Dict[str, Any]:
        """
        Validate and analyze the loaded test data.

        Args:
            conversations: List of loaded conversations

        Returns:
            Dict with validation statistics
        """
        stats = {
            'total_cases': len(conversations),
            'valid_cases': 0,
            'cases_with_esi': 0,
            'esi_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            'empty_conversations': 0,
            'case_types': {}
        }

        for conversation in conversations:
            # Count valid cases
            if conversation.turns:
                stats['valid_cases'] += 1
            else:
                stats['empty_conversations'] += 1

            # Count cases with ESI levels
            if conversation.expected_esi:
                stats['cases_with_esi'] += 1
                stats['esi_distribution'][conversation.expected_esi] += 1

            # Count case types
            case_type = conversation.case_type or 'Unknown'
            stats['case_types'][case_type] = stats['case_types'].get(case_type, 0) + 1

        return stats