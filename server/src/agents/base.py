"""
Base agent class for emergency triage assessment.
"""
import time
from abc import ABC, abstractmethod
from typing import Dict, Any
from models.esi_assessment import MedicalConversation, ESIAssessment


class BaseTriageAgent(ABC):
    """
    Abstract base class for all triage agents.

    All triage agents must implement the triage method to assess
    medical conversations and return ESI level predictions.
    """

    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize the triage agent.

        Args:
            name: Human-readable name for the agent
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}

    @abstractmethod
    def triage(self, conversation: MedicalConversation) -> ESIAssessment:
        """
        Assess a medical conversation and return an ESI level prediction.

        Args:
            conversation: The medical conversation to assess

        Returns:
            ESIAssessment with predicted ESI level and optional details
        """
        pass

    def triage_with_timing(self, conversation: MedicalConversation) -> tuple[ESIAssessment, float]:
        """
        Assess a conversation with timing information.

        Args:
            conversation: The medical conversation to assess

        Returns:
            Tuple of (ESIAssessment, processing_time_seconds)
        """
        start_time = time.time()
        assessment = self.triage(conversation)
        processing_time = time.time() - start_time

        # Ensure agent name is set in the assessment
        assessment.agent_name = self.name

        return assessment, processing_time

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', config={self.config})"