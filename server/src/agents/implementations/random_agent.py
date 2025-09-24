"""
Random triage agent for baseline testing.
"""
import random
from typing import Dict, Any
from agents.base import BaseTriageAgent
from models.esi_assessment import MedicalConversation, ESIAssessment


class RandomTriageAgent(BaseTriageAgent):
    """
    A baseline triage agent that assigns random ESI levels.

    This agent is used to establish a baseline performance for comparison
    with more sophisticated agents. It randomly assigns ESI levels with
    configurable distribution weights.
    """

    def __init__(self, name: str = "Random", config: Dict[str, Any] = None):
        """
        Initialize the random triage agent.

        Args:
            name: Name for the agent
            config: Configuration dict with optional 'weights' key for ESI level probabilities
        """
        super().__init__(name, config)

        # Default uniform distribution across ESI levels
        default_weights = [1, 1, 1, 1, 1]  # ESI levels 1-5
        self.weights = config.get('weights', default_weights) if config else default_weights

        # Validate weights
        if len(self.weights) != 5:
            raise ValueError("Weights must be a list of 5 values for ESI levels 1-5")

        # Set random seed if provided for reproducible results
        if config and 'seed' in config:
            random.seed(config['seed'])

    def triage(self, conversation: MedicalConversation) -> ESIAssessment:
        """
        Randomly assign an ESI level based on configured weights.

        Args:
            conversation: The medical conversation (unused by this agent)

        Returns:
            ESIAssessment with randomly selected ESI level
        """
        # Randomly select ESI level (1-5) based on weights
        esi_level = random.choices(range(1, 6), weights=self.weights)[0]

        # Random confidence between 0.1 and 0.9 (never completely confident)
        confidence = random.uniform(0.1, 0.9)

        return ESIAssessment(
            esi_level=esi_level,
            confidence=confidence,
            rationale=f"Randomly assigned ESI level {esi_level}",
            agent_name=self.name
        )