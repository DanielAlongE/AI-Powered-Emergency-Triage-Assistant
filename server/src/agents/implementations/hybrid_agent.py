"""
Hybrid triage agent combining rule-based and LLM approaches for ESI assessment.
"""
from typing import Dict, Any, Optional
from agents.base import BaseTriageAgent
from agents.implementations.rule_based_agent import RuleBasedTriageAgent
from agents.implementations.llm_agent import LLMTriageAgent
from models.esi_assessment import MedicalConversation, ESIAssessment


class HybridTriageAgent(BaseTriageAgent):
    """
    Work in Progress, not ready to use yet.
    A hybrid agent that combines rule-based and LLM-based approaches.
    """

    def __init__(self, name: str = "Hybrid", config: Dict[str, Any] = None):
        """
        Initialize the hybrid triage agent.

        Args:
            name: Name for the agent
            config: Configuration dict
        """
        super().__init__(name, config)

        # Initialize sub-agents
        self.rule_agent = RuleBasedTriageAgent("Rule-Component", config)
        self.llm_agent = LLMTriageAgent("LLM-Component", config)

        # Configuration for hybrid decision making
        self.rule_weight = config.get('rule_weight', 0.3) if config else 0.3
        self.llm_weight = config.get('llm_weight', 0.7) if config else 0.7

    def triage(self, conversation: MedicalConversation) -> ESIAssessment:
        """
        Assess conversation using hybrid rule-based and LLM approach.

        Args:
            conversation: The medical conversation to assess

        Returns:
            ESIAssessment combining both approaches
        """
        try:
            # Get assessments from both agents
            rule_assessment = self.rule_agent.triage(conversation)
            llm_assessment = self.llm_agent.triage(conversation)

            # Combine the assessments
            combined_esi = self._combine_esi_levels(rule_assessment.esi_level, llm_assessment.esi_level)
            combined_confidence = self._combine_confidences(rule_assessment.confidence, llm_assessment.confidence)
            combined_rationale = self._combine_rationales(rule_assessment, llm_assessment)

            return ESIAssessment(
                esi_level=combined_esi,
                confidence=combined_confidence,
                rationale=combined_rationale,
                agent_name=self.name
            )

        except Exception as e:
            print(f"Error in hybrid triage assessment: {e}")
            # Fallback to rule-based assessment
            return self.rule_agent.triage(conversation)

    def _combine_esi_levels(self, rule_esi: int, llm_esi: int) -> int:
        """
        Combine ESI levels from rule and LLM agents.

        Args:
            rule_esi: ESI level from rule-based agent
            llm_esi: ESI level from LLM agent

        Returns:
            Combined ESI level
        """
        # For critical cases (ESI 1-2), take the more urgent assessment
        if rule_esi <= 2 or llm_esi <= 2:
            return min(rule_esi, llm_esi)

        # For non-critical cases, use weighted average (rounded to nearest integer)
        weighted_avg = (rule_esi * self.rule_weight) + (llm_esi * self.llm_weight)
        return max(1, min(5, round(weighted_avg)))

    def _combine_confidences(self, rule_conf: Optional[float], llm_conf: Optional[float]) -> float:
        """
        Combine confidence scores from both agents.

        Args:
            rule_conf: Confidence from rule-based agent
            llm_conf: Confidence from LLM agent

        Returns:
            Combined confidence score
        """
        if rule_conf is None and llm_conf is None:
            return 0.5

        if rule_conf is None:
            return llm_conf

        if llm_conf is None:
            return rule_conf

        # Weighted average of confidences
        return (rule_conf * self.rule_weight) + (llm_conf * self.llm_weight)

    def _combine_rationales(self, rule_assessment: ESIAssessment, llm_assessment: ESIAssessment) -> str:
        """
        Combine rationales from both agents.

        Args:
            rule_assessment: Assessment from rule-based agent
            llm_assessment: Assessment from LLM agent

        Returns:
            Combined rationale string
        """
        rule_part = f"Rule-based: {rule_assessment.rationale}" if rule_assessment.rationale else ""
        llm_part = f"LLM analysis: {llm_assessment.rationale}" if llm_assessment.rationale else ""

        parts = [part for part in [rule_part, llm_part] if part]
        return " | ".join(parts) if parts else "Combined assessment completed."