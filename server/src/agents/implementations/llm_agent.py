"""
LLM-based triage agent using Ollama/LLama for ESI assessment.
"""
import json
from typing import Dict, Any, Optional
import ollama
from agents.base import BaseTriageAgent
from models.esi_assessment import MedicalConversation, ESIAssessment


class LLMTriageAgent(BaseTriageAgent):
    """
    Work in Progress, not ready to use yet.
    An LLM-based triage agent that uses Ollama/LLama to assess conversations
    and assign ESI levels with rationale.
    """

    def __init__(self, name: str = "LLM-Based", config: Dict[str, Any] = None):
        """
        Initialize the LLM-based triage agent.

        Args:
            name: Name for the agent
            config: Configuration dict with optional model and prompt customization
        """
        super().__init__(name, config)

        # Model configuration
        self.model = config.get('model', 'llama3.2') if config else 'llama3.2'
        self.temperature = config.get('temperature', 0.3) if config else 0.3  # Lower temperature for consistency

        # System prompt for ESI assessment
        self.system_prompt = self._get_system_prompt()

        # Response format for structured output
        self.response_format = {
            "esi_level": 3,
            "confidence": 0.75,
            "rationale": "Based on the symptoms presented, this appears to be a moderate urgency case requiring assessment within 30 minutes.",
            "key_factors": [
                "Patient symptom 1",
                "Clinical finding 2"
            ]
        }

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for ESI assessment.

        Returns:
            System prompt string
        """
        return """You are an experienced emergency triage nurse with expertise in the Emergency Severity Index (ESI) system. Your task is to analyze medical conversations and assign appropriate ESI levels.

ESI Level Guidelines:
- ESI 1 (Resuscitation): Life-threatening conditions requiring immediate care (e.g., cardiac arrest, respiratory failure, severe trauma, shock, unconscious)
- ESI 2 (Emergent): High-risk situations requiring care within 10 minutes (e.g., chest pain, stroke symptoms, severe breathing difficulty, significant bleeding)
- ESI 3 (Urgent): Moderate urgency requiring care within 30 minutes (e.g., moderate pain, fever with concerning symptoms, minor injuries with complications)
- ESI 4 (Less Urgent): Lower urgency requiring care within 1-2 hours (e.g., minor injuries, stable chronic conditions, mild symptoms)
- ESI 5 (Non-urgent): Can wait, routine care (e.g., prescription refills, routine follow-ups, minor stable concerns)

Consider these factors:
1. Immediate life threats (ESI 1)
2. High-risk presentations (ESI 2)
3. Symptom severity and acuity
4. Patient stability and vital signs mentioned
5. Resource requirements for care

Provide your assessment as JSON with esi_level (1-5), confidence (0.0-1.0), rationale explaining your decision, and key_factors that influenced your assessment."""

    def triage(self, conversation: MedicalConversation) -> ESIAssessment:
        """
        Assess conversation using LLM analysis.

        Args:
            conversation: The medical conversation to assess

        Returns:
            ESIAssessment with LLM-predicted ESI level
        """
        try:
            # Prepare the conversation text
            conversation_text = self._format_conversation(conversation)

            # Create the assessment prompt
            prompt = f"""
            Analyze this medical conversation and assign an appropriate ESI level:

            Conversation:
            {conversation_text}

            Respond with JSON in this exact format:
            {json.dumps(self.response_format, indent=2)}
            """

            # Get LLM response
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': self.temperature,
                    'top_p': 0.9,
                },
                format="json"
            )

            # Parse the response
            analysis = self._parse_llm_response(response.message.content)

            return ESIAssessment(
                esi_level=analysis['esi_level'],
                confidence=analysis['confidence'],
                rationale=analysis['rationale'],
                follow_up_questions=analysis.get('follow_up_questions', []),
                agent_name=self.name
            )

        except Exception as e:
            print(f"Error in LLM triage assessment: {e}")
            # Fallback to safe default
            return ESIAssessment(
                esi_level=3,  # Moderate urgency as safe default
                confidence=0.1,  # Low confidence due to error
                rationale=f"LLM assessment failed: {str(e)}. Defaulting to ESI 3 for safety.",
                agent_name=self.name
            )

    def _format_conversation(self, conversation: MedicalConversation) -> str:
        """
        Format conversation for LLM analysis.

        Args:
            conversation: The conversation to format

        Returns:
            Formatted conversation string
        """
        lines = []
        for turn in conversation.turns:
            lines.append(f"{turn.speaker.title()}: {turn.message}")

        return "\n".join(lines)

    def _parse_llm_response(self, response_content: str) -> Dict[str, Any]:
        """
        Parse and validate LLM response.

        Args:
            response_content: Raw LLM response content

        Returns:
            Parsed and validated response dict
        """
        try:
            analysis = json.loads(response_content)
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM JSON response: {e}")
            print(f"Raw response: {response_content[:200]}...")
            # Return safe default
            return {
                'esi_level': 3,
                'confidence': 0.2,
                'rationale': 'Failed to parse LLM response. Using safe default.',
                'key_factors': []
            }

        # Validate and sanitize the response
        esi_level = analysis.get('esi_level', 3)
        if not isinstance(esi_level, int) or esi_level < 1 or esi_level > 5:
            esi_level = 3

        confidence = analysis.get('confidence', 0.5)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            confidence = 0.5

        rationale = analysis.get('rationale', 'No rationale provided')
        if not isinstance(rationale, str):
            rationale = 'Invalid rationale format'

        key_factors = analysis.get('key_factors', [])
        if not isinstance(key_factors, list):
            key_factors = []

        return {
            'esi_level': esi_level,
            'confidence': float(confidence),
            'rationale': rationale,
            'key_factors': key_factors
        }

    def test_connection(self) -> bool:
        """
        Test connection to Ollama service.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'user', 'content': 'Hello, please respond with "OK"'}
                ],
                options={'temperature': 0.1}
            )
            return 'ok' in response.message.content.lower()
        except Exception as e:
            print(f"Ollama connection test failed: {e}")
            return False

