"""
HandbookRagOpenAiAgent - Replicates the reasoning logic from project/app/services/reasoner.py
"""

import asyncio
from typing import Dict, Any, List
from agents.base import BaseTriageAgent
from models.esi_assessment import ESIAssessment, MedicalConversation
from config import get_settings
from logger import get_logger
from services.openai_client import get_hosted_model_gateway
from services.rag import get_protocol_rag
from services.red_flags import RedFlag, RedFlagDetector

logger = get_logger(__name__)

JSON_SCHEMA = {
    "name": "triage_response",
    "schema": {
        "type": "object",
        "properties": {
            "follow_up_questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "priority": {"type": "integer"},
                        "rationale": {"type": "string"},
                        "escalation": {"type": "boolean"},
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
            "esi_level": {
                "type": ["integer", "null"],
                "minimum": 1,
                "maximum": 5,
            },
            "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "rationale": {"type": ["string", "null"]},
            "escalation_required": {"type": "boolean"},
        },
        "required": ["follow_up_questions", "escalation_required"],
        "additionalProperties": False,
    },
}


class HandbookRagOpenAiAgent(BaseTriageAgent):
    """
    Emergency triage agent that uses RAG, red-flag detection, and OpenAI
    to assess conversations and provide ESI recommendations.

    This replicates the exact logic from project/app/services/reasoner.py
    """

    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        settings = get_settings()

        # Initialize logger for this specific instance
        configure_logging = getattr(__import__('logger'), 'configure_logging', None)
        if configure_logging:
            configure_logging(settings.log_level)

        # Initialize components (same as ReasoningEngine)
        self._gateway = get_hosted_model_gateway()
        self._rag = get_protocol_rag()
        self._red_flag_detector = RedFlagDetector(settings.red_flag_lexicon_path)
        self._max_questions = settings.max_follow_up_questions

        # Agent-specific configuration (improved temperature for better calibration)
        self.temperature = config.get('temperature', 0.2)  # Lower for more consistent reasoning
        self.max_questions = config.get('max_questions', 3)
        # Use higher-capacity model for better medical reasoning (default to gpt-4o for accuracy)
        self.model_override = config.get('model_override', 'gpt-4o')

        logger.info("handbook_rag_openai_agent_initialized",
                   name=name,
                   max_questions=self._max_questions,
                   temperature=self.temperature)

    def triage(self, conversation: MedicalConversation) -> ESIAssessment:
        """
        Main triage method - runs async logic in sync context
        """
        try:
            # Run async generation in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._generate_assessment(conversation))
                return result
            finally:
                loop.close()
        except Exception as e:
            logger.error("triage_error", error=str(e), agent=self.name)
            # Return safe fallback
            return ESIAssessment(
                esi_level=3,
                confidence=0.1,
                rationale=f"Error during assessment: {str(e)}",
                agent_name=self.name
            )

    async def _generate_assessment(self, conversation: MedicalConversation) -> ESIAssessment:
        """
        Generate triage assessment - replicates ReasoningEngine.generate()
        """
        # Extract conversation text
        combined_text = conversation.get_full_text()

        # Red flag detection
        red_flag_matches: List[RedFlag] = self._red_flag_detector.scan(combined_text or "")
        red_flag_terms = list({flag.term for flag in red_flag_matches})

        # RAG retrieval
        rag_docs = self._rag.query(combined_text or "triage assessment guidance")
        rag_context = "\n\n".join(f"- {doc.page_content}" for doc in rag_docs)

        # Build prompt (same structure as reasoner.py)
        prompt = self._build_prompt(combined_text, red_flag_terms, rag_context)

        logger.info(
            "handbook_agent_prompt",
            red_flags=red_flag_terms,
            rag_docs=len(rag_docs),
            transcript_chars=len(combined_text),
            agent=self.name
        )

        # Call OpenAI
        response = await self._gateway.stream_completion(
            prompt=prompt,
            json_schema=JSON_SCHEMA,
            temperature=self.temperature,
            model_override=self.model_override
        )

        logger.info("handbook_agent_response_received", agent=self.name)

        # Convert to ESIAssessment
        return self._to_esi_assessment(response, red_flag_matches)

    def _build_prompt(self, transcript: str, red_flag_terms: List[str], rag_context: str) -> str:
        """
        Build prompt - enhanced with structured ESI reasoning
        """
        instructions = [
            "You are an emergency department triage copilot assisting a nurse.",
            "Summarize the key symptoms, identify missing critical questions, and estimate an ESI level when possible.",
            "Use the provided protocol context and red-flag list to inform your judgment.",
            "",
            "ESI Level Guidelines:",
            "• Level 1 (Life-threatening): Requires immediate lifesaving intervention (airway, breathing, circulation, neuro)",
            "• Level 2 (High risk): High-risk presentation, severe pain/distress WITH vital sign abnormalities OR high-risk mechanism, or imminent danger of deterioration",
            "• Level 3 (Moderate): Stable, needs 2 or more resources",
            "• Level 4 (Lower urgency): Stable, needs 1 resource",
            "• Level 5 (Non-urgent): Stable, needs 0 resources",
            "",
            "Resource Definitions (count carefully):",
            "Resources (count towards 2+): Lab tests, X-rays/imaging, IV fluids, IV medications, specialty consults, procedures requiring sedation/anesthesia",
            "NOT Resources: Simple prescriptions, tetanus shots, simple wound care, routine vital signs, basic examinations, discharge instructions",
            "",
            "Pain Score Guidelines:",
            "• Severe pain (9-10/10) ALONE only warrants ESI 2 if ALSO accompanied by:",
            "  - Vital sign abnormalities (hypotension, tachycardia, fever)",
            "  - High-risk mechanism (trauma, post-surgical complications)",
            "  - Signs of deterioration or systemic involvement",
            "• Otherwise, severe pain in stable patients typically warrants ESI 3-5 based on resources needed",
            "",
            "Age-Specific Considerations:",
            "• Neonates (<28 days): Any fever = ESI 2 minimum",
            "• Infants (1-24 months): Lower threshold for concerning symptoms",
            "• Elderly (>65): Consider baseline frailty and medication interactions",
            "• Pediatric vital signs: Higher HR and RR are normal",
            "",
            "Key Decision Factors:",
            "• Vital sign stability and concerning trends",
            "• Patient appearance and demeanor (distressed, anxious, pale, diaphoretic)",
            "• Potential for clinical deterioration based on mechanism and symptoms",
            "• Age-appropriate assessment of severity",
            "",
            "Respond with raw JSON only, matching the schema keys: follow_up_questions (list of objects), esi_level, confidence, rationale, escalation_required.",
            f"The follow_up_questions array should contain at most {self.max_questions} items, each with question, priority, rationale, escalation.",
        ]
        prompt_sections = [
            "\n".join(f"- {item}" for item in instructions),
            f"Transcript:\n{transcript if transcript else 'No transcript yet.'}",
            f"Red-flag terms detected: {', '.join(red_flag_terms) if red_flag_terms else 'none'}",
            f"Protocol context:\n{rag_context if rag_context else 'No relevant passages.'}",
        ]
        return "\n\n".join(prompt_sections)

    def _to_esi_assessment(self, llm_response: dict, matches: List[RedFlag]) -> ESIAssessment:
        """
        Convert LLM response to ESIAssessment - same logic as reasoner.py
        """
        # Handle confidence normalization
        confidence = None
        if isinstance(llm_response, dict):
            raw_confidence = llm_response.get("confidence")
            if isinstance(raw_confidence, str):
                normalized = raw_confidence.strip().lower()
                mapping = {"high": 0.9, "medium": 0.6, "low": 0.3}
                confidence = mapping.get(normalized)
            else:
                confidence = raw_confidence

        # Extract ESI level and rationale
        esi_level = None
        rationale = None
        escalation_required = False

        if isinstance(llm_response, dict):
            esi_level = llm_response.get("esi_level")
            rationale = llm_response.get("rationale")
            escalation_required = bool(llm_response.get("escalation_required", False))

        # Extract follow-up questions from LLM response
        follow_up_questions = []
        if isinstance(llm_response, dict):
            questions_payload = llm_response.get("follow_up_questions", [])
            for idx, item in enumerate(questions_payload):
                if idx >= self.max_questions:
                    break
                if isinstance(item, dict) and "question" in item:
                    follow_up_questions.append(item["question"])
                elif isinstance(item, str):
                    follow_up_questions.append(item)

        # Add red flag questions if no questions from LLM and we have red flag matches
        if not follow_up_questions and matches:
            # Use red flag detector's suggested questions if available
            if hasattr(self._red_flag_detector, 'suggested_questions'):
                for follow_up in self._red_flag_detector.suggested_questions(matches):
                    if len(follow_up_questions) >= self.max_questions:
                        break
                    follow_up_questions.append(follow_up)

        # Handle red flag escalation
        if matches and any(flag.escalation for flag in matches):
            escalation_required = True
            if rationale:
                rationale += f" Red flags detected: {', '.join([flag.term for flag in matches])}"
            else:
                rationale = f"Red flags detected: {', '.join([flag.term for flag in matches])}"

        return ESIAssessment(
            esi_level=esi_level,
            confidence=confidence,
            rationale=rationale,
            follow_up_questions=follow_up_questions,
            agent_name=self.name
        )