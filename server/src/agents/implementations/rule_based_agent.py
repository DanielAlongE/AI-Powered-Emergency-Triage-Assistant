"""
Rule-based triage agent using keyword matching and clinical rules.
"""
import re
from typing import Dict, Any, List, Set
from agents.base import BaseTriageAgent
from models.esi_assessment import MedicalConversation, ESIAssessment


class RuleBasedTriageAgent(BaseTriageAgent):
    """
    A rule-based triage agent that uses keyword matching and clinical rules
    to assign ESI levels based on symptoms and presentations.
    """

    def __init__(self, name: str = "Rule-Based", config: Dict[str, Any] = None):
        """
        Initialize the rule-based triage agent.

        Args:
            name: Name for the agent
            config: Configuration dict with optional rule customization
        """
        super().__init__(name, config)

        # Define symptom keywords for each ESI level
        self.esi_keywords = self._get_default_keywords()

        # Override with custom keywords if provided
        if config and 'keywords' in config:
            self.esi_keywords.update(config['keywords'])

        # Compile regex patterns for efficiency
        self.compiled_patterns = self._compile_patterns()

    def _get_default_keywords(self) -> Dict[int, Dict[str, List[str]]]:
        """
        Get default keyword patterns for ESI classification.

        Returns:
            Dict mapping ESI levels to categories and their keywords
        """
        return {
            1: {  # Life-threatening - immediate care
                'cardiac_arrest': ['cardiac arrest', 'no pulse', 'cpr', 'asystole', 'v-fib', 'vfib'],
                'respiratory_failure': ['not breathing', 'respiratory arrest', 'apnea', 'blue', 'cyanotic'],
                'severe_trauma': ['major trauma', 'multiple injuries', 'penetrating', 'gunshot', 'stab wound'],
                'shock': ['shock', 'hypotensive', 'blood pressure', 'systolic', 'unresponsive'],
                'overdose': ['overdose', 'unconscious', 'unresponsive', 'coma', 'altered mental']
            },
            2: {  # High risk - within 10 minutes
                'chest_pain': ['chest pain', 'crushing', 'pressure', 'heart attack', 'mi', 'angina'],
                'stroke': ['stroke', 'weakness', 'numbness', 'slurred speech', 'facial droop'],
                'severe_pain': ['severe pain', '10/10', 'excruciating', 'unbearable'],
                'breathing_difficulty': ['shortness of breath', 'difficulty breathing', 'dyspnea', 'wheezing'],
                'bleeding': ['bleeding', 'blood loss', 'hemorrhage', 'heavy bleeding']
            },
            3: {  # Moderate urgency - 30 minutes
                'moderate_pain': ['pain', 'hurts', 'ache', 'sore', 'discomfort'],
                'fever': ['fever', 'temperature', 'hot', 'chills', 'feverish'],
                'nausea': ['nausea', 'vomiting', 'throwing up', 'sick to stomach'],
                'minor_injury': ['cut', 'scrape', 'bruise', 'sprain', 'twisted'],
                'infection': ['infection', 'infected', 'pus', 'swollen', 'red']
            },
            4: {  # Lower urgency - 1-2 hours
                'minor_symptoms': ['tired', 'fatigue', 'headache', 'dizzy', 'lightheaded'],
                'skin_issues': ['rash', 'itchy', 'skin problem', 'hives'],
                'digestive': ['stomach', 'diarrhea', 'constipation', 'gas'],
                'cold_symptoms': ['cold', 'cough', 'runny nose', 'congestion', 'sneezing']
            },
            5: {  # Non-urgent - can wait
                'routine': ['check up', 'follow up', 'routine', 'prescription refill'],
                'minor_concerns': ['worried about', 'concerned', 'questions'],
                'chronic_stable': ['chronic', 'stable', 'usual', 'same as always']
            }
        }

    def _compile_patterns(self) -> Dict[int, List[re.Pattern]]:
        """
        Compile regex patterns for efficient matching.

        Returns:
            Dict mapping ESI levels to compiled regex patterns
        """
        compiled = {}
        for esi_level, categories in self.esi_keywords.items():
            patterns = []
            for category, keywords in categories.items():
                for keyword in keywords:
                    # Create case-insensitive word boundary pattern
                    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                    patterns.append(pattern)
            compiled[esi_level] = patterns
        return compiled

    def triage(self, conversation: MedicalConversation) -> ESIAssessment:
        """
        Assess conversation using rule-based keyword matching.

        Args:
            conversation: The medical conversation to assess

        Returns:
            ESIAssessment with predicted ESI level based on rules
        """
        # Get full conversation text
        text = conversation.get_full_text().lower()

        # Check for matches at each ESI level (starting from most urgent)
        esi_scores = {}
        matched_keywords = []

        for esi_level in [1, 2, 3, 4, 5]:
            score = 0
            level_matches = []

            for pattern in self.compiled_patterns[esi_level]:
                matches = pattern.findall(text)
                if matches:
                    score += len(matches)
                    level_matches.extend(matches)

            esi_scores[esi_level] = score
            if level_matches:
                matched_keywords.append(f"ESI {esi_level}: {', '.join(level_matches[:3])}...")  # Show first 3 matches

        # Determine ESI level based on highest scoring level (most urgent wins)
        predicted_esi = self._determine_esi_level(esi_scores)

        # Calculate confidence based on score distribution
        confidence = self._calculate_confidence(esi_scores, predicted_esi)

        # Create rationale
        rationale = self._create_rationale(predicted_esi, esi_scores, matched_keywords)

        return ESIAssessment(
            esi_level=predicted_esi,
            confidence=confidence,
            rationale=rationale,
            agent_name=self.name
        )

    def _determine_esi_level(self, esi_scores: Dict[int, int]) -> int:
        """
        Determine ESI level based on keyword match scores.

        Args:
            esi_scores: Dict mapping ESI levels to match scores

        Returns:
            Predicted ESI level
        """
        # If any critical (ESI 1) keywords found, assign ESI 1
        if esi_scores[1] > 0:
            return 1

        # If high-risk (ESI 2) keywords found, assign ESI 2
        if esi_scores[2] > 0:
            return 2

        # For ESI 3-5, use highest scoring level
        max_score = max(esi_scores[3], esi_scores[4], esi_scores[5])

        if max_score == 0:
            # No keywords matched - default to ESI 4 (lower urgency)
            return 4

        # Return the most urgent level with the highest score
        for esi_level in [3, 4, 5]:
            if esi_scores[esi_level] == max_score:
                return esi_level

        return 4  # Default fallback

    def _calculate_confidence(self, esi_scores: Dict[int, int], predicted_esi: int) -> float:
        """
        Calculate confidence score based on keyword match distribution.

        Args:
            esi_scores: Dict mapping ESI levels to match scores
            predicted_esi: The predicted ESI level

        Returns:
            Confidence score between 0.1 and 0.95
        """
        total_score = sum(esi_scores.values())

        if total_score == 0:
            return 0.3  # Low confidence when no keywords match

        predicted_score = esi_scores[predicted_esi]
        confidence_raw = predicted_score / total_score

        # Scale confidence to reasonable range (0.1 to 0.95)
        confidence = 0.1 + (confidence_raw * 0.85)

        # Boost confidence for critical cases (ESI 1-2)
        if predicted_esi in [1, 2] and predicted_score > 0:
            confidence = min(0.95, confidence + 0.2)

        return round(confidence, 3)

    def _create_rationale(self, predicted_esi: int, esi_scores: Dict[int, int],
                         matched_keywords: List[str]) -> str:
        """
        Create human-readable rationale for the prediction.

        Args:
            predicted_esi: Predicted ESI level
            esi_scores: ESI scores for each level
            matched_keywords: List of matched keywords by level

        Returns:
            Rationale string
        """
        esi_descriptions = {
            1: "life-threatening symptoms requiring immediate care",
            2: "high-risk symptoms requiring urgent care",
            3: "moderate urgency symptoms",
            4: "lower urgency symptoms",
            5: "non-urgent symptoms"
        }

        rationale = f"Assigned ESI {predicted_esi} based on {esi_descriptions[predicted_esi]}. "

        predicted_score = esi_scores[predicted_esi]
        total_matches = sum(esi_scores.values())

        if predicted_score > 0:
            rationale += f"Detected {predicted_score} relevant keywords. "

        if matched_keywords:
            rationale += f"Key matches: {'; '.join(matched_keywords[:2])}."
        else:
            rationale += "No specific keywords matched - using default classification."

        return rationale

    def add_custom_keywords(self, esi_level: int, category: str, keywords: List[str]):
        """
        Add custom keywords for a specific ESI level and category.

        Args:
            esi_level: ESI level (1-5)
            category: Category name
            keywords: List of keywords to add
        """
        if esi_level not in self.esi_keywords:
            self.esi_keywords[esi_level] = {}

        if category not in self.esi_keywords[esi_level]:
            self.esi_keywords[esi_level][category] = []

        self.esi_keywords[esi_level][category].extend(keywords)

        # Recompile patterns
        self.compiled_patterns = self._compile_patterns()