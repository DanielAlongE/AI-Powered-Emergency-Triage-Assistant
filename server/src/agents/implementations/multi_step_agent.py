import re
import asyncio
from typing import List, Dict, Any, Optional
from agents.base import BaseTriageAgent
from models.esi_assessment import MedicalConversation, ESIAssessment
from services.ollama_client import get_ollama_gateway


class MultiStepESIAgent(BaseTriageAgent):
    """
    Multi-step reasoning process following the Emergency Severity Index 
    (ESI) v5 decision algorithm to determine ESI level.

    This agent queries a large language model (LLM) at each major decision point:
    A. Immediate life-saving intervention required?
    B. High-risk situation (severe distress/pain or risk of deterioration)?
    C. How many resources will the patient likely need?
    D. Are vital signs in a dangerous range that would upgrade acuity?
    """
    def __init__(self, name: str = "MultiStepESI", config: Dict[str, Any] = None):
        """
        Initialize the multi-step ESI triage agent.
        Args:
            name: Name for the agent (default "MultiStepESI")
            config: Optional configuration dictionary (unused in this agent)
        """
        super().__init__(name, config or {})
        self.llm_gateway = get_ollama_gateway()
        self.model_name = "gpt-oss:20b-cloud"
        self.temperature = 0.1
        self.non_resource_keywords = ["tetanus", "eye drop", "slit lamp", 
                                      "visual acuity", "splint", "crutch", "prescription", "exam"]
    
    def _parse_yes_no(self, text: str) -> (bool, str):
        """
        Helper to interpret a yes/no question answer from model output.
        Returns a tuple (decision_bool, explanation_string).
        """
        text = text.strip()
        # Look for 'yes' or 'no' as a separate word (case-insensitive)
        decision = None
        explanation = text
        # Check common patterns at start (e.g., "Yes, ...", "No. ...")
        match = re.match(r'^\s*(yes|no)\b[\s:,.]*', text, flags=re.IGNORECASE)
        if match:
            decision_word = match.group(1).lower()
            if decision_word == "yes":
                decision = True
            elif decision_word == "no":
                decision = False
            # Strip the leading yes/no from explanation for clarity
            explanation = text[match.end():].strip()
        # If no clear yes/no found, try to infer from text content
        if decision is None:
            if "yes" in text.lower():
                decision = True
            elif "no" in text.lower():
                decision = False
        # Clean up explanation text
        if explanation is None:
            explanation = ""
        explanation = explanation.strip().strip('"').strip()
        return decision if decision is not None else False, explanation
    
    def _count_resources_from_list(self, resources: List[str]) -> int:
        """
        Count the number of distinct ESI resources needed, excluding items that are not counted as resources.
        """
        count = 0
        for res in resources:
            res_lower = res.lower().strip()
            if not res_lower:
                continue
            # Skip any item containing a non-resource keyword
            skip = False
            for keyword in self.non_resource_keywords:
                if keyword in res_lower:
                    skip = True
                    break
            if skip:
                continue
            # Skip oral medications/prescriptions (not ESI resources)
            if "oral" in res_lower and ("medication" in res_lower or "antibiotic" in res_lower or "medications" in res_lower):
                continue
            # Count this as a resource
            count += 1
        return count
    
    def _extract_vitals(self, text: str, age: Optional[int]) -> Dict[str, float]:
        """
        Extract vital sign values from text. Returns a dict with any of 'hr', 'rr', 'o2' keys if found.
        """
        vitals = {}
        # Heart rate (HR or pulse)
        hr_match = re.search(r'(heart rate|HR|pulse)\D*?(\d+)', text, flags=re.IGNORECASE)
        if hr_match:
            vitals['hr'] = float(hr_match.group(2))
        # Respiratory rate (RR)
        rr_match = re.search(r'(respiratory rate|RR)\D*?(\d+)', text, flags=re.IGNORECASE)
        if rr_match:
            vitals['rr'] = float(rr_match.group(2))
        # Oxygen saturation (SpO2 or O2 sat)
        o2_match = re.search(r'(O2 sat|oxygen saturation|SpO2)\D*?(\d+)', text, flags=re.IGNORECASE)
        if o2_match:
            vitals['o2'] = float(o2_match.group(2))
        return vitals
    
    def _get_age_from_conversation(self, text: str) -> Optional[int]:
        """
        Attempt to extract patient age from the conversation text.
        Returns the age in years if found, otherwise None.
        """
        # Patterns like "I'm 27 years old" or "27-year-old"
        age_match = re.search(r'(\d+)\s*(?:years? old|y/o)', text, flags=re.IGNORECASE)
        if age_match:
            try:
                return int(age_match.group(1))
            except ValueError:
                return None
        age_match2 = re.search(r'(\d+)-year-old', text, flags=re.IGNORECASE)
        if age_match2:
            try:
                return int(age_match2.group(1))
            except ValueError:
                return None
        return None
    
    def triage(self, conversation: MedicalConversation) -> ESIAssessment:
        """
        Assess the medical conversation using ESI v5 algorithm via multi-step LLM reasoning.
        Returns an ESIAssessment containing ESI level, confidence, rationale, and follow-up questions.
        """
        # Combine conversation turns into a single text prompt
        conv_text = conversation.get_full_text()
        # Determine patient age if mentioned (for vital sign thresholds)
        age = self._get_age_from_conversation(conv_text)
        
        # Step A: Check for immediate life-saving intervention
        prompt_a = (
            f"{conv_text}\n\n"
            f"Question A: Does this patient require any immediate, life-saving intervention "
            f"(e.g., to support airway, breathing, circulation, or neurological function)? "
            f"Answer 'Yes' or 'No' and explain briefly."
        )
        response_a = asyncio.run(self.llm_gateway.stream_completion(
            prompt=prompt_a,
            model_override=self.model_name,
            temperature=self.temperature
        ))
        text_a = response_a.get("text", "") if isinstance(response_a, dict) else str(response_a)
        life_saving_needed, reason_a = self._parse_yes_no(text_a)
        if life_saving_needed:
            # ESI 1: needs immediate resuscitation
            esi_level = 1
            rationale = f"Requires immediate lifesaving intervention. {reason_a}" if reason_a else \
                        "Requires immediate lifesaving intervention."
            confidence = 0.99  # very high confidence for ESI-1
            follow_up_questions: List[str] = []
            # No further steps needed for ESI 1
            return ESIAssessment(
                esi_level=esi_level,
                confidence=confidence,
                rationale=rationale.strip(),
                follow_up_questions=follow_up_questions
            )
        
        # Step B: Check for high-risk scenario or severe pain/distress
        prompt_b = (
            f"{conv_text}\n\n"
            f"Question B: Is this patient high-risk or in severe pain or distress? "
            f"Consider factors like altered mental status, severe pain (≥7/10), signs of stroke, "
            f"respiratory distress, uncontrolled bleeding, or any condition that could quickly worsen. "
            f"Answer 'Yes' or 'No' and explain briefly."
        )
        response_b = asyncio.run(self.llm_gateway.stream_completion(
            prompt=prompt_b,
            model_override=self.model_name,
            temperature=self.temperature
        ))
        text_b = response_b.get("text", "") if isinstance(response_b, dict) else str(response_b)
        high_risk, reason_b = self._parse_yes_no(text_b)
        if high_risk:
            # ESI 2: high-risk presentation (needs rapid care, but not immediate resuscitation)
            esi_level = 2
            rationale = f"High risk. {reason_b}" if reason_b else "High risk situation requiring rapid care."
            confidence = 0.95  # high confidence for clear high-risk cases
            follow_up_questions: List[str] = []
            return ESIAssessment(
                esi_level=esi_level,
                confidence=confidence,
                rationale=rationale.strip(),
                follow_up_questions=follow_up_questions
            )
        
        # Step C: Estimate number of resources needed (for ESI 3-5 determination)
        prompt_c = (
            f"{conv_text}\n\n"
            f"Question C: List the medical resources or interventions this patient will likely need in the ED to reach a diagnosis and treatment. "
            f"Resources can include lab tests, imaging (X-ray, CT, etc.), IV fluids, IV/IM medications, procedures, or consults. "
            f"If no resources beyond the physician exam are needed, provide an empty list. "
            f'Respond with a JSON object in this exact format: {{"resources": ["resource1", "resource2"]}} or {{"resources": []}} if none needed.'
        )
        schema_c = {
            "type": "object",
            "properties": {
                "resources": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["resources"]
        }
        response_c = asyncio.run(self.llm_gateway.stream_completion(
            prompt=prompt_c,
            model_override=self.model_name,
            temperature=self.temperature,
            json_schema=schema_c
        ))
        resources_list: List[str] = []
        if isinstance(response_c, dict):
            # Model returned structured JSON
            resources_list = response_c.get("resources", [])
        if not isinstance(resources_list, list):
            resources_list = []
        # Ensure all items are strings
        resources_list = [str(item).strip() for item in resources_list if isinstance(item, (str,))]
        resource_count = self._count_resources_from_list(resources_list)
        # Assign ESI based on resource count (none=5, one=4, >=2=3)
        if resource_count == 0:
            esi_level = 5
        elif resource_count == 1:
            esi_level = 4
        else:
            esi_level = 3
        
        # Step D: If provisional ESI is 3, check vital signs for possible up-triage to ESI 2
        vitals = self._extract_vitals(conv_text, age)
        if esi_level == 3:
            # Determine danger zone thresholds for high-risk vitals by age group
            danger_zone = {"hr": 100, "rr": 20, "o2": 92}  # adult defaults
            if age is not None:
                if age < 1:
                    danger_zone["hr"] = 190; danger_zone["rr"] = 60
                elif 1 <= age < 12:
                    # 1-12 months
                    danger_zone["hr"] = 180; danger_zone["rr"] = 55
                elif 1 <= age < 3:
                    # 1-3 years (toddler)
                    danger_zone["hr"] = 180; danger_zone["rr"] = 40
                elif 3 <= age < 6:
                    # 3-5 years
                    danger_zone["hr"] = 140; danger_zone["rr"] = 35
                elif 6 <= age < 13:
                    # 5-12 years
                    danger_zone["hr"] = 120; danger_zone["rr"] = 30
                elif 13 <= age < 19:
                    # 12-18 years
                    danger_zone["hr"] = 120; danger_zone["rr"] = 20
                else:
                    # adult (≥18)
                    danger_zone["hr"] = 100; danger_zone["rr"] = 20
            # Flag if any vital exceeds danger zone thresholds
            vital_flag = False
            if 'hr' in vitals and vitals['hr'] > danger_zone['hr']:
                vital_flag = True
            if 'rr' in vitals and vitals['rr'] > danger_zone['rr']:
                vital_flag = True
            if 'o2' in vitals and vitals['o2'] < danger_zone['o2']:
                vital_flag = True
            if vital_flag:
                esi_level = 2  # upgrade to ESI 2 due to abnormal vitals
        
        # Construct the rationale from decisions and resource analysis
        rationale_parts: List[str] = []
        if esi_level == 5:
            rationale_parts.append("No resources.")
        elif esi_level == 4:
            rationale_parts.append("One resource.")
        elif esi_level == 3:
            rationale_parts.append("Two or more resources.")
        elif esi_level == 2:
            # ESI 2 could come from high risk (B) or vital sign upgrade (D)
            if high_risk:
                rationale_parts.append("High risk.")
            else:
                rationale_parts.append("High risk vital signs.")
        
        # Include additional details in rationale
        explained_resources = [res for res in resources_list
                               if not any(keyword in res.lower() for keyword in self.non_resource_keywords)]
        if esi_level == 5:
            if explained_resources:
                rationale_parts.append("This patient will be evaluated and discharged without requiring any ED resources.")
            else:
                rationale_parts.append("This patient will be evaluated and does not require any ED resources.")
        elif esi_level == 4:
            if explained_resources:
                rationale_parts.append(f"The patient will need {explained_resources[0]} – one resource.")
        elif esi_level == 3:
            if len(explained_resources) >= 2:
                rationale_parts.append(f"At least {explained_resources[0]} and {explained_resources[1]} will be required – two or more resources.")
            elif len(explained_resources) == 1:
                rationale_parts.append(f"{explained_resources[0]} and additional workup will be required – two or more resources.")
        if esi_level == 2:
            if high_risk and reason_b:
                rationale_parts.append(reason_b)
            elif esi_level == 2 and life_saving_needed is False:
                rationale_parts.append("Vital signs are outside normal range indicating potential instability.")
        
        rationale = " ".join(part.strip() for part in rationale_parts if part).strip()
        
        # Estimate confidence score
        if esi_level == 2:
            confidence = 0.9
        elif esi_level == 3:
            confidence = 0.8
        elif esi_level == 4:
            confidence = 0.85
        elif esi_level == 5:
            confidence = 0.9
        else:
            confidence = 0.99  # (ESI 1 handled earlier)
        # Adjust confidence if the model's explanations showed uncertainty
        combined_reasons = " ".join([reason_a, reason_b])
        uncertainty_markers = ["possibly", "maybe", "probably", "not sure", "uncertain", "not certain", "perhaps"]
        for marker in uncertainty_markers:
            if marker in combined_reasons.lower():
                confidence = max(0.5, confidence - 0.3)
                break
        
        # Prepare follow-up questions if more info could clarify the triage
        follow_up_questions: List[str] = []
        # If vital signs were not mentioned and patient is not already critical, ask for vitals
        vitals_mentioned = bool(re.search(r'vital signs|heart rate|respiratory rate|O2 sat|SpO2|blood pressure', conv_text, flags=re.IGNORECASE))
        if not vitals_mentioned and esi_level >= 3:
            follow_up_questions.append("What are the patient's vital signs?")
        # If pain is mentioned but no numeric rating given, ask for pain scale
        if re.search(r'\bpain\b', conv_text, flags=re.IGNORECASE) and not re.search(r'/10', conv_text):
            follow_up_questions.append("On a scale of 1 to 10, how severe is the pain?")
        # If female with abdominal pain and no mention of pregnancy, ask about pregnancy (clarifies high-risk possibility)
        if re.search(r'abdominal pain', conv_text, flags=re.IGNORECASE) and re.search(r'female|woman|girl', conv_text, flags=re.IGNORECASE):
            if not re.search(r'pregnant|pregnancy|LMP', conv_text, flags=re.IGNORECASE):
                follow_up_questions.append("Is there any possibility the patient could be pregnant?")
        
        # Return the assessment object with all fields
        return ESIAssessment(
            esi_level=esi_level,
            confidence=confidence,
            rationale=rationale,
            follow_up_questions=follow_up_questions
        )
