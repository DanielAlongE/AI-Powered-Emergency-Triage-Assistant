"""
ESI Assessment data models for emergency triage.
"""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, validator


class ESIAssessment(BaseModel):
    """
    Emergency Severity Index (ESI) assessment model.

    ESI Levels:
    1 - Life-threatening (immediate care)
    2 - High risk/critical (within 10 minutes)
    3 - Moderate urgency (30 minutes)
    4 - Lower urgency (1-2 hours)
    5 - Non-urgent (can wait)
    """
    esi_level: int = Field(..., ge=1, le=5, description="ESI level from 1 (most urgent) to 5 (least urgent)")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score for the assessment")
    rationale: Optional[str] = Field(None, description="Explanation of the ESI level assignment")
    follow_up_questions: Optional[List[str]] = Field(default_factory=list, description="Suggested follow-up questions")
    red_flag_terms: Optional[List[str]] = Field(default_factory=list, description="Red flag terms detected in conversation")
    timestamp: datetime = Field(default_factory=datetime.now, description="Assessment timestamp")
    agent_name: Optional[str] = Field(None, description="Name of the agent that made the assessment")

    @validator('esi_level')
    def validate_esi_level(cls, v):
        if v not in range(1, 6):
            raise ValueError('ESI level must be between 1 and 5')
        return v

    def __str__(self) -> str:
        return f"ESI Level {self.esi_level} (confidence: {self.confidence:.2f})" if self.confidence else f"ESI Level {self.esi_level}"


class ConversationTurn(BaseModel):
    """A single turn in a medical conversation."""
    speaker: str = Field(..., description="Who is speaking (nurse, patient, paramedic, etc.)")
    message: str = Field(..., description="The message content")

    def __str__(self) -> str:
        return f"{self.speaker}: {self.message}"


class MedicalConversation(BaseModel):
    """A complete medical conversation for triage assessment."""
    turns: List[ConversationTurn] = Field(..., description="List of conversation turns")

    def get_full_text(self) -> str:
        """Get the conversation as a single text string."""
        return "\n".join([f"{turn.speaker}: {turn.message}" for turn in self.turns])

    def __str__(self) -> str:
        return f"Conversation ({len(self.turns)} turns)"


