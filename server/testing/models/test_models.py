"""
Testing-specific data models for emergency triage testing infrastructure.
"""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, validator

from models.esi_assessment import ConversationTurn


class TestMedicalConversation(BaseModel):
    """A complete medical conversation for triage assessment testing."""
    turns: List[ConversationTurn] = Field(..., description="List of conversation turns")
    case_id: Optional[str] = Field(None, description="Unique identifier for the case")
    expected_esi: Optional[int] = Field(None, ge=1, le=5, description="Expected ESI level for testing")
    case_type: Optional[str] = Field(None, description="Type of case (e.g., 'Practice')")
    source_case_text: Optional[str] = Field(None, description="Original source case text from meta field")

    @validator('expected_esi')
    def validate_expected_esi(cls, v):
        if v is not None and v not in range(1, 6):
            raise ValueError('Expected ESI level must be between 1 and 5')
        return v

    def get_full_text(self) -> str:
        """Get the conversation as a single text string."""
        return "\n".join([f"{turn.speaker}: {turn.message}" for turn in self.turns])

    def __str__(self) -> str:
        return f"Conversation ({len(self.turns)} turns)" + (f" - Expected ESI: {self.expected_esi}" if self.expected_esi else "")


class TestResult(BaseModel):
    """Result of testing an agent against a conversation."""
    case_id: str = Field(..., description="Case identifier")
    expected_esi: int = Field(..., ge=1, le=5, description="Expected ESI level")
    predicted_esi: int = Field(..., ge=1, le=5, description="Predicted ESI level")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Prediction confidence")
    correct: bool = Field(..., description="Whether the prediction was correct")
    agent_name: str = Field(..., description="Name of the agent that made the prediction")
    processing_time: float = Field(..., description="Time taken to process in seconds")
    rationale: Optional[str] = Field(None, description="Agent's rationale for the decision")

    def __str__(self) -> str:
        status = "✓" if self.correct else "✗"
        return f"{status} {self.case_id}: {self.predicted_esi} vs {self.expected_esi} ({self.agent_name})"


class TestSummary(BaseModel):
    """Summary of test results for an agent."""
    agent_name: str = Field(..., description="Name of the tested agent")
    total_cases: int = Field(..., description="Total number of test cases")
    correct_predictions: int = Field(..., description="Number of correct predictions")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Overall accuracy")
    avg_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Average confidence score")
    avg_processing_time: float = Field(..., description="Average processing time in seconds")
    results_by_esi: dict = Field(default_factory=dict, description="Results broken down by ESI level")
    confusion_matrix: List[List[int]] = Field(default_factory=list, description="Confusion matrix as 5x5 list")

    def __str__(self) -> str:
        return f"{self.agent_name}: {self.accuracy:.2%} accuracy ({self.correct_predictions}/{self.total_cases})"