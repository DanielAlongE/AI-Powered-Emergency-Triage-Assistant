from typing import List, Optional
from models.esi_assessment import ConversationTurn
from pydantic import BaseModel
from uuid import UUID
from datetime import datetime

class ChatResponse(BaseModel):
    role: str
    content: str


class ConversationResponse(BaseModel):
    conversation: List[ChatResponse]

class ConversationRequest(BaseModel):
    transcript: str

class TranscriptionRequest(BaseModel):
    audio_content: bytes
    mime_type: Optional[str] = None

class TranscriptionResponse(BaseModel):
    transcript: str


class TriageSummaryRequest(BaseModel):
    turns: List[ConversationTurn]

class SessionCreate(BaseModel):
    name: str

class SessionResponse(BaseModel):
    id: UUID
    name: str
    transcript: Optional[str]
    conversation: Optional[str]
    summary: Optional[str]
    created_at: datetime

class SessionUpdate(BaseModel):
    name: Optional[str] = None
    transcript: Optional[str] = None
    conversation: Optional[str] = None
    summary: Optional[str] = None

class AuditLogCreate(BaseModel):
    session_id: UUID
    suggestion: str
    response: str
    similarity: int

class AuditLogResponse(BaseModel):
    id: UUID
    session_id: UUID
    suggestion: str
    response: str
    similarity: int
    created_at: datetime

class AuditLogUpdate(BaseModel):
    suggestion: Optional[str] = None
    response: Optional[str] = None
    similarity: Optional[int] = None
