from typing import List, Optional
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

class SessionCreate(BaseModel):
    name: str

class SessionResponse(BaseModel):
    id: UUID
    name: str
    created_at: datetime

class SessionUpdate(BaseModel):
    name: Optional[str] = None
