from typing import List, Optional
from pydantic import BaseModel

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
