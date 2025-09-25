from typing import List
from pydantic import BaseModel

class ChatResponse(BaseModel):
    role: str
    content: str


class ConversationResponse(BaseModel):
    conversation: List[ChatResponse]

class ConversationRequest(BaseModel):
    transcript: str