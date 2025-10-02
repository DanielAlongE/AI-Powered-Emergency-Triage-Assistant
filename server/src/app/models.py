from sqlalchemy import Column, String, DateTime, Uuid, Text
from datetime import datetime, timezone
import uuid
from .database import Base

class Session(Base):
    __tablename__ = "sessions"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    transcript = Column(Text)
    conversation = Column(Text)
    summary = Column(Text)
    updated_at = Column(DateTime)
