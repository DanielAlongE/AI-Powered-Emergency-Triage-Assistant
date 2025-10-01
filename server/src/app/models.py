from sqlalchemy import Column, String, DateTime, Uuid
from datetime import datetime
import uuid
from .database import Base

class Session(Base):
    __tablename__ = "sessions"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
