from agents.implementations.handbook_rag_ollama_agent import HandbookRagOllamaAgent
from agents.implementations.handbook_rag_openai_agent import HandbookRagOpenAiAgent
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends
from models.esi_assessment import ConversationTurn, ESIAssessment, MedicalConversation
from sqlalchemy.orm import Session as DBSession
from typing import List, Optional
from uuid import UUID
from datetime import datetime, timezone
import asyncio
import json
import numpy as np
from app.schemas import ConversationResponse, ConversationRequest, TranscriptionRequest, TranscriptionResponse, SessionCreate, SessionResponse, SessionUpdate, TriageSummaryRequest, AuditLogCreate, AuditLogResponse, AuditLogUpdate
from models.llama import ConversationAnalizer, MODEL_GEMMA_3, MODEL_GPT_4O
from models.transcription import VoskTranscriber
from .database import SessionLocal
from .models import Session as SessionModel, AuditLog
from config import get_settings


router = APIRouter(prefix="/api", tags=["triage"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# receive chunks of audio byte and return a string of text and a chat like conversation using pyannote
@router.post("/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    audio: UploadFile = File(...),
):
    try:
        audio_data = await audio.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio file: {str(e)}")

    # Use asyncio subprocess to run FFmpeg for conversion
    try:
        process = await asyncio.create_subprocess_exec(
            'ffmpeg', '-f', 'webm', '-i', 'pipe:0', '-f', 's16le', '-ac', '1', '-ar', '16000', 'pipe:1',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        out, err = await process.communicate(input=audio_data)

        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"FFmpeg conversion failed: {err.decode()}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")

    try:
        audio_np_array = np.frombuffer(out, dtype=np.int16)
        sample_rate = 16000
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio data conversion failed: {str(e)}")

    # Transcribe
    try:
        transcriber = VoskTranscriber()
        transcript = transcriber.transcribe((sample_rate, audio_np_array))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    return TranscriptionResponse(transcript=transcript)


@router.post("/v1/triage-summary", response_model=ESIAssessment)
async def triage_summary(request: TriageSummaryRequest, session_id: Optional[UUID] = None, db: DBSession = Depends(get_db)):
    option = {}

    if get_settings().online_mode:
        agent = HandbookRagOpenAiAgent('OpenAi Agent', option)
    else:
        agent = HandbookRagOllamaAgent('Ollama Agent', option)

    # Run the synchronous triage method in a thread pool to avoid blocking
    result = await asyncio.to_thread(agent.triage, conversation=MedicalConversation(turns=request.turns))

    # Save result to session summary if session_id provided and rationale does not contain "Error"
    if session_id and "Error" not in result.rationale:
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if session:
            session.summary = json.dumps(result.model_dump(mode='json'))
            session.updated_at = datetime.now(timezone.utc)
            db.commit()

    return result

    
        

# given a transcription text, return an array of chat like conversation between the nurse and patient
# we have the option of either using the conversation_analizer based on llama3.2
@router.post("/v1/conversation", response_model=ConversationResponse)
async def conversation(request: ConversationRequest, session_id: Optional[UUID] = None, db: DBSession = Depends(get_db)) -> ConversationResponse:
    try:
        model = MODEL_GPT_4O if get_settings().online_mode else MODEL_GEMMA_3
        conversation_result = ConversationAnalizer(model).analyze(request.transcript)

        # Update session with transcript and conversation if session_id is provided
        if session_id:
            session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
            if session:
                session.transcript = request.transcript
                session.conversation = json.dumps(conversation_result)
                session.updated_at = datetime.now(timezone.utc)
                db.commit()

        return ConversationResponse(**conversation_result)
    except Exception as e:
        print(e)
        return ConversationResponse(conversation=[])


# Session CRUD endpoints
@router.post("/v1/sessions", response_model=SessionResponse)
async def create_session(session: SessionCreate, db: DBSession = Depends(get_db)):
    db_session = SessionModel(**session.model_dump())
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

@router.get("/v1/sessions", response_model=List[SessionResponse])
async def list_sessions(db: DBSession = Depends(get_db)):
    return db.query(SessionModel).order_by(SessionModel.created_at.desc()).all()

@router.get("/v1/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: UUID, db: DBSession = Depends(get_db)):
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.put("/v1/sessions/{session_id}", response_model=SessionResponse)
async def update_session(session_id: UUID, session_update: SessionUpdate, db: DBSession = Depends(get_db)):
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    for key, value in session_update.model_dump(exclude_unset=True).items():
        setattr(session, key, value)
    session.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(session)
    return session

@router.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: UUID, db: DBSession = Depends(get_db)):
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    db.delete(session)
    db.commit()
    return {"message": "Session deleted"}

# AuditLog CRUD endpoints
@router.post("/v1/audit-logs", response_model=AuditLogResponse)
async def create_audit_log(audit_log: AuditLogCreate, db: DBSession = Depends(get_db)):
    model = MODEL_GPT_4O if get_settings().online_mode else MODEL_GEMMA_3
    try:
        result = ConversationAnalizer(model).similarity_analysis(audit_log.suggestion, audit_log.response)
        similarity = result.get('similarity', 5) if isinstance(result, dict) else 5
    except Exception as e:
        print(e)
        similarity = 5  # Default similarity in case of error

    db_audit_log = AuditLog(**audit_log.model_dump(exclude={'similarity'}), similarity=similarity)
    db.add(db_audit_log)
    db.commit()
    db.refresh(db_audit_log)
    return db_audit_log

@router.get("/v1/audit-logs", response_model=List[AuditLogResponse])
async def list_audit_logs(db: DBSession = Depends(get_db)):
    return db.query(AuditLog).order_by(AuditLog.created_at.desc()).all()

@router.get("/v1/audit-logs/{audit_log_id}", response_model=AuditLogResponse)
async def get_audit_log(audit_log_id: UUID, db: DBSession = Depends(get_db)):
    audit_log = db.query(AuditLog).filter(AuditLog.id == audit_log_id).first()
    if not audit_log:
        raise HTTPException(status_code=404, detail="Audit log not found")
    return audit_log

@router.put("/v1/audit-logs/{audit_log_id}", response_model=AuditLogResponse)
async def update_audit_log(audit_log_id: UUID, audit_log_update: AuditLogUpdate, db: DBSession = Depends(get_db)):
    audit_log = db.query(AuditLog).filter(AuditLog.id == audit_log_id).first()
    if not audit_log:
        raise HTTPException(status_code=404, detail="Audit log not found")
    for key, value in audit_log_update.model_dump(exclude_unset=True).items():
        setattr(audit_log, key, value)
    db.commit()
    db.refresh(audit_log)
    return audit_log

@router.delete("/v1/audit-logs/{audit_log_id}")
async def delete_audit_log(audit_log_id: UUID, db: DBSession = Depends(get_db)):
    audit_log = db.query(AuditLog).filter(AuditLog.id == audit_log_id).first()
    if not audit_log:
        raise HTTPException(status_code=404, detail="Audit log not found")
    db.delete(audit_log)
    db.commit()
    return {"message": "Audit log deleted"}
